"""HTTP API for Votum on vLLM (request and response shapes of TypeSafe / SystemOne, see votum.request).

    votum serve --model <org>/Votum-8B | <dir> [--name votum-8b] [--host 127.0.0.1] [--port 8011] [--api-key-env VOTUM_API_KEY]

POST /v1/systemone   {"model": str, "state": str | object, "questions": {qid: {"type", "instructions", "criteria"}}}
                  -> {"model": str, "answers": {qid: answer}, "usage": {"input_tokens": int, "output_tokens": 0}}
GET  /v1/models, GET /health

The questions of one request share one header and one state block (one packed prompt). Concurrent requests are batched:
a collector gathers whatever arrives within --batch-window-ms (or up to --max-batch questions) and runs one vLLM call.
An API key is required only if the named environment variable is set (Authorization: Bearer <key>).
"""

import argparse, asyncio, os, threading, time, uuid

from .. import request as R


class Batcher:
    """collects the requests of concurrent callers and runs them through one vLLM call per batch (worker thread)."""

    def __init__(self, vm, window_ms: float, max_batch: int) -> None:
        self.vm, self.window, self.max_batch = vm, window_ms / 1000, max_batch
        self.q: list = []; self.cv = threading.Condition()
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, rows: list[dict]) -> asyncio.Future:
        loop = asyncio.get_running_loop(); fut = loop.create_future()
        with self.cv: self.q.append((rows, uuid.uuid4().hex, fut, loop)); self.cv.notify()
        return fut

    def _loop(self) -> None:
        while True:
            with self.cv:
                while not self.q: self.cv.wait()
                t_end = time.monotonic() + self.window
                while sum(len(x[0]) for x in self.q) < self.max_batch and (rem := t_end - time.monotonic()) > 0: self.cv.wait(rem)
                batch, n = [], 0
                while self.q and (not batch or n + len(self.q[0][0]) <= self.max_batch):
                    batch.append(self.q.pop(0)); n += len(batch[-1][0])
            rows, keys, spans = [], [], []
            for grp, key, _, _ in batch:
                spans.append((len(rows), len(rows) + len(grp))); rows += grp; keys += [key] * len(grp)
            try:
                outs, err = self.vm.decide_rows(rows, keys), None
            except Exception as e:   # surface engine errors to every waiting request
                outs, err = None, e
            for (_, _, fut, loop), (a, b) in zip(batch, spans):
                loop.call_soon_threadsafe(_resolve, fut, None if err else outs[a:b], err)


def _resolve(fut, o, e):
    if fut.done(): return
    fut.set_exception(e) if e else fut.set_result(o)


def build_app(vm, model_name: str, api_key: str | None, window_ms: float, max_batch: int):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    app = FastAPI(title="Votum"); batcher = Batcher(vm, window_ms, max_batch)

    def err(code, msg): return JSONResponse({"error": {"message": msg}}, status_code=code)

    @app.get("/health")
    async def health(): return {"status": "ok"}

    @app.get("/v1/models")
    async def models(): return {"object": "list", "data": [{"id": model_name, "object": "model", "owned_by": "local"}]}

    @app.post("/v1/systemone")
    async def systemone(request: Request):
        if api_key and request.headers.get("authorization", "") != f"Bearer {api_key}": return err(401, "invalid API key")
        try: req = await request.json()
        except Exception: return err(400, "body must be JSON")
        if (msg := R.validate(req)): return err(400, msg)
        qs = req["questions"]
        rows = [R.to_row(req["state"], q, rid=str(qid)) for qid, q in qs.items()]
        try: res = await batcher.submit(rows)
        except Exception as e: return err(500, str(e))
        answers, used = {}, 0
        for (qid, q), r in zip(qs.items(), res):
            if r is None: return err(413, f"{qid}: prompt longer than the maximum model length ({vm.max_model_len} tokens) or not renderable")
            answers[qid] = R.answer(q, r["probs"]); used += r["n_tokens"]
        return {"model": req.get("model", model_name), "answers": answers, "usage": {"input_tokens": used, "output_tokens": 0}}

    return app


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="votum serve"); ap.add_argument("--model", required=True); ap.add_argument("--name", default="votum-8b")
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8011); ap.add_argument("--api-key-env", default="VOTUM_API_KEY")
    ap.add_argument("--batch-window-ms", type=float, default=5.0); ap.add_argument("--max-batch", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=8192); ap.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    ap.add_argument("--max-num-seqs", type=int, default=256, help="sequences per engine step (exact/LoRA checkpoints use at most 32)")
    a = ap.parse_args(argv)
    import uvicorn
    from .client import VotumVLLM
    vm = VotumVLLM(a.model, max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_memory_utilization, max_num_seqs=a.max_num_seqs)
    uvicorn.run(build_app(vm, a.name, os.environ.get(a.api_key_env), a.batch_window_ms, a.max_batch), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
