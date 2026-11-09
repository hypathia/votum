"""HTTP API for Votum on vLLM (request and response shapes of TypeSafe / SystemOne, see votum.request).

    votum serve --model <org>/Votum-8B | <dir> [--name votum-8b] [--host 127.0.0.1] [--port 8011] [--api-key-env VOTUM_API_KEY]

POST /v1/systemone   {"model": str, "state": str | object, "questions": {qid: {"type", "instructions", "criteria"}}}
                  -> {"model": str, "answers": {qid: answer}, "usage": {"input_tokens": int, "output_tokens": 0}}
GET  /v1/models, GET /health

The questions of one request share one header and one state block (one packed prompt). Concurrent requests are batched:
a collector gathers whatever arrives within --batch-window-ms (or up to --max-batch questions) and runs one vLLM call.
An API key is required only if the named environment variable is set (Authorization: Bearer <key>).
Limits (environment): VOTUM_MAX_BODY_MB (request body, default 8), VOTUM_MAX_QUESTIONS (per request, default 64),
VOTUM_MAX_OPTIONS (per question, default 2000), VOTUM_MAX_QUEUE (questions waiting, default 4096; beyond it: HTTP 503).
"""

import argparse, sys, asyncio, hmac, json, logging, os, threading, time, uuid

log = logging.getLogger("votum.serve")

from .. import request as R


class Batcher:
    """collects the requests of concurrent callers and runs them through one vLLM call per batch (worker thread)."""

    def __init__(self, vm, window_ms: float, max_batch: int, max_queue: int = 4096) -> None:
        self.vm, self.window, self.max_batch, self.max_queue = vm, window_ms / 1000, max_batch, max_queue
        self.q: list = []; self.cv = threading.Condition()
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, rows: list[dict]) -> asyncio.Future | None:
        """None when the queue is full (the caller answers 503)."""
        loop = asyncio.get_running_loop()
        with self.cv:
            if sum(len(x[0]) for x in self.q) + len(rows) > self.max_queue: return None
            fut = loop.create_future(); self.q.append((rows, uuid.uuid4().hex, fut, loop)); self.cv.notify()
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


def _env_int(name: str, default: int) -> int:
    try: return max(1, int(float(os.environ.get(name, default))))
    except ValueError: sys.exit(f"votum serve: {name} must be a number")


async def _body(request, limit: int) -> bytes | None:
    """the request body, or None if it is larger than limit bytes (checked while reading, not only from the header)."""
    try:
        if int(request.headers.get("content-length", 0)) > limit: return None
    except ValueError: pass
    buf = bytearray()
    async for chunk in request.stream():
        buf += chunk
        if len(buf) > limit: return None
    return bytes(buf)


def build_app(vm, model_name: str, api_key: str | None, window_ms: float, max_batch: int):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    max_body = _env_int("VOTUM_MAX_BODY_MB", 8) * 1024 * 1024
    max_q, max_opts = _env_int("VOTUM_MAX_QUESTIONS", 64), _env_int("VOTUM_MAX_OPTIONS", 2000)
    app = FastAPI(title="Votum"); batcher = Batcher(vm, window_ms, max_batch, _env_int("VOTUM_MAX_QUEUE", 4096))
    want = f"Bearer {api_key}".encode() if api_key else None

    def err(code, msg): return JSONResponse({"error": {"message": msg}}, status_code=code)

    @app.get("/health")
    async def health(): return {"status": "ok"}

    @app.get("/v1/models")
    async def models(): return {"object": "list", "data": [{"id": model_name, "object": "model", "owned_by": "local"}]}

    @app.post("/v1/systemone")
    async def systemone(request: Request):
        if want and not hmac.compare_digest(request.headers.get("authorization", "").encode(), want): return err(401, "invalid API key")
        raw = await _body(request, max_body)
        if raw is None: return err(413, f"request body larger than {max_body // (1024 * 1024)} MB (VOTUM_MAX_BODY_MB)")
        try: req = json.loads(raw)
        except Exception: return err(400, "body must be JSON")
        if (msg := R.validate(req, max_q, max_opts)): return err(400, msg)
        qs = req["questions"]
        try: rows = [R.to_row(req["state"], q, rid=str(qid)) for qid, q in qs.items()]
        except Exception: return err(400, "the request could not be read as questions and options")
        fut = batcher.submit(rows)
        if fut is None: return err(503, "the server is busy (queue full); retry shortly")
        try: res = await fut
        except Exception:    # the engine failed: the details go to the log, not to the caller
            log.exception("engine error"); return err(500, "internal error while scoring the request")
        answers, used = {}, 0
        for (qid, q), r in zip(qs.items(), res):
            if r is None: return err(413, f"{qid}: prompt longer than the maximum model length ({vm.max_model_len} tokens)")
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
    import torch
    if not torch.cuda.is_available(): sys.exit("votum serve needs a CUDA GPU (vLLM); none is visible to torch")
    import uvicorn
    from .client import VotumVLLM
    vm = VotumVLLM(a.model, max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_memory_utilization, max_num_seqs=a.max_num_seqs)
    key = os.environ.get(a.api_key_env)
    if not key and a.host not in ("127.0.0.1", "localhost", "::1"): print(f"votum serve: no API key ({a.api_key_env} unset) on a non-local host {a.host}: anyone who can reach the port can use the model", file=sys.stderr, flush=True)
    print(f"votum serve: ready at http://{a.host}:{a.port}/v1/systemone (model {a.name!r}, auth {'Bearer key from $' + a.api_key_env if key else 'off'})", flush=True)
    uvicorn.run(build_app(vm, a.name, key, a.batch_window_ms, a.max_batch), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
