"""Offline vLLM client for Votum.

    from votum.vllm import VotumVLLM
    vm = VotumVLLM("<org>/Votum-8B")                # the Hub repo, or a directory built by `votum build`
    vm.decide_rows([{"type": "choice", "state": "...", "question": "...", "options": [{"id": "a", "text": "..."}, ...]}])
    -> [{"ids": [...], "scores": [...], "probs": {"a": 0.91, ...}, "n_tokens": 312}]

Prompts are rendered by votum.prompt (the same renderer as the PyTorch reference) with their request header, then the questions that share a
state block (system turn + header + state) are packed into one prompt [header][state][question 1]...[question J] up to
max_model_len tokens (isolate.build_prompts); a persistent prefix cache reuses header and state blocks across calls.
Probabilities = softmax(scores / T[type]) with the per-type temperatures of the serving config. The engine runs without
chunked prefill and without vLLM's prefix caching: a bidirectional layer needs the whole prompt in one step.
"""
from __future__ import annotations

import json, math, os
from pathlib import Path
from typing import Any

from .. import prompt as P
from .. import request as R

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")   # the client registers prompt layouts in-process


def register() -> None:
    """vLLM plugin hook (entry point vllm.general_plugins) and in-process registration."""
    from vllm import ModelRegistry
    if "VotumDecisionModel" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("VotumDecisionModel", "votum.vllm.model:VotumDecisionModel")


class VotumVLLM:
    def __init__(self, path: str | Path, *, max_model_len: int = 8192, gpu_memory_utilization: float = 0.75,
                 max_num_seqs: int = 256, pack: bool = True, prefix_cache_mb: float | None = None,
                 state_cache_mb: float | None = None, warmup: bool = True, **engine_kwargs: Any) -> None:
        from transformers import AutoTokenizer
        from vllm import LLM
        from vllm.lora.request import LoRARequest
        from ..build import TOKENIZER_FILES, resolve
        from . import isolate
        register()
        self.path = resolve(str(path), None, None, ["config.json", "votum_config.json", "model*.safetensors", "model.safetensors.index.json",
                                                    "lora/*", *TOKENIZER_FILES])      # a local directory or a Hub repo (not its adapter/)
        self.cfg = json.loads((self.path / "config.json").read_text())
        self.T = self.cfg["decision_temperatures"]; h = self.cfg["decision_header"]
        self.header = dict(max_q=int(h.get("max_q", 8)), max_opts=h.get("max_opts"),
                           sort=bool(h.get("sort", True)), chunk=os.environ.get("VOTUM_HEADER_CHUNK", "1" if h.get("chunk") else "0") == "1")
        self.tok = AutoTokenizer.from_pretrained(self.path); self.max_model_len = max_model_len; self.pack = pack
        self.lora = None
        if (self.path / "lora" / "adapter_config.json").exists():   # unmerged adapter: served through vLLM's LoRA kernels
            r = json.loads((self.path / "lora" / "adapter_config.json").read_text())["r"]
            self.lora = LoRARequest("votum", 1, str(self.path / "lora"))
            engine_kwargs.update(enable_lora=True, max_lora_rank=max(8, r), max_loras=1)
            max_num_seqs = min(max_num_seqs, 32)   # LoRA buffers + warm-up peak do not fit a 24 GB card at 256 (measured)
        self.llm = LLM(model=str(self.path), runner="pooling", max_model_len=max_model_len, dtype="bfloat16",
                       enable_chunked_prefill=False, enable_prefix_caching=False,
                       max_num_batched_tokens=max(max_model_len, engine_kwargs.pop("max_num_batched_tokens", 8192)),
                       max_num_seqs=max_num_seqs, gpu_memory_utilization=gpu_memory_utilization, **engine_kwargs)
        mb = prefix_cache_mb if prefix_cache_mb is not None else float(os.environ.get("VOTUM_PREFIX_CACHE_MB", "512"))
        smb = state_cache_mb if state_cache_mb is not None else float(os.environ.get("VOTUM_STATE_CACHE_MB", "0"))
        isolate.CACHE = isolate.PrefixCache(mb, smb) if mb > 0 or smb > 0 else None
        if warmup: self.warmup()

    def warmup(self) -> None:
        """compile the attention kernels for every shape class once (short and long steps, one and several questions,
        with and without cached prefix blocks), so that no real request pays for a compilation."""
        words = "the request names a team a date an amount and a reason for the decision".split()
        for n_words, n_q in ((4, 1), (60, 1), (300, 1), (1500, 1), (60, 4), (300, 4), (1500, 8), (4, 3)):
            for rep in range(2):          # the second pass reuses cached header blocks
                st = " ".join(words[(i + rep) % len(words)] for i in range(n_words))
                rows = [{"type": "choice", "state": st, "question": f"Warm-up question {j}?",
                         "options": [{"id": "a", "text": "first"}, {"id": "b", "text": "second"}]} for j in range(n_q)]
                self.decide_rows(rows, keys=[f"warmup{n_words}-{n_q}-{rep}"] * n_q)

    def headers(self, rows: list[dict], keys: list | None = None) -> list[str | None]:
        h = self.header
        return P.request_headers(rows, keys, h["max_q"], h["max_opts"], h["sort"], h["chunk"])

    def scores(self, rows: list[dict], keys: list | None = None) -> list[dict | None]:
        """raw scores per row (None if it does not fit). keys: request id per row (rows of one request share a header);
        None = the whole call is one request."""
        from . import isolate
        hs = self.headers(rows, keys); encs = []
        for r, h in zip(rows, hs):
            try:
                e = P.render(self.tok, r, header=h)
                encs.append(e if len(e.ids) <= self.max_model_len else None)
            except ValueError:
                encs.append(None)
        cache = isolate.CACHE
        prompts, members, used, saved = isolate.build_prompts(encs, self.pack, self.max_model_len, cache)
        keys_reg = [isolate.register(ids, lay) for ids, lay in prompts]; ok = False
        try:
            outs = self.llm.encode([{"prompt_token_ids": ids} for ids, _ in prompts], pooling_task="token_classify", use_tqdm=False,
                                   lora_request=self.lora); ok = True
        finally:
            isolate.unregister(keys_reg)
            if cache is not None: cache.finish(used, saved, ok)
        res: list[dict | None] = [None] * len(rows)
        for mem, o in zip(members, outs):
            z = o.outputs.data.float().cpu().tolist(); k = 0
            need = sum(len(encs[i].ends) for i in mem)
            if len(z) != need: raise RuntimeError(f"packed prompt: {len(z)} scores for {need} options")
            for i in mem:
                e = encs[i]; n = len(e.ends)
                res[i] = dict(ids=[str(o["id"]) for o in rows[i]["options"]], scores=z[k:k + n], n_tokens=len(e.ids)); k += n
        return res

    def decide_rows(self, rows: list[dict], keys: list | None = None) -> list[dict | None]:
        out = self.scores(rows, keys)
        for r, o in zip(rows, out):
            if o is None: continue
            T = float(self.T.get(r["type"], 1.0)); z = [s / T for s in o["scores"]]; m = max(z); e = [math.exp(x - m) for x in z]; s = sum(e)
            o["probs"] = {i: v / s for i, v in zip(o["ids"], e)}
        return out

    def decide(self, request: dict) -> dict:
        """one request in the SystemOne shape (votum.request) -> {"answers": ..., "usage": ...}."""
        return R.decide(request, self.decide_rows, self.max_model_len)
