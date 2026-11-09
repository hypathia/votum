"""Reference PyTorch implementation of Votum (transformers + PEFT; runs on GPU or CPU).

Model = the Apertus-1.5-8B text stack, loaded with transformers' mainline `apertus` classes, with the Votum LoRA either
merged into its weights (the default: the Votum repository) or applied with PEFT on the swiss-ai base (exact=True: the
repository's adapter/ folder; key mapping model.language_model.* -> *), plus a pointer head:
    score_j = < q0, W_o LN(h_end_j) > / sqrt(r),   r = 256,   LN = LayerNorm without affine (fp32)
over the final-norm hidden states h at every option-end token; softmax(score / T[type]) gives the option probabilities.

Attention (one dense boolean mask per layer kind, applied through SDPA by a forward pre-hook on every attention module):
    causal layers          past & ~cross
    bidirectional layers   past & ~cross  |  future & ~cross & ~(non-option query -> option key)
                                                    & ~(state-block query -> question-block key)
                                                    & ~(system/header query -> state key)
    cross = query and key in two different option blocks. Bidirectional layers: config.bidirectional_layers (3, 7, ..., 31).
Option blocks share position ids (prompt.shared_positions). This is the training-time rule exactly; the vLLM plugin
(votum.vllm) computes the same function with FlexAttention and packs the questions of a request around one state block.
"""
from __future__ import annotations

import inspect, json, math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import prompt as P
from .build import HEAD_NAME, TOKENIZER_FILES, adapter_dir, resolve
from .config import CONFIG_NAME, VotumConfig


def attention_masks(encs: list[P.Encoding], cfg: VotumConfig, T: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """(causal-layer mask, bidirectional-layer mask), bool [B, 1, T, T] (True = visible), right padding."""
    B = len(encs); i = torch.arange(T, device=device)
    blk = torch.full((B, T), -1, dtype=torch.long, device=device)
    lens = torch.tensor([len(e.ids) for e in encs], device=device)
    S = torch.full((B,), T, dtype=torch.long, device=device); H = torch.full((B,), T, dtype=torch.long, device=device)
    for b, e in enumerate(encs):
        blk[b, :len(e.ids)] = torch.tensor(P.option_blocks(e, cfg.close_last), device=device)
        if e.qstart is not None: S[b] = e.qstart
        if e.hend is not None: H[b] = e.hend
    q, k = i[None, :, None], i[None, None, :]
    past = k <= q
    bq, bk = blk[:, :, None], blk[:, None, :]
    cross = (bq >= 0) & (bk >= 0) & (bq != bk)
    fut = ~past & ~cross & ~((bq < 0) & (bk >= 0))
    fut = fut & ~((q < S[:, None, None]) & (k >= S[:, None, None])) & ~((q < H[:, None, None]) & (k >= H[:, None, None]))
    real = k < lens[:, None, None]
    causal = past & ~cross & real
    bidir = ((past & ~cross) | fut) & real
    pad_q = ~(q < lens[:, None, None]).expand(B, T, T)          # padding query rows: see key 0 only (outputs discarded)
    first = (k == 0).expand(B, T, T)
    causal = torch.where(pad_q, first, causal); bidir = torch.where(pad_q, first, bidir)
    return causal[:, None], bidir[:, None]


def _text_config(base_dir: Path):
    from transformers import ApertusConfig
    d = json.loads((base_dir / "config.json").read_text()); d = dict(d.get("text_config", d))
    for k in ("model_type", "architectures", "output_vocab_size", "transformers_version", "hidden_dropout", "post_norm", "qk_norm", "mlp_bias"):
        d.pop(k, None)
    params = inspect.signature(ApertusConfig.__init__).parameters
    if "rope_parameters" in d and "rope_parameters" not in params:          # transformers 4.x: rope_theta + rope_scaling
        rp = dict(d.pop("rope_parameters")); d["rope_theta"] = rp.pop("rope_theta"); d["rope_scaling"] = rp
    if "rope_theta" in d and "rope_parameters" in params:                    # transformers 5.x: one rope_parameters dict
        d["rope_parameters"] = {**(d.pop("rope_scaling", None) or {}), "rope_theta": d.pop("rope_theta")}
    if "dtype" in d and "dtype" not in params: d["torch_dtype"] = d.pop("dtype")
    if not any(v.kind == v.VAR_KEYWORD for v in params.values()): d = {k: v for k, v in d.items() if k in params or k == "rope_parameters"}
    cfg = ApertusConfig(**d); cfg._attn_implementation = "sdpa"; cfg.use_cache = False
    return cfg


def load_backbone(base: str, revision: str | None, device, dtype=torch.bfloat16, token=None):
    """the Apertus-1.5-8B text stack (ApertusModel) from the multimodal swiss-ai checkpoint (or a local copy of it)."""
    from transformers import ApertusModel
    d = resolve(base, revision, token, ["config.json", "model.safetensors.index.json", "*.safetensors", *TOKENIZER_FILES])
    cfg = _text_config(d)
    kw = dict(config=cfg, key_mapping={r"^model\.language_model\.": ""}, device_map={"": str(device)})
    kw["dtype" if _tf_major() >= 5 else "torch_dtype"] = dtype
    m = ApertusModel.from_pretrained(str(d), **kw)
    m.eval(); m.requires_grad_(False)
    return m, d


def _tf_major() -> int:
    import transformers
    return int(transformers.__version__.split(".")[0])


class _Wrap(nn.Module):
    """gives the backbone the module path of the training-time causal LM (base_model.model.model.layers.*), so the PEFT
    adapter's keys load unchanged."""

    def __init__(self, backbone):
        super().__init__(); self.model = backbone


def _load(p: Path) -> dict:
    from safetensors.torch import load_file
    return load_file(str(p))


def _with_adapter(backbone, ad: Path):
    """the backbone with the PEFT LoRA adapter of directory ad applied (unmerged, fp32 adapter weights)."""
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    lcfg = LoraConfig.from_pretrained(str(ad)); lcfg.task_type = None; lcfg.inference_mode = True
    lcfg.base_model_name_or_path = None                    # the backbone is passed in, not loaded by PEFT
    net = get_peft_model(_Wrap(backbone), lcfg)
    sd = _load(ad / "adapter_model.safetensors")
    res = set_peft_model_state_dict(net, sd)
    if getattr(res, "unexpected_keys", None): raise ValueError(f"adapter keys not used: {res.unexpected_keys[:4]} ...")
    n_lora = sum(1 for n, _ in net.named_parameters() if "lora_" in n)
    if n_lora != len(sd): raise ValueError(f"adapter: {len(sd)} tensors in the file, {n_lora} LoRA parameters in the model")
    return net


class Votum(nn.Module):
    def __init__(self, wrapped, tokenizer, cfg: VotumConfig, o_weight: torch.Tensor, q0: torch.Tensor):
        super().__init__()
        self.net, self.tok, self.cfg = wrapped, tokenizer, cfg
        self.o = nn.Parameter(o_weight.float(), requires_grad=False); self.q0 = nn.Parameter(q0.float(), requires_grad=False)
        self._masks = None
        bidir = set(cfg.bidirectional_layers)
        for li, layer in enumerate(self.backbone.layers):
            attn = layer.self_attn; attn.is_causal = False
            attn.register_forward_pre_hook(self._hook(li in bidir), with_kwargs=True)

    @property
    def backbone(self):
        m = self.net
        while not hasattr(m, "layers"): m = m.model if hasattr(m, "model") else m.base_model
        return m

    @property
    def device(self):
        return self.o.device

    def _hook(self, bidir: bool):
        def pre(module, args, kwargs):
            if self._masks is None: return args, kwargs
            m = self._masks[1] if bidir else self._masks[0]
            if "attention_mask" in kwargs or len(args) <= 2: kwargs["attention_mask"] = m
            else: args = args[:2] + (m,) + args[3:]
            return args, kwargs
        return pre

    @classmethod
    def from_pretrained(cls, repo: str, *, exact: bool = False, base: str | None = None, revision: str | None = None,
                        device: str | None = None, dtype=torch.bfloat16, token=None) -> "Votum":
        """repo: the Votum repo id or a local checkpoint directory (votum build). Default: the merged weights in the repo
        (one download, no PEFT). exact=True: the unmerged form, the Apertus base (base: repo id or local snapshot; default
        the revision pinned in votum_config.json) plus the LoRA adapter in the repo's adapter/ folder, as in training."""
        from safetensors import safe_open
        from transformers import ApertusModel, AutoTokenizer
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if exact:
            ad = adapter_dir(repo, token); cfg = VotumConfig.load(ad)
            bb, bd = load_backbone(base or cfg.base_model, None if base else cfg.base_revision, device, dtype, token)
            net = _with_adapter(bb, ad); tok = AutoTokenizer.from_pretrained(str(bd))
            head = _load(ad / HEAD_NAME)
            o, q0 = head["o.weight"], head["q0"]
        else:
            rd = resolve(repo, revision, token, ["config.json", CONFIG_NAME, "model*.safetensors", "model.safetensors.index.json", *TOKENIZER_FILES])
            cfg = VotumConfig.load(rd); tcfg = _text_config(rd)
            kw = dict(config=tcfg, device_map={"": str(device)}); kw["dtype" if _tf_major() >= 5 else "torch_dtype"] = dtype
            bb = ApertusModel.from_pretrained(str(rd), **kw); bb.eval(); bb.requires_grad_(False)
            net = _with_adapter(bb, rd / "lora") if (rd / "lora" / "adapter_config.json").exists() else _Wrap(bb)   # votum build --exact
            tok = AutoTokenizer.from_pretrained(str(rd))
            wmap = json.loads((rd / "model.safetensors.index.json").read_text())["weight_map"]
            with safe_open(str(rd / wmap["pointer.o.weight"]), framework="pt") as f: o, q0 = f.get_tensor("pointer.o.weight"), f.get_tensor("pointer.q0")
        if tok.convert_tokens_to_ids(cfg.opt_end_token) != cfg.opt_end_id:
            raise ValueError(f"tokenizer maps {cfg.opt_end_token} to {tok.convert_tokens_to_ids(cfg.opt_end_token)}, expected {cfg.opt_end_id}")
        net.to(device); net.eval()
        return cls(net, tok, cfg, o, q0).to(device)

    # ------------------------------------------------------------------------------------------------------ forward
    @torch.no_grad()
    def score_encodings(self, encs: list[P.Encoding], max_batch_tokens: int = 16384) -> list[torch.Tensor]:
        """raw option scores (fp32, the row's option order) per encoding; one forward pass per question, batched by length."""
        out: list = [None] * len(encs)
        order = sorted(range(len(encs)), key=lambda i: len(encs[i].ids))
        batch: list = []
        for i in order + [None]:
            if i is not None and (not batch or (len(batch) + 1) * len(encs[i].ids) <= max_batch_tokens):
                batch.append(i); continue
            if batch:
                for j, z in zip(batch, self._forward([encs[j] for j in batch])): out[j] = z
            batch = [i] if i is not None else []
        return out

    def _forward(self, encs: list[P.Encoding]) -> list[torch.Tensor]:
        dev = self.device; T = max(len(e.ids) for e in encs); B = len(encs)
        ids = torch.full((B, T), int(self.tok.pad_token_id or 0), dtype=torch.long)
        pos = torch.zeros((B, T), dtype=torch.long)
        for b, e in enumerate(encs):
            n = len(e.ids); ids[b, :n] = torch.tensor(e.ids)
            p = P.shared_positions(e, self.cfg.close_last) if self.cfg.shared_positions else list(range(n))
            pos[b, :n] = torch.tensor(p)
            if n < T: pos[b, n:] = torch.arange(T - n) + max(p) + 1
        ids, pos = ids.to(dev), pos.to(dev)
        self._masks = attention_masks(encs, self.cfg, T, dev)
        try:
            h = self.backbone(input_ids=ids, position_ids=pos, use_cache=False).last_hidden_state
        finally:
            self._masks = None
        out = []
        for b, e in enumerate(encs):
            he = F.layer_norm(h[b, e.ends].float(), (h.shape[-1],))
            out.append((he @ self.o.T) @ self.q0 / math.sqrt(self.cfg.pointer_rank))
        return out

    # ------------------------------------------------------------------------------------------------------ decisions
    def encode(self, rows: list[dict], keys: list | None = None) -> list[P.Encoding | None]:
        """render every row with its request header (rows sharing a key and a state form one request); None for a row
        that does not fit in cfg.max_tokens or cannot be rendered."""
        h = self.cfg.header
        heads = P.request_headers(rows, keys, int(h.get("max_q", 8)), h.get("max_opts"), bool(h.get("sort", True)), bool(h.get("chunk", False)))
        out = []
        for r, hd in zip(rows, heads):
            try:
                e = P.render(self.tok, r, header=hd)
                out.append(e if len(e.ids) <= self.cfg.max_tokens else None)
            except ValueError:
                out.append(None)
        return out

    def decide_rows(self, rows: list[dict], keys: list | None = None) -> list[dict | None]:
        """rows {"type", "state", "question", "options": [{"id", "text"?}]} -> {"ids", "scores", "probs", "n_tokens"} per
        row (in the row's option order), or None for a row that does not fit."""
        encs = self.encode(rows, keys); live = [i for i, e in enumerate(encs) if e is not None]
        zs = self.score_encodings([encs[i] for i in live]); out: list = [None] * len(rows)
        for i, z in zip(live, zs):
            e, r = encs[i], rows[i]; T = float(self.cfg.temperatures.get(r["type"], 1.0))
            p = torch.softmax(z / T, 0).tolist(); ids = [str(o["id"]) for o in r["options"]]
            out[i] = {"ids": ids, "scores": z.tolist(), "probs": dict(zip(ids, p)), "n_tokens": len(e.ids)}
        return out

    def decide(self, request: dict) -> dict:
        """one request in the SystemOne shape (votum.request) -> {"answers": {qid: answer}, "usage": {...}}."""
        from . import request as R
        return R.decide(request, self.decide_rows, self.cfg.max_tokens)
