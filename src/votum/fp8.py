"""FP8 weights for Votum: make a quantized checkpoint, and load one in PyTorch.

    votum quantize --model <org>/Votum-8B --out Votum-8B-FP8          # the published variant; also for your own fine-tune
    votum quantize --model <dir> --out <dir> --activations [--ignore-layers 0,1] [--ignore-proj up_proj,o_proj]

The output is a checkpoint in the compressed-tensors format. `votum serve` and `Votum.from_pretrained` both load it and
need no flag. Weights of the attention and MLP projections become e4m3 with one fp32 scale per output channel
(scale = max |w| / 448); the activations stay bf16, so memory roughly halves and speed stays about the same.
--activations also quantizes the activations (faster on recent GPUs, but more answers change, and only vLLM runs it).
--ignore-layers / --ignore-proj keep layers or projection types in bf16. The pointer head, the norms, the xIELU
parameters and the embeddings are never quantized. The output keeps the decision_* fields of config.json,
votum_config.json and the tokenizer; it has no adapter/ folder. The conversion is deterministic and runs on the CPU,
shard by shard, in about a minute.
"""
from __future__ import annotations

import argparse, json, shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .build import TOKENIZER_FILES, resolve

FP8_MAX = 448.0
PROJ = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.up_proj", "mlp.down_proj")
COPY = ["votum_config.json", "LICENSE", "NOTICE", *TOKENIZER_FILES]


def quantize(model: str, out: str | Path, activations: bool = False, ignore_layers: set[int] | None = None,
             ignore_proj: set[str] | None = None, token=None) -> Path:
    from safetensors import safe_open
    from safetensors.torch import save_file
    ign_layers, ign_proj = set(ignore_layers or ()), set(ignore_proj or ())
    src = resolve(model, None, token, ["config.json", "votum_config.json", "model*.safetensors", "model.safetensors.index.json", "LICENSE", "NOTICE", *TOKENIZER_FILES])
    cfg = json.loads((src / "config.json").read_text())
    if cfg.get("quantization_config"): raise SystemExit(f"votum quantize: {model} is already quantized; start from the full-precision checkpoint")
    if (src / "lora" / "adapter_config.json").exists():
        raise SystemExit(f"votum quantize: {model} keeps its adapter unmerged (votum build --exact); quantize a merged checkpoint")
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    idx = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]; shards = sorted(set(idx.values()))

    def quantized(k: str) -> bool:
        if not (k.startswith("model.layers.") and any(k.endswith(p + ".weight") for p in PROJ)): return False
        return int(k.split(".")[2]) not in ign_layers and k.split(".")[-2] not in ign_proj

    new_map, n_q, n_keep, total = {}, 0, 0, 0
    for si, sh in enumerate(shards):
        tensors = {}
        with safe_open(str(src / sh), "pt") as f:
            for k in f.keys():
                w = f.get_tensor(k)
                if quantized(k):
                    wf = w.float(); s = wf.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / FP8_MAX
                    tensors[k] = (wf / s).to(torch.float8_e4m3fn).contiguous(); tensors[k[:-len("weight")] + "weight_scale"] = s.contiguous(); n_q += 1
                else:
                    tensors[k] = w.contiguous(); n_keep += 1
        name = f"model-{si + 1:05d}-of-{len(shards):05d}.safetensors"
        save_file(tensors, str(out / name), metadata={"format": "pt"}); total += sum(t.numel() * t.element_size() for t in tensors.values())
        new_map.update({k: name for k in tensors}); print(f"{name}: {len(tensors)} tensors", flush=True)
    json.dump({"metadata": {"total_size": total}, "weight_map": new_map}, open(out / "model.safetensors.index.json", "w"), indent=1)

    ignore = ["model.embed_tokens", "lm_head", "pointer.o"] + [f"model.layers.{i}.{p}" for i in sorted(ign_layers) for p in PROJ] + [f"re:.*\\.{p}$" for p in sorted(ign_proj)]
    acts = {"num_bits": 8, "type": "float", "symmetric": True, "strategy": "token", "dynamic": True, "observer": None} if activations else None
    cfg["quantization_config"] = {
        "quant_method": "compressed-tensors", "format": "float-quantized", "quantization_status": "compressed", "kv_cache_scheme": None,
        "sparsity_config": {}, "ignore": ignore,
        "config_groups": {"group_0": {"targets": ["Linear"], "weights": {"num_bits": 8, "type": "float", "symmetric": True, "strategy": "channel", "dynamic": False, "observer": "minmax"},
                                      "input_activations": acts, "output_activations": None}},
        "votum_fp8": {"source": Path(model).name,   # the name only: a local path must not end up in a published config
                      "weights": "fp8 e4m3, per output channel", "activations": "fp8 per token (dynamic)" if activations else "bf16",
                      "ignore_layers": sorted(ign_layers), "ignore_proj": sorted(ign_proj)}}
    (out / "config.json").write_text(json.dumps(cfg, indent=1) + "\n")
    for f in COPY:
        if (src / f).exists(): shutil.copy(src / f, out / f)
    print(f"{n_q} projections quantized, {n_keep} tensors kept in their dtype; {total / 1e9:.1f} GB -> {out}")
    print(f"serve it:  votum serve --model {out}")
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="votum quantize", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Votum checkpoint: Hub repo id or local directory (merged bf16 weights)")
    ap.add_argument("--out", required=True, help="directory to write the FP8 checkpoint to")
    ap.add_argument("--activations", action="store_true", help="also quantize activations (W8A8, dynamic per token); default weights only (W8A16)")
    ap.add_argument("--ignore-layers", default="", help="comma-separated layer indices kept in bf16")
    ap.add_argument("--ignore-proj", default="", help="comma-separated projections kept in bf16 in every layer, e.g. up_proj,o_proj")
    ap.add_argument("--token", default=None)
    a = ap.parse_args(argv)
    quantize(a.model, a.out, a.activations, {int(x) for x in a.ignore_layers.split(",") if x.strip()},
             {x.strip() for x in a.ignore_proj.split(",") if x.strip()}, a.token)


# ---------------------------------------------------------------------------------------------- PyTorch loading
class FP8Linear(nn.Module):
    """a linear layer without bias whose weight stays in FP8 (e4m3, one fp32 scale per output channel). The weight is
    expanded for each call and freed again, so the model keeps the memory footprint of the file."""

    def __init__(self, weight: torch.Tensor, scale: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("weight", weight); self.register_buffer("weight_scale", scale.float())
        self.out_features, self.in_features = weight.shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, (self.weight.float() * self.weight_scale).to(x.dtype))

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, fp8"


def is_quantized(rd: Path) -> bool:
    return bool(json.loads((Path(rd) / "config.json").read_text()).get("quantization_config"))


def load_backbone(rd: Path, tcfg, device, dtype=torch.bfloat16):
    """the text stack of a weight-only FP8 Votum checkpoint: quantized projections as FP8Linear, the rest in dtype."""
    from accelerate import init_empty_weights
    from safetensors import safe_open
    from transformers import ApertusModel
    qc = json.loads((rd / "config.json").read_text())["quantization_config"]
    groups = list((qc.get("config_groups") or {}).values())
    if qc.get("format") != "float-quantized" or any(g.get("input_activations") or g.get("output_activations") for g in groups):
        raise ValueError(f"{rd}: the PyTorch path loads weight-only FP8 checkpoints (votum quantize without --activations). "
                         "Serve this one with `votum serve`.")
    reset = {c: c.reset_parameters for c in (nn.Linear, nn.Embedding)}
    try:                                       # the weights come from the file: skip the random initialisation of 8B values
        for c in reset: c.reset_parameters = lambda self: None
        with init_empty_weights(): bb = ApertusModel(tcfg)
    finally:
        for c, f in reset.items(): c.reset_parameters = f
    wmap = json.loads((rd / "model.safetensors.index.json").read_text())["weight_map"]
    sd, q, scales = {}, {}, {}
    for sh in sorted(set(wmap.values())):
        with safe_open(str(rd / sh), framework="pt", device=str(device)) as f:
            for k in f.keys():
                if not k.startswith("model."): continue          # the pointer head is read by the caller
                name, t = k[len("model."):], f.get_tensor(k)
                if name.endswith(".weight_scale"): scales[name[:-len(".weight_scale")]] = t
                elif t.dtype == torch.float8_e4m3fn: q[name[:-len(".weight")]] = t
                else: sd[name] = t.to(dtype) if t.is_floating_point() else t
    if set(q) != set(scales): raise ValueError(f"{rd}: FP8 weights and their scales do not match")
    for name, w in q.items():
        parent, _, leaf = name.rpartition(".")
        setattr(bb.get_submodule(parent), leaf, FP8Linear(w, scales[name]))
    res = bb.load_state_dict(sd, strict=False, assign=True)
    missing = [k for k in res.missing_keys if not k.endswith((".weight_scale",)) and k.rpartition(".")[0] not in q]
    if missing or res.unexpected_keys:
        raise ValueError(f"{rd}: weights do not fit the model (missing {missing[:3]}, unexpected {list(res.unexpected_keys)[:3]})")
    bb.eval(); bb.requires_grad_(False)
    return bb


if __name__ == "__main__":
    main()
