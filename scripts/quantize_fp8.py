"""Quantize a Votum checkpoint to FP8 in the compressed-tensors format that vLLM loads.

    python scripts/quantize_fp8.py --model {{ORG}}/Votum-8B --out Votum-8B-FP8                 # the published variant: weights only
    python scripts/quantize_fp8.py --model <dir> --out <dir> --activations [--ignore-layers 0,1] [--ignore-proj up_proj,o_proj]

Weights of the attention and MLP projections become e4m3 with one fp32 scale per output channel (scale = max |w| / 448).
By default the activations stay bf16 (W8A16: vLLM runs the Marlin FP8 kernel; memory halves, speed is about bf16).
--activations adds dynamic per-token FP8 activations (W8A8: faster on Ada and Hopper, but it changes more answers; see the
model card of Votum-8B-FP8 for the measurements). --ignore-layers / --ignore-proj keep layers or projection types in
bf16. The pointer head, the norms, the xIELU parameters and the embeddings are never quantized. The output keeps the
decision_* fields of config.json, votum_config.json and the tokenizer; it has no adapter/ folder (the exact form stays
in the source repo). The conversion is deterministic, so the published variant can be rebuilt byte for byte.
Runs on the CPU, shard by shard; needs torch and safetensors."""
from __future__ import annotations

import argparse, json, shutil, sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from votum.build import TOKENIZER_FILES, resolve  # noqa: E402

FP8_MAX = 448.0
PROJ = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.up_proj", "mlp.down_proj")
COPY = ["votum_config.json", "LICENSE", "NOTICE", *TOKENIZER_FILES]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Votum checkpoint: Hub repo id or local directory (merged bf16 weights)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--activations", action="store_true", help="also quantize activations (W8A8, dynamic per token); default weights only (W8A16)")
    ap.add_argument("--ignore-layers", default="", help="comma-separated layer indices kept in bf16")
    ap.add_argument("--ignore-proj", default="", help="comma-separated projections kept in bf16 in every layer, e.g. up_proj,o_proj")
    ap.add_argument("--token", default=None)
    a = ap.parse_args(argv)
    src = resolve(a.model, None, a.token, ["config.json", "votum_config.json", "model*.safetensors", "model.safetensors.index.json", "LICENSE", "NOTICE", *TOKENIZER_FILES])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((src / "config.json").read_text())
    ign_layers = {int(x) for x in a.ignore_layers.split(",") if x.strip()}; ign_proj = {x.strip() for x in a.ignore_proj.split(",") if x.strip()}
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
    acts = {"num_bits": 8, "type": "float", "symmetric": True, "strategy": "token", "dynamic": True, "observer": None} if a.activations else None
    cfg["quantization_config"] = {
        "quant_method": "compressed-tensors", "format": "float-quantized", "quantization_status": "compressed", "kv_cache_scheme": None,
        "sparsity_config": {}, "ignore": ignore,
        "config_groups": {"group_0": {"targets": ["Linear"], "weights": {"num_bits": 8, "type": "float", "symmetric": True, "strategy": "channel", "dynamic": False, "observer": "minmax"},
                                      "input_activations": acts, "output_activations": None}},
        "votum_fp8": {"source": a.model, "weights": "fp8 e4m3, per output channel", "activations": "fp8 per token (dynamic)" if a.activations else "bf16",
                      "ignore_layers": sorted(ign_layers), "ignore_proj": sorted(ign_proj)}}
    (out / "config.json").write_text(json.dumps(cfg, indent=1) + "\n")
    for f in COPY:
        if (src / f).exists(): shutil.copy(src / f, out / f)
    print(f"{n_q} projections quantized, {n_keep} tensors kept in their dtype; {total / 2**30:.2f} GiB -> {out}")


if __name__ == "__main__":
    main()
