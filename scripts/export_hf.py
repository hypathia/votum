"""Training checkpoint -> Hugging Face repo directory (weights and configs; the model card is written separately).

    python scripts/export_hf.py --ckpt <run>/ckpt_600 --out <dir> [--base <Apertus-1.5-8B dir>] [--temperatures '{...}']

Writes <out>/adapter/ (the exact form): adapter_config.json + adapter_model.safetensors (PEFT LoRA, fp32, unchanged values;
target modules restricted to those that exist in Apertus), pointer_head.safetensors (o.weight [256, 4096] and the constant
query q0 [256], fp32) and votum_config.json; then the merged checkpoint at <out>/ (votum.build: LoRA folded into the
base weights in bf16, pointer head, tokenizer, config.json, votum_config.json). The checkpoint's layout files
(opt_isolate.json, header.json, shared_positions.json, gate.pt, the constant query q0) are checked against the one
layout votum implements; any other checkpoint is refused."""
from __future__ import annotations

import argparse, hashlib, json, sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from votum.build import LORA, build  # noqa: E402
from votum.config import VotumConfig  # noqa: E402



def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=None, help="Apertus-1.5-8B snapshot (default: the pinned Hub revision)")
    ap.add_argument("--temperatures", default=None, help="JSON per question type (default: the released ones in VotumConfig)")
    a = ap.parse_args(); ck, root = Path(a.ckpt), Path(a.out); out = root / "adapter"; out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((ck / "train_meta.json").read_text())
    # layout checks: votum implements exactly this layout
    gate = torch.load(ck / "gate.pt", map_location="cpu")["beta"].tolist()
    assert all(b == 0.0 or b <= -1e8 for b in gate), "learned (non-binary) bidirectionality gate: not supported"
    bidir = [i for i, b in enumerate(gate) if b == 0.0]
    for f in ("opt_isolate.json", "shared_positions.json", "header.json"): assert (ck / f).exists(), f"{f} missing: layout not supported"
    hj = json.loads((ck / "header.json").read_text()); assert hj.get("mode") == "group" and hj.get("blind"), hj
    span = torch.load(ck / "span_head.pt", map_location="cpu"); assert "q0" in span, "checkpoint with a decision marker: not supported"
    # adapter
    acfg = json.loads((ck / "adapter" / "adapter_config.json").read_text())
    lora = load_file(str(ck / "adapter" / "adapter_model.safetensors"))
    targets = sorted({LORA.fullmatch(k)[3] for k in lora}); assert all(LORA.fullmatch(k) for k in lora)
    assert len(lora) == 2 * len(gate) * len(targets), (len(lora), targets)   # A and B per target per layer
    keep = ("r", "lora_alpha", "lora_dropout", "bias", "fan_in_fan_out", "init_lora_weights", "use_rslora", "use_dora", "peft_type", "peft_version")
    acfg2 = {k: acfg[k] for k in keep if k in acfg}
    defaults = VotumConfig()
    acfg2.update(base_model_name_or_path=defaults.base_model, revision=defaults.base_revision, target_modules=targets, task_type=None, inference_mode=True,
                 modules_to_save=None, layers_to_transform=None, layers_pattern=None, rank_pattern={}, alpha_pattern={})
    (out / "adapter_config.json").write_text(json.dumps(acfg2, indent=2, sort_keys=True) + "\n")
    save_file({k: v.contiguous() for k, v in lora.items()}, str(out / "adapter_model.safetensors"), metadata={"format": "pt"})
    # pointer head (fp32): o.weight and the constant query q0 (the checkpoint's unused query projection q.weight is dropped)
    save_file({"o.weight": span["o.weight"].float().contiguous(), "q0": span["q0"].float().contiguous()}, str(out / "pointer_head.safetensors"),
              metadata={"format": "pt"})
    cfg = VotumConfig(bidirectional_layers=bidir, pointer_rank=int(span["o.weight"].shape[0]),
                      header=dict(defaults.header, max_q=int(hj.get("max_q", 8))),
                      temperatures=json.loads(a.temperatures) if a.temperatures else defaults.temperatures)
    cfg.save(out)
    # round trip
    back = load_file(str(out / "adapter_model.safetensors")); assert all(torch.equal(back[k], lora[k]) for k in lora)
    hd = load_file(str(out / "pointer_head.safetensors"))
    assert torch.equal(hd["o.weight"], span["o.weight"].float()) and torch.equal(hd["q0"], span["q0"].float())
    for f in ("adapter_model.safetensors", "pointer_head.safetensors"): print(f"{f}: sha256 {sha256(out / f)}")
    print(f"{out}: {len(lora)} LoRA tensors on {targets}, bidirectional layers {bidir}, step {meta.get('step')}")
    build(root, str(out), a.base)


if __name__ == "__main__":
    main()
