"""Build a Votum checkpoint directory from the adapter form (LoRA adapter + pointer head + votum_config.json) and the
Apertus-1.5-8B base checkpoint. Both the PyTorch reference (votum.Votum) and the vLLM server (votum.vllm) load it.

    votum build --adapter <dir | <org>/Votum-8B> --out <dir> [--base swiss-ai/Apertus-v1.5-8B | <dir>] [--exact] [--link]

Default (merged): W <- bf16(W + (alpha / r) * B @ A), computed in fp32 and stored in bf16. On the evaluation sets this
gives the same accuracy and benchmark scores as the unmerged model (98 % identical answers), is about 10 % faster on
vLLM and needs no LoRA support. --exact: the base weights are written unchanged and the adapter goes to lora/, served
through vLLM's LoRA kernels (the training-time computation exactly).
Written: the base's text stack only (model.language_model.* -> model.*; vision / audio tokenizers and lm_head are not
needed), the pointer head (pointer.o.weight, pointer.q0; fp32), the tokenizer files, votum_config.json, and a config.json
with model_type "apertus" and the decision_* fields read by votum.vllm. --link (with --exact): shards of a local base
that hold only text-stack tensors are symlinked instead of rewritten."""
from __future__ import annotations

import argparse, json, os, re, shutil
from pathlib import Path

from .config import CONFIG_NAME, VotumConfig

TEXT = "model.language_model."
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja"]
HEAD_NAME = "pointer_head.safetensors"
ADAPTER_FILES = [CONFIG_NAME, HEAD_NAME, "adapter_config.json", "adapter_model.safetensors"]
LORA = re.compile(r"base_model\.model\.model\.layers\.(\d+)\.(self_attn|mlp)\.(\w+)\.lora_([AB])\.weight")


def resolve(path_or_repo: str, revision: str | None = None, token=None, patterns: list[str] | None = None, subfolder: str = "") -> Path:
    """a local directory, or a Hub repo downloaded with the given file patterns (inside subfolder)."""
    p = Path(path_or_repo)
    if p.exists(): return p / subfolder if subfolder else p
    from huggingface_hub import snapshot_download
    pats = [f"{subfolder}/{x}" if subfolder else x for x in (patterns or ["*"])]
    d = Path(snapshot_download(path_or_repo, revision=revision, token=token, allow_patterns=pats))
    return d / subfolder if subfolder else d


def adapter_dir(path_or_repo: str, token=None) -> Path:
    """the adapter form: a directory with the four adapter files, or the adapter/ subfolder of a Votum repo."""
    p = Path(path_or_repo)
    if p.exists(): return p if (p / "adapter_model.safetensors").exists() else p / "adapter"
    return resolve(path_or_repo, None, token, ADAPTER_FILES, subfolder="adapter")


def serving_config(base_cfg: dict, cfg: VotumConfig, tok) -> dict:
    t = base_cfg.get("text_config", base_cfg)
    rp = dict(t.get("rope_parameters") or {**(t.get("rope_scaling") or {}), "rope_theta": t["rope_theta"]})
    h = cfg.header
    return {
        "architectures": ["VotumDecisionModel"], "model_type": "apertus", "torch_dtype": "bfloat16",
        "hidden_size": t["hidden_size"], "intermediate_size": t["intermediate_size"], "num_hidden_layers": t["num_hidden_layers"],
        "num_attention_heads": t["num_attention_heads"], "num_key_value_heads": t["num_key_value_heads"],
        "hidden_act": t["hidden_act"], "rms_norm_eps": t["rms_norm_eps"], "vocab_size": t["vocab_size"],
        "max_position_embeddings": t["max_position_embeddings"], "attention_bias": t.get("attention_bias", False),
        "mlp_bias": t.get("mlp_bias", False), "qk_norm": t.get("qk_norm", True), "post_norm": t.get("post_norm", False),
        "tie_word_embeddings": False, "bos_token_id": t["bos_token_id"], "eos_token_id": t["eos_token_id"], "pad_token_id": t["pad_token_id"],
        "rope_theta": rp.pop("rope_theta"), "rope_scaling": rp,
        "decision_opt_end_id": cfg.opt_end_id, "decision_bidirectional_layers": cfg.bidirectional_layers,
        "decision_pointer_rank": cfg.pointer_rank, "decision_temperatures": cfg.temperatures,
        "decision_shared_positions": cfg.shared_positions, "decision_close_last": cfg.close_last,
        "decision_options_header_ids": tok("\n\nOptions:\n", add_special_tokens=False).input_ids[-2:],     # "Options", ":\n"
        "decision_question_header_ids": tok("\n\nQuestion:\n", add_special_tokens=False).input_ids[-2:],   # "Question", ":\n"
        "decision_header": {"max_q": int(h.get("max_q", 8)), "max_opts": h.get("max_opts"), "sort": bool(h.get("sort", True)),
                            "chunk": bool(h.get("chunk", False))},
    }


def build(out: Path, adapter: str, base: str | None = None, exact: bool = False, link: bool = False, token=None) -> None:
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer
    out.mkdir(parents=True, exist_ok=True)
    ad = adapter_dir(adapter, token); cfg = VotumConfig.load(ad)
    bd = resolve(base or cfg.base_model, None if base else cfg.base_revision, token,
                 ["config.json", "model.safetensors.index.json", "*.safetensors", *TOKENIZER_FILES])
    for f in TOKENIZER_FILES:
        if (bd / f).exists(): shutil.copy(bd / f, out / f)
    tok = AutoTokenizer.from_pretrained(str(out))
    assert tok.convert_tokens_to_ids(cfg.opt_end_token) == cfg.opt_end_id
    delta, scale = {}, None
    if not exact:
        acfg = json.loads((ad / "adapter_config.json").read_text()); scale = acfg["lora_alpha"] / acfg["r"]
        for k, v in load_file(str(ad / "adapter_model.safetensors")).items():
            m = LORA.fullmatch(k); assert m, k
            delta.setdefault(f"model.layers.{m[1]}.{m[2]}.{m[3]}.weight", {})[m[4]] = v
        link = False
    wmap = json.loads((bd / "model.safetensors.index.json").read_text())["weight_map"]
    plan = []                                             # (base shard, {base key: our key}, all keys of the shard)
    for shard in sorted(set(wmap.values())):
        with safe_open(str(bd / shard), framework="pt") as f: names = list(f.keys())   # every tensor in the file (vLLM reads them all)
        text = {k: "model." + k[len(TEXT):] for k in names if k.startswith(TEXT)}
        text.update({k: k for k in names if k.startswith("model.") and not k.startswith(TEXT) and k.split(".")[1] in ("layers", "embed_tokens", "norm")})
        if text: plan.append((shard, text, names))
    n = len(plan) + 1; index = {}
    for i, (shard, text, names) in enumerate(plan, 1):
        fn = f"model-{i:05d}-of-{n:05d}.safetensors"; dst = out / fn
        if dst.exists() or dst.is_symlink(): dst.unlink()
        if link and len(text) == len(names) and all(k == v for k, v in text.items()):   # a pure text-stack shard: link it unchanged
            os.symlink((bd / shard).resolve(), dst); index.update({k: fn for k in text}); print(f"{fn}: linked ({len(text)} tensors)", flush=True)
            continue
        tensors = {}
        with safe_open(str(bd / shard), framework="pt") as f:
            for k, name in text.items():
                w = f.get_tensor(k)
                if name in delta:
                    d = delta.pop(name); w = (w.float() + scale * (d["B"].float() @ d["A"].float())).to(torch.bfloat16)
                tensors[name] = w.contiguous()
        save_file(tensors, str(dst), metadata={"format": "pt"}); index.update({k: fn for k in tensors}); print(f"{fn}: {len(tensors)} tensors", flush=True)
    assert any(k.startswith("model.layers.") for k in index), "no text-stack weights found in the base checkpoint"
    assert not delta, f"{len(delta)} LoRA targets not found in the base"
    head = load_file(str(ad / HEAD_NAME)); fn = f"model-{n:05d}-of-{n:05d}.safetensors"
    save_file({"pointer.o.weight": head["o.weight"].float().contiguous(), "pointer.q0": head["q0"].float().contiguous()}, str(out / fn),
              metadata={"format": "pt"})
    index.update({"pointer.o.weight": fn, "pointer.q0": fn})
    (out / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": index}, indent=1))
    if exact:
        (out / "lora").mkdir(exist_ok=True); shutil.copy(ad / "adapter_model.safetensors", out / "lora" / "adapter_model.safetensors")
        acfg = json.loads((ad / "adapter_config.json").read_text()); acfg.update(base_model_name_or_path=out.name, task_type="CAUSAL_LM")
        (out / "lora" / "adapter_config.json").write_text(json.dumps(acfg, indent=1))
    shutil.copy(ad / CONFIG_NAME, out / CONFIG_NAME)
    (out / "config.json").write_text(json.dumps(serving_config(json.loads((bd / "config.json").read_text()), cfg, tok), indent=1))
    print(f"{out}: {'exact (LoRA under lora/)' if exact else 'merged'} checkpoint ready")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="votum build"); ap.add_argument("--out", required=True)
    ap.add_argument("--adapter", required=True, help="adapter directory, or a Votum repo id / directory (uses its adapter/)")
    ap.add_argument("--base", default=None); ap.add_argument("--exact", action="store_true"); ap.add_argument("--link", action="store_true")
    a = ap.parse_args(argv)
    build(Path(a.out), a.adapter, a.base, a.exact, a.link)


if __name__ == "__main__":
    main()
