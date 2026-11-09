"""Fine-tune Votum-8B on your own decision data: start from the released adapter and pointer head, train with the released
layout and recipe, then export and calibrate.

    python finetune.py --from <Votum-8B repo id or local checkpoint> --data data/mine --tag mine            # train
    python finetune.py --from ... --data data/mine --tag mine --probe                                        # 2 steps, memory
    python finetune.py --from ... --data data/mine --tag mine --dry-run                                      # print the command

--data is the directory written by prepare_data.py (train.jsonl.gz, val.jsonl.gz, stats.json). The run goes to
$VOTUM_RUNS/<tag>/ (default training/runs/<tag>/): log.jsonl, state_<step>.pt and ckpt_<step>/, as in train.py. The
layout flags are fixed to the released layout (option isolation, shared positions, bidirectional layers 3, 7, ..., 31,
request headers, FlexAttention); everything the trainer accepts can still be overridden through --train-arg.

Two modes. "merged" (default): the backbone is the Votum checkpoint itself (the released adapter is already folded into its
weights) and a new LoRA is trained on top, starting from zero, with the released pointer head; this runs in the same
environment as serving (mainline transformers, no Apertus download). "adapter": the released adapter is continued on the
Apertus base, as the released run was trained; it needs the base in HF_HOME and the training fork (see Environment).

Hardware. The backbone is loaded in bf16 (about 16 GB) with gradient checkpointing; the LoRA and head add under 1 GB.
What remains is activation memory, which grows with --micro-tokens, the padded size of one micro-batch. The released run
used 16,384 on a 96 GB card and peaked at 46.9 GiB. The automatic choice looks at the FREE memory of the card (a display or
another process counts). Rows longer than --micro-tokens are never trained (the data plan
skips them), so a small micro-batch also caps the prompt length you can learn from. --micro-tokens auto picks a value
from the GPU's memory (see MICRO_BY_GB); use --probe to measure the peak on your card before a long run."""
from __future__ import annotations

import argparse, json, os, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent / "src"))
import common as C                                          # noqa: E402

# GPU memory (GiB, total) -> micro-batch tokens. Measured: 16,384 peaks at 46.9 GiB (released run, 96 GB card); on an
# RTX 4090 (24 GB) 2,048 peaks at 17.6 GiB and 3,072 at 18.5 GiB, while 4,096 runs out of memory. The 32 GB and 48 GB
# entries are interpolated from those points, not measured; check them with --probe.
MICRO_BY_GB = [(70, 16384), (44, 8192), (30, 4096), (21, 3072)]
LAYOUT = ["--opt-block", "--opt-isolate", "--shared-positions", "--bidir-layers", "--attn", "flex",
          "--header", "group", "--header-blind", "--header-max-opts", "999", "--loss-norm", "constant", "--tok-field", "n_tok_span", "--repack"]


def gpu_gib() -> tuple[float, float] | None:
    """(free, total) GiB of GPU 0, or None without CUDA"""
    try:
        import torch
        if not torch.cuda.is_available(): return None
        free, total = torch.cuda.mem_get_info(0); return free / 2**30, total / 2**30
    except Exception: return None


def pick_micro(mem: tuple[float, float] | None) -> int:
    """from the FREE memory (a display or another process on the card counts), not the card's size"""
    if mem is None: raise SystemExit("no CUDA device found; pass --micro-tokens explicitly to build a command for another machine")
    free, total = mem
    for floor, mt in MICRO_BY_GB:
        if free >= floor: return mt
    if free >= 19.5: return 2048                                 # a 24 GB card that also drives a display: 2,048 peaks at 17.6 GiB
    raise SystemExit(f"{free:.1f} GiB of GPU memory free ({total:.0f} GiB card): below the smallest measured configuration (19.5 GiB free, 2,048-token micro-batches). Free the card or pass --micro-tokens")


def init_checkpoint(src: str, out: Path, mode: str, token=None) -> Path:
    """a light checkpoint train.py starts from (--init-weights). mode "merged": only the released pointer head, read from
    the Votum checkpoint's shards; the LoRA starts from zero on the merged weights. mode "adapter": the released adapter
    and head, to be applied to the Apertus base. (The head's unused query projection is zero; the trainer never reads it.)"""
    import json as J
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    from votum.build import TOKENIZER_FILES, adapter_dir, resolve
    from votum.config import VotumConfig
    out.mkdir(parents=True, exist_ok=True)
    if mode == "adapter":
        ad = adapter_dir(src, token=token); cfg = VotumConfig.load(ad)
        (out / "adapter").mkdir(exist_ok=True)
        for f in ("adapter_config.json", "adapter_model.safetensors"): shutil.copy(ad / f, out / "adapter" / f)
        head = load_file(str(ad / "pointer_head.safetensors")); o, q0 = head["o.weight"], head["q0"]; backbone = None
    else:
        rd = resolve(src, None, token, ["config.json", "votum_config.json", "model*.safetensors", "model.safetensors.index.json", *TOKENIZER_FILES])
        cfg = VotumConfig.load(rd); wm = J.loads((rd / "model.safetensors.index.json").read_text())["weight_map"]
        with safe_open(str(rd / wm["pointer.o.weight"]), "pt") as f: o = f.get_tensor("pointer.o.weight")
        with safe_open(str(rd / wm["pointer.q0"]), "pt") as f: q0 = f.get_tensor("pointer.q0")
        backbone = str(rd)
    torch.save({"q.weight": torch.zeros(cfg.pointer_rank, o.shape[1]), "o.weight": o.float(), "q0": q0.float()}, out / "span_head.pt")
    C.jdump(dict(source=str(src), mode=mode, backbone=backbone, base_model=cfg.base_model, base_revision=cfg.base_revision, temperatures=cfg.temperatures), out / "init_meta.json")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="src", default=os.environ.get("VOTUM_REPO"), help="Votum-8B, or your own merged export: Hub repo id or a local checkpoint directory with config.json, model*.safetensors and votum_config.json (default $VOTUM_REPO)")
    ap.add_argument("--mode", choices=["merged", "adapter"], default="merged",
                    help="merged (default): train a new LoRA on the Votum checkpoint itself, one environment, no Apertus download; "
                         "adapter: continue the released adapter on the Apertus base (needs the base in HF_HOME and the training fork)")
    ap.add_argument("--data", required=True, help="directory from prepare_data.py")
    ap.add_argument("--tag", required=True, help="run name: output in $VOTUM_RUNS/<tag>/")
    ap.add_argument("--steps", type=int, default=None, help="optimizer steps (default: --epochs over the training rows)")
    ap.add_argument("--epochs", type=float, default=1.0, help="passes over the training rows when --steps is not given (default 1)")
    ap.add_argument("--lr", type=float, default=5e-5, help="peak learning rate of the LoRA (default 5e-5, half the released run's 1e-4: the adapter is already trained)")
    ap.add_argument("--lr-head", type=float, default=5e-4, help="peak learning rate of the pointer head (default 5e-4)")
    ap.add_argument("--tokens-per-step", type=int, default=C.RECIPE["tokens_per_step"], help="padded tokens per optimizer step (default 65,536)")
    ap.add_argument("--micro-tokens", default="auto", help="padded tokens per micro-batch: auto (from GPU memory) or a number")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--train-arg", action="append", default=[], help="extra train.py argument, repeatable (e.g. --train-arg=--lora-r=32 is NOT possible: the released adapter is rank 16)")
    ap.add_argument("--probe", action="store_true", help="run 2 steps without saving and report the peak GPU memory and step time (in runs/<tag>_probe)")
    ap.add_argument("--resume", action="store_true", help="continue the run <tag> from its last saved state (after a crash or a stop)")
    ap.add_argument("--dry-run", action="store_true", help="print the train.py command and exit")
    ap.add_argument("--token", default=None, help="Hub token (if the model repo is gated and you are not logged in)")
    a = ap.parse_args(argv)
    if not a.src: raise SystemExit("--from is required: the Votum-8B repo id or a local checkpoint directory")
    if a.probe: a.tag += "_probe"                                 # its own directory: the real run's log stays clean
    data = Path(a.data).resolve(); st = json.loads((data / "stats.json").read_text()); tr = st["train"]
    if a.mode == "adapter":
        try: C.backbone_path()
        except FileNotFoundError as e: raise SystemExit(f"{e}\nDownload swiss-ai/Apertus-v1.5-8B into HF_HOME first (huggingface-cli download swiss-ai/Apertus-v1.5-8B); it is gated on the Apertus Acceptable Use Policy")

    mem = gpu_gib(); mt = pick_micro(mem) if a.micro_tokens == "auto" else int(a.micro_tokens)
    if mt > a.tokens_per_step: raise SystemExit("--micro-tokens cannot exceed --tokens-per-step")
    # tokens a step really holds: tokens_per_step // micro_tokens micro-batches, each with as many typical rows as fit
    # (a 1,100-token row in a 2,048-token micro-batch fits alone, so such a step carries ~35k tokens, not 65k)
    typical = tr["tokens_p50"] + 16; fit = max(1, mt // typical); eff = min(a.tokens_per_step * 0.85, (a.tokens_per_step // mt) * fit * tr["tokens_p50"])
    want = a.epochs * tr["tokens_total"] / eff
    steps = a.steps or max(10, round(want)); passes = steps * eff / max(1, tr["tokens_total"])
    warm = min(25, max(2, steps // 10)); decay = min(100, max(1, steps // 4))
    too_long = tr["tokens_p90"] > mt
    print((f"GPU: {mem[0]:.1f} of {mem[1]:.0f} GiB free" if mem else "GPU: none detected") + f" | micro-batch {mt:,} tokens | {steps} steps x {a.tokens_per_step:,} tokens "
          f"(about {eff:,.0f} used per step with {fit} typical row{'s' if fit > 1 else ''} per micro-batch) = about {passes:.1f} passes over {tr['rows']:,} rows ({tr['tokens_total']:,} tokens) | warm-up {warm}, cosine decay over the last {decay}", flush=True)
    if not a.steps and want < 10: print(f"NOTE: {a.epochs:g} epoch would be {want:.1f} steps; the minimum is 10, so this is {passes:.0f} passes over your rows. Small sets overfit quickly: check the validation accuracy from calibrate.py, or pass --steps", flush=True)
    if a.tokens_per_step > 4 * max(1, tr["tokens_total"]): print("NOTE: one step is larger than your whole dataset; consider --tokens-per-step 16384 for a small set", flush=True)
    if too_long: print(f"WARNING: at least 10 % of the rows are longer than {mt:,} tokens (p90 = {tr['tokens_p90']}); the data plan skips rows longer than one micro-batch, so they will not be trained on", flush=True)

    run = C.RUNS / a.tag; init = run / "init"
    if a.dry_run and not (init / "span_head.pt").exists():
        backbone = ["--backbone", a.src] if a.mode == "merged" else []       # nothing is written in a dry run
    else:
        run.mkdir(parents=True, exist_ok=True); C.disk_guard(run, need_gb=2.0)
        if not (init / "span_head.pt").exists():
            print(f"preparing the start checkpoint from {a.src} ({a.mode})", flush=True); init_checkpoint(a.src, init, a.mode, token=a.token)
        meta = json.loads((init / "init_meta.json").read_text())
        backbone = ["--backbone", meta["backbone"]] if meta.get("backbone") else []
    argv_t = [*backbone, "--lr", str(a.lr), "--lr-span", str(a.lr_head), "--seed", str(a.seed), "--tag", a.tag, "--train", str(data / "train.jsonl.gz"),
              "--steps", str(steps), "--warmup", str(warm), "--final-decay", str(decay), "--tokens-per-step", str(a.tokens_per_step),
              "--micro-tokens", str(mt), "--save-every", str(a.save_every), "--init-weights", str(init), *LAYOUT, *a.train_arg]
    if a.probe: argv_t += ["--steps", "2", "--until", "2", "--no-save", "--print-every", "1", "--final-decay", "1", "--warmup", "1"]
    if a.resume: argv_t.append("--resume")
    cmd = [sys.executable, str(ROOT / "train.py"), *argv_t]
    print("train.py " + " ".join(argv_t), flush=True)
    if a.dry_run: return
    if a.probe and (run / "log.jsonl").exists(): (run / "log.jsonl").unlink()      # a probe directory only ever holds probe lines
    rc = subprocess.call(cmd, cwd=str(ROOT))
    if rc: raise SystemExit(rc)
    if a.probe:
        recs = [json.loads(l) for l in open(run / "log.jsonl")]
        pk = max(r["peak_gib"] for r in recs); sec = recs[-1]["sec"]
        print(f"\nprobe: peak {pk:.1f} GiB with {mt:,}-token micro-batches; last step {sec:.0f} s for {recs[-1]['n_q']} questions "
              f"({recs[-1]['padded_tok_s']:,} padded tokens/s); {steps} steps would take about {steps * sec / 3600:.1f} h")
        if mem and pk > 0.92 * mem[0]: print("the peak is within 8 % of the free memory: use a smaller --micro-tokens")
        return
    ck = sorted(run.glob("ckpt_*"), key=lambda p: int(p.name.split("_")[1]))[-1]
    print(f"\ndone. Next:\n  python ../scripts/export_hf.py --ckpt {ck} --out <out dir>           # merged checkpoint + adapter/ form"
          f"\n  python calibrate.py --model <out dir> --val {data / 'val.jsonl.gz'}      # refit the temperatures on your validation rows")


if __name__ == "__main__":
    main()
