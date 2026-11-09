"""Refit the per-type temperatures of an exported Votum checkpoint on labelled validation rows, and write them into it;
with --eval-only, just report accuracy, cross-entropy and calibration error on any labelled file (a held-out test set).

    python calibrate.py --model <exported checkpoint dir> --val data/mine/val.jsonl.gz [--limit 2000] [--dry-run]

Votum's probabilities are softmax(scores / T) with one T per question type (choice, noul, score). After fine-tuning,
the released temperatures no longer match the new scores, so they are refitted here by minimising the cross-entropy
against the rows' targets over a log-spaced grid (0.25 to 20), the procedure that produced the released values. The
result is written to votum_config.json (root and adapter/) and to the decision_temperatures field of config.json, which
the vLLM server reads. Scoring runs the PyTorch reference (votum.Votum) on the GPU: about 20 rows/s on an RTX 4090 for
400-token prompts; use --limit for a quick fit. A few hundred rows per type give a stable temperature; a type with
fewer than --min-rows (50) keeps the released value, and a fit that hits the grid edge is flagged."""
from __future__ import annotations

import argparse, gzip, json, math, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from votum import Votum                    # noqa: E402
from votum.config import CONFIG_NAME, VotumConfig  # noqa: E402

GRID = np.exp(np.linspace(np.log(0.25), np.log(20.0), 300))


def read_rows(path: Path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip(): yield json.loads(line)


def nll(zs: list[np.ndarray], ts: list[np.ndarray], T: float) -> float:
    tot = 0.0
    for z, t in zip(zs, ts):
        lz = z / T; lz -= lz.max(); lp = lz - math.log(np.exp(lz).sum()); tot -= float((t * lp).sum())
    return tot / len(zs)


def ece(zs, ts, T, bins=15) -> float:
    conf, hit = [], []
    for z, t in zip(zs, ts):
        p = np.exp(z / T - (z / T).max()); p /= p.sum(); k = int(p.argmax()); conf.append(float(p[k])); hit.append(float(t.argmax() == k))
    conf, hit = np.array(conf), np.array(hit); e = 0.0
    for b in range(bins):
        m = (conf > b / bins) & (conf <= (b + 1) / bins)
        if m.any(): e += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return float(e)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Votum checkpoint: Hub repo id or local directory (scripts/export_hf.py output; an --adapter-only directory needs --exact)")
    ap.add_argument("--val", required=True, help="validation rows with targets (prepare_data.py's val.jsonl.gz)")
    ap.add_argument("--limit", type=int, default=None, help="use at most this many rows per type (default: all)")
    ap.add_argument("--batch", type=int, default=4, help="rows per forward pass (default 4; 8 fits a card with nothing else on it)"); ap.add_argument("--device", default=None)
    ap.add_argument("--exact", action="store_true", help="load the adapter/ form (export_hf.py --adapter-only) on the checkpoint it was trained on")
    ap.add_argument("--dry-run", "--eval-only", dest="dry_run", action="store_true", help="report accuracy, NLL and ECE with the current temperatures and the refitted ones, write nothing (use this to evaluate on a held-out test file)")
    ap.add_argument("--min-rows", type=int, default=50, help="a type with fewer scored rows keeps its current temperature (default 50); --eval-only always reports metrics")
    ap.add_argument("--force", action="store_true", help="write a temperature even if the fit hit the edge of the grid (does not override --min-rows)")
    a = ap.parse_args(argv)
    by_type = defaultdict(list)
    for r in read_rows(Path(a.val)):
        if a.limit is None or len(by_type[r["type"]]) < a.limit: by_type[r["type"]].append(r)
    # the questions of one state are scored together, with the header they would share in a request: whole groups per
    # call, whatever --batch is, so the result does not depend on the batch size
    groups = defaultdict(list)
    for t in ("choice", "noul", "score"):
        for r in by_type[t]: groups[str(r.get("group") or r["id"])].append(r)
    calls, cur = [], []
    for g in groups.values():
        cur += g
        if len(cur) >= a.batch: calls.append(cur); cur = []
    if cur: calls.append(cur)
    rows = [r for c in calls for r in c]
    print(f"{len(rows)} rows: " + ", ".join(f"{t} {len(by_type[t])}" for t in by_type), flush=True)
    from votum.build import resolve
    if not Path(a.model).exists() and not a.dry_run:
        raise SystemExit(f"{a.model} is not a local directory. Temperatures are written into the checkpoint's files, so download it first "
                         f"(hf download {a.model} --local-dir <dir>) and pass that directory; a Hub id can only be evaluated (--eval-only)")
    root = resolve(a.model, None, None, ["config.json", CONFIG_NAME, "model*.safetensors", "model.safetensors.index.json", "adapter/*",
                                         "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja"])
    adapter_only = not (root / CONFIG_NAME).exists() and (root / "adapter" / CONFIG_NAME).exists()
    if adapter_only and not a.exact: raise SystemExit(f"{root} holds only the adapter form (export_hf.py --adapter-only): pass --exact")
    vm = Votum.from_pretrained(str(root), exact=a.exact, device=a.device); cfg = VotumConfig.load(root / "adapter" if adapter_only else root)
    import torch
    if torch.cuda.is_available(): torch.cuda.empty_cache()                 # loading leaves transient buffers; a 24 GB card with a display is close to the edge
    zs, ts, skipped, t0 = defaultdict(list), defaultdict(list), 0, time.time()
    for n_call, chunk in enumerate(calls):
        i = sum(len(c) for c in calls[:n_call])
        for r, o in zip(chunk, vm.decide_rows(chunk, keys=[str(r.get("group") or r["id"]) for r in chunk])):
            if o is None: skipped += 1; continue
            pos = {oid: k for k, oid in enumerate(o["ids"])}; tgt = np.zeros(len(o["ids"]))
            for oid, p in zip([x["id"] for x in r["options"]], r["target"]): tgt[pos[str(oid)]] = p
            zs[r["type"]].append(np.array(o["scores"], dtype=np.float64)); ts[r["type"]].append(tgt)
        if n_call % 25 == 0: print(f"  {i + len(chunk)}/{len(rows)} scored ({time.time() - t0:.0f} s)", flush=True)
    new = dict(cfg.temperatures)
    print(f"\n{'type':7s} {'rows':>5s} {'acc':>6s} {'T now':>9s} {'T fitted':>8s} {'NLL now':>10s} {'NLL fit':>9s} {'ECE now':>10s} {'ECE fit':>9s}")
    for t in ("choice", "noul", "score"):
        if not zs[t]: continue
        T0 = cfg.temperatures[t]; T1 = float(min(GRID, key=lambda T: nll(zs[t], ts[t], T)))
        edge = T1 <= GRID[0] * 1.001 or T1 >= GRID[-1] * 0.999
        kept = not (len(zs[t]) >= a.min_rows and (not edge or a.force))
        if not kept: new[t] = T1
        elif not a.dry_run:
            why = f"only {len(zs[t])} rows (< --min-rows {a.min_rows})" if len(zs[t]) < a.min_rows else f"the fit hit the grid edge ({T1:.3f}); rows too few or too easy (--force to write it anyway)"
            print(f"  {t}: temperature kept at {T0:.3f}: {why}")
        acc = 100 * float(np.mean([z.argmax() == y.argmax() for z, y in zip(zs[t], ts[t])]))
        print(f"{t:7s} {len(zs[t]):5d} {acc:5.1f}% {T0:9.3f} {T1:8.3f} {nll(zs[t], ts[t], T0):10.4f} {nll(zs[t], ts[t], T1):9.4f} {ece(zs[t], ts[t], T0):10.4f} {ece(zs[t], ts[t], T1):9.4f}"
              + ("   (fitted value not written)" if kept and not a.dry_run else ""))
    if skipped: print(f"{skipped} rows did not fit the context and were skipped")
    if a.dry_run: print("dry run: nothing written"); return
    cfg.temperatures = new; written = []
    if (root / CONFIG_NAME).exists(): cfg.save(root); written.append(CONFIG_NAME)
    if (root / "adapter" / CONFIG_NAME).exists(): cfg.save(root / "adapter"); written.append(f"adapter/{CONFIG_NAME}")
    cj = root / "config.json"
    if cj.exists(): d = json.loads(cj.read_text()); d["decision_temperatures"] = new; cj.write_text(json.dumps(d, indent=1)); written.append("config.json")
    print(f"wrote temperatures {json.dumps({k: round(v, 4) for k, v in new.items()})} to {root}: {', '.join(written)}")


if __name__ == "__main__":
    main()
