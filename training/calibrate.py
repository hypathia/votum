"""Refit the per-type temperatures of an exported Votum checkpoint on labelled validation rows, and write them into it.

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
    ap.add_argument("--model", required=True, help="exported checkpoint directory (scripts/export_hf.py output)")
    ap.add_argument("--val", required=True, help="validation rows with targets (prepare_data.py's val.jsonl.gz)")
    ap.add_argument("--limit", type=int, default=None, help="use at most this many rows per type (default: all)")
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--device", default=None)
    ap.add_argument("--exact", action="store_true", help="load the adapter/ form (export_hf.py --adapter-only) on the checkpoint it was trained on")
    ap.add_argument("--dry-run", action="store_true", help="fit and report, but do not write")
    ap.add_argument("--min-rows", type=int, default=50, help="a type with fewer scored rows keeps its released temperature (default 50)")
    a = ap.parse_args(argv); root = Path(a.model)
    by_type = defaultdict(list)
    for r in read_rows(Path(a.val)):
        if a.limit is None or len(by_type[r["type"]]) < a.limit: by_type[r["type"]].append(r)
    rows = [r for t in ("choice", "noul", "score") for r in by_type[t]]
    print(f"{len(rows)} rows: " + ", ".join(f"{t} {len(by_type[t])}" for t in by_type), flush=True)
    vm = Votum.from_pretrained(str(root), exact=a.exact, device=a.device); cfg = VotumConfig.load(root / 'adapter' if a.exact and not (root / CONFIG_NAME).exists() else root)
    zs, ts, skipped, t0 = defaultdict(list), defaultdict(list), 0, time.time()
    for i in range(0, len(rows), a.batch):
        chunk = rows[i:i + a.batch]
        for r, o in zip(chunk, vm.decide_rows(chunk)):
            if o is None: skipped += 1; continue
            pos = {oid: k for k, oid in enumerate(o["ids"])}; tgt = np.zeros(len(o["ids"]))
            for oid, p in zip([x["id"] for x in r["options"]], r["target"]): tgt[pos[str(oid)]] = p
            zs[r["type"]].append(np.array(o["scores"], dtype=np.float64)); ts[r["type"]].append(tgt)
        if (i // a.batch) % 25 == 0: print(f"  {min(i + a.batch, len(rows))}/{len(rows)} scored ({time.time() - t0:.0f} s)", flush=True)
    new = dict(cfg.temperatures)
    print(f"\n{'type':7s} {'rows':>5s} {'acc':>6s} {'T before':>9s} {'T after':>8s} {'NLL before':>10s} {'NLL after':>9s} {'ECE before':>10s} {'ECE after':>9s}")
    for t in ("choice", "noul", "score"):
        if len(zs[t]) < a.min_rows: print(f"{t:7s} {len(zs[t]):5d}  only {len(zs[t])} rows (< --min-rows {a.min_rows}): temperature kept at {cfg.temperatures[t]:.3f}"); continue
        T0 = cfg.temperatures[t]; T1 = float(min(GRID, key=lambda T: nll(zs[t], ts[t], T))); new[t] = T1
        if T1 <= GRID[0] * 1.001 or T1 >= GRID[-1] * 0.999: print(f"  warning: {t} temperature hit the grid edge ({T1:.3f}); the validation rows are probably too few or too easy")
        acc = 100 * float(np.mean([z.argmax() == y.argmax() for z, y in zip(zs[t], ts[t])]))
        print(f"{t:7s} {len(zs[t]):5d} {acc:5.1f}% {T0:9.3f} {T1:8.3f} {nll(zs[t], ts[t], T0):10.4f} {nll(zs[t], ts[t], T1):9.4f} {ece(zs[t], ts[t], T0):10.4f} {ece(zs[t], ts[t], T1):9.4f}")
    if skipped: print(f"{skipped} rows did not fit the context and were skipped")
    if a.dry_run: print("dry run: nothing written"); return
    cfg.temperatures = new; cfg.save(root)
    if (root / "adapter" / CONFIG_NAME).exists(): cfg.save(root / "adapter")
    cj = root / "config.json"; d = json.loads(cj.read_text()); d["decision_temperatures"] = new; cj.write_text(json.dumps(d, indent=1))
    print(f"wrote temperatures {json.dumps({k: round(v, 4) for k, v in new.items()})} to {root / CONFIG_NAME}, adapter/{CONFIG_NAME} and config.json")


if __name__ == "__main__":
    main()
