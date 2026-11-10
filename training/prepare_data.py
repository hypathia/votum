"""Turn decision rows into the trainer's input: validated rows with token counts, split into train and validation files.

    python prepare_data.py --pool <Votum-Pool dir or Hub id> [--config default|share_alike] --out data/pool
    python prepare_data.py --rows my_rows.jsonl[.gz] [--val my_val.jsonl] [--val-frac 0.05] --out data/mine
    python prepare_data.py --csv tickets.csv --text-col text --label-col team --question "Which team should handle this ticket?" --out data/tickets

Rows use the Votum Pool schema (hf.co/datasets/votum-ai/Votum-Pool): id, type (choice | noul | score), state (text or JSON
object), question, options [{id, text}], target (one probability per option, in option order), and optionally group
(rows about the same state; a group never spans train and validation), lang, source, license. Yes/no rows have option
ids "true" and "false"; score rows list their levels from low to high. Anything else is passed through untouched.

Every row gets n_tok_span, the length in tokens of its training prompt (system turn, a header listing the other questions
asked about the same state, the state, the question and the options), which train.py's data plan packs on
(--tok-field n_tok_span). Rows longer than --max-tokens are dropped and counted. stats.json summarises the output and
estimates how many optimizer steps one pass over the data takes at the recipe's 65,536 tokens per step."""
from __future__ import annotations

import argparse, gzip, hashlib, json, math, os, random, sys, time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent / "src"))
import common as C                         # noqa: E402
from votum import prompt as P              # noqa: E402

TYPES = ("choice", "noul", "score")


def read_rows(path: Path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try: yield json.loads(line)
                except json.JSONDecodeError as e: raise SystemExit(f"{path}:{n}: not JSON ({e})")


def read_parquet(files: list[Path]):
    try: import pyarrow.parquet as pq
    except ImportError: raise SystemExit("reading parquet needs pyarrow: pip install pyarrow")
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=4096):
            yield from batch.to_pylist()


def pool_files(pool: str, config: str, split: str) -> list[Path]:
    """parquet files of one config and split, from a local dataset directory or the Hub."""
    d = Path(pool)
    if not d.is_dir():
        from huggingface_hub import snapshot_download
        d = Path(snapshot_download(pool, repo_type="dataset", allow_patterns=[f"data/{config}/{split}-*.parquet"]))
    files = sorted((d / "data" / config).glob(f"{split}-*.parquet"))
    if not files: raise SystemExit(f"no {split} parquet files for config {config!r} under {d / 'data' / config}")
    return files


def csv_rows(path: Path, text_col: str, label_col: str, question: str, labels: str | None, group_col: str | None):
    """a classification table -> choice rows: every text gets the same question and the same option list (all labels seen
    in the file, or the --labels mapping), with a one-hot target on the row's label."""
    import csv
    with open(path, newline="", encoding="utf-8") as f: recs = list(csv.DictReader(f))
    if not recs: raise SystemExit(f"{path}: empty")
    for c in (text_col, label_col):
        if c not in recs[0]: raise SystemExit(f"{path}: no column {c!r}; columns are {list(recs[0])}")
    if labels:
        bad = [kv for kv in labels.split(";") if kv.strip() and "=" not in kv]
        if bad: raise SystemExit(f"--labels: {bad[0].strip()!r} has no '='; the form is \"id=Description;id2=Description 2\"")
        opts = [{"id": kv.split("=", 1)[0].strip(), "text": kv.split("=", 1)[1].strip()} for kv in labels.split(";") if kv.strip()]
    else:
        opts = [{"id": v, "text": v} for v in sorted({r[label_col].strip() for r in recs})]
    ids = [o["id"] for o in opts]
    unknown = sorted({r[label_col].strip() for r in recs} - set(ids))
    if unknown: raise SystemExit(f"{path}: labels not in --labels: {unknown[:8]}")
    stem = path.stem
    for i, r in enumerate(recs):
        # without a group column, rows with the same text form one group, so a repeated text never sits on both sides of the split
        y = r[label_col].strip(); grp = r[group_col].strip() if group_col else f"{stem}/" + hashlib.sha1(r[text_col].strip().encode()).hexdigest()[:16]
        yield dict(id=f"{stem}/{i}", group=grp, type="choice", state=r[text_col], question=question, options=[dict(o) for o in opts],
                   target=[1.0 if k == y else 0.0 for k in ids])


def check(r: dict, where: str) -> dict:
    """validate one row in place; raises SystemExit with the row id on the first problem."""
    rid = r.get("id") or where
    for k in ("id", "type", "state", "question", "options", "target"):
        if k not in r: raise SystemExit(f"{rid}: missing field {k!r}")
    if r["type"] not in TYPES: raise SystemExit(f"{rid}: type must be one of {TYPES}, got {r['type']!r}")
    opts = r["options"]
    if not isinstance(opts, list) or len(opts) < 2: raise SystemExit(f"{rid}: at least two options are needed")
    for o in opts:
        if not isinstance(o, dict) or "id" not in o: raise SystemExit(f"{rid}: every option is an object with an id (and usually a text)")
        o["id"] = str(o["id"])
    ids = [o["id"] for o in opts]
    if len(set(ids)) != len(ids): raise SystemExit(f"{rid}: option ids repeat: {ids}")
    if r["type"] == "noul" and sorted(ids) != ["false", "true"]: raise SystemExit(f"{rid}: a noul row has exactly the options true and false, got {ids}")
    t = r["target"]
    if not isinstance(t, list) or len(t) != len(opts): raise SystemExit(f"{rid}: target needs one probability per option ({len(opts)}), got {len(t) if isinstance(t, list) else t!r}")
    t = [float(x) for x in t]
    if min(t) < 0 or sum(t) <= 0: raise SystemExit(f"{rid}: target probabilities must be non-negative and not all zero")
    s = sum(t); r["target"] = [x / s for x in t]                            # normalised to one
    if not isinstance(r["question"], str) or not r["question"].strip(): raise SystemExit(f"{rid}: empty question")
    if isinstance(r["state"], str) and not r["state"].strip(): raise SystemExit(f"{rid}: empty state")
    if r.get("group") is None: r["group"] = r["id"]
    r.setdefault("soft", any(0 < x < 1 for x in r["target"]))
    return r


def split_rows(rows: list[dict], frac: float, seed: int):
    """deterministic split by group: a hash of the group name decides, so rows about one state stay together."""
    tr, va = [], []
    for r in rows:
        h = int(hashlib.sha1(f"{seed}\x1f{r['group']}".encode()).hexdigest()[:8], 16) / 2**32
        (va if h < frac else tr).append(r)
    return tr, va


def count_tokens(tok, rows: list[dict], max_tokens: int, label: str):
    """n_tok_span per row = its full training prompt, with a header listing the other questions of the same state (the
    trainer's --header group behaviour: the questions sharing a state in the file). Returns the kept rows."""
    heads = P.request_headers(rows, keys=[r["group"] for r in rows], max_q=8, max_opts=None, sort=True)
    kept, dropped, t0 = [], Counter(), time.time()
    for i, (r, h) in enumerate(zip(rows, heads)):
        try: n = len(P.render(tok, r, header=h).ids)
        except ValueError as e: dropped["render"] += 1; print(f"  skip {r['id']}: {e}", file=sys.stderr); continue
        if n > max_tokens: dropped["too_long"] += 1; continue
        r["n_tok_span"] = n; r["n_tok"] = n; kept.append(r)
        if (i + 1) % 20000 == 0: print(f"  {label}: {i + 1}/{len(rows)} rows tokenised ({time.time() - t0:.0f} s)", flush=True)
    return kept, dropped


def write(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")


def stats(rows: list[dict], tokens_per_step: int) -> dict:
    n = [r["n_tok_span"] for r in rows]; n.sort()
    q = lambda p: n[min(len(n) - 1, int(p * len(n)))] if n else 0
    tot = sum(n)
    return dict(rows=len(rows), types=dict(Counter(r["type"] for r in rows)), groups=len({r["group"] for r in rows}),
                languages=dict(Counter(r.get("lang") or "?" for r in rows).most_common(12)), options_mean=round(sum(len(r["options"]) for r in rows) / max(1, len(rows)), 1),
                tokens_total=tot, tokens_p50=q(.5), tokens_p90=q(.9), tokens_max=n[-1] if n else 0,
                steps_per_epoch=math.ceil(tot / (tokens_per_step * 0.85)) if tot else 0)   # 0.85: packing efficiency of train.Plan


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pool", help="Votum-Pool dataset: local directory or Hub id")
    src.add_argument("--rows", help="your rows, JSONL (optionally gzipped), Pool schema")
    src.add_argument("--csv", help="a classification CSV: one text column and one label column (see --text-col, --label-col, --question, --labels)")
    ap.add_argument("--text-col", default="text"); ap.add_argument("--label-col", default="label")
    ap.add_argument("--question", default=None, help="with --csv: the question asked about every text, e.g. 'Which team should handle this ticket?'")
    ap.add_argument("--labels", default=None, help="with --csv: option texts per label, 'id=text;id=text' (default: the label values themselves)")
    ap.add_argument("--group-col", default=None, help="with --csv: column whose value groups rows about the same text (default: each row its own group)")
    ap.add_argument("--mix-pool", type=int, default=0, help="add this many randomly sampled Votum-Pool training rows (--pool-dir, config default) to the training file, to keep the general behaviour while fine-tuning on a narrow set")
    ap.add_argument("--pool-dir", default="votum-ai/Votum-Pool", help="with --mix-pool: Votum-Pool dataset, local directory or Hub id")
    ap.add_argument("--config", default="default", choices=["default", "share_alike"], help="Pool config (with --pool)")
    ap.add_argument("--val", help="validation rows, a JSONL file in the Pool schema (with --rows or --csv); default: a --val-frac split by group")
    ap.add_argument("--val-frac", type=float, default=0.05, help="validation share when --val is not given (default 0.05)")
    ap.add_argument("--out", required=True, help="output directory: train.jsonl.gz, val.jsonl.gz, stats.json")
    ap.add_argument("--tokenizer", default=None, help="tokenizer directory (default: the Apertus snapshot in HF_HOME, or $VOTUM_TOKENIZER)")
    ap.add_argument("--max-tokens", type=int, default=8192 - 64, help="drop rows whose prompt is longer (default 8128: the model's 8,192 context with a margin)")
    ap.add_argument("--tokens-per-step", default="auto", help="for the steps-per-epoch estimate: auto (finetune.py's rule: 65,536 for 2M+ tokens, 32,768 for 500k+, else 16,384) or a number")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv); out = Path(a.out)

    if a.csv:
        if not a.question: raise SystemExit("--csv needs --question (the question asked about every text)")
        train = [check(r, f"{a.csv}:{i}") for i, r in enumerate(csv_rows(Path(a.csv), a.text_col, a.label_col, a.question, a.labels, a.group_col), 1)]
        if a.val: val = [check(r, f"{a.val}:{i}") for i, r in enumerate(read_rows(Path(a.val)), 1)]
        else: train, val = split_rows(train, a.val_frac, a.seed)
    elif a.pool:
        train = [check(r, f"{a.pool}:train") for r in read_parquet(pool_files(a.pool, a.config, "train"))]
        val = [check(r, f"{a.pool}:validation") for r in read_parquet(pool_files(a.pool, a.config, "validation"))]
    else:
        train = [check(r, f"{a.rows}:{i}") for i, r in enumerate(read_rows(Path(a.rows)), 1)]
        if a.val: val = [check(r, f"{a.val}:{i}") for i, r in enumerate(read_rows(Path(a.val)), 1)]
        else: train, val = split_rows(train, a.val_frac, a.seed)
    if a.mix_pool:
        rng = random.Random(a.seed); pool = [check(r, f"{a.pool_dir}:train") for r in read_parquet(pool_files(a.pool_dir, "default", "train"))]
        mix = rng.sample(pool, min(a.mix_pool, len(pool))); train += mix
        print(f"mixed in {len(mix)} Votum-Pool training rows", flush=True)
    random.Random(a.seed).shuffle(train)             # always shuffled: a file sorted by label must not train label by label
    if a.val and not a.pool:                         # an explicit validation file is not split by group, so check it
        gt = {r["group"] for r in train}; st_tr = {json.dumps(r["state"], sort_keys=True, ensure_ascii=False) for r in train}
        leak = [r["id"] for r in val if r["group"] in gt or json.dumps(r["state"], sort_keys=True, ensure_ascii=False) in st_tr]
        if leak: print(f"WARNING: {len(leak)} validation rows share a group or a state with training rows (e.g. {leak[:3]}); their accuracy will look better than it is", flush=True)
    if not val: print("WARNING: no validation rows (raise --val-frac or pass --val): calibrate.py will have nothing to fit or report on", flush=True)
    ids = Counter(r["id"] for r in train + val); dup = [k for k, v in ids.items() if v > 1]
    if dup: raise SystemExit(f"row ids repeat: {dup[:5]}{' ...' if len(dup) > 5 else ''}")
    print(f"{len(train)} training rows, {len(val)} validation rows", flush=True)

    tok_dir = a.tokenizer or os.environ.get("VOTUM_TOKENIZER")
    if tok_dir is None:
        try: tok_dir = C.backbone_path(weights=False)[1]
        except FileNotFoundError: raise SystemExit("no tokenizer: pass --tokenizer <dir> (any Votum-8B or Apertus-1.5-8B directory) or set VOTUM_TOKENIZER")
    print(f"tokenizer: {tok_dir} (the first load of the Apertus tokenizer takes a few minutes)", flush=True)
    tok = C.load_tokenizer(Path(tok_dir))
    train, d1 = count_tokens(tok, train, a.max_tokens, "train"); val, d2 = count_tokens(tok, val, a.max_tokens, "val")
    write(train, out / "train.jsonl.gz"); write(val, out / "val.jsonl.gz")
    tot = sum(r["n_tok_span"] for r in train)
    tps = int(a.tokens_per_step) if a.tokens_per_step != "auto" else (65536 if tot >= 2_000_000 else 32768 if tot >= 500_000 else 16384)
    st = dict(source=a.pool or a.rows or a.csv, config=a.config if a.pool else None, mixed_pool_rows=a.mix_pool, tokenizer=str(tok_dir), max_tokens=a.max_tokens,
              tokens_per_step_estimate=tps, dropped=dict(train=dict(d1), val=dict(d2)), train=stats(train, tps), val=stats(val, tps))
    C.jdump(st, out / "stats.json")
    t = st["train"]
    print(f"wrote {out}/train.jsonl.gz ({t['rows']} rows, {t['tokens_total']:,} tokens, p50 {t['tokens_p50']}, p90 {t['tokens_p90']}) and val.jsonl.gz ({st['val']['rows']} rows)")
    print(f"dropped: {dict(d1)} / {dict(d2)}; one pass over the training rows is about {t['steps_per_epoch']} steps of {tps:,} tokens (finetune.py's default step size for this amount of data; a minimum of 40 steps applies)")


if __name__ == "__main__":
    main()
