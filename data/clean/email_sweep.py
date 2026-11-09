"""Step 3: replace every e-mail address with [EMAIL], except public court and cantonal addresses (*.be.ch, *.zh.ch) and
the reserved example.com/org/net domains, in the state, question and option texts of a pool directory (train.jsonl.gz,
val.jsonl.gz), in place.

    python -I email_sweep.py --pool-dir <dir written by pii_pass.py>

It catches addresses the source datasets' own masking missed (mostly in old Usenet posts)."""
import argparse, gzip, json, re
from pathlib import Path

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
KEEP = ("be.ch", "zh.ch", "example.com", "example.org", "example.net")   # public court / cantonal domains, placeholders


def _kept(host):
    return any(host == d or host.endswith("." + d) for d in KEEP)


def sweep(x):
    if isinstance(x, str):
        return EMAIL.sub(lambda m: m.group(0) if _kept(m.group(0).split("@")[1].lower()) else "[EMAIL]", x)
    if isinstance(x, list): return [sweep(v) for v in x]
    if isinstance(x, dict): return {k: sweep(v) for k, v in x.items()}
    return x


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pool-dir", required=True); a = ap.parse_args()
    for split in ("train", "val"):
        p = Path(a.pool_dir) / f"{split}.jsonl.gz"; out, changed = [], 0
        with gzip.open(p, "rt") as f:
            for line in f:
                r = json.loads(line); new = dict(r, state=sweep(r["state"]), question=sweep(r["question"]), options=sweep(r["options"]))
                if new != r: changed += 1; line = json.dumps(new, ensure_ascii=False) + "\n"
                out.append(line)
        with gzip.open(p, "wt") as f: f.writelines(out)
        print(f"{p}: {changed} rows changed")


if __name__ == "__main__":
    main()
