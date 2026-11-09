"""Step 3: replace every e-mail address with [EMAIL], except the listed public mailboxes of courts and cantonal offices
(KEEP_ADDR: shared office addresses, each checked by hand; a person's mailbox on the same domain is masked) and the
reserved example.com/org/net domains, in the state, question and option texts of a pool directory (train.jsonl.gz,
val.jsonl.gz), in place.

    python -I email_sweep.py --pool-dir <dir written by pii_pass.py>

It catches addresses the source datasets' own masking missed (mostly in old Usenet posts)."""
import argparse, gzip, json, os, re
from pathlib import Path

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*\.[A-Za-z]{2,}")
KEEP_ADDR = {   # office mailboxes of Bernese courts and cantonal legal services, and Zurich's register of convictions
    "obergericht-straf.bern@justice.be.ch", "obergericht-zivil.bern@justice.be.ch", "coursupreme-penal.berne@justice.be.ch",
    "handelsgericht.bern@justice.be.ch", "info.ra.bvd@be.ch", "info.ra.gsi@be.ch", "vostra-pdf@ji.zh.ch",
}
KEEP_HOST = ("example.com", "example.org", "example.net")                 # reserved placeholder domains


def _kept(addr):
    addr = addr.lower(); host = addr.split("@")[1]
    return addr in KEEP_ADDR or any(host == d or host.endswith("." + d) for d in KEEP_HOST)


def sweep(x):
    if isinstance(x, str):
        return EMAIL.sub(lambda m: m.group(0) if _kept(m.group(0)) else "[EMAIL]", x)
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
        tmp = p.with_name(p.name + ".tmp")
        with gzip.open(tmp, "wt") as f: f.writelines(out)
        os.replace(tmp, p)                                    # never leaves a half-written file behind
        print(f"{p}: {changed} rows changed")


if __name__ == "__main__":
    main()
