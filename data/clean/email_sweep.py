#!/usr/bin/env python3
"""Step 3: replace every e-mail address left in the pool with [EMAIL], in place.

    python3 -I email_sweep.py --pool-dir clean_pii

Covers the state, question and option texts of train.jsonl.gz and val.jsonl.gz in --pool-dir (the directory written
by pii_pass.py), in every source, so it also catches the addresses the source datasets' own masking missed (mostly
in old Usenet posts). Two kinds of address are kept:
  - KEEP_ADDR: shared office mailboxes of courts and cantonal services, each checked by hand (a person's mailbox on
    the same domain is still masked);
  - the reserved placeholder domains example.com, example.org and example.net.

Rerunning it on its own output changes nothing. Standard library only.
"""
import argparse
import gzip
import json
import os
import re
from pathlib import Path

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*\.[A-Za-z]{2,}")
KEEP_ADDR = {   # office mailboxes of Bernese courts and cantonal legal services, and Zurich's register of convictions
    "obergericht-straf.bern@justice.be.ch", "obergericht-zivil.bern@justice.be.ch", "coursupreme-penal.berne@justice.be.ch",
    "handelsgericht.bern@justice.be.ch", "info.ra.bvd@be.ch", "info.ra.gsi@be.ch", "vostra-pdf@ji.zh.ch",
}
KEEP_HOST = ("example.com", "example.org", "example.net")                 # reserved placeholder domains


def _kept(addr):
    addr = addr.lower()
    host = addr.split("@")[1]
    return addr in KEEP_ADDR or any(host == d or host.endswith("." + d) for d in KEEP_HOST)


def sweep(x):
    """Mask the addresses in every string inside x (nested dicts and lists)."""
    if isinstance(x, str):
        return EMAIL.sub(lambda m: m.group(0) if _kept(m.group(0)) else "[EMAIL]", x)
    if isinstance(x, list):
        return [sweep(v) for v in x]
    if isinstance(x, dict):
        return {k: sweep(v) for k, v in x.items()}
    return x


def sweep_file(path):
    """Rewrite one .jsonl.gz file; unchanged rows keep their original line. -> number of rows changed"""
    out, changed = [], 0
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            new = dict(r, state=sweep(r["state"]), question=sweep(r["question"]), options=sweep(r["options"]))
            if new != r:
                changed += 1
                line = json.dumps(new, ensure_ascii=False) + "\n"
            out.append(line)
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(tmp, "wt") as f:
        f.writelines(out)
    os.replace(tmp, path)                                   # never leaves a half-written file behind
    return changed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool-dir", required=True, help="directory with train.jsonl.gz and val.jsonl.gz (rewritten in place)")
    a = ap.parse_args()
    for split in ("train", "val"):
        p = Path(a.pool_dir) / f"{split}.jsonl.gz"
        print(f"{p}: {sweep_file(p)} rows changed")


if __name__ == "__main__":
    main()
