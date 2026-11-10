#!/usr/bin/env python3
"""Step 4: write the Hugging Face dataset files (two parquet configs and SOURCES.csv) from the masked pool.

    python3 -I build_hf.py --pool-dir clean_pii --policy pii_policy.json --out-dir Votum-Pool [--shard-mb 200]

Writes into --out-dir:
  data/default/{train,validation}-XXXXX-of-YYYYY.parquet       sources whose policy says "config": "default"
  data/share_alike/{train,validation}-XXXXX-of-YYYYY.parquet   sources under share-alike terms
  SOURCES.csv                                                  one line per source
and build_hf_stats.json (the counts the dataset card quotes) into --pool-dir, so that --out-dir holds only repository
files. The dataset card (README.md) is written by hand.

Rows keep the fields in KEEP; the other pool fields (token counts, annotation details, ...) are dropped and counted.
A JSON state is stored as a JSON string with sorted keys. A policy entry may carry
"license_fix": [[text found in the row's license_detail, licence tag], ...] (first match wins; "" matches every row):
the tag replaces the row's `license` where the pool's own tag was wrong. The build stops if a share-alike row would
land in the `default` config or if an excluded source is still in the input.

Needs pyarrow.
"""
import argparse
import collections
import csv
import glob
import gzip
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq

KEEP = ["id", "source", "license", "license_detail", "lang", "type", "state", "question", "options", "target", "soft", "group", "kind"]
SCHEMA = pa.schema([("id", pa.string()), ("source", pa.string()), ("license", pa.string()), ("license_detail", pa.string()),
                    ("lang", pa.string()), ("type", pa.string()), ("state", pa.string()), ("question", pa.string()),
                    ("options", pa.list_(pa.struct([("id", pa.string()), ("text", pa.string())]))),
                    ("target", pa.list_(pa.float64())), ("soft", pa.bool_()), ("group", pa.string()), ("kind", pa.string())])
SHARE_ALIKE = ("share-alike", "cc-by-sa", "odbl", "gpl")       # licence-tag substrings that must not reach `default`
CONFIGS = ("default", "share_alike")


class Sharder:
    """Write one split of one config as parquet shards of roughly `shard_bytes`, named *-XXXXX-of-YYYYY on close."""

    def __init__(self, d, split, shard_bytes):
        self.d, self.split, self.lim = d, split, shard_bytes
        self.n, self.files, self.w, self.sz = 0, [], None, 0

    def _open(self):
        p = os.path.join(self.d, f"{self.split}-tmp{self.n:05d}.parquet")
        self.files.append(p)
        self.w = pq.ParquetWriter(p, SCHEMA, compression="zstd", compression_level=9)
        self.sz = 0
        self.n += 1

    def write(self, rows):
        if not rows:
            return
        if self.w is None:
            self._open()
        t = pa.Table.from_pylist(rows, schema=SCHEMA)
        self.w.write_table(t, row_group_size=2000)
        self.sz += t.nbytes // 3                                # rough compressed size (zstd ratio of about 3)
        if self.sz >= self.lim:
            self.w.close()
            self.w = None

    def close(self):
        """-> number of shards"""
        if self.w:
            self.w.close()
            self.w = None
        n = len(self.files)
        for i, p in enumerate(self.files):
            os.rename(p, os.path.join(self.d, f"{self.split}-{i:05d}-of-{n:05d}.parquet"))
        return n


def conv(r, fix=()):
    """A pool row -> a dataset row (KEEP fields, typed for SCHEMA, licence correction applied)."""
    st = r["state"]
    o = {k: r.get(k) for k in KEEP}
    for needle, tag in fix:
        if needle in (r.get("license_detail") or ""):
            o["license"] = tag
            break
    o["state"] = st if isinstance(st, str) else json.dumps(st, sort_keys=True, ensure_ascii=False)
    o["options"] = [{"id": str(x["id"]), "text": x["text"]} for x in r["options"]]
    o["target"] = [float(x) for x in r["target"]]
    o["soft"] = bool(r.get("soft"))
    return o


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool-dir", required=True, help="directory with train.jsonl.gz and val.jsonl.gz (after email_sweep.py)")
    ap.add_argument("--policy", required=True, help="pii_policy.json")
    ap.add_argument("--out-dir", required=True, help="the dataset repository directory")
    ap.add_argument("--shard-mb", type=int, default=200, help="approximate parquet shard size")
    a = ap.parse_args()
    with open(a.policy) as f:
        pol = json.load(f)

    rows = collections.Counter()
    dropped_fields = collections.Counter()
    per_src = collections.defaultdict(collections.Counter)
    langs = collections.Counter()
    lics = collections.defaultdict(collections.Counter)
    shards = {}
    for cfg in CONFIGS:
        os.makedirs(os.path.join(a.out_dir, "data", cfg), exist_ok=True)
        for f in glob.glob(os.path.join(a.out_dir, "data", cfg, "*.parquet")):
            os.remove(f)

    for split, fn in (("train", "train"), ("validation", "val")):
        sh = {c: Sharder(os.path.join(a.out_dir, "data", c), split, a.shard_mb * 1024 * 1024) for c in CONFIGS}
        buf = {c: [] for c in CONFIGS}
        with gzip.open(os.path.join(a.pool_dir, f"{fn}.jsonl.gz"), "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                p = pol[r["source"]]
                if p["excluded"]:
                    raise SystemExit(f"{r['id']}: source {r['source']!r} is excluded by the policy; run pii_pass.py first")
                c = p["config"]
                for k in r:
                    if k not in KEEP:
                        dropped_fields[k] += 1
                o = conv(r, p.get("license_fix") or ())
                if c == "default" and any(k in (o["license"] or "").lower() for k in SHARE_ALIKE):
                    raise SystemExit(f"{r['id']} ({r['source']}): licence {o['license']!r} is share-alike but the policy puts the source in `default`")
                buf[c].append(o)
                rows[(c, split)] += 1
                per_src[r["source"]][split] += 1
                langs[(c, r.get("lang") or "unknown")] += 1
                lics[c][o["license"]] += 1
                if len(buf[c]) >= 5000:
                    sh[c].write(buf[c])
                    buf[c] = []
        for c in CONFIGS:
            sh[c].write(buf[c])
            shards[(c, split)] = sh[c].close()

    with open(os.path.join(a.out_dir, "SOURCES.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "config", "origin", "licence_tags", "train_rows", "validation_rows", "names_masked", "upstream_release", "description"])
        for s in sorted(per_src):
            p = pol[s]
            w.writerow([s, p["config"], p["origin"], p["license"], per_src[s]["train"], per_src[s]["validation"],
                        "yes" if (p["origin"] == "self-collected" and p["names"]) else "no", p.get("upstream", ""), p.get("description", "")])
    out = {"rows": {f"{c}/{s}": v for (c, s), v in rows.items()},
           "shards": {f"{c}/{s}": v for (c, s), v in shards.items()},
           "dropped_fields": dict(dropped_fields),
           "lang": {f"{c}/{lang}": v for (c, lang), v in langs.items()},
           "lic": {c: dict(v) for c, v in lics.items()}}
    with open(os.path.join(a.pool_dir, "build_hf_stats.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: out[k] for k in ("rows", "shards", "dropped_fields")}, indent=1))


if __name__ == "__main__":
    main()
