#!/usr/bin/env python3
"""Stage the Hugging Face dataset directory (parquet configs + SOURCES.csv) from the masked pool.

usage: python3 -I build_hf.py --pool-dir clean_pii --policy pii_policy.json --out-dir Votum-Pool [--shard-mb 200]

Writes data/{default,share_alike}/{train,validation}-XXXXX-of-YYYYY.parquet (zstd) and SOURCES.csv into --out-dir, and
build_hf_stats.json (counts used by the dataset card) into --pool-dir, so --out-dir holds only repository files.
README.md is written separately. Needs pyarrow.

A policy entry may carry "license_fix": [[text found in the row's license_detail, licence tag], ...] (first match wins,
"" matches every row): the tag replaces the row's `license` where the pool's own tag was wrong. A row that lands in the
`default` config with a share-alike licence stops the build.
"""
import argparse, collections, csv, gzip, json, os, glob
import pyarrow as pa, pyarrow.parquet as pq

KEEP = ["id", "source", "license", "license_detail", "lang", "type", "state", "question", "options", "target", "soft", "group", "kind"]
SCHEMA = pa.schema([("id", pa.string()), ("source", pa.string()), ("license", pa.string()), ("license_detail", pa.string()), ("lang", pa.string()),
                    ("type", pa.string()), ("state", pa.string()), ("question", pa.string()),
                    ("options", pa.list_(pa.struct([("id", pa.string()), ("text", pa.string())]))), ("target", pa.list_(pa.float64())),
                    ("soft", pa.bool_()), ("group", pa.string()), ("kind", pa.string())])


class Sharder:
    def __init__(self, d, split, shard_bytes):
        self.d, self.split, self.lim, self.n, self.files, self.w, self.sz = d, split, shard_bytes, 0, [], None, 0

    def _open(self):
        p = os.path.join(self.d, f"{self.split}-tmp{self.n:05d}.parquet"); self.files.append(p)
        self.w = pq.ParquetWriter(p, SCHEMA, compression="zstd", compression_level=9); self.sz = 0; self.n += 1

    def write(self, rows):
        if not rows: return
        if self.w is None: self._open()
        t = pa.Table.from_pylist(rows, schema=SCHEMA); self.w.write_table(t, row_group_size=2000)
        self.sz += t.nbytes // 3          # rough estimate of the compressed size (zstd ratio of about 3)
        if self.sz >= self.lim: self.w.close(); self.w = None

    def close(self):
        if self.w: self.w.close(); self.w = None
        N = len(self.files)
        for i, p in enumerate(self.files):
            os.rename(p, os.path.join(self.d, f"{self.split}-{i:05d}-of-{N:05d}.parquet"))
        return N


SHARE_ALIKE = ("share-alike", "cc-by-sa", "odbl", "gpl")


def conv(r, fix=()):
    st = r["state"]
    o = {k: r.get(k) for k in KEEP}
    for needle, tag in fix:
        if needle in (r.get("license_detail") or ""): o["license"] = tag; break
    o["state"] = st if isinstance(st, str) else json.dumps(st, sort_keys=True, ensure_ascii=False)
    o["options"] = [{"id": str(x["id"]), "text": x["text"]} for x in r["options"]]
    o["target"] = [float(x) for x in r["target"]]
    o["soft"] = bool(r.get("soft"))
    return o


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool-dir", required=True); ap.add_argument("--policy", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shard-mb", type=int, default=200)
    a = ap.parse_args()
    pol = json.load(open(a.policy))
    stats = {"rows": collections.Counter(), "dropped_fields": collections.Counter(), "src": collections.defaultdict(lambda: collections.Counter()),
             "lang": collections.Counter(), "lic": collections.defaultdict(lambda: collections.Counter())}
    shards = {}
    for cfg in ("default", "share_alike"):
        os.makedirs(os.path.join(a.out_dir, "data", cfg), exist_ok=True)
        for f in glob.glob(os.path.join(a.out_dir, "data", cfg, "*.parquet")): os.remove(f)
    for split, fn in (("train", "train"), ("validation", "val")):
        sh = {c: Sharder(os.path.join(a.out_dir, "data", c), split, a.shard_mb * 1024 * 1024) for c in ("default", "share_alike")}
        buf = {c: [] for c in sh}
        with gzip.open(os.path.join(a.pool_dir, f"{fn}.jsonl.gz"), "rt", encoding="utf-8") as f:
            for l in f:
                r = json.loads(l); p = pol[r["source"]]
                assert not p["excluded"], r["source"]
                c = p["config"]
                for k in r:
                    if k not in KEEP: stats["dropped_fields"][k] += 1
                o = conv(r, p.get("license_fix") or ())
                if c == "default" and any(k in (o["license"] or "").lower() for k in SHARE_ALIKE):
                    raise SystemExit(f"{r['id']} ({r['source']}): licence {o['license']!r} is share-alike but the policy puts the source in `default`")
                buf[c].append(o); stats["rows"][(c, split)] += 1; stats["src"][r["source"]][split] += 1
                stats["lang"][(c, r.get("lang") or "unknown")] += 1; stats["lic"][c][o["license"]] += 1
                if len(buf[c]) >= 5000: sh[c].write(buf[c]); buf[c] = []
        for c in sh:
            sh[c].write(buf[c]); shards[(c, split)] = sh[c].close()
    with open(os.path.join(a.out_dir, "SOURCES.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["source", "config", "origin", "licence_tags", "train_rows", "validation_rows", "names_masked", "upstream_release", "description"])
        for s in sorted(stats["src"]):
            p = pol[s]; w.writerow([s, p["config"], p["origin"], p["license"], stats["src"][s]["train"], stats["src"][s]["validation"],
                                    "yes" if (p["origin"] == "self-collected" and p["names"]) else "no", p.get("upstream", ""), p.get("description", "")])
    out = {"rows": {f"{c}/{s}": v for (c, s), v in stats["rows"].items()}, "shards": {f"{c}/{s}": v for (c, s), v in shards.items()},
           "dropped_fields": dict(stats["dropped_fields"]), "lang": {f"{c}/{l}": v for (c, l), v in stats["lang"].items()},
           "lic": {c: dict(v) for c, v in stats["lic"].items()}}
    json.dump(out, open(os.path.join(a.pool_dir, "build_hf_stats.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: out[k] for k in ("rows", "shards", "dropped_fields")}, indent=1))


if __name__ == "__main__":
    main()
