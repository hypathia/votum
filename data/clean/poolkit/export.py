"""Stage `export`: write the Hugging Face dataset files from the masked pool.

Writes into the output directory:
  data/<config>/{train,validation}-XXXXX-of-YYYYY.parquet   one directory per dataset config; a source's entry in
                                                             the config file says which one its rows go to
  SOURCES.csv                                                one line per source
and export_stats.json (the counts the dataset card quotes) into the pool directory, so that the output directory
holds only repository files. The dataset card (README.md) is written by hand.

Rows keep the fields in KEEP; the other pool fields (token counts, annotation details, ...) are dropped and counted.
A JSON state is stored as a JSON string with sorted keys. A source's entry may carry
"license_fix": [[text found in the row's license_detail, licence tag], ...] (first match wins; "" matches every row):
the tag replaces the row's `license` where the pool's own tag was wrong. A flag (name, file of row ids) adds a boolean
column that is true for the rows whose id is in the file. The export stops if a dropped source is still in the input,
or if a row's licence contains a substring that `export.license_guard` forbids in its config.

Rows are converted in parallel; each config is written by its own thread, in input order.
"""
import collections
import csv
import glob
import gzip
import json
import os
import queue
import threading
import time

import pyarrow as pa
import pyarrow.parquet as pq

from . import engine

KEEP = ["id", "source", "license", "license_detail", "lang", "type", "state", "question", "options", "target", "soft", "group", "kind"]
SCHEMA = pa.schema([("id", pa.string()), ("source", pa.string()), ("license", pa.string()), ("license_detail", pa.string()),
                    ("lang", pa.string()), ("type", pa.string()), ("state", pa.string()), ("question", pa.string()),
                    ("options", pa.list_(pa.struct([("id", pa.string()), ("text", pa.string())]))),
                    ("target", pa.list_(pa.float64())), ("soft", pa.bool_()), ("group", pa.string()), ("kind", pa.string())])
HF_SPLIT = {"train": "train", "val": "validation"}
WRITE_ROWS, ROW_GROUP = 5000, 2000                  # rows handed to the parquet writer at a time; rows per row group
G = {}                                              # state shared with the workers (set before a pool is made)


def read_ids(path):
    """The row ids of a text file (one per line, plain or gzip)."""
    with (gzip.open if str(path).endswith(".gz") else open)(path, "rt", encoding="utf-8") as f:
        return {line.rstrip("\n") for line in f if line.strip()}


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


def _to_ipc(table):
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as w:
        w.write_table(table)
    return sink.getvalue().to_pybytes()


def _export_worker(lines):
    """-> ({config: the chunk's rows as an Arrow table, serialised}, the chunk's counts)"""
    cfg, keep, flags, guard = G["cfg"], G["keep"], G["flags"], G["guard"]
    per = collections.defaultdict(list)
    n = {"src": collections.Counter(), "lang": collections.Counter(), "lic": collections.Counter(),
         "dropped_fields": collections.Counter(), "flag": collections.Counter()}
    for line in lines:
        r = json.loads(line)
        p = engine.policy(cfg, r["source"], r.get("id"))
        if p.get("drop"):
            raise engine.PoolError(f"{r['id']}: source {r['source']!r} is dropped by the config; run the clean and mask stages first")
        c = p["config"]
        for k in r:
            if k not in keep:
                n["dropped_fields"][k] += 1
        o = conv(r, p.get("license_fix") or ())
        if any(k in (o["license"] or "").lower() for k in guard.get(c, ())):
            raise engine.PoolError(f"{r['id']} ({r['source']}): licence {o['license']!r} is not allowed in the config `{c}` "
                                   "(export.license_guard)")
        for name, ids in flags:
            o[name] = r["id"] in ids
            n["flag"][(name, c)] += o[name]
        per[c].append(o)
        n["src"][r["source"]] += 1
        n["lang"][(c, r.get("lang") or "unknown")] += 1
        n["lic"][(c, o["license"])] += 1
    return {c: _to_ipc(pa.Table.from_pylist(rows, schema=G["schema"])) for c, rows in per.items()}, n


class ShardWriter(threading.Thread):
    """Writes one split of one config as parquet shards of roughly `shard_bytes`, named *-XXXXX-of-YYYYY at the end.
    Tables are queued by put() and written by this thread, WRITE_ROWS rows at a time."""

    def __init__(self, directory, split, shard_bytes, schema):
        super().__init__(daemon=True)
        self.d, self.split, self.lim, self.schema = directory, split, shard_bytes, schema
        self.q, self.error = queue.Queue(maxsize=16), None
        self.files, self.w, self.sz = [], None, 0
        self.start()

    def put(self, table):
        if self.error:
            raise self.error
        self.q.put(table)

    def finish(self):
        """-> number of shards"""
        self.q.put(None)
        self.join()
        if self.error:
            raise self.error
        n = len(self.files)
        for i, p in enumerate(self.files):
            os.rename(p, os.path.join(self.d, f"{self.split}-{i:05d}-of-{n:05d}.parquet"))
        return n

    def _write(self, t):
        if self.w is None:
            p = os.path.join(self.d, f"{self.split}-tmp{len(self.files):05d}.parquet")
            self.files.append(p)
            self.w, self.sz = pq.ParquetWriter(p, self.schema, compression="zstd", compression_level=9), 0
        t = t.combine_chunks()
        self.w.write_table(t, row_group_size=ROW_GROUP)
        self.sz += t.nbytes // 3                            # rough compressed size (zstd ratio of about 3)
        if self.sz >= self.lim:
            self.w.close()
            self.w = None

    def run(self):
        buf, rows = [], 0
        try:
            while True:
                t = self.q.get()
                if t is None:
                    break
                buf.append(t)
                rows += t.num_rows
                if rows >= WRITE_ROWS:
                    whole = pa.concat_tables(buf)
                    for a in range(0, rows - rows % WRITE_ROWS, WRITE_ROWS):
                        self._write(whole.slice(a, WRITE_ROWS))
                    rest = whole.slice(rows - rows % WRITE_ROWS)
                    buf, rows = ([rest] if rest.num_rows else []), rest.num_rows
            if rows:
                self._write(pa.concat_tables(buf))
            if self.w:
                self.w.close()
        except BaseException as e:                          # handed to the main thread by put() or finish()
            self.error = e
            while self.q.get() is not None:
                pass


def run(cfg, pool_dir, out_dir, flags=(), shard_mb=200, workers=None):
    """Run the stage on <pool_dir>/{train,val}.jsonl.gz. flags: (column name, path of an ids file) pairs. -> the stats"""
    t_start = time.time()
    ex = cfg.get("export", {})
    configs = list(ex.get("configs") or sorted({p["config"] for p in cfg["sources"].values() if not p.get("drop")}))
    flags = [(name, read_ids(path)) for name, path in list(ex.get("flags", {}).items()) + list(flags)]
    schema = SCHEMA
    for name, _ in flags:
        schema = schema.append(pa.field(name, pa.bool_()))
    G.update(cfg=cfg, keep=frozenset(KEEP), flags=flags, schema=schema,
             guard={c: tuple(x.lower() for x in v) for c, v in ex.get("license_guard", {}).items()})
    for c in configs:
        os.makedirs(os.path.join(out_dir, "data", c), exist_ok=True)
        for f in glob.glob(os.path.join(out_dir, "data", c, "*.parquet")):
            os.remove(f)

    rows, shards, per_src = collections.Counter(), {}, collections.defaultdict(collections.Counter)
    total = {"lang": collections.Counter(), "lic": collections.Counter(), "dropped_fields": collections.Counter(), "flag": collections.Counter()}
    flag_split = collections.Counter()
    with engine.Workers(workers or engine.default_workers()) as w:
        for split in engine.SPLITS:
            hf = HF_SPLIT[split]
            writers = {c: ShardWriter(os.path.join(out_dir, "data", c), hf, shard_mb * 1024 * 1024, schema) for c in configs}
            try:
                for tables, n in w.imap(_export_worker, engine.chunks(engine.split_path(pool_dir, split))):
                    for c, buf in tables.items():
                        if c not in writers:
                            raise engine.PoolError(f"config {c!r} is not listed in export.configs")
                        t = pa.ipc.open_stream(buf).read_all()
                        rows[(c, hf)] += t.num_rows
                        writers[c].put(t)
                    for s, v in n["src"].items():
                        per_src[s][hf] += v
                    for k in ("lang", "lic", "dropped_fields"):
                        total[k].update(n[k])
                    for (name, c), v in n["flag"].items():
                        flag_split[(name, c, hf)] += v
            finally:
                for c, wr in writers.items():
                    shards[(c, hf)] = wr.finish()

    with open(os.path.join(out_dir, "SOURCES.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["source", "config", "origin", "licence_tags", "train_rows", "validation_rows", "names_masked", "upstream_release", "description"])
        for s in sorted(per_src):
            p = cfg["sources"][s]
            wr.writerow([s, p["config"], p["origin"], p["license"], per_src[s]["train"], per_src[s]["validation"],
                         "yes" if (p["origin"] == "self-collected" and p.get("names")) else "no", p.get("upstream", ""), p.get("description", "")])
    lic = collections.defaultdict(dict)
    for (c, tag), v in total["lic"].items():
        lic[c][tag] = v
    out = {"rows": {f"{c}/{s}": v for (c, s), v in rows.items()},
           "shards": {f"{c}/{s}": v for (c, s), v in shards.items()},
           "dropped_fields": dict(total["dropped_fields"]),
           **{name: {f"{c}/{s}": v for (nm, c, s), v in flag_split.items() if nm == name} for name, _ in flags},
           "lang": {f"{c}/{lang}": v for (c, lang), v in total["lang"].items()},
           "lic": dict(lic)}
    with open(os.path.join(pool_dir, "export_stats.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"export: {sum(rows.values()):,} rows in {sum(shards.values())} parquet files, {len(per_src)} sources, {time.time() - t_start:.0f} s", flush=True)
    return out
