"""What every stage shares: the config file, reading and writing .jsonl.gz in parallel, and walking the text of a row.

Rows travel as bytes lines. A stage reads a file once in the parent process, hands chunks of lines to worker
processes and writes what they return in the same order, so the output never depends on the number of workers.
Workers compress their own output: a .jsonl.gz written here is a series of gzip members, which every gzip reader
reads as one file.
"""
import gzip
import itertools
import json
import multiprocessing as mp
import os
import threading

CHUNK = 2000                                          # rows per unit of parallel work
SPLITS = ("train", "val")                             # a pool is a directory with <split>.jsonl.gz for each of these
STAGES = ("clean", "mask")                            # where a source's "drop" can take effect
ORIGINS = ("redistributed", "self-collected")


class PoolError(Exception):
    """A problem with the data or the config; the command line reports it without a traceback."""


# ----------------------------------------------------------------------------------------------- config
def load_config(path):
    """Read pool.json. -> the config dict, with `sources` checked and file paths made absolute."""
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    here = os.path.dirname(os.path.abspath(path))
    for name, p in cfg.get("sources", {}).items():
        if p.get("drop") is not None:
            if p["drop"] not in STAGES:
                raise PoolError(f"{path}: source {name!r}: \"drop\" must be one of {STAGES}")
            if not p.get("drop_reason"):
                raise PoolError(f"{path}: source {name!r} is dropped without a \"drop_reason\"")
        elif p.get("origin") not in ORIGINS or "config" not in p:
            raise PoolError(f"{path}: source {name!r} needs \"origin\" ({' or '.join(ORIGINS)}) and \"config\", or a \"drop\"")
    mask = cfg.setdefault("mask", {})
    if mask.get("stop_spans"):
        mask["stop_spans"] = os.path.join(here, mask["stop_spans"])
    return cfg


def policy(cfg, source, row_id=""):
    """The config entry of a source; every source in the data must have one."""
    p = cfg["sources"].get(source)
    if p is None:
        raise PoolError(f"source missing from the config: {source!r}" + (f" (row {row_id})" if row_id else ""))
    return p


def dropped_at(cfg, stage):
    """{source: reason} of the sources the config drops at this stage."""
    return {s: p["drop_reason"] for s, p in cfg["sources"].items() if p.get("drop") == stage}


# ----------------------------------------------------------------------------------------------- files
def split_path(pool_dir, split):
    return os.path.join(pool_dir, f"{split}.jsonl.gz")


def lines_of(path):
    """The lines of a .jsonl or .jsonl.gz file, as bytes, each ending in a newline."""
    with (gzip.open(path, "rb") if str(path).endswith(".gz") else open(path, "rb")) as f:
        for line in f:
            yield line if line.endswith(b"\n") else line + b"\n"


def chunks(path, rows=CHUNK):
    """-> lists of at most `rows` lines, in file order."""
    it = lines_of(path)
    while True:
        buf = list(itertools.islice(it, rows))
        if not buf:
            return
        yield buf


def gz_member(lines):
    """One gzip member holding these lines (no timestamp, so the same lines always give the same bytes)."""
    return gzip.compress(b"".join(lines), compresslevel=6, mtime=0) if lines else b""


class GzOut:
    """A .jsonl.gz written member by member; it appears under its name only once it is complete."""

    def __init__(self, path):
        self.path, self.tmp = str(path), str(path) + ".tmp"
        self.f, self.empty = open(self.tmp, "wb"), True

    def write(self, member):
        if member:
            self.f.write(member)
            self.empty = False

    def close(self):
        if self.empty:
            self.f.write(gzip.compress(b"", mtime=0))
        self.f.close()
        os.replace(self.tmp, self.path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        if exc_type is None:
            self.close()
        else:
            self.f.close()
            os.remove(self.tmp)


# ----------------------------------------------------------------------------------------------- workers
class Workers:
    """A pool of forked worker processes with an ordered map that reads its input only as fast as results are taken,
    so a file larger than memory can stream through. Workers see the module globals as they were when the pool was
    made: set a stage's shared state first, then create the pool. With n = 1 everything runs in this process."""

    def __init__(self, n):
        self.n = max(1, int(n))
        self.pool = mp.get_context("fork").Pool(self.n) if self.n > 1 else None

    def imap(self, fn, items):
        if self.pool is None:
            yield from map(fn, items)
            return
        slots, stop = threading.Semaphore(self.n * 3), threading.Event()

        def feed():
            for item in items:
                while not slots.acquire(timeout=0.2):
                    if stop.is_set():
                        return
                yield item

        try:
            for res in self.pool.imap(fn, feed(), chunksize=1):
                slots.release()
                yield res
        finally:
            stop.set()

    def close(self):
        if self.pool is not None:
            self.pool.terminate()
            self.pool.join()
            self.pool = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def default_workers():
    return max(1, (os.cpu_count() or 4) - 2)


# ----------------------------------------------------------------------------------------------- the text of a row
def strings(x, out):
    """Append every string inside x (nested dicts and lists) to out. -> out"""
    if isinstance(x, str):
        out.append(x)
    elif isinstance(x, dict):
        for v in x.values():
            strings(v, out)
    elif isinstance(x, (list, tuple)):
        for v in x:
            strings(v, out)
    return out


def map_strings(x, fn):
    """Apply fn to every string inside x, keeping the structure."""
    if isinstance(x, dict):
        return {k: map_strings(v, fn) for k, v in x.items()}
    if isinstance(x, list):
        return [map_strings(v, fn) for v in x]
    return fn(x) if isinstance(x, str) else x


def row_texts(row):
    """The texts of a row: every string of `state`, the `question`, and each option's `text`."""
    out = strings(row.get("state"), [])
    out.append(row.get("question"))
    for o in row.get("options") or []:
        out.append(o.get("text"))
    return [t for t in out if isinstance(t, str)]


def dump(row):
    """A row as an output line."""
    return (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
