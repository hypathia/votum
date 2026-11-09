#!/usr/bin/env python3
"""Build the cleaned Votum training pool from the pool v7 files (step 1 of data/clean/README.md).

CPU only, no network.  Needs numpy.
usage: python3 -I build_clean.py --train TRAIN --val VAL --out-dir OUT --bench FILE [FILE ...] [options; see --help]

Rules (applied in this order; rows are never reordered or rewritten, kept lines are
copied byte for byte so the trainer reads them exactly as before):

  R1  source drop      whole sources removed (second licence review, bias_in_bios,
                       hh_redteam, expertqa, HelpSteer2/3, Aegis 2.0).  The list below is
                       explicit; the script also checks it against the name patterns and
                       aborts if a pattern matches a source that is not listed.
  R2  ContractNLI test every row with source == "ContractNLI" and upstream_split == "test".
                       The pool carries the upstream split on every ContractNLI row.
  R3  document-level   a row is dropped if its STATE shares at least one normalised word
      decontamination  8-gram with the state/context/passage text of any benchmark item
                       (questions are never used).  Normalisation: lowercase, split on
                       runs of word characters (so punctuation and whitespace vanish).
                       Boilerplate 8-grams (occurring in more than ``--stop-min-rows``
                       pool rows from at least ``--stop-min-sources`` sources) are put on
                       an automatic stop-list and ignored; the stop-list is written to
                       disk for review.  Per source the rate without and with the
                       stop-list is reported, and the top offending 8-grams are listed
                       for every source that would lose more than ``--warn-frac`` rows.

Benchmark files (``--bench``) are jsonl or jsonl.gz.  The state text of a record is read
from ``record.state`` (benchmark records), ``row.state`` (evaluation item exports and
out-of-distribution sets) or ``state`` (suite rows); every string
leaf is hashed separately so no 8-gram spans two fields.  A benchmark label is
``<file stem>`` plus ``/<_evaluation.dataset or set>`` when the record names one.
Benchmark labels given with ``--info-labels`` are only measured (overlap is reported), never used to
drop rows.  Items with fewer than 8 normalised words cannot match; their number is
reported per benchmark.

Hashing: every word -> 64-bit blake2b (stable across runs), 8-gram -> polynomial
combination mod 2**64.  A collision would drop a row spuriously; at ~1e8 benchmark grams
the expected number of false matches is below 1e-3 per run.
"""
import argparse
import collections
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import re
import sys
import time

import numpy as np

N = 8
MULT = np.uint64(0x9E3779B97F4A7C15)
WRE = re.compile(r"\w+", re.U)

# ---- Rule 1: exact source names, taken from the actual `source` values of pool v7 ----
# The 13 sources removed by the second licence review (19,950 training rows in total); they are present in the
# pool v7 files but were excluded from the build specification after review.
LICENCE_REVIEW_13 = [
    "expand/nhtsa", "tasksource/circa", "rm_quotidiana", "HopeEDI Tamil + Malayalam (LT-EDI)",
    "offenseval_dravidian", "decider/bias_in_bios", "prdect_id", "de_fr_it/frenchmedmcqa",
    "tasksource/disrpt/eng.dep.scidtb.rels", "tasksource/fracas", "de_fr_it/medexpqa",
    "tasksource/maven-arg/role", "tasksource/scone",
]
OTHER_DROPS = {
    "bias_in_bios (privacy; excluded in the build specification)": ["decider/bias_in_bios"],
    "hh_redteam (card: not for fine-tuning)": ["expand/hh_redteam"],
    "expertqa (GPT-4/Bing generated)": ["expand/expertqa"],
    "HelpSteer2 (LLM-generated)": [
        "lab/helpsteer2", "lab/helpsteer2_pref", "tasksource/HelpSteer2/coherence",
        "tasksource/HelpSteer2/complexity", "tasksource/HelpSteer2/correctness",
        "tasksource/HelpSteer2/helpfulness", "tasksource/HelpSteer2/verbosity"],
    "HelpSteer3 (LLM-generated)": [
        "lab/helpsteer3_pref", "tasksource/HelpSteer3/edit_quality", "tasksource/HelpSteer3/feedback",
        "tasksource/HelpSteer3/preference_strength", "tasksource/HelpSteer3/principle"],
    "Aegis 2.0": ["expand/aegis2"],
}
# Sanity patterns (case-insensitive substrings): every source they match must be listed above.
NAME_PATTERNS = ["hh_redteam", "expertqa", "helpsteer2", "helpsteer3", "aegis", "bias_in_bios"]


def drop_table():
    t = {}
    for s in LICENCE_REVIEW_13:
        t[s] = "second licence review (13 sources)"
    for why, names in OTHER_DROPS.items():
        for s in names:
            t.setdefault(s, why)
    return t


# ---------------------------------------------------------------- hashing helpers
_WC = {}


def _wh(w):
    h = _WC.get(w)
    if h is None:
        h = int.from_bytes(hashlib.blake2b(w.encode("utf-8", "surrogatepass"), digest_size=8).digest(), "little")
        _WC[w] = h
    return h


def leaves(x, out):
    if isinstance(x, str):
        out.append(x)
    elif isinstance(x, dict):
        for v in x.values():
            leaves(v, out)
    elif isinstance(x, (list, tuple)):
        for v in x:
            leaves(v, out)
    return out


def gram_hashes(text):
    """uint64 hashes of every normalised word 8-gram of ``text`` (may contain repeats)."""
    ws = WRE.findall(text.lower())
    n = len(ws)
    if n < N:
        return np.empty(0, np.uint64)
    h = np.fromiter((_wh(w) for w in ws), dtype=np.uint64, count=n)
    acc = np.zeros(n - N + 1, np.uint64)
    with np.errstate(over="ignore"):
        for k in range(N):
            acc = acc * MULT + h[k:n - N + 1 + k]
    return acc


def state_grams(state):
    parts = [gram_hashes(t) for t in leaves(state, [])]
    parts = [p for p in parts if len(p)]
    return np.unique(np.concatenate(parts)) if parts else np.empty(0, np.uint64)


def gram_texts(state, want):
    """{hash: 'w1 .. w8'} for the hashes in ``want`` found in ``state``."""
    out = {}
    for t in leaves(state, []):
        ws = WRE.findall(t.lower())
        if len(ws) < N:
            continue
        g = gram_hashes(t)
        for i in np.nonzero(np.isin(g, want))[0]:
            out[int(g[i])] = " ".join(ws[i:i + N])
    return out


def opener(p):
    return gzip.open(p, "rt", encoding="utf-8") if p.endswith(".gz") else open(p, encoding="utf-8")


def batches(path, size):
    buf, start = [], 0
    with opener(path) as f:
        for i, line in enumerate(f):
            buf.append(line)
            if len(buf) == size:
                yield start, buf
                buf, start = [], i + 1
    if buf:
        yield start, buf


# ---------------------------------------------------------------- benchmark side
def bench_state(rec):
    if isinstance(rec.get("record"), dict) and "state" in rec["record"]:
        return rec["record"]["state"]
    if isinstance(rec.get("row"), dict) and "state" in rec["row"]:
        return rec["row"]["state"]
    if "state" in rec:
        return rec["state"]
    return None


def bench_label(stem, rec):
    sub = None
    if isinstance(rec.get("_evaluation"), dict):
        sub = rec["_evaluation"].get("dataset")
    if sub is None:
        sub = rec.get("set")
    return f"{stem}/{sub}" if sub else stem


def bench_instructions(rec):
    """Question instruction strings of a record (only used by the fallback for items without state text)."""
    qs = rec.get("questions")
    if qs is None and isinstance(rec.get("record"), dict):
        qs = rec["record"].get("questions")
    out = []
    if isinstance(qs, dict):
        for q in qs.values():
            if isinstance(q, dict) and isinstance(q.get("instructions"), str):
                out.append(q["instructions"])
    return out


def _bench_worker(args):
    stem, lines = args
    per, items, short, unknown = {}, collections.Counter(), collections.Counter(), 0
    for line in lines:
        rec = json.loads(line)
        st = bench_state(rec)
        lab = bench_label(stem, rec)
        if st is None:
            unknown += 1
            continue
        g = state_grams(st)
        items[lab] += 1
        if not len(g):
            short[lab] += 1
            if G.get("fallback"):
                # Many benchmark datasets (ANLI, VAST, MuSR, MMLU, HellaSwag, ...) keep the whole
                # item text in question.instructions and have an empty state.  Without this fallback
                # they could not be decontaminated at all.
                fl = lab + "[instr]"
                items[fl] += 1
                parts = [gram_hashes(t) for t in bench_instructions(rec)]
                parts = [p for p in parts if len(p)]
                if parts:
                    per.setdefault(fl, []).append(np.unique(np.concatenate(parts)))
                else:
                    short[fl] += 1
        else:
            per.setdefault(lab, []).append(g)
    per = {k: np.unique(np.concatenate(v)) for k, v in per.items()}
    return per, items, short, unknown


def build_bench(paths, pool, tag):
    """-> (sorted unique hashes, pair hashes, pair bench ids, labels, item stats)."""
    labels, parts, stats = [], collections.defaultdict(list), {}
    unknown = collections.Counter()
    for p in paths:
        stem = os.path.basename(p).split(".")[0]
        t0 = time.time()
        for per, items, short, unk in pool.imap(_bench_worker, ((stem, b) for _, b in batches(p, 200)), chunksize=1):
            unknown[stem] += unk
            for lab, n in items.items():
                s = stats.setdefault(lab, {"items": 0, "items_lt8_words": 0, "file": p})
                s["items"] += n
                s["items_lt8_words"] += short.get(lab, 0)
            for lab, a in per.items():
                parts[lab].append(a)
        print(f"[{tag}] {p}: {time.time() - t0:.0f}s", flush=True)
    hs, bs = [], []
    for lab, lst in parts.items():
        a = np.unique(np.concatenate(lst))
        labels.append(lab)
        hs.append(a)
        bs.append(np.full(len(a), len(labels) - 1, np.uint16))
        stats[lab]["distinct_8grams"] = int(len(a))
    for lab in stats:
        stats[lab].setdefault("distinct_8grams", 0)
        if lab not in labels:
            labels.append(lab)
    if hs:
        h, b = np.concatenate(hs), np.concatenate(bs)
        o = np.argsort(h, kind="stable")
        h, b = h[o], b[o]
        uh = np.unique(h)
    else:
        h, b, uh = (np.empty(0, np.uint64), np.empty(0, np.uint16), np.empty(0, np.uint64))
    return uh, h, b, labels, stats, dict(unknown)


# ---------------------------------------------------------------- pool side
G = {}


def _isin_sorted(uh, q):
    if not len(uh) or not len(q):
        return np.zeros(len(q), bool)
    i = np.searchsorted(uh, q)
    i[i == len(uh)] = len(uh) - 1
    return uh[i] == q


def _pool_worker(args):
    start, lines = args
    drop = G["drop"]
    out = []
    for k, line in enumerate(lines):
        r = json.loads(line)
        src = r.get("source")
        reason, hit, info = None, None, ()
        if src in drop:
            reason = "R1:" + drop[src]
        elif src == "ContractNLI" and r.get("upstream_split") == "test":
            reason = "R2:ContractNLI test split"
        else:
            g = state_grams(r.get("state"))
            if len(g):
                m = g[_isin_sorted(G["uh"], g)]
                hit = m if len(m) else None
                if len(G["iuh"]):
                    mi = g[_isin_sorted(G["iuh"], g)]
                    if len(mi):
                        lo = np.searchsorted(G["ih"], mi, "left")
                        hi = np.searchsorted(G["ih"], mi, "right")
                        info = tuple(sorted({int(G["ib"][j]) for a, c in zip(lo, hi) for j in range(a, c)}))
        out.append((start + k, src, r.get("license"), reason, hit, info, r.get("group")))
    return out


def scan_pool(path, pool, tag):
    rows = []
    t0 = time.time()
    for res in pool.imap(_pool_worker, batches(path, 400), chunksize=1):
        rows.extend(res)
    print(f"[{tag}] scanned {len(rows)} rows in {time.time() - t0:.0f}s", flush=True)
    return rows


def expand(hits, h, b):
    """hits: list of (row_pos, uint64 array) -> arrays (row_pos, hash, bench) with one line per bench."""
    rp, hh = [], []
    for pos, a in hits:
        rp.append(np.full(len(a), pos, np.int64))
        hh.append(a)
    if not rp:
        z = np.empty(0, np.int64)
        return z, np.empty(0, np.uint64), np.empty(0, np.int64)
    rp, hh = np.concatenate(rp), np.concatenate(hh)
    lo, hi = np.searchsorted(h, hh, "left"), np.searchsorted(h, hh, "right")
    cnt = hi - lo
    rr = np.repeat(rp, cnt)
    hr = np.repeat(hh, cnt)
    idx = np.repeat(lo, cnt) + (np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt))
    return rr, hr, b[idx].astype(np.int64)


def top_texts(path, rows_needed, want):
    """re-read ``path`` and recover the 8-gram text of the hashes in ``want`` from the given row indices."""
    need = set(rows_needed)
    want = np.array(sorted(want), np.uint64)
    out = {}
    with opener(path) as f:
        for i, line in enumerate(f):
            if i in need:
                out.update(gram_texts(json.loads(line).get("state"), want))
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True, help="pool train.jsonl.gz")
    ap.add_argument("--val", required=True, help="pool val.jsonl.gz")
    ap.add_argument("--out-dir", required=True, help="writes train.jsonl.gz, val.jsonl.gz, REPORT.json, REPORT.md, stoplist.tsv, dropped.tsv.gz")
    ap.add_argument("--bench", nargs="+", required=True, help="benchmark jsonl[.gz] files (state text is decontaminated against)")
    ap.add_argument("--info-labels", nargs="*", default=[], help="benchmark labels (see above, e.g. <file stem>/<set>) that are only measured, never used to drop")
    ap.add_argument("--val-released", default=None, help="optional: released val split, to check the cleaned val against")
    ap.add_argument("--stop-min-rows", type=int, default=50)
    ap.add_argument("--stop-min-sources", type=int, default=5)
    ap.add_argument("--warn-frac", type=float, default=0.10)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--no-instruction-fallback", action="store_true", help="do not use question.instructions of benchmark items that have no state text")
    ap.add_argument("--report-only", action="store_true", help="do not write train/val.jsonl.gz (sensitivity runs)")
    ap.add_argument("--no-stoplist", action="store_true", help="disable the stop-list (plain 8-gram rule)")
    a = ap.parse_args()
    T0 = time.time()
    os.makedirs(a.out_dir, exist_ok=True)
    drop = drop_table()

    # --- check names against the data
    src_train = collections.Counter()
    with opener(a.train) as f:
        for line in f:
            src_train[json.loads(line)["source"]] += 1
    src_val = collections.Counter()
    with opener(a.val) as f:
        for line in f:
            src_val[json.loads(line)["source"]] += 1
    allsrc = set(src_train) | set(src_val)
    missing = [s for s in drop if s not in allsrc]
    pat = sorted(s for s in allsrc if any(p in s.lower() for p in NAME_PATTERNS))
    unlisted = [s for s in pat if s not in drop]
    if unlisted:
        sys.exit(f"name patterns match sources that are not in the drop list: {unlisted}")
    print("R1 matched sources:", sorted(s for s in drop if s in allsrc), flush=True)
    print("R1 listed but absent from the data:", missing, flush=True)

    G["drop"] = drop
    G["fallback"] = not a.no_instruction_fallback
    G.update(uh=np.empty(0, np.uint64), iuh=np.empty(0, np.uint64), ih=np.empty(0, np.uint64), ib=np.empty(0, np.uint16))
    ctx = mp.get_context("fork")
    pool = ctx.Pool(a.workers)
    uh, h, b, labels, bstats, unk = build_bench(a.bench, pool, "bench")
    pool.close(); pool.join()
    info_ids = [i for i, l in enumerate(labels) if l in set(a.info_labels)]
    for l in a.info_labels:
        if l not in labels:
            print("WARNING: --info-labels entry not found among labels:", l, flush=True)
    info_labels = [labels[i] for i in info_ids]
    isinfo = np.isin(b, np.array(info_ids, np.uint16)) if info_ids else np.zeros(len(b), bool)
    ih = h[isinfo]
    ib = np.array([info_ids.index(int(x)) for x in b[isinfo]], np.uint16) if isinfo.any() else np.empty(0, np.uint16)
    iuh = np.unique(ih)
    h, b = h[~isinfo], b[~isinfo]
    uh = np.unique(h)
    for l in info_labels:
        bstats[l]["info_only"] = True
    print(f"benchmark grams: {len(uh)} distinct (drop), {len(iuh)} (info), {len(labels)} labels", flush=True)
    G.update(uh=uh, iuh=iuh, ih=ih, ib=ib)
    pool = ctx.Pool(a.workers)

    rep = {"args": {k: v for k, v in vars(a).items()}, "rule_names": {"R1_matched": {s: drop[s] for s in sorted(drop) if s in allsrc},
                                                                           "R1_listed_but_absent": missing}}
    rep["benchmarks"] = {"files": a.bench, "info_labels": a.info_labels, "unreadable_records": unk,
                         "per_label": bstats, "distinct_8grams_total": int(len(uh))}

    res = {}
    for name, path in (("train", a.train), ("val", a.val)):
        res[name] = scan_pool(path, pool, name)
    pool.close(); pool.join()

    # --- stop-list from train rows that survived R1/R2
    tr = res["train"]
    hits = [(r[0], r[4]) for r in tr if r[3] is None and r[4] is not None]
    srcid = {s: i for i, s in enumerate(sorted(src_train))}
    row_src = np.array([srcid[r[1]] for r in tr], np.int32)
    hrow = np.concatenate([np.full(len(x), p, np.int64) for p, x in hits]) if hits else np.empty(0, np.int64)
    hhash = np.concatenate([x for _, x in hits]) if hits else np.empty(0, np.uint64)
    o = np.argsort(hhash, kind="stable")
    hhash, hrow = hhash[o], hrow[o]
    uhs, first, df = np.unique(hhash, return_index=True, return_counts=True)
    pair = np.unique(np.stack([hhash, row_src[hrow].astype(np.uint64)], 1), axis=0) if len(hhash) else np.empty((0, 2), np.uint64)
    nsrc = np.zeros(len(uhs), np.int64)
    if len(pair):
        u2, c2 = np.unique(pair[:, 0], return_counts=True)
        nsrc[np.searchsorted(uhs, u2)] = c2
    is_stop = (df > a.stop_min_rows) & (nsrc >= a.stop_min_sources)
    if a.no_stoplist:
        is_stop[:] = False
    stop = uhs[is_stop]
    print(f"stop-list: {len(stop)} grams of {len(uhs)} matched distinct grams", flush=True)

    def decide(rows, stopset):
        """-> (plain, final): plain = set of row indices with a benchmark 8-gram match (no stop-list); final = {row index: matching
        8-gram hashes that are not on the stop-list}, the rows that R3 drops."""
        plain, final = set(), {}
        for r in rows:
            if r[3] is None and r[4] is not None:
                plain.add(r[0])
                keep = r[4][~np.isin(r[4], stopset)] if len(stopset) else r[4]
                if len(keep):
                    final[r[0]] = keep
        return plain, final

    out_info = {}
    for name in ("train", "val"):
        rows = res[name]
        plain, final = decide(rows, stop)
        out_info[name] = (plain, final)

    # --- per-source plain rate, top offenders for sources over warn_frac
    tr_after12 = collections.Counter(r[1] for r in tr if r[3] is None)
    plain_tr, final_tr = out_info["train"]
    plain_by_src = collections.Counter(tr[i][1] for i in plain_tr)
    final_by_src = collections.Counter(tr[i][1] for i in final_tr)
    flagged = {s: (plain_by_src[s], tr_after12[s]) for s in tr_after12 if tr_after12[s] and plain_by_src[s] / tr_after12[s] > a.warn_frac}
    # top offending grams per flagged source
    top_per_src = {}
    need_hash, need_rows = set(), set()
    if len(hhash):
        hs_src = row_src[hrow]
        for s in flagged:
            m = hs_src == srcid[s]
            u, i0, c = np.unique(hhash[m], return_index=True, return_counts=True)
            top = np.argsort(-c)[:5]
            top_per_src[s] = [(int(u[t]), int(c[t]), int(hrow[m][i0[t]])) for t in top]
            for hv, _, rw in top_per_src[s]:
                need_hash.add(hv); need_rows.add(rw)
    # overall top and stop-list text
    order = np.argsort(-df)
    stop_order = [i for i in order if is_stop[i]]
    for i in stop_order:
        need_hash.add(int(uhs[i])); need_rows.add(int(hrow[first[i]]))
    for i in order[:20]:
        need_hash.add(int(uhs[i])); need_rows.add(int(hrow[first[i]]))
    texts = top_texts(a.train, need_rows, need_hash) if need_hash else {}

    def tx(hv):
        return texts.get(int(hv), f"<hash {int(hv):x}>")

    rep["stoplist"] = {"rule": f"8-grams that also occur in a benchmark and in more than {a.stop_min_rows} train rows from at least {a.stop_min_sources} sources",
                       "disabled": a.no_stoplist, "size": int(len(stop)), "matched_distinct_grams": int(len(uhs)),
                       "top": [{"text": tx(uhs[i]), "rows": int(df[i]), "sources": int(nsrc[i]), "stop": bool(is_stop[i])} for i in order[:20]]}
    with open(os.path.join(a.out_dir, "stoplist.tsv"), "w") as f:
        f.write("hash\ttrain_rows\tsources\ttext\n")
        for i in np.nonzero(is_stop)[0]:
            f.write(f"{int(uhs[i]):x}\t{int(df[i])}\t{int(nsrc[i])}\t{texts.get(int(uhs[i]), '')}\n")
    rep["over_warn_frac_plain_rule"] = {s: {"rows": n, "plain_drop": d, "plain_frac": round(d / n, 4),
                                            "with_stoplist_drop": final_by_src[s],
                                            "top_8grams": [{"text": tx(hv), "rows_in_source": c} for hv, c, _ in top_per_src.get(s, [])]}
                                        for s, (d, n) in sorted(flagged.items(), key=lambda kv: -kv[1][0] / kv[1][1])}

    # --- attribution per benchmark (train, after stop-list)
    def attribute(final):
        hh = [(i, v) for i, v in final.items()]
        rr, _, br = expand(hh, h, b)
        per_any, per_first = collections.Counter(), collections.Counter()
        if len(rr):
            u = np.unique(np.stack([rr, br], 1), axis=0)
            for bid, c in zip(*np.unique(u[:, 1], return_counts=True)):
                per_any[labels[bid]] = int(c)
            fb = {}
            for r_, b_ in u:  # u is sorted by row then bench: first bench = lowest label id
                fb.setdefault(int(r_), int(b_))
            for b_ in fb.values():
                per_first[labels[b_]] += 1
            rb = collections.defaultdict(list)
            for r_, b_ in u:
                rb[int(r_)].append(labels[b_])
        else:
            rb = {}
        return per_any, per_first, rb

    attr = {}
    for name in ("train", "val"):
        attr[name] = attribute(out_info[name][1])

    # --- write outputs
    summary = {}
    dropped_log = gzip.open(os.path.join(a.out_dir, "dropped.tsv.gz"), "wt")
    dropped_log.write("split\tindex\tid\tsource\trule\tbenchmarks\n")
    for name, path in (("train", a.train), ("val", a.val)):
        rows = res[name]
        plain, final = out_info[name]
        rb = attr[name][2]
        keep_lic, keep_src, in_src = collections.Counter(), collections.Counter(), collections.Counter()
        rule_cnt = collections.Counter()
        lost = collections.defaultdict(collections.Counter)
        n_out = 0
        by_idx = {r[0]: r for r in rows}
        import contextlib
        fout_cm = contextlib.nullcontext() if a.report_only else gzip.open(os.path.join(a.out_dir, f"{name}.jsonl.gz"), "wt", compresslevel=6)
        with opener(path) as fin, fout_cm as fout:
            for i, line in enumerate(fin):
                r = by_idx[i]
                in_src[r[1]] += 1
                rule = r[3]
                if rule is None and i in final:
                    rule = "R3:benchmark 8-gram"
                if rule is None:
                    if fout is not None:
                        fout.write(line)
                    n_out += 1
                    keep_src[r[1]] += 1
                    keep_lic[r[2]] += 1
                else:
                    key = rule.split(":")[0]
                    rule_cnt[key] += 1
                    lost[r[1]][key] += 1
                    idv = json.loads(line).get("id") if name == "val" or key == "R3" else None
                    dropped_log.write(f"{name}\t{i}\t{idv or ''}\t{r[1]}\t{rule}\t{'|'.join(rb.get(i, []))}\n")
        n_in = len(rows)
        summary[name] = {
            "rows_in": n_in, "rows_out": n_out,
            "rows_removed_by_rule": {"R1_source_drop": rule_cnt["R1"], "R2_contractnli_test": rule_cnt["R2"], "R3_benchmark_8gram": rule_cnt["R3"]},
            "rows_after_R1": n_in - rule_cnt["R1"], "rows_after_R2": n_in - rule_cnt["R1"] - rule_cnt["R2"],
            "R3_plain_rule_would_remove (no stop-list)": len(plain),
            "R3_stoplist_saved_rows": len(plain) - len(final),
            "sources_in": len(in_src), "sources_out": len(keep_src),
            "license_counts_out": dict(keep_lic.most_common()),
            "per_source": {s: {"before": in_src[s], "after": keep_src.get(s, 0), "removed": in_src[s] - keep_src.get(s, 0),
                               "by_rule": dict(lost.get(s, {}))} for s in sorted(in_src)},
            "R3_rows_per_benchmark_any_match": dict(attr[name][0].most_common()),
            "R3_rows_per_benchmark_first_match": dict(attr[name][1].most_common()),
            "R3_rows_matching_info_only_benchmarks": collections.Counter(
                (info_labels[j]) for r in rows if r[3] is None for j in r[5]),
        }
        summary[name]["R3_rows_matching_info_only_benchmarks"] = dict(summary[name]["R3_rows_matching_info_only_benchmarks"])
    dropped_log.close()
    rep["summary"] = summary
    tr_s = summary["train"]["per_source"]
    rep["top30_sources_by_rows_lost"] = [{"source": s, "before": v["before"], "after": v["after"], "removed": v["removed"],
                                          "removed_frac": round(v["removed"] / v["before"], 4), "by_rule": v["by_rule"]}
                                         for s, v in sorted(tr_s.items(), key=lambda kv: -kv[1]["removed"])[:30]]
    rep["R3_sources_losing_most_to_R3"] = sorted(
        [{"source": s, "removed": v["by_rule"].get("R3", 0), "frac": round(v["by_rule"].get("R3", 0) / v["before"], 4)}
         for s, v in tr_s.items() if v["by_rule"].get("R3")], key=lambda d: -d["removed"])[:30]

    # --- optional checks
    if a.val_released and not a.report_only:
        ids = set()
        with opener(a.val_released) as f:
            for line in f:
                ids.add(json.loads(line)["id"])
        with opener(os.path.join(a.out_dir, "val.jsonl.gz")) as f:
            mine = {json.loads(line)["id"] for line in f}
        rep["val_vs_released"] = {"released": len(ids), "clean_val": len(mine), "in_both": len(ids & mine),
                                  "released_not_in_clean": len(ids - mine), "clean_not_in_released": len(mine - ids)}
    rep["runtime_seconds"] = round(time.time() - T0)
    json.dump(rep, open(os.path.join(a.out_dir, "REPORT.json"), "w"), indent=1, default=str)
    write_md(rep, os.path.join(a.out_dir, "REPORT.md"))
    print("done in", rep["runtime_seconds"], "s", flush=True)


def write_md(rep, path):
    s, L = rep["summary"], []
    for nm in ("train", "val"):
        x = s[nm]
        L += [f"## {nm}", f"- rows in {x['rows_in']:,}, out {x['rows_out']:,}; sources {x['sources_in']} -> {x['sources_out']}",
              f"- removed by rule: {x['rows_removed_by_rule']}",
              f"- R3 plain rule (no stop-list) would remove {x['R3_plain_rule_would_remove (no stop-list)']:,}; stop-list saved {x['R3_stoplist_saved_rows']:,}",
              f"- licences out: {x['license_counts_out']}", ""]
    L += ["## R1 sources dropped", ""] + [f"- `{k}`: {v}" for k, v in rep["rule_names"]["R1_matched"].items()]
    L += ["", f"Listed but absent from the data: {rep['rule_names']['R1_listed_but_absent']}", ""]
    L += ["## Stop-list", "", f"{rep['stoplist']['rule']}: {rep['stoplist']['size']} 8-grams (of {rep['stoplist']['matched_distinct_grams']} matched).", "", "Top matched 8-grams by pool rows:", ""]
    L += [f"- {t['rows']} rows / {t['sources']} sources / stop={t['stop']}: `{t['text']}`" for t in rep["stoplist"]["top"]]
    L += ["", "## Sources over the warn fraction under the plain rule", ""]
    for k, v in rep["over_warn_frac_plain_rule"].items():
        L.append(f"- `{k}`: {v['plain_drop']}/{v['rows']} ({v['plain_frac']:.1%}) plain, {v['with_stoplist_drop']} with stop-list; top: " +
                 "; ".join(f"`{g['text']}` ({g['rows_in_source']})" for g in v["top_8grams"]))
    L += ["", "## R3 rows per benchmark (train, any match / first match)", ""]
    ta, tf = s["train"]["R3_rows_per_benchmark_any_match"], s["train"]["R3_rows_per_benchmark_first_match"]
    L += [f"- {k}: {v} / {tf.get(k, 0)}" for k, v in ta.items()]
    L += ["", "## Top 30 sources by rows lost (train)", ""]
    L += [f"- `{d['source']}`: {d['before']} -> {d['after']} (-{d['removed']}, {d['removed_frac']:.1%}) {d['by_rule']}" for d in rep["top30_sources_by_rows_lost"]]
    L += ["", "## Benchmarks", "", f"files: {rep['benchmarks']['files']}", f"info only labels: {rep['benchmarks']['info_labels']}", ""]
    L += [f"- {k}: {v['items']} items, {v['items_lt8_words']} with <8 words, {v['distinct_8grams']} distinct 8-grams" for k, v in rep["benchmarks"]["per_label"].items()]
    for k in ("val_vs_released",):
        if k in rep:
            L += ["", f"## {k}", "", json.dumps(rep[k])]
    L += ["", f"runtime {rep['runtime_seconds']} s"]
    open(path, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
