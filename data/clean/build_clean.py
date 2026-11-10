#!/usr/bin/env python3
"""Step 1: remove the sources and rows that should not be in the pool (pool v7 -> clean).

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
                       (for benchmark items without state text, with their question
                       instructions instead).  Normalisation: lowercase, split on
                       runs of word characters (so punctuation and whitespace vanish).
                       Scripts written without spaces (Chinese, Japanese, Thai, Lao,
                       Khmer, Burmese) have no words to split on, so their characters
                       are also matched as 16-character grams.
                       The row's QUESTION is checked the same way, but a matching gram
                       counts only if it occurs in the questions of at most
                       ``--question-max-rows`` pool rows: a question template shared by
                       many rows is not an item quoted from a benchmark.
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
import contextlib
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
NC = 16                                            # gram length, in characters, for scripts written without spaces
MULT = np.uint64(0x9E3779B97F4A7C15)
WRE = re.compile(r"\w+", re.U)
# Thai, Lao, Burmese, Khmer, kana, CJK ideographs (Hangul is written with spaces and stays with the word rule)
CRE = re.compile("[\u0e00-\u0eff\u1000-\u109f\u1780-\u17ff\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002ebef]")

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


def _poly(h, k):
    n = len(h)
    acc = np.zeros(n - k + 1, np.uint64)
    with np.errstate(over="ignore"):
        for j in range(k):
            acc = acc * MULT + h[j:n - k + 1 + j]
    return acc


def word_grams(text):
    """(words, uint64 hash of every normalised word 8-gram), in text order."""
    ws = WRE.findall(text.lower())
    if len(ws) < N:
        return ws, np.empty(0, np.uint64)
    return ws, _poly(np.fromiter((_wh(w) for w in ws), dtype=np.uint64, count=len(ws)), N)


def char_grams(text):
    """(characters, uint64 hash of every 16-character gram) over the characters of scripts written without spaces,
    taken in order with everything else removed. The "\\x00" prefix keeps these hashes apart from the word hashes."""
    cs = CRE.findall(text)
    if len(cs) < NC:
        return cs, np.empty(0, np.uint64)
    return cs, _poly(np.fromiter((_wh("\x00" + c) for c in cs), dtype=np.uint64, count=len(cs)), NC)


def gram_hashes(text):
    """uint64 hashes of every normalised word 8-gram of ``text``, then of its 16-character grams (may contain repeats)."""
    w, c = word_grams(text)[1], char_grams(text)[1]
    return np.concatenate([w, c]) if len(c) else w


def state_grams(state):
    parts = [gram_hashes(t) for t in leaves(state, [])]
    parts = [p for p in parts if len(p)]
    return np.unique(np.concatenate(parts)) if parts else np.empty(0, np.uint64)


def gram_texts(state, want):
    """{hash: 'w1 .. w8'} for the hashes in ``want`` found in ``state``."""
    out = {}
    for t in leaves(state, []):
        ws, g = word_grams(t)
        for i in np.nonzero(np.isin(g, want))[0]:
            out[int(g[i])] = " ".join(ws[i:i + N])
        cs, g = char_grams(t)
        for i in np.nonzero(np.isin(g, want))[0]:
            out[int(g[i])] = "".join(cs[i:i + NC])
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
        reason, hit, info, qhit = None, None, (), None
        if src in drop:
            reason = "R1:" + drop[src]
        elif src == "ContractNLI" and r.get("upstream_split") == "test":
            reason = "R2:ContractNLI test split"
        else:
            gq = state_grams(r.get("question"))
            if len(gq):
                mq = gq[_isin_sorted(G["uh"], gq)]
                qhit = mq if len(mq) else None
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
        out.append((start + k, src, r.get("license"), reason, hit, info, r.get("group"), qhit))
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
                r = json.loads(line)
                out.update(gram_texts([r.get("state"), r.get("question")], want))
    return out


# ---------------------------------------------------------------- steps of main()
def count_sources(path):
    c = collections.Counter()
    with opener(path) as f:
        for line in f:
            c[json.loads(line)["source"]] += 1
    return c


def check_drop_list(drop, allsrc):
    """Stop if a name pattern matches a source that is not in the drop list. -> listed sources absent from the data."""
    missing = [s for s in drop if s not in allsrc]
    pat = sorted(s for s in allsrc if any(p in s.lower() for p in NAME_PATTERNS))
    unlisted = [s for s in pat if s not in drop]
    if unlisted:
        sys.exit(f"name patterns match sources that are not in the drop list: {unlisted}")
    print("R1 matched sources:", sorted(s for s in drop if s in allsrc), flush=True)
    print("R1 listed but absent from the data:", missing, flush=True)
    return missing


def load_benchmarks(a, ctx):
    """Hash every benchmark file and split off the measure-only labels (--info-labels).
    -> dict: h, b (drop grams with their benchmark ids, sorted), uh (distinct drop grams), ih, ib, iuh (the same for
    the measure-only grams), labels, info_labels, stats, unknown."""
    pool = ctx.Pool(a.workers)
    _, h, b, labels, stats, unknown = build_bench(a.bench, pool, "bench")
    pool.close()
    pool.join()
    info_ids = [i for i, lab in enumerate(labels) if lab in set(a.info_labels)]
    for lab in a.info_labels:
        if lab not in labels:
            print("WARNING: --info-labels entry not found among labels:", lab, flush=True)
    info_labels = [labels[i] for i in info_ids]
    isinfo = np.isin(b, np.array(info_ids, np.uint16)) if info_ids else np.zeros(len(b), bool)
    ih = h[isinfo]
    ib = np.array([info_ids.index(int(x)) for x in b[isinfo]], np.uint16) if isinfo.any() else np.empty(0, np.uint16)
    h, b = h[~isinfo], b[~isinfo]
    for lab in info_labels:
        stats[lab]["info_only"] = True
    return {"h": h, "b": b, "uh": np.unique(h), "ih": ih, "ib": ib, "iuh": np.unique(ih), "labels": labels,
            "info_labels": info_labels, "stats": stats, "unknown": unknown}


def build_stoplist(tr, src_train, a):
    """The automatic stop-list, from the benchmark grams found in the states of train rows that survived R1/R2: a gram in
    more than --stop-min-rows rows from at least --stop-min-sources sources is boilerplate, not a quoted item."""
    hits = [(r[0], r[4]) for r in tr if r[3] is None and r[4] is not None]
    srcid = {s: i for i, s in enumerate(sorted(src_train))}
    row_src = np.array([srcid[r[1]] for r in tr], np.int32)
    hrow = np.concatenate([np.full(len(x), p, np.int64) for p, x in hits]) if hits else np.empty(0, np.int64)
    hhash = np.concatenate([x for _, x in hits]) if hits else np.empty(0, np.uint64)
    o = np.argsort(hhash, kind="stable")
    hhash, hrow = hhash[o], hrow[o]
    uhs, first, df = np.unique(hhash, return_index=True, return_counts=True)        # df: train rows per gram
    pair = np.unique(np.stack([hhash, row_src[hrow].astype(np.uint64)], 1), axis=0) if len(hhash) else np.empty((0, 2), np.uint64)
    nsrc = np.zeros(len(uhs), np.int64)                                              # sources per gram
    if len(pair):
        u2, c2 = np.unique(pair[:, 0], return_counts=True)
        nsrc[np.searchsorted(uhs, u2)] = c2
    is_stop = (df > a.stop_min_rows) & (nsrc >= a.stop_min_sources)
    if a.no_stoplist:
        is_stop[:] = False
    stop = uhs[is_stop]
    print(f"stop-list: {len(stop)} grams of {len(uhs)} matched distinct grams", flush=True)
    return {"hhash": hhash, "hrow": hrow, "row_src": row_src, "srcid": srcid, "uhs": uhs, "first": first, "df": df,
            "nsrc": nsrc, "is_stop": is_stop, "stop": stop}


def rare_question_grams(res, a):
    """Benchmark grams found in the questions of at most --question-max-rows pool rows (train and val together)."""
    qall = [r[7] for name in ("train", "val") for r in res[name] if r[3] is None and r[7] is not None]
    if qall:
        qu, qdf = np.unique(np.concatenate(qall), return_counts=True)
    else:
        qu, qdf = np.empty(0, np.uint64), np.empty(0, np.int64)
    qrare = qu[qdf <= a.question_max_rows]
    print(f"question matches: {len(qu)} distinct grams in {len(qall)} rows; {len(qrare)} occur in at most "
          f"{a.question_max_rows} rows", flush=True)
    return qrare


def decide(rows, stopset, qrare):
    """-> (plain, final, question_only).
    plain: the row indices with a benchmark gram in the state (the rule without a stop-list);
    final: {row index: the gram hashes R3 drops the row for}: state grams off the stop-list, and question grams that are
           off the stop-list and rare among the pool's questions;
    question_only: how many rows of `final` are there for a question match alone."""
    plain, final, question_only = set(), {}, 0
    for r in rows:
        if r[3] is not None:
            continue
        if r[4] is not None:
            plain.add(r[0])
            keep = r[4][~np.isin(r[4], stopset)] if len(stopset) else r[4]
            if len(keep):
                final[r[0]] = keep
        if r[7] is not None:
            kq = r[7][np.isin(r[7], qrare)]
            kq = kq[~np.isin(kq, stopset)] if len(stopset) else kq
            if len(kq):
                if r[0] not in final:
                    question_only += 1
                    final[r[0]] = kq
                else:
                    final[r[0]] = np.unique(np.concatenate([final[r[0]], kq]))
    return plain, final, question_only


def stoplist_report(a, tr, sl, plain_tr, final_tr):
    """Report entries for the stop-list and for the sources the plain rule would hit hardest; writes stoplist.tsv."""
    uhs, df, nsrc, is_stop, first = sl["uhs"], sl["df"], sl["nsrc"], sl["is_stop"], sl["first"]
    hhash, hrow = sl["hhash"], sl["hrow"]
    tr_after12 = collections.Counter(r[1] for r in tr if r[3] is None)
    plain_by_src = collections.Counter(tr[i][1] for i in plain_tr)
    final_by_src = collections.Counter(tr[i][1] for i in final_tr)
    flagged = {s: (plain_by_src[s], tr_after12[s]) for s in tr_after12
               if tr_after12[s] and plain_by_src[s] / tr_after12[s] > a.warn_frac}

    # the grams whose text the report shows: the top 5 per flagged source, the stop-list, the overall top 20
    top_per_src = {}
    need_hash, need_rows = set(), set()
    if len(hhash):
        hs_src = sl["row_src"][hrow]
        for s in flagged:
            m = hs_src == sl["srcid"][s]
            u, i0, c = np.unique(hhash[m], return_index=True, return_counts=True)
            top = np.argsort(-c)[:5]
            top_per_src[s] = [(int(u[t]), int(c[t]), int(hrow[m][i0[t]])) for t in top]
            for hv, _, rw in top_per_src[s]:
                need_hash.add(hv)
                need_rows.add(rw)
    order = np.argsort(-df)
    for i in [i for i in order if is_stop[i]]:
        need_hash.add(int(uhs[i]))
        need_rows.add(int(hrow[first[i]]))
    for i in order[:20]:
        need_hash.add(int(uhs[i]))
        need_rows.add(int(hrow[first[i]]))
    texts = top_texts(a.train, need_rows, need_hash) if need_hash else {}

    def tx(hv):
        return texts.get(int(hv), f"<hash {int(hv):x}>")

    stoplist = {"rule": f"8-grams that also occur in a benchmark and in more than {a.stop_min_rows} train rows from at least {a.stop_min_sources} sources",
                "disabled": a.no_stoplist, "size": int(len(sl["stop"])), "matched_distinct_grams": int(len(uhs)),
                "top": [{"text": tx(uhs[i]), "rows": int(df[i]), "sources": int(nsrc[i]), "stop": bool(is_stop[i])} for i in order[:20]]}
    with open(os.path.join(a.out_dir, "stoplist.tsv"), "w") as f:
        f.write("hash\ttrain_rows\tsources\ttext\n")
        for i in np.nonzero(is_stop)[0]:
            f.write(f"{int(uhs[i]):x}\t{int(df[i])}\t{int(nsrc[i])}\t{texts.get(int(uhs[i]), '')}\n")
    over_warn = {s: {"rows": n, "plain_drop": d, "plain_frac": round(d / n, 4), "with_stoplist_drop": final_by_src[s],
                     "top_8grams": [{"text": tx(hv), "rows_in_source": c} for hv, c, _ in top_per_src.get(s, [])]}
                 for s, (d, n) in sorted(flagged.items(), key=lambda kv: -kv[1][0] / kv[1][1])}
    return stoplist, over_warn


def attribute(final, h, b, labels):
    """Which benchmarks the dropped rows match. -> (rows per benchmark, rows per first-matching benchmark,
    {row index: [benchmark labels]})"""
    rr, _, br = expand(list(final.items()), h, b)
    per_any, per_first = collections.Counter(), collections.Counter()
    rb = collections.defaultdict(list)
    if len(rr):
        u = np.unique(np.stack([rr, br], 1), axis=0)
        for bid, c in zip(*np.unique(u[:, 1], return_counts=True)):
            per_any[labels[bid]] = int(c)
        fb = {}
        for r_, b_ in u:                            # u is sorted by row, then benchmark: the first is the lowest label id
            fb.setdefault(int(r_), int(b_))
        for b_ in fb.values():
            per_first[labels[b_]] += 1
        for r_, b_ in u:
            rb[int(r_)].append(labels[b_])
    return per_any, per_first, rb


def write_split(name, path, rows, plain, final, question_only, attr, info_labels, out_dir, report_only, dropped_log):
    """Copy the kept lines byte for byte, log the dropped ones. -> the split's summary for the report."""
    per_any, per_first, rb = attr
    keep_lic, keep_src, in_src = collections.Counter(), collections.Counter(), collections.Counter()
    rule_cnt = collections.Counter()
    lost = collections.defaultdict(collections.Counter)
    n_out = 0
    by_idx = {r[0]: r for r in rows}
    fout_cm = contextlib.nullcontext() if report_only else gzip.open(os.path.join(out_dir, f"{name}.jsonl.gz"), "wt", compresslevel=6)
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
    info_hits = collections.Counter(info_labels[j] for r in rows if r[3] is None for j in r[5])
    return {
        "rows_in": n_in, "rows_out": n_out,
        "rows_removed_by_rule": {"R1_source_drop": rule_cnt["R1"], "R2_contractnli_test": rule_cnt["R2"], "R3_benchmark_8gram": rule_cnt["R3"]},
        "rows_after_R1": n_in - rule_cnt["R1"], "rows_after_R2": n_in - rule_cnt["R1"] - rule_cnt["R2"],
        "R3_plain_rule_would_remove (no stop-list)": len(plain),
        "R3_stoplist_saved_rows": len(plain) - (len(final) - question_only),
        "R3_rows_dropped_for_the_question_only": question_only,
        "sources_in": len(in_src), "sources_out": len(keep_src),
        "license_counts_out": dict(keep_lic.most_common()),
        "per_source": {s: {"before": in_src[s], "after": keep_src.get(s, 0), "removed": in_src[s] - keep_src.get(s, 0),
                           "by_rule": dict(lost.get(s, {}))} for s in sorted(in_src)},
        "R3_rows_per_benchmark_any_match": dict(per_any.most_common()),
        "R3_rows_per_benchmark_first_match": dict(per_first.most_common()),
        "R3_rows_matching_info_only_benchmarks": dict(info_hits),
    }


def check_val_released(path, out_dir):
    with opener(path) as f:
        ids = {json.loads(line)["id"] for line in f}
    with opener(os.path.join(out_dir, "val.jsonl.gz")) as f:
        mine = {json.loads(line)["id"] for line in f}
    return {"released": len(ids), "clean_val": len(mine), "in_both": len(ids & mine),
            "released_not_in_clean": len(ids - mine), "clean_not_in_released": len(mine - ids)}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True, help="pool train.jsonl.gz")
    ap.add_argument("--val", required=True, help="pool val.jsonl.gz")
    ap.add_argument("--out-dir", required=True, help="writes train.jsonl.gz, val.jsonl.gz, REPORT.json, REPORT.md, stoplist.tsv, dropped.tsv.gz")
    ap.add_argument("--bench", nargs="+", required=True, help="benchmark jsonl[.gz] files (state text is decontaminated against)")
    ap.add_argument("--info-labels", nargs="*", default=[], help="benchmark labels (see above, e.g. <file stem>/<set>) that are only measured, never used to drop")
    ap.add_argument("--val-released", default=None, help="optional: released val split, to check the cleaned val against")
    ap.add_argument("--stop-min-rows", type=int, default=50, help="stop-list: a gram must occur in more than this many train rows ...")
    ap.add_argument("--stop-min-sources", type=int, default=5, help="... from at least this many sources")
    ap.add_argument("--question-max-rows", type=int, default=5, help="a benchmark 8-gram found in a row's question drops the row only if it occurs in the questions of at most this many pool rows (templates are shared by many)")
    ap.add_argument("--warn-frac", type=float, default=0.10, help="list the top 8-grams of every source the plain rule would cut by more than this fraction")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--no-instruction-fallback", action="store_true", help="do not use question.instructions of benchmark items that have no state text")
    ap.add_argument("--report-only", action="store_true", help="do not write train/val.jsonl.gz (sensitivity runs)")
    ap.add_argument("--no-stoplist", action="store_true", help="disable the stop-list (plain 8-gram rule)")
    a = ap.parse_args()
    t_start = time.time()
    os.makedirs(a.out_dir, exist_ok=True)

    # R1 names against the data
    drop = drop_table()
    src_train, src_val = count_sources(a.train), count_sources(a.val)
    allsrc = set(src_train) | set(src_val)
    missing = check_drop_list(drop, allsrc)

    # benchmark grams (the workers inherit G through fork)
    G["drop"] = drop
    G["fallback"] = not a.no_instruction_fallback
    G.update(uh=np.empty(0, np.uint64), iuh=np.empty(0, np.uint64), ih=np.empty(0, np.uint64), ib=np.empty(0, np.uint16))
    ctx = mp.get_context("fork")
    bm = load_benchmarks(a, ctx)
    print(f"benchmark grams: {len(bm['uh'])} distinct (drop), {len(bm['iuh'])} (info), {len(bm['labels'])} labels", flush=True)
    G.update(uh=bm["uh"], iuh=bm["iuh"], ih=bm["ih"], ib=bm["ib"])

    rep = {"args": dict(vars(a)),
           "rule_names": {"R1_matched": {s: drop[s] for s in sorted(drop) if s in allsrc}, "R1_listed_but_absent": missing}}
    rep["benchmarks"] = {"files": a.bench, "info_labels": a.info_labels, "unreadable_records": bm["unknown"],
                         "per_label": bm["stats"], "distinct_8grams_total": int(len(bm["uh"]))}

    # scan the pool
    pool = ctx.Pool(a.workers)
    res = {name: scan_pool(path, pool, name) for name, path in (("train", a.train), ("val", a.val))}
    pool.close()
    pool.join()

    # R3 decisions
    sl = build_stoplist(res["train"], src_train, a)
    qrare = rare_question_grams(res, a)
    decided = {name: decide(res[name], sl["stop"], qrare) for name in ("train", "val")}
    rep["stoplist"], rep["over_warn_frac_plain_rule"] = stoplist_report(a, res["train"], sl, *decided["train"][:2])
    attr = {name: attribute(decided[name][1], bm["h"], bm["b"], bm["labels"]) for name in ("train", "val")}

    # outputs
    summary = {}
    with gzip.open(os.path.join(a.out_dir, "dropped.tsv.gz"), "wt") as dropped_log:
        dropped_log.write("split\tindex\tid\tsource\trule\tbenchmarks\n")
        for name, path in (("train", a.train), ("val", a.val)):
            summary[name] = write_split(name, path, res[name], *decided[name], attr[name], bm["info_labels"],
                                        a.out_dir, a.report_only, dropped_log)
    rep["summary"] = summary
    tr_s = summary["train"]["per_source"]
    rep["top30_sources_by_rows_lost"] = [{"source": s, "before": v["before"], "after": v["after"], "removed": v["removed"],
                                          "removed_frac": round(v["removed"] / v["before"], 4), "by_rule": v["by_rule"]}
                                         for s, v in sorted(tr_s.items(), key=lambda kv: -kv[1]["removed"])[:30]]
    rep["R3_sources_losing_most_to_R3"] = sorted(
        [{"source": s, "removed": v["by_rule"].get("R3", 0), "frac": round(v["by_rule"].get("R3", 0) / v["before"], 4)}
         for s, v in tr_s.items() if v["by_rule"].get("R3")], key=lambda d: -d["removed"])[:30]
    if a.val_released and not a.report_only:
        rep["val_vs_released"] = check_val_released(a.val_released, a.out_dir)
    rep["runtime_seconds"] = round(time.time() - t_start)
    with open(os.path.join(a.out_dir, "REPORT.json"), "w") as f:
        json.dump(rep, f, indent=1, default=str)
    write_md(rep, os.path.join(a.out_dir, "REPORT.md"))
    print("done in", rep["runtime_seconds"], "s", flush=True)


def write_md(rep, path):
    s, L = rep["summary"], []
    for nm in ("train", "val"):
        x = s[nm]
        L += [f"## {nm}", f"- rows in {x['rows_in']:,}, out {x['rows_out']:,}; sources {x['sources_in']} -> {x['sources_out']}",
              f"- removed by rule: {x['rows_removed_by_rule']}",
              f"- R3 plain rule (no stop-list) would remove {x['R3_plain_rule_would_remove (no stop-list)']:,}; stop-list saved "
              f"{x['R3_stoplist_saved_rows']:,}; dropped for a question match alone {x.get('R3_rows_dropped_for_the_question_only', 0):,}",
              f"- licences out: {x['license_counts_out']}", ""]
    L += ["## R1 sources dropped", ""] + [f"- `{k}`: {v}" for k, v in rep["rule_names"]["R1_matched"].items()]
    L += ["", f"Listed but absent from the data: {rep['rule_names']['R1_listed_but_absent']}", ""]
    L += ["## Stop-list", "", f"{rep['stoplist']['rule']}: {rep['stoplist']['size']} 8-grams (of {rep['stoplist']['matched_distinct_grams']} matched).",
          "", "Top matched 8-grams by pool rows:", ""]
    L += [f"- {t['rows']} rows / {t['sources']} sources / stop={t['stop']}: `{t['text']}`" for t in rep["stoplist"]["top"]]
    L += ["", "## Sources over the warn fraction under the plain rule", ""]
    for k, v in rep["over_warn_frac_plain_rule"].items():
        L.append(f"- `{k}`: {v['plain_drop']}/{v['rows']} ({v['plain_frac']:.1%}) plain, {v['with_stoplist_drop']} with stop-list; top: "
                 + "; ".join(f"`{g['text']}` ({g['rows_in_source']})" for g in v["top_8grams"]))
    L += ["", "## R3 rows per benchmark (train, any match / first match)", ""]
    ta, tf = s["train"]["R3_rows_per_benchmark_any_match"], s["train"]["R3_rows_per_benchmark_first_match"]
    L += [f"- {k}: {v} / {tf.get(k, 0)}" for k, v in ta.items()]
    L += ["", "## Top 30 sources by rows lost (train)", ""]
    L += [f"- `{d['source']}`: {d['before']} -> {d['after']} (-{d['removed']}, {d['removed_frac']:.1%}) {d['by_rule']}"
          for d in rep["top30_sources_by_rows_lost"]]
    L += ["", "## Benchmarks", "", f"files: {rep['benchmarks']['files']}", f"info only labels: {rep['benchmarks']['info_labels']}", ""]
    L += [f"- {k}: {v['items']} items, {v['items_lt8_words']} with <8 words, {v['distinct_8grams']} distinct 8-grams"
          for k, v in rep["benchmarks"]["per_label"].items()]
    if "val_vs_released" in rep:
        L += ["", "## val_vs_released", "", json.dumps(rep["val_vs_released"])]
    L += ["", f"runtime {rep['runtime_seconds']} s"]
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
