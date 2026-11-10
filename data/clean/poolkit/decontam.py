"""Stage `clean`: remove the sources and rows that should not be in the pool.

Three rules, applied in this order. Rows are never reordered or rewritten: kept lines are copied byte for byte.

  R1  sources          every row of a source whose config entry says "drop": "clean".
  R2  row filters      the rows matched by the config's `clean.drop_rows` (a source, a field and the value to drop).
  R3  decontamination  a row is dropped if its STATE shares at least one normalised word 8-gram with the state text
                       of any benchmark item (for benchmark items without state text, with their question
                       instructions). Normalisation: lowercase, split on runs of word characters, so punctuation and
                       whitespace vanish. Scripts written without spaces (Chinese, Japanese, Thai, Lao, Khmer,
                       Burmese) have no words to split on, so their characters are also matched as 16-character grams.
                       The row's QUESTION is checked the same way, but a matching gram counts only if it occurs in the
                       questions of at most `question_max_rows` pool rows: a question template shared by many rows is
                       not an item quoted from a benchmark.
                       Boilerplate grams (in more than `stop_min_rows` train rows from at least `stop_min_sources`
                       sources) go on an automatic stop-list and are ignored; the stop-list is written out for review.

Benchmark files are jsonl or jsonl.gz. The state text of a record is read from `record.state`, `row.state` or
`state`; every string in it is hashed on its own, so no gram spans two fields. A benchmark label is the file stem,
plus `/<_evaluation.dataset or set>` when the record names one. Labels listed as info labels are only measured.

Hashing: every word -> 64-bit blake2b, a gram -> polynomial combination mod 2**64. A collision would drop a row
spuriously; at 1e8 benchmark grams the expected number of false matches is below 1e-3 per run.

The work is two parallel passes over the pool: one that hashes every row and looks its grams up, one that writes.
Between them the stop-list and the rare question grams are derived from the first pass, which is why R3 cannot be
decided row by row.
"""
import collections
import contextlib
import gzip
import hashlib
import json
import os
import re
import time

import numpy as np

from . import engine

DEFAULTS = {"word_gram": 8, "char_gram": 16, "stop_min_rows": 50, "stop_min_sources": 5, "question_max_rows": 5,
            "warn_frac": 0.10, "instruction_fallback": True}
MULT = np.uint64(0x9E3779B97F4A7C15)
WRE = re.compile(r"\w+", re.U)
# Thai, Lao, Burmese, Khmer, kana, CJK ideographs (Hangul is written with spaces and stays with the word rule)
CRE = re.compile("[\u0e00-\u0eff\u1000-\u109f\u1780-\u17ff\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002ebef]")
G: dict = {"N": 8, "NC": 16}                              # state shared with the workers (set before a pool is made)


# ----------------------------------------------------------------------------------------------- hashing
_WC = {}


def _wh(w):
    h = _WC.get(w)
    if h is None:
        h = int.from_bytes(hashlib.blake2b(w.encode("utf-8", "surrogatepass"), digest_size=8).digest(), "little")
        _WC[w] = h
    return h


def _poly(h, k):
    n = len(h)
    acc = np.zeros(n - k + 1, np.uint64)
    with np.errstate(over="ignore"):
        for j in range(k):
            acc = acc * MULT + h[j:n - k + 1 + j]
    return acc


def word_grams(text):
    """(words, uint64 hash of every normalised word gram), in text order."""
    ws = WRE.findall(text.lower())
    if len(ws) < G["N"]:
        return ws, np.empty(0, np.uint64)
    return ws, _poly(np.fromiter((_wh(w) for w in ws), dtype=np.uint64, count=len(ws)), G["N"])


def char_grams(text):
    """(characters, uint64 hash of every character gram) over the characters of scripts written without spaces, taken
    in order with everything else removed. The "\\x00" prefix keeps these hashes apart from the word hashes."""
    cs = CRE.findall(text)
    if len(cs) < G["NC"]:
        return cs, np.empty(0, np.uint64)
    return cs, _poly(np.fromiter((_wh("\x00" + c) for c in cs), dtype=np.uint64, count=len(cs)), G["NC"])


def gram_hashes(text):
    """uint64 hashes of every word gram of `text`, then of its character grams (may contain repeats)."""
    w, c = word_grams(text)[1], char_grams(text)[1]
    return np.concatenate([w, c]) if len(c) else w


def state_grams(state):
    """The distinct gram hashes of every string inside `state`, sorted."""
    parts = [gram_hashes(t) for t in engine.strings(state, [])]
    parts = [p for p in parts if len(p)]
    return np.unique(np.concatenate(parts)) if parts else np.empty(0, np.uint64)


def gram_texts(state, want):
    """{hash: the gram's text} for the hashes in `want` found in `state`."""
    out = {}
    for t in engine.strings(state, []):
        ws, g = word_grams(t)
        for i in np.nonzero(np.isin(g, want))[0]:
            out[int(g[i])] = " ".join(ws[i:i + G["N"]])
        cs, g = char_grams(t)
        for i in np.nonzero(np.isin(g, want))[0]:
            out[int(g[i])] = "".join(cs[i:i + G["NC"]])
    return out


def _isin_sorted(uh, q):
    if not len(uh) or not len(q):
        return np.zeros(len(q), bool)
    i = np.searchsorted(uh, q)
    i[i == len(uh)] = len(uh) - 1
    return uh[i] == q


# ----------------------------------------------------------------------------------------------- benchmarks
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
    """Question instruction strings of a record (the fallback for items without state text)."""
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
        if len(g):
            per.setdefault(lab, []).append(g)
            continue
        short[lab] += 1
        if G["fallback"]:
            # Many benchmark datasets (ANLI, VAST, MuSR, MMLU, HellaSwag, ...) keep the whole item text in
            # question.instructions and have an empty state. Without this they could not be decontaminated at all.
            fl = lab + "[instr]"
            items[fl] += 1
            parts = [gram_hashes(t) for t in bench_instructions(rec)]
            parts = [p for p in parts if len(p)]
            if parts:
                per.setdefault(fl, []).append(np.unique(np.concatenate(parts)))
            else:
                short[fl] += 1
    return {k: np.unique(np.concatenate(v)) for k, v in per.items()}, items, short, unknown


def _uniq_sorted(x):
    return x[np.concatenate(([True], x[1:] != x[:-1]))] if len(x) else x


def _bench_index(paths, workers):
    """-> (h, b, labels, stats, unknown): every (gram hash, benchmark label id) pair once, sorted by hash, then label."""
    labels, parts, stats, unknown = [], collections.defaultdict(list), {}, collections.Counter()
    tasks = ((os.path.basename(p).split(".")[0], p, b) for p in paths for b in engine.chunks(p, 200))
    for (stem, p, _), (per, items, short, unk) in _zip_tasks(workers, _bench_worker, tasks):
        unknown[stem] += unk
        for lab, n in items.items():
            s = stats.setdefault(lab, {"items": 0, "items_too_short": 0, "file": p})
            s["items"] += n
            s["items_too_short"] += short.get(lab, 0)
        for lab, a in per.items():
            parts[lab].append(a)
    labels = list(parts)
    if labels:
        h = np.concatenate([a for lab in labels for a in parts[lab]])
        b = np.concatenate([np.full(len(a), i, np.uint16) for i, lab in enumerate(labels) for a in parts[lab]])
        o = np.argsort(h, kind="stable")            # equal hashes stay in label order, so equal pairs end up adjacent
        h, b = h[o], b[o]
        keep = np.concatenate(([True], (h[1:] != h[:-1]) | (b[1:] != b[:-1])))
        h, b = h[keep], b[keep]
    else:
        h, b = np.empty(0, np.uint64), np.empty(0, np.uint16)
    per_label = np.bincount(b, minlength=len(labels))
    for i, lab in enumerate(labels):
        stats[lab]["distinct_grams"] = int(per_label[i])
    for lab in stats:
        if lab not in parts:
            stats[lab]["distinct_grams"] = 0
            labels.append(lab)
    return h, b, labels, stats, dict(unknown)


def load_benchmarks(paths, info_labels, workers, cache_dir=None):
    """Hash every benchmark file and split off the measure-only labels. The index is kept in cache_dir and reused as long
    as the files (path, size, time of change) and the gram settings are the same.
    -> dict: h, b (drop grams with their benchmark ids, sorted by hash), uh (distinct drop grams), ih, ib, iuh (the same
    for the measure-only grams), labels, info_labels, stats, unknown."""
    key = hashlib.blake2b(json.dumps([G["N"], G["NC"], G["fallback"]] + [[os.path.abspath(p), os.path.getsize(p), os.stat(p).st_mtime_ns]
                                                                             for p in paths]).encode(), digest_size=16).hexdigest()
    npz, meta = (os.path.join(cache_dir, f"bench_index.{ext}") for ext in ("npz", "json")) if cache_dir else (None, None)
    index = None
    if meta and os.path.exists(meta) and os.path.exists(npz):
        with open(meta) as f:
            m = json.load(f)
        if m.get("key") == key:
            with np.load(npz) as z:
                index = z["h"], z["b"], m["labels"], m["stats"], m["unknown"]
    if index is None:
        index = _bench_index(paths, workers)
        if cache_dir:
            np.savez(npz, h=index[0], b=index[1])
            with open(meta, "w") as f:
                json.dump({"key": key, "labels": index[2], "stats": index[3], "unknown": index[4]}, f)
    h, b, labels, stats, unknown = index
    info_ids = [i for i, lab in enumerate(labels) if lab in set(info_labels)]
    for lab in info_labels:
        if lab not in labels:
            print("WARNING: info label not found among the benchmark labels:", lab, flush=True)
    isinfo = np.isin(b, np.array(info_ids, np.uint16)) if info_ids else np.zeros(len(b), bool)
    ih = h[isinfo]
    ib = np.array([info_ids.index(int(x)) for x in b[isinfo]], np.uint16) if isinfo.any() else np.empty(0, np.uint16)
    h, b = h[~isinfo], b[~isinfo]
    for i in info_ids:
        stats[labels[i]]["info_only"] = True
    return {"h": h, "b": b, "uh": _uniq_sorted(h), "ih": ih, "ib": ib, "iuh": _uniq_sorted(ih), "labels": labels,
            "info_labels": [labels[i] for i in info_ids], "stats": stats, "unknown": unknown}


def _zip_tasks(workers, fn, tasks):
    """Run fn over the tasks' (stem, lines) and give each result back with its task."""
    sent = collections.deque()

    def feed():
        for t in tasks:
            sent.append(t)
            yield (t[0], t[2])

    for res in workers.imap(fn, feed()):
        yield sent.popleft(), res


# ----------------------------------------------------------------------------------------------- pass 1: scan
def _scan_worker(lines):
    """-> one tuple per row: (source, licence, rule or None, state hits or None, info benchmark ids, question hits or None)"""
    drop, filters, out = G["drop"], G["filters"], []
    for line in lines:
        r = json.loads(line)
        src = r.get("source")
        reason, hit, info, qhit = None, None, (), None
        if src in drop:
            reason = "R1:" + drop[src]
        else:
            for field, value, why in filters.get(src, ()) + filters.get("*", ()):
                if r.get(field) == value:
                    reason = "R2:" + why
                    break
        if reason is None:
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
        out.append((src, r.get("license"), reason, hit, info, qhit))
    return out


def scan(path, workers, name):
    """-> rows: one tuple (index, source, licence, rule, state hits, info ids, question hits) per row of the file."""
    t0, rows = time.time(), []
    for res in workers.imap(_scan_worker, engine.chunks(path)):
        n = len(rows)
        rows.extend((n + k,) + r for k, r in enumerate(res))
    print(f"[{name}] scanned {len(rows):,} rows in {time.time() - t0:.0f} s", flush=True)
    return rows


# ----------------------------------------------------------------------------------------------- deciding R3
def build_stoplist(tr, opt):
    """The automatic stop-list, from the benchmark grams found in the states of train rows that survived R1 and R2:
    a gram in more than stop_min_rows rows from at least stop_min_sources sources is boilerplate, not a quoted item."""
    hits = [(r[0], r[4]) for r in tr if r[3] is None and r[4] is not None]
    srcid = {s: i for i, s in enumerate(sorted({r[1] for r in tr}))}
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
    is_stop = (df > opt["stop_min_rows"]) & (nsrc >= opt["stop_min_sources"])
    if opt.get("no_stoplist"):
        is_stop[:] = False
    stop = uhs[is_stop]
    print(f"stop-list: {len(stop)} grams of {len(uhs):,} matched distinct grams", flush=True)
    return {"hhash": hhash, "hrow": hrow, "row_src": row_src, "srcid": srcid, "uhs": uhs, "first": first, "df": df,
            "nsrc": nsrc, "is_stop": is_stop, "stop": stop}


def rare_question_grams(res, opt):
    """Benchmark grams found in the questions of at most question_max_rows pool rows (all splits together)."""
    qall = [r[6] for rows in res.values() for r in rows if r[3] is None and r[6] is not None]
    if qall:
        qu, qdf = np.unique(np.concatenate(qall), return_counts=True)
    else:
        qu, qdf = np.empty(0, np.uint64), np.empty(0, np.int64)
    qrare = qu[qdf <= opt["question_max_rows"]]
    print(f"question matches: {len(qu)} distinct grams in {len(qall)} rows; {len(qrare)} occur in at most "
          f"{opt['question_max_rows']} rows", flush=True)
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
        if r[6] is not None:
            kq = r[6][np.isin(r[6], qrare)]
            kq = kq[~np.isin(kq, stopset)] if len(stopset) else kq
            if len(kq):
                if r[0] not in final:
                    question_only += 1
                    final[r[0]] = kq
                else:
                    final[r[0]] = np.unique(np.concatenate([final[r[0]], kq]))
    return plain, final, question_only


def expand(hits, h, b):
    """hits: list of (row index, uint64 array) -> arrays (row index, hash, benchmark id), one line per benchmark."""
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


def report_needs(tr, sl, plain_tr, opt):
    """What the report shows text for: the top 5 grams of every source the plain rule would cut by more than warn_frac,
    the stop-list, and the 20 most frequent grams. -> (flagged sources, top grams per source, hashes, train row indices)"""
    uhs, df, is_stop, first, hhash, hrow = sl["uhs"], sl["df"], sl["is_stop"], sl["first"], sl["hhash"], sl["hrow"]
    after12 = collections.Counter(r[1] for r in tr if r[3] is None)
    plain_by_src = collections.Counter(tr[i][1] for i in plain_tr)
    flagged = {s: (plain_by_src[s], after12[s]) for s in after12 if after12[s] and plain_by_src[s] / after12[s] > opt["warn_frac"]}
    top_per_src, need_hash, need_rows = {}, set(), set()
    if len(hhash):
        hs_src = sl["row_src"][hrow]
        for s in flagged:
            m = hs_src == sl["srcid"][s]
            u, i0, c = np.unique(hhash[m], return_index=True, return_counts=True)
            top_per_src[s] = [(int(u[t]), int(c[t]), int(hrow[m][i0[t]])) for t in np.argsort(-c)[:5]]
            for hv, _, rw in top_per_src[s]:
                need_hash.add(hv)
                need_rows.add(rw)
    order = np.argsort(-df)
    for i in [i for i in order if is_stop[i]] + list(order[:20]):
        need_hash.add(int(uhs[i]))
        need_rows.add(int(hrow[first[i]]))
    return flagged, top_per_src, need_hash, need_rows


def stoplist_report(out_dir, opt, tr, sl, final_tr, flagged, top_per_src, texts):
    """Report entries for the stop-list and for the flagged sources; writes stoplist.tsv."""
    uhs, df, nsrc, is_stop = sl["uhs"], sl["df"], sl["nsrc"], sl["is_stop"]
    final_by_src = collections.Counter(tr[i][1] for i in final_tr)
    order = np.argsort(-df)

    def tx(hv):
        return texts.get(int(hv), f"<hash {int(hv):x}>")

    stoplist = {"rule": f"grams that also occur in a benchmark and in more than {opt['stop_min_rows']} train rows from at "
                        f"least {opt['stop_min_sources']} sources",
                "disabled": bool(opt.get("no_stoplist")), "size": int(len(sl["stop"])), "matched_distinct_grams": int(len(uhs)),
                "top": [{"text": tx(uhs[i]), "rows": int(df[i]), "sources": int(nsrc[i]), "stop": bool(is_stop[i])} for i in order[:20]]}
    with open(os.path.join(out_dir, "stoplist.tsv"), "w") as f:
        f.write("hash\ttrain_rows\tsources\ttext\n")
        for i in np.nonzero(is_stop)[0]:
            f.write(f"{int(uhs[i]):x}\t{int(df[i])}\t{int(nsrc[i])}\t{texts.get(int(uhs[i]), '')}\n")
    over_warn = {s: {"rows": n, "plain_drop": d, "plain_frac": round(d / n, 4), "with_stoplist_drop": final_by_src[s],
                     "top_grams": [{"text": tx(hv), "rows_in_source": c} for hv, c, _ in top_per_src.get(s, [])]}
                 for s, (d, n) in sorted(flagged.items(), key=lambda kv: -kv[1][0] / kv[1][1])}
    return stoplist, over_warn


# ----------------------------------------------------------------------------------------------- pass 2: write
def _write_worker(args):
    """lines, {position: rule} of the rows to drop, the positions whose id the log wants, the positions whose gram
    text the report wants. -> (gzip member of the kept lines, number of lines, {position: id}, {hash: gram text})"""
    lines, dropped, want_id, want_text = args
    ids = {k: json.loads(lines[k]).get("id") for k in want_id}
    texts = {}
    for k in want_text:
        r = json.loads(lines[k])
        texts.update(gram_texts([r.get("state"), r.get("question")], G["want"]))
    member = engine.gz_member([ln for k, ln in enumerate(lines) if k not in dropped]) if G["write"] else b""
    return member, len(lines), ids, texts


def write_split(name, path, out_dir, rows, decided, attr, info_labels, workers, need_rows, dropped_log, report_only):
    """Copy the kept lines, log the dropped ones. -> (the split's summary for the report, {hash: gram text})"""
    plain, final, question_only = decided
    per_any, per_first, rb = attr
    rule = [r[3] if r[3] is not None else ("R3:benchmark overlap" if r[0] in final else None) for r in rows]

    def tasks():
        pos = 0
        for lines in engine.chunks(path):
            idx = range(pos, pos + len(lines))
            dropped = {i - pos: rule[i] for i in idx if rule[i] is not None}
            want_id = [k for k, why in dropped.items() if name != "train" or why.startswith("R3")]
            yield lines, dropped, want_id, [i - pos for i in idx if i in need_rows]
            pos += len(lines)

    texts, pos = {}, 0
    with (contextlib.nullcontext() if report_only else engine.GzOut(engine.split_path(out_dir, name))) as out:
        for member, n, ids, tx in workers.imap(_write_worker, tasks()):
            if out is not None:
                out.write(member)
            texts.update(tx)
            for i in range(pos, pos + n):
                if rule[i] is not None:
                    dropped_log.write(f"{name}\t{i}\t{ids.get(i - pos) or ''}\t{rows[i][1]}\t{rule[i]}\t{'|'.join(rb.get(i, []))}\n")
            pos += n

    in_src = collections.Counter(r[1] for r in rows)
    keep_src = collections.Counter(r[1] for r, why in zip(rows, rule) if why is None)
    keep_lic = collections.Counter(r[2] for r, why in zip(rows, rule) if why is None)
    rule_cnt = collections.Counter(why.split(":")[0] for why in rule if why is not None)
    lost = collections.defaultdict(collections.Counter)
    for r, why in zip(rows, rule):
        if why is not None:
            lost[r[1]][why.split(":")[0]] += 1
    n_in = len(rows)
    info_hits = collections.Counter(info_labels[j] for r in rows if r[3] is None for j in r[5])
    return {
        "rows_in": n_in, "rows_out": sum(keep_src.values()),
        "rows_removed_by_rule": {"R1_sources": rule_cnt["R1"], "R2_row_filters": rule_cnt["R2"], "R3_benchmark_overlap": rule_cnt["R3"]},
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
    }, texts


# ----------------------------------------------------------------------------------------------- the stage
def check_patterns(cfg, drop, sources):
    """Stop if one of the config's must_drop_patterns matches a source that is in the data but not dropped."""
    pats = [p.lower() for p in cfg.get("clean", {}).get("must_drop_patterns", [])]
    unlisted = sorted(s for s in sources if any(p in s.lower() for p in pats) and s not in drop)
    if unlisted:
        raise engine.PoolError(f"must_drop_patterns match sources that the config does not drop at the clean stage: {unlisted}")


def run(cfg, pool_dir, out_dir, bench, info_labels=(), workers=None, report_only=False, no_stoplist=False):
    """Run the stage on <pool_dir>/{train,val}.jsonl.gz. Writes the cleaned splits, REPORT.json, REPORT.md,
    stoplist.tsv and dropped.tsv.gz into out_dir. -> the report"""
    t_start = time.time()
    os.makedirs(out_dir, exist_ok=True)
    n_workers = workers or engine.default_workers()
    opt = dict(DEFAULTS, **cfg.get("clean", {}).get("decontaminate", {}), no_stoplist=no_stoplist)
    drop = engine.dropped_at(cfg, "clean")
    filters = collections.defaultdict(tuple)
    for f in cfg.get("clean", {}).get("drop_rows", []):
        filters[f.get("source", "*")] += ((f["field"], f["equals"], f["reason"]),)
    paths = {name: engine.split_path(pool_dir, name) for name in engine.SPLITS}

    G.update(N=opt["word_gram"], NC=opt["char_gram"], fallback=opt["instruction_fallback"], drop=drop, filters=dict(filters))
    t0 = time.time()
    with engine.Workers(n_workers) as w:
        bm = load_benchmarks(bench, list(info_labels), w, out_dir)
    print(f"benchmark grams: {len(bm['uh']):,} distinct (drop), {len(bm['iuh']):,} (info), {len(bm['labels'])} labels, "
          f"{time.time() - t0:.0f} s", flush=True)

    seconds = {"benchmarks": round(time.time() - t0, 1)}
    G.update(uh=bm["uh"], iuh=bm["iuh"], ih=bm["ih"], ib=bm["ib"])
    t0 = time.time()
    with engine.Workers(n_workers) as w:
        res = {name: scan(path, w, name) for name, path in paths.items()}
    present = {r[1] for rows in res.values() for r in rows}
    check_patterns(cfg, drop, present)
    absent = sorted(s for s in drop if s not in present)

    seconds["scan"] = round(time.time() - t0, 1)
    t0 = time.time()
    sl = build_stoplist(res["train"], opt)
    qrare = rare_question_grams(res, opt)
    decided = {name: decide(res[name], sl["stop"], qrare) for name in res}
    attr = {name: attribute(decided[name][1], bm["h"], bm["b"], bm["labels"]) for name in res}
    flagged, top_per_src, need_hash, need_rows = report_needs(res["train"], sl, decided["train"][0], opt)

    seconds["decide"] = round(time.time() - t0, 1)
    t0 = time.time()
    G.update(want=np.array(sorted(need_hash), np.uint64), write=not report_only)
    summary, texts = {}, {}
    with engine.Workers(n_workers) as w, gzip.open(os.path.join(out_dir, "dropped.tsv.gz"), "wt") as dropped_log:
        dropped_log.write("split\tindex\tid\tsource\trule\tbenchmarks\n")
        for name, path in paths.items():
            summary[name], tx = write_split(name, path, out_dir, res[name], decided[name], attr[name], bm["info_labels"], w,
                                            need_rows if name == "train" else (), dropped_log, report_only)
            texts.update(tx)

    seconds["write"] = round(time.time() - t0, 1)
    rep = {"options": opt, "pool": os.path.abspath(pool_dir),
           "rules": {"R1_sources": {s: drop[s] for s in sorted(drop) if s in present}, "R1_listed_but_absent": absent,
                     "R2_row_filters": cfg.get("clean", {}).get("drop_rows", [])},
           "benchmarks": {"files": list(bench), "info_labels": list(info_labels), "unreadable_records": bm["unknown"],
                          "per_label": bm["stats"], "distinct_grams_total": int(len(bm["uh"]))}}
    rep["stoplist"], rep["over_warn_frac_plain_rule"] = stoplist_report(out_dir, opt, res["train"], sl, decided["train"][1],
                                                                        flagged, top_per_src, texts)
    rep["summary"] = summary
    tr_s = summary["train"]["per_source"]
    rep["top30_sources_by_rows_lost"] = [{"source": s, "before": v["before"], "after": v["after"], "removed": v["removed"],
                                          "removed_frac": round(v["removed"] / v["before"], 4), "by_rule": v["by_rule"]}
                                         for s, v in sorted(tr_s.items(), key=lambda kv: -kv[1]["removed"])[:30]]
    rep["R3_sources_losing_most_to_R3"] = sorted(
        [{"source": s, "removed": v["by_rule"].get("R3", 0), "frac": round(v["by_rule"].get("R3", 0) / v["before"], 4)}
         for s, v in tr_s.items() if v["by_rule"].get("R3")], key=lambda d: -d["removed"])[:30]
    rep["runtime_seconds"], rep["seconds_per_step"] = round(time.time() - t_start), seconds
    with open(os.path.join(out_dir, "REPORT.json"), "w") as f:
        json.dump(rep, f, indent=1, default=str)
    write_md(rep, os.path.join(out_dir, "REPORT.md"))
    print(f"clean: {summary['train']['rows_in']:,} -> {summary['train']['rows_out']:,} training rows, "
          f"{summary['val']['rows_in']:,} -> {summary['val']['rows_out']:,} validation rows, {rep['runtime_seconds']} s {seconds}", flush=True)
    return rep


def write_md(rep, path):
    s, L = rep["summary"], []
    for nm, x in s.items():
        L += [f"## {nm}", f"- rows in {x['rows_in']:,}, out {x['rows_out']:,}; sources {x['sources_in']} -> {x['sources_out']}",
              f"- removed by rule: {x['rows_removed_by_rule']}",
              f"- R3 plain rule (no stop-list) would remove {x['R3_plain_rule_would_remove (no stop-list)']:,}; stop-list saved "
              f"{x['R3_stoplist_saved_rows']:,}; dropped for a question match alone {x['R3_rows_dropped_for_the_question_only']:,}",
              f"- licences out: {x['license_counts_out']}", ""]
    L += ["## R1 sources dropped", ""] + [f"- `{k}`: {v}" for k, v in rep["rules"]["R1_sources"].items()]
    L += ["", f"Listed but absent from the data: {rep['rules']['R1_listed_but_absent']}", ""]
    L += ["## R2 row filters", ""] + [f"- {f}" for f in rep["rules"]["R2_row_filters"]] + [""]
    L += ["## Stop-list", "", f"{rep['stoplist']['rule']}: {rep['stoplist']['size']} grams (of {rep['stoplist']['matched_distinct_grams']} matched).",
          "", "Top matched grams by pool rows:", ""]
    L += [f"- {t['rows']} rows / {t['sources']} sources / stop={t['stop']}: `{t['text']}`" for t in rep["stoplist"]["top"]]
    L += ["", "## Sources over the warn fraction under the plain rule", ""]
    for k, v in rep["over_warn_frac_plain_rule"].items():
        L.append(f"- `{k}`: {v['plain_drop']}/{v['rows']} ({v['plain_frac']:.1%}) plain, {v['with_stoplist_drop']} with stop-list; top: "
                 + "; ".join(f"`{g['text']}` ({g['rows_in_source']})" for g in v["top_grams"]))
    L += ["", "## R3 rows per benchmark (train, any match / first match)", ""]
    ta, tf = s["train"]["R3_rows_per_benchmark_any_match"], s["train"]["R3_rows_per_benchmark_first_match"]
    L += [f"- {k}: {v} / {tf.get(k, 0)}" for k, v in ta.items()]
    L += ["", "## Top 30 sources by rows lost (train)", ""]
    L += [f"- `{d['source']}`: {d['before']} -> {d['after']} (-{d['removed']}, {d['removed_frac']:.1%}) {d['by_rule']}"
          for d in rep["top30_sources_by_rows_lost"]]
    L += ["", "## Benchmarks", "", f"files: {rep['benchmarks']['files']}", f"info only labels: {rep['benchmarks']['info_labels']}", ""]
    L += [f"- {k}: {v['items']} items, {v['items_too_short']} too short to match, {v['distinct_grams']} distinct grams"
          for k, v in rep["benchmarks"]["per_label"].items()]
    L += ["", f"runtime {rep['runtime_seconds']} s"]
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
