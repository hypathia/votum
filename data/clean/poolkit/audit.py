"""An independent audit of the mask stage on a fixed-seed sample of masked rows from the self-collected sources.

Run the steps in this order (the audit directory is scratch space for the intermediate files):

  sample   draws n masked self-collected rows (train and val), the same number from each dataset config, into
           sample.jsonl
  regex    residual structured identifiers (e-mail, phone, ID, URL, street address): the mask stage's own patterns,
           plus two loose patterns of this module that over-report on purpose, so that a form the stage cannot see
           still shows up
  spacy    PERSON entities on the English rows (needs spaCy with en_core_web_trf)
  gliner   person and address spans on the non-English rows, at a lower threshold than the mask stage (needs gliner)
  report   hits per kind and config; names are split by whether the config masks names in that source

Hits are stored with at most about 60 characters of context in hits_*.json. They contain residual personal data:
keep them local and read them; a hit is a candidate, not a confirmed miss.
"""
import collections
import json
import os
import random
import re

from . import engine, pii

# independent of the mask stage: a "+" or "00" number of nine or more digits, or eight or more digits right after a telephone word
LOOSE_PHONE = re.compile(r"(?i)(?<![\w+])(?:\+|00)[1-9](?:[\s./()-]{0,2}\d){8,13}(?!\d)|"
                         r"(?:\b(?:tel|phone|fax|mobile?|natel|handy|call)\b\D{0,12})\d(?:[\s./()-]{0,2}\d){7,13}(?!\d)")
LOOSE_EMAIL = re.compile(r"[^\s@<>()\[\],;:\"']+@[^\s@<>()\[\],;:\"']+\.[A-Za-z]{2,}")
G = {}                                              # state shared with the workers (set before a pool is made)


def texts_of(row):
    return engine.strings(row.get("state"), []) + [row.get("question") or ""] + [o.get("text") or "" for o in row.get("options") or []]


def strip_ph(t):
    """Blank out the placeholders so that they cannot be part of a hit."""
    return pii.PLACEHOLDER.sub(" ", t)


def load_sample(d):
    with open(os.path.join(d, "sample.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_hits(d, name, hits):
    with open(os.path.join(d, f"hits_{name}.json"), "w") as f:
        json.dump(hits, f, ensure_ascii=False)


def _config_worker(lines):
    """-> per line, the dataset config of a self-collected row, else None"""
    out = []
    for line in lines:
        p = engine.policy(G["cfg"], json.loads(line)["source"])
        out.append(p["config"] if p.get("origin") == "self-collected" and not p.get("drop") else None)
    return out


def sample(cfg, pool_dir, audit_dir, n=500, seed=20261107, workers=None):
    """Draw the sample: the same rows for the same pool, seed and n."""
    G.update(cfg=cfg)
    where = collections.defaultdict(list)           # config -> [(split, row index)] in file order
    with engine.Workers(workers or engine.default_workers()) as w:
        for split in engine.SPLITS:
            i = 0
            for res in w.imap(_config_worker, engine.chunks(engine.split_path(pool_dir, split))):
                for c in res:
                    if c is not None:
                        where[c].append((split, i))
                    i += 1
    configs = list(cfg.get("export", {}).get("configs") or sorted(where))
    shares = [n // len(configs)] * (len(configs) - 1) + [n - n // len(configs) * (len(configs) - 1)]
    rng, picked = random.Random(seed), []
    for c, k in zip(configs, shares):
        if k > len(where[c]):
            print(f"only {len(where[c])} self-collected rows in {c}; sampling all of them")
            k = len(where[c])
        picked += [(c, pos) for pos in rng.sample(where[c], k)]
    want = {pos for _, pos in picked}
    found = {}
    for split in engine.SPLITS:
        for i, line in enumerate(engine.lines_of(engine.split_path(pool_dir, split))):
            if (split, i) in want:
                found[(split, i)] = line
    os.makedirs(audit_dir, exist_ok=True)
    with open(os.path.join(audit_dir, "sample.jsonl"), "w", encoding="utf-8") as f:
        for c, pos in picked:
            r = json.loads(found[pos])
            r["_config"] = c
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print({c: len(v) for c, v in where.items()}, "sampled", len(picked))


def regex(audit_dir):
    hits = []
    for r in load_sample(audit_dir):
        rid = r["id"]
        for t in texts_of(r):
            t = strip_ph(t)

            def ctx(m, before=30, after=10):
                return t[max(0, m.start() - before):m.end() + after]

            for m in pii.EMAIL.finditer(t):
                if not (pii.ROLE_LOCAL.match(m.group(0).split("@")[0]) or re.search(r"@(?:[\w-]+\.)*example\.", m.group(0), re.I)):
                    hits.append((rid, "email", ctx(m)))
            for m in pii.PHONE.finditer(t):
                if pii._phone_ok(m):
                    hits.append((rid, "phone", ctx(m)))
            for rx in (pii.PHONE_INTL, pii.PHONE_NATL):
                for m in rx.finditer(t):
                    if pii._phone2_ok(m):
                        hits.append((rid, "phone", ctx(m)))
            for m in pii.PHONE_BARE.finditer(t):
                hits.append((rid, "phone", ctx(m, 10)))
            for m in LOOSE_PHONE.finditer(t):
                hits.append((rid, "phone (loose)", ctx(m)))
            for m in LOOSE_EMAIL.finditer(t):
                if not pii.EMAIL.fullmatch(m.group(0)):
                    hits.append((rid, "email (loose)", ctx(m)))
            for rx in [pii.IBAN, pii.SSN] + pii.ID_BARE:
                for m in rx.finditer(t):
                    hits.append((rid, "id", ctx(m)))
            for m in pii.ID_LABEL.finditer(t):
                if pii._id_ok(m.group(3)):
                    hits.append((rid, "id", ctx(m, 10)))
            for m in pii.CARD.finditer(t):
                d = re.sub(r"\D", "", m.group(0))
                if 13 <= len(d) <= 19 and pii._luhn(d) and len(set(d)) > 2:
                    hits.append((rid, "id", ctx(m)))
            for m in pii.URL.finditer(t):
                if pii._url_ok(m):
                    hits.append((rid, "url", ctx(m, 10, 0)[:60]))
            for rx in pii.ADDR:
                for m in rx.finditer(t):
                    hits.append((rid, "address", ctx(m, 20)))
    save_hits(audit_dir, "regex", hits)
    print(collections.Counter(h[1] for h in hits))


def spacy(audit_dir):
    import spacy as sp
    nlp = sp.load("en_core_web_trf")
    hits = []
    for r in load_sample(audit_dir):
        if r.get("lang") != "en":
            continue
        for t in texts_of(r):
            if len(t) < 4:
                continue
            for i in range(0, len(t), 3000):
                seg = t[i:i + 3000]
                for e in nlp(seg).ents:
                    if e.label_ == "PERSON" and not re.search(r"\[(?:[A-Z]+)$|^[A-Z]+\]", e.text) and "[" not in e.text:
                        hits.append((r["id"], "name", seg[max(0, e.start_char - 25):e.end_char + 10]))
    save_hits(audit_dir, "spacy", hits)
    print(len(hits))


def gliner(cfg, audit_dir, device="cuda", threshold=0.3):
    name = cfg.get("mask", {}).get("model") or pii.DEFAULTS["model"]
    model, filters = pii.Model(name, device, threshold=threshold, workers=1), pii.Filters(threshold, threshold, threshold)
    hits = []
    for r in load_sample(audit_dir):
        if r.get("lang") == "en":
            continue
        ts = [t for t in (strip_ph(t) for t in texts_of(r)) if pii.needs_ner(t)]
        for t, sp in zip(ts, pii.find_spans(model, filters, ts, True)):
            for s, e, k in sp:
                hits.append((r["id"], "name" if k == "NAME" else "address", t[max(0, s - 25):e + 10]))
    model.close()
    save_hits(audit_dir, "gliner", hits)
    print(collections.Counter(h[1] for h in hits))


def report(cfg, audit_dir):
    rows = {r["id"]: r for r in load_sample(audit_dir)}
    out = collections.defaultdict(collections.Counter)
    n = collections.Counter(r["_config"] for r in rows.values())
    for fn in ("regex", "spacy", "gliner"):
        p = os.path.join(audit_dir, f"hits_{fn}.json")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            hits = json.load(f)
        for rid, kind, _ in hits:
            r = rows[rid]
            if kind == "name":
                kind = "name (policy: mask)" if cfg["sources"][r["source"]].get("names") else "name (policy: keep)"
            out[r["_config"]][kind + " / " + fn] += 1
    print("sample sizes", dict(n))
    for c in out:
        print(c, dict(out[c]))
