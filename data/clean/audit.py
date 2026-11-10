#!/usr/bin/env python3
"""Step 5: an independent audit of the PII pass on a fixed-seed sample of masked rows from the self-collected sources.

    python3 -I audit.py {sample,regex,spacy,gliner,report} --audit-dir AUDIT [options]

Run the subcommands in this order (AUDIT is a scratch directory for the intermediate files):

  sample  --pool-dir clean_pii --policy pii_policy.json [--n 500] [--seed 20261107]
          draws n masked self-collected rows (train and val), half from each config, into AUDIT/sample.jsonl
  regex   [--pass-script pii_pass.py]
          residual structured identifiers (e-mail, phone, ID, URL, street address): the pass's own patterns, plus two
          loose patterns of this script that over-report on purpose, so a form the pass cannot see still shows up
  spacy   PERSON entities on the English rows (needs spaCy with en_core_web_trf)
  gliner  [--model M] [--device cuda] [--threshold 0.3]
          person and address spans on the non-English rows, at a lower threshold than the pass (needs gliner, torch)
  report  --policy pii_policy.json
          hits per kind and config; names are split by whether the policy masks names in that source

Hits are stored with at most about 60 characters of context in AUDIT/hits_*.json. They contain residual personal
data: keep them local and read them, a hit is a candidate, not a confirmed miss.
"""
import argparse
import collections
import gzip
import importlib.util
import json
import os
import random
import re

HERE = os.path.dirname(os.path.abspath(__file__))

# independent of pii_pass.py: a "+" or "00" number of nine or more digits, or eight or more digits right after a telephone word
LOOSE_PHONE = re.compile(r"(?i)(?<![\w+])(?:\+|00)[1-9](?:[\s./()-]{0,2}\d){8,13}(?!\d)|"
                         r"(?:\b(?:tel|phone|fax|mobile?|natel|handy|call)\b\D{0,12})\d(?:[\s./()-]{0,2}\d){7,13}(?!\d)")
LOOSE_EMAIL = re.compile(r"[^\s@<>()\[\],;:\"']+@[^\s@<>()\[\],;:\"']+\.[A-Za-z]{2,}")
PLACEHOLDER = re.compile(r"\[(?:EMAIL|PHONE|IBAN|ID|CARD|IP|URL|ADDRESS|NAME)\]")


def load_pass(path):
    """Import pii_pass.py from a path (works under python -I, which leaves the script directory off sys.path)."""
    spec = importlib.util.spec_from_file_location("pii_pass", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def texts_of(row):
    out = []

    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, str):
            out.append(x)

    walk(row.get("state"))
    out.append(row.get("question") or "")
    out += [o.get("text") or "" for o in row.get("options") or []]
    return out


def strip_ph(t):
    """Blank out the placeholders so that they cannot be part of a hit."""
    return PLACEHOLDER.sub(" ", t)


def load_sample(d):
    with open(os.path.join(d, "sample.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_hits(a, name, hits):
    with open(os.path.join(a.audit_dir, f"hits_{name}.json"), "w") as f:
        json.dump(hits, f, ensure_ascii=False)


def cmd_sample(a):
    with open(a.policy) as f:
        pol = json.load(f)
    rng = random.Random(a.seed)
    pool = {"default": [], "share_alike": []}
    for fn in ("train", "val"):
        with gzip.open(os.path.join(a.pool_dir, fn + ".jsonl.gz"), "rt", encoding="utf-8") as f:
            for line in f:
                p = pol[json.loads(line)["source"]]
                if p["origin"] == "self-collected" and not p["excluded"]:
                    pool[p["config"]].append(line)
    rows = []
    for c, n in (("default", a.n // 2), ("share_alike", a.n - a.n // 2)):
        if n > len(pool[c]):
            print(f"only {len(pool[c])} self-collected rows in {c}; sampling all of them")
            n = len(pool[c])
        rows += [(c, line) for line in rng.sample(pool[c], n)]
    os.makedirs(a.audit_dir, exist_ok=True)
    with open(os.path.join(a.audit_dir, "sample.jsonl"), "w", encoding="utf-8") as f:
        for c, line in rows:
            r = json.loads(line)
            r["_config"] = c
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print({c: len(v) for c, v in pool.items()}, "sampled", len(rows))


def cmd_regex(a):
    pp = load_pass(a.pass_script)
    hits = []
    for r in load_sample(a.audit_dir):
        rid = r["id"]
        for t in texts_of(r):
            t = strip_ph(t)

            def ctx(m, before=30, after=10):
                return t[max(0, m.start() - before):m.end() + after]

            for m in pp.EMAIL.finditer(t):
                if not (pp.ROLE_LOCAL.match(m.group(0).split("@")[0]) or re.search(r"@(?:[\w-]+\.)*example\.", m.group(0), re.I)):
                    hits.append((rid, "email", ctx(m)))
            for m in pp.PHONE.finditer(t):
                if pp._phone_ok(m):
                    hits.append((rid, "phone", ctx(m)))
            for rx in (pp.PHONE_INTL, pp.PHONE_NATL):
                for m in rx.finditer(t):
                    if pp._phone2_ok(m):
                        hits.append((rid, "phone", ctx(m)))
            for m in pp.PHONE_BARE.finditer(t):
                hits.append((rid, "phone", ctx(m, 10)))
            for m in LOOSE_PHONE.finditer(t):
                hits.append((rid, "phone (loose)", ctx(m)))
            for m in LOOSE_EMAIL.finditer(t):
                if not pp.EMAIL.fullmatch(m.group(0)):
                    hits.append((rid, "email (loose)", ctx(m)))
            for rx in [pp.IBAN, pp.SSN] + pp.ID_BARE:
                for m in rx.finditer(t):
                    hits.append((rid, "id", ctx(m)))
            for m in pp.ID_LABEL.finditer(t):
                if pp._id_ok(m.group(3)):
                    hits.append((rid, "id", ctx(m, 10)))
            for m in pp.CARD.finditer(t):
                d = re.sub(r"\D", "", m.group(0))
                if 13 <= len(d) <= 19 and pp._luhn(d) and len(set(d)) > 2:
                    hits.append((rid, "id", ctx(m)))
            for m in pp.URL.finditer(t):
                if pp._url_ok(m):
                    hits.append((rid, "url", ctx(m, 10, 0)[:60]))
            for rx in pp.ADDR:
                for m in rx.finditer(t):
                    hits.append((rid, "address", ctx(m, 20)))
    save_hits(a, "regex", hits)
    print(collections.Counter(h[1] for h in hits))


def cmd_spacy(a):
    import spacy
    nlp = spacy.load("en_core_web_trf")
    hits = []
    for r in load_sample(a.audit_dir):
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
    save_hits(a, "spacy", hits)
    print(len(hits))


def cmd_gliner(a):
    pp = load_pass(a.pass_script)
    ner = pp.NER(a.model, a.device, a.threshold, a.threshold, th_user=a.threshold)
    hits = []
    for r in load_sample(a.audit_dir):
        if r.get("lang") == "en":
            continue
        ts = [t for t in (strip_ph(t) for t in texts_of(r)) if pp.needs_ner(t)]
        for t, sp in zip(ts, ner.spans(ts, True)):
            for s, e, k in sp:
                hits.append((r["id"], "name" if k == "NAME" else "address", t[max(0, s - 25):e + 10]))
    save_hits(a, "gliner", hits)
    print(collections.Counter(h[1] for h in hits))


def cmd_report(a):
    with open(a.policy) as f:
        pol = json.load(f)
    rows = {r["id"]: r for r in load_sample(a.audit_dir)}
    out = collections.defaultdict(collections.Counter)
    n = collections.Counter(r["_config"] for r in rows.values())
    for fn in ("regex", "spacy", "gliner"):
        p = os.path.join(a.audit_dir, f"hits_{fn}.json")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            hits = json.load(f)
        for rid, kind, _ in hits:
            r = rows[rid]
            if kind == "name":
                kind = "name (policy: mask)" if pol[r["source"]]["names"] else "name (policy: keep)"
            out[r["_config"]][kind + " / " + fn] += 1
    print("sample sizes", dict(n))
    for c in out:
        print(c, dict(out[c]))


COMMANDS = {"sample": cmd_sample, "regex": cmd_regex, "spacy": cmd_spacy, "gliner": cmd_gliner, "report": cmd_report}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=list(COMMANDS))
    ap.add_argument("--audit-dir", required=True)
    ap.add_argument("--pool-dir", help="sample: the masked pool directory")
    ap.add_argument("--policy", help="sample, report: pii_policy.json")
    ap.add_argument("--n", type=int, default=500, help="sample: rows to draw")
    ap.add_argument("--seed", type=int, default=20261107, help="sample: random seed")
    ap.add_argument("--pass-script", default=os.path.join(HERE, "pii_pass.py"), help="regex, gliner: the pii_pass.py to audit")
    ap.add_argument("--model", default="urchade/gliner_multi_pii-v1", help="gliner: model id or path")
    ap.add_argument("--device", default="cuda", help="gliner: cuda or cpu")
    ap.add_argument("--threshold", type=float, default=0.3, help="gliner: span score threshold")
    a = ap.parse_args()
    need = {"sample": ("pool_dir", "policy"), "report": ("policy",)}.get(a.cmd, ())
    missing = ["--" + k.replace("_", "-") for k in need if getattr(a, k) is None]
    if missing:
        ap.error(f"{a.cmd} needs {' '.join(missing)}")
    COMMANDS[a.cmd](a)


if __name__ == "__main__":
    main()
