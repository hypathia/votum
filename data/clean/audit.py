#!/usr/bin/env python3
"""Independent audit of the PII pass on a fixed-seed sample of masked rows from the SELF-COLLECTED sources.

usage: python3 -I audit.py {sample,regex,spacy,gliner,report} --audit-dir AUDIT [options]

Subcommands (run in this order; AUDIT = scratch directory for the intermediate files):
  sample  --pool-dir clean_pii --policy pii_policy.json --audit-dir AUDIT [--n 500] [--seed 20261107]
          draws n masked self-collected rows (train+val), half from each config (default / share_alike), writes AUDIT/sample.jsonl
  regex   --audit-dir AUDIT --pass-script pii_pass.py      residual structured identifiers (email, phone, id, url, street address)
  spacy   --audit-dir AUDIT        (needs spaCy with en_core_web_trf; PERSON on English rows)
  gliner  --audit-dir AUDIT [--threshold 0.3]   (needs gliner and torch; person/address on non-English rows at a lower threshold)
  report  --audit-dir AUDIT --policy pii_policy.json      residual rate per 500 rows by kind and config
Hits are stored with a <= 60 character context in AUDIT/hits_*.json (keep them local; they contain residual personal data).
"""
import argparse, collections, gzip, importlib.util, json, os, random, re


# independent of pii_pass.py: a "+" or "00" number of nine or more digits, or eight or more digits right after a telephone word
LOOSE_PHONE = re.compile(r"(?i)(?<![\w+])(?:\+|00)[1-9](?:[\s./()-]{0,2}\d){8,13}(?!\d)|(?:\b(?:tel|phone|fax|mobile?|natel|handy|call)\b\D{0,12})\d(?:[\s./()-]{0,2}\d){7,13}(?!\d)")
LOOSE_EMAIL = re.compile(r"[^\s@<>()\[\],;:\"']+@[^\s@<>()\[\],;:\"']+\.[A-Za-z]{2,}")


def load_pass(path):
    spec = importlib.util.spec_from_file_location("pii_pass", path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def texts_of(row):
    out = []
    def walk(x):
        if isinstance(x, dict): [walk(v) for v in x.values()]
        elif isinstance(x, list): [walk(v) for v in x]
        elif isinstance(x, str): out.append(x)
    walk(row.get("state")); out.append(row.get("question") or "")
    out += [o.get("text") or "" for o in row.get("options") or []]
    return out


def strip_ph(t):
    return re.sub(r"\[(?:EMAIL|PHONE|IBAN|ID|CARD|IP|URL|ADDRESS|NAME)\]", " ", t)


def cmd_sample(a):
    pol = json.load(open(a.policy)); rng = random.Random(a.seed); pool = {"default": [], "share_alike": []}
    for fn in ("train", "val"):
        with gzip.open(os.path.join(a.pool_dir, fn + ".jsonl.gz"), "rt", encoding="utf-8") as f:
            for l in f:
                r = json.loads(l); p = pol[r["source"]]
                if p["origin"] == "self-collected" and not p["excluded"]: pool[p["config"]].append(l)
    rows = []
    for c, n in (("default", a.n // 2), ("share_alike", a.n - a.n // 2)):
        rows += [(c, l) for l in rng.sample(pool[c], n)]
    os.makedirs(a.audit_dir, exist_ok=True)
    with open(os.path.join(a.audit_dir, "sample.jsonl"), "w", encoding="utf-8") as f:
        for c, l in rows:
            r = json.loads(l); r["_config"] = c; f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print({c: len(v) for c, v in pool.items()}, "sampled", len(rows))


def load_sample(d):
    return [json.loads(l) for l in open(os.path.join(d, "sample.jsonl"), encoding="utf-8")]


def cmd_regex(a):
    pp = load_pass(a.pass_script); hits = []
    for r in load_sample(a.audit_dir):
        for t in texts_of(r):
            t = strip_ph(t)
            for m in pp.EMAIL.finditer(t):
                if not (pp.ROLE_LOCAL.match(m.group(0).split("@")[0]) or re.search(r"@(?:[\w-]+\.)*example\.", m.group(0), re.I)):
                    hits.append((r["id"], "email", t[max(0, m.start() - 30):m.end() + 10]))
            for m in pp.PHONE.finditer(t):
                if pp._phone_ok(m): hits.append((r["id"], "phone", t[max(0, m.start() - 30):m.end() + 10]))
            for rx in (pp.PHONE_INTL, pp.PHONE_NATL):
                for m in rx.finditer(t):
                    if pp._phone2_ok(m): hits.append((r["id"], "phone", t[max(0, m.start() - 30):m.end() + 10]))
            for m in pp.PHONE_BARE.finditer(t): hits.append((r["id"], "phone", t[max(0, m.start() - 10):m.end() + 10]))
            # two checks that do not use the pass's own patterns, so a form the pass cannot see still shows up here
            # (they over-report on purpose; read the hits)
            for m in LOOSE_PHONE.finditer(t): hits.append((r["id"], "phone (loose)", t[max(0, m.start() - 30):m.end() + 10]))
            for m in LOOSE_EMAIL.finditer(t):
                if not pp.EMAIL.fullmatch(m.group(0)): hits.append((r["id"], "email (loose)", t[max(0, m.start() - 30):m.end() + 10]))
            for rx in [pp.IBAN, pp.SSN] + pp.ID_BARE:
                for m in rx.finditer(t): hits.append((r["id"], "id", t[max(0, m.start() - 30):m.end() + 10]))
            for m in pp.ID_LABEL.finditer(t):
                if pp._id_ok(m.group(3)): hits.append((r["id"], "id", t[max(0, m.start() - 10):m.end() + 10]))
            for m in pp.CARD.finditer(t):
                d = re.sub(r"\D", "", m.group(0))
                if 13 <= len(d) <= 19 and pp._luhn(d) and len(set(d)) > 2: hits.append((r["id"], "id", t[max(0, m.start() - 30):m.end() + 10]))
            for m in pp.URL.finditer(t):
                if pp._url_ok(m): hits.append((r["id"], "url", t[max(0, m.start() - 10):m.end()][:60]))
            for rx in pp.ADDR:
                for m in rx.finditer(t): hits.append((r["id"], "address", t[max(0, m.start() - 20):m.end() + 10]))
    json.dump(hits, open(os.path.join(a.audit_dir, "hits_regex.json"), "w"), ensure_ascii=False)
    print(collections.Counter(h[1] for h in hits))


def cmd_spacy(a):
    import spacy
    nlp = spacy.load("en_core_web_trf"); hits = []
    for r in load_sample(a.audit_dir):
        if r.get("lang") != "en": continue
        for t in texts_of(r):
            if len(t) < 4: continue
            for i in range(0, len(t), 3000):
                seg = t[i:i + 3000]; doc = nlp(seg)
                for e in doc.ents:
                    if e.label_ == "PERSON" and not re.search(r"\[(?:[A-Z]+)$|^[A-Z]+\]", e.text) and "[" not in e.text:
                        hits.append((r["id"], "name", seg[max(0, e.start_char - 25):e.end_char + 10]))
    json.dump(hits, open(os.path.join(a.audit_dir, "hits_spacy.json"), "w"), ensure_ascii=False)
    print(len(hits))


def cmd_gliner(a):
    pp = load_pass(a.pass_script)
    ner = pp.NER("urchade/gliner_multi_pii-v1", "cuda", a.threshold, a.threshold, th_user=a.threshold)
    rows = [r for r in load_sample(a.audit_dir) if r.get("lang") != "en"]; hits = []
    for r in rows:
        ts = [strip_ph(t) for t in texts_of(r)]
        ts = [t for t in ts if pp.needs_ner(t)]
        for t, sp in zip(ts, ner.spans(ts, True)):
            for s, e, k in sp: hits.append((r["id"], "name" if k == "NAME" else "address", t[max(0, s - 25):e + 10]))
    json.dump(hits, open(os.path.join(a.audit_dir, "hits_gliner.json"), "w"), ensure_ascii=False)
    print(collections.Counter(h[1] for h in hits))


def cmd_report(a):
    pol = json.load(open(a.policy)); rows = {r["id"]: r for r in load_sample(a.audit_dir)}
    out = collections.defaultdict(collections.Counter); n = collections.Counter(r["_config"] for r in rows.values())
    for fn in ("regex", "spacy", "gliner"):
        p = os.path.join(a.audit_dir, f"hits_{fn}.json")
        if not os.path.exists(p): continue
        for rid, kind, ctx in json.load(open(p)):
            r = rows[rid]; names = pol[r["source"]]["names"]
            k = kind if kind != "name" else ("name (policy: mask)" if names else "name (policy: keep)")
            out[r["_config"]][k + " / " + fn] += 1
    print("sample sizes", dict(n))
    for c in out: print(c, dict(out[c]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["sample", "regex", "spacy", "gliner", "report"])
    ap.add_argument("--pool-dir"); ap.add_argument("--policy"); ap.add_argument("--audit-dir", required=True)
    ap.add_argument("--n", type=int, default=500); ap.add_argument("--seed", type=int, default=20261107)
    ap.add_argument("--pass-script", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "pii_pass.py"))
    ap.add_argument("--threshold", type=float, default=0.3)
    a = ap.parse_args(); globals()["cmd_" + a.cmd](a)


if __name__ == "__main__":
    main()
