#!/usr/bin/env python3
"""PII pass for the cleaned Votum pool (step 2 of data/clean/README.md: clean -> clean_pii).

What it does
  1. Drops every row of a source marked `excluded` in the policy file (licence terms not cleared for redistribution).
  2. REDISTRIBUTED sources (reformatted copies of published dataset releases): rows are copied byte for byte
     (they already carry the builder's regex masks). No NER.
  3. SELF-COLLECTED sources (text the builder scraped/extracted itself): every string of `state` (plain string or every
     string value of a JSON state), `question` and every option `text` gets
       - regex masks: [EMAIL] [PHONE] [IBAN] [ID] [CARD] [IP] (the placeholder style of the pool's original masking),
         [URL] (URLs carrying a personal identifier; plain domain URLs are kept), [ADDRESS] (street addresses),
         [NAME] for wiki user references (User:xyz, Benutzer:xyz);
       - NER (GLiNER multilingual PII model, GPU): spans labelled address (kept only when they contain a digit) -> [ADDRESS]
         for every self-collected source; person (+ username) -> [NAME] only for sources whose policy says names = yes.
     If masking makes two option texts of a row identical, the row is dropped (counted, ids in pii_dropped.tsv).
  4. Writes train/val .jsonl.gz in the original row format (all fields), a report and a drop list.

Per-chunk checkpoints (CHUNK input rows each) are written under OUT/ckpt; re-running the same command resumes.

Usage (needs gliner and torch):
  python3 -I pii_pass.py \\
      --train clean/train.jsonl.gz --val clean/val.jsonl.gz \\
      --policy pii_policy.json --stop-spans pii_stop_spans.txt --out-dir clean_pii \\
      [--device cuda] [--chunk 4000] [--no-ner] [--part k/N] [--mem-fraction F]

Arguments
  --train/--val   input jsonl.gz (one JSON row per line; fields id, source, state, question, options[{id,text}], ...)
  --policy        pii_policy.json: per source {excluded, origin, names, config, reason}
  --out-dir       output directory (train.jsonl.gz, val.jsonl.gz, pii_report.json, pii_dropped.tsv, ckpt/)
  --model         GLiNER model id or path (default urchade/gliner_multi_pii-v1, Apache-2.0)
  --stop-spans    text file, one lowercase span per line, never masked as a name (role words such as 'claimant')
  --device        cuda or cpu;  --no-ner  regex layer only (tests);  --th-person / --th-address  NER thresholds
  --part k/N      process only chunks with index % N == k (run N processes, then rerun without --part to assemble)
  --mem-fraction  cap on the GPU memory share per process
"""
import argparse, collections, gzip, json, os, re, time

# ----------------------------------------------------------------------------------------------- regex layer
EMAIL = re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9_.-]+\.[A-Za-z]{2,}(?![\w-])")
PHONE = re.compile(r"(?<![\w./-])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?)?\d{2,4}(?:[\s.-]\d{2,4}){1,4}(?![\w./-])")
IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){3,7}(?: ?[A-Z0-9]{1,3})?\b")
SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
IPV4 = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")
_PRIVATE_IP = re.compile(r"^(?:10\.|127\.|0\.|192\.168\.|172\.(?:1[6-9]|2\d|3[01])\.|255\.|169\.254\.|8\.8\.|1\.1\.1\.1$|1\.0\.0\.1$|224\.|239\.)")
_ISBN = re.compile(r"^97[89]")
ROLE_LOCAL = re.compile(r"^(?:info|contact|kontakt|support|press|presse|media|office|help|service|admin|webmaster|noreply|no-reply|post|mail|"
                        r"secretariat|sekretariat|greffe|registry|registrar|enquiries|inquiries|poststelle|sales|hello|team|complaints)$", re.I)
ID_BARE = [
    re.compile(r"\b756\.\d{4}\.\d{4}\.\d{2}\b"),                                        # Swiss AHV/AVS number
    re.compile(r"\b\d{8}-?[A-HJ-NP-TV-Z]\b(?<!\d{8}A)"),                                # Spanish DNI
    re.compile(r"\b[XYZ]-?\d{7}-?[A-Z]\b"),                                             # Spanish NIE
    re.compile(r"\b[A-CEGHJ-PR-TW-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b"),            # UK national insurance number
]
ID_LABEL = re.compile(
    r"\b((?-i:AHV(?:-Nr\.?)?|AVS|NIF|NIE|DNI|SSN|BSN)|passport(?: n(?:o|r|umber))?|passeport|reisepass(?:nummer)?|account(?: n(?:o|r|umber))?|acct\.?|"
    r"konto(?:nummer)?|numéro de compte|social security(?: n(?:o|r|umber))?|sozialversicherungsnummer|tax (?:id|number|no\.?)|steuernummer|"
    r"driver'?s? licen[cs]e(?: n(?:o|r|umber))?|national id(?: number)?|identity card(?: n(?:o|r|umber))?|policy (?:number|no\.?)|"
    r"customer (?:number|id|no\.?)|member (?:number|id)|insurance (?:number|no\.?))"
    r"(\s*[:#]?\s*)((?-i:[A-Z0-9][A-Z0-9 ./-]{3,22}[A-Z0-9]))", re.I)
_PROD_CTX = re.compile(r"(?i)(?:NDC|UPC|UDI|GTIN|EAN|SKU|\(01\)|lot|batch|serial|catalog(?:ue)?|model|part|ref|code|barcode|CIG|CUP|CPV|nomenclature|tariff|HTS)\W{0,8}$")
URL = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>\"'\]\[)}{|]+")
URL_PERSONAL = re.compile(
    r"(?i)(?:@|%40)|/(?:users?|profiles?|people|persons?|members?|u|in|pub|author|authors|account|accounts|~[\w.-]+|@[\w.-]+)(?:/|$)|"
    r"(?:Special:Contributions|User(?:_talk)?:|Benutzer(?:in)?(?:_Diskussion)?:|Utilisateur:)|"
    r"[?&;](?:uid|user|userid|user_id|username|email|account|acct|token|key|session|sid|profile|member)=|\d{9,}|/[a-z0-9-]*[a-z]\d[a-z0-9]{10,}")
WIKI_USER = re.compile(r"(?i)((?:\[\[:?)?\b(?:User(?: talk)?|Benutzer(?:in)?(?: Diskussion)?|Utilisateur(?:rice)?|Usuario|Gebruiker)\s*:\s*)"
                       r"(?!\[)([^\s|\]\[)(,;/]+(?: [A-Z][\w.-]+)?)")
_STREET_W = r"[A-Z][\w'.-]*"
ADDR = [
    re.compile(r"\b\d{1,5}(?:[-/]\d{1,4})?[A-Za-z]?,?\s+(?:" + _STREET_W + r"\s+){1,3}(?:Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Drive|Dr|Boulevard|Blvd|"
               r"Court|Ct|Place|Pl|Way|Terrace|Close|Crescent|Gardens|Square|Highway|Hwy|Parkway|Pkwy)\b\.?(?:,?\s+(?:Apt|Unit|Suite|Flat|Ste|#)\.?\s*\w+)?"),
    re.compile(r"\b[A-ZÄÖÜ][\wäöüß.-]*(?:strasse|straße|str\.|gasse|weg|platz|allee|damm|ufer|quai|promenade)\s+\d{1,4}[a-z]?\b"),
    re.compile(r"\b[A-Z][a-zà-ÿ]+(?:straat|laan|plein|gracht|singel|kade)\s+\d{1,4}[a-z]?\b"),
    re.compile(r"(?i)\b\d{1,4}(?:\s?(?:bis|ter))?,?\s+(?:rue|avenue|av\.|boulevard|bd|chemin|route|place|impasse|allée|quai|cours)\s+"
               r"(?:de\s+la\s+|de\s+l'|du\s+|des\s+|de\s+|d')?[A-ZÉÈ][\w'-]+(?:\s+[A-ZÉÈ][\w'-]+){0,2}"),
    re.compile(r"\b(?:[Cc]alle|C/|[Aa]venida|Avda\.?|[Pp]aseo|[Pp]laza|[Rr]ua|[Tt]ravessa)\s+(?:de\s+(?:la\s+|los\s+)?)?[A-ZÁÉÍÓÚ][\w'-]+"
               r"(?:\s+[A-ZÁÉÍÓÚ][\w'-]+){0,2},?\s+(?:n[º°.]?\s*)?\d{1,4}"),
    re.compile(r"\bul\.\s+[A-ZŁŚŻŹĆŃÓ][\w.-]+(?:\s+[A-ZŁŚŻŹĆŃÓ][\w.-]+)?\s+\d{1,4}[a-z]?\b"),            # Polish "ul. Name 12"
]
PLACEHOLDER = re.compile(r"\[(?:EMAIL|PHONE|IBAN|ID|CARD|IP|URL|ADDRESS|NAME)\]")


def _luhn(d):
    s = 0
    for i, c in enumerate(reversed(d)):
        n = int(c)
        if i % 2: n = n * 2 - 9 if n * 2 > 9 else n * 2
        s += n
    return s % 10 == 0


def _phone_ok(m):
    t = m.group(0); d = re.sub(r"\D", "", t)
    if not 9 <= len(d) <= 15: return False
    if _ISBN.match(d) and len(d) in (10, 13): return False
    if _PROD_CTX.search(m.string[max(0, m.start() - 24):m.start()]): return False
    if re.fullmatch(r"(?:(?:19|20)\d\d[\s./-]?){3,}", t): return False
    if re.match(r"(?:19|20)\d\d[-./]\d{1,2}[-./]\d{1,2}", t) or m.string[m.end():m.end() + 1] == ":": return False
    if re.fullmatch(r"(?:19|20)\d\d[-./]\d{1,2}[-./]\d{1,2}(?:[ T]\d{1,2}[:.]\d\d(?:[:.]\d\d)?)?", t): return False
    if re.fullmatch(r"\d{1,2}[-./]\d{1,2}[-./](?:19|20)\d\d", t): return False
    if re.fullmatch(r"(?:19|20)\d\d[\s.-](?:19|20)\d\d", t): return False
    if re.fullmatch(r"\d{1,4}(?:\.\d{1,4}){2,}", t): return False
    return bool(t.startswith("+") or re.search(r"[\s-]", t))


def _url_ok(m):
    u = m.group(0).rstrip(".,;:!?")
    path = re.sub(r"^(?i:https?://|www\.)[^/?#]*", "", u)
    return bool(path and path not in ("/",) and URL_PERSONAL.search(path))


def regex_mask(s, names=False):
    """-> (masked text, {kind: count}). Order matters: URLs and e-mails first so their parts are not re-matched."""
    n = collections.Counter()

    def sub(rx, tag, s, ok=lambda m: True):
        def f(m):
            if not ok(m): return m.group(0)
            n[tag] += 1
            return f"[{tag}]"
        return rx.sub(f, s)

    def url_f(m):
        if not _url_ok(m): return m.group(0)
        n["URL"] += 1
        return "[URL]" + m.group(0)[len(m.group(0).rstrip(".,;:!?")):]
    s = URL.sub(url_f, s)
    s = sub(EMAIL, "EMAIL", s, lambda m: not (ROLE_LOCAL.match(m.group(0).split("@")[0]) or re.search(r"@(?:[\w-]+\.)*example\.(?:com|org|net)$", m.group(0), re.I)))
    s = sub(IBAN, "IBAN", s, lambda m: sum(c.isdigit() for c in m.group(0)) >= 10)
    s = sub(SSN, "ID", s)
    for rx in ID_BARE: s = sub(rx, "ID", s)
    s = ID_LABEL.sub(lambda m: (n.update(ID=1) or m.group(1) + m.group(2) + "[ID]") if sum(c.isdigit() for c in m.group(3)) >= 4 else m.group(0), s)
    s = sub(CARD, "CARD", s, lambda m: 13 <= len(re.sub(r"\D", "", m.group(0))) <= 19 and _luhn(re.sub(r"\D", "", m.group(0)))
            and len(set(re.sub(r"\D", "", m.group(0)))) > 2 and not _ISBN.match(re.sub(r"\D", "", m.group(0)))
            and m.string[max(0, m.start() - 1):m.start()] not in ".," and not _PROD_CTX.search(m.string[max(0, m.start() - 24):m.start()]))
    s = sub(IPV4, "IP", s, lambda m: not _PRIVATE_IP.match(m.group(0)) and max(map(int, m.group(0).split("."))) >= 32)
    s = sub(PHONE, "PHONE", s, _phone_ok)
    for rx in ADDR: s = sub(rx, "ADDRESS", s)
    if names:
        s = WIKI_USER.sub(lambda m: (n.update(NAME=1) or m.group(1) + "[NAME]"), s)
    return s, dict(n)


# ----------------------------------------------------------------------------------------------- NER layer
class NER:
    def __init__(self, model, device, th_person, th_address, th_user=0.7, batch=24, words=110, stride=90, stop=(), mem_fraction=None):
        from gliner import GLiNER
        import torch
        if mem_fraction and device.startswith("cuda"): torch.cuda.set_per_process_memory_fraction(mem_fraction)
        self.m = GLiNER.from_pretrained(model).to(device)
        self.stop = set(stop)
        self.thu = th_user
        self.thp, self.tha, self.batch, self.words, self.stride = th_person, th_address, batch, words, stride

    MAXC = 900
    _TITLE = {'monsieur', 'madame', 'mme', 'm', 'maître', 'me', 'docteur', 'dr', 'mr', 'mrs', 'ms', 'miss', 'herr', 'frau', 'prof', 'professor', 'sir', 'mevrouw', 'de', 'heer', 'meneer', 'mw', 'dhr'}
    _ART = {"the", "a", "an", "der", "die", "das", "den", "dem", "des", "ein", "eine", "einen", "einem", "de", "het", "een", "le", "la", "les", "l", "du", "un", "une", "el", "los", "las", "il", "lo", "gli"}
    _PART = {"de", "van", "von", "der", "den", "ten", "ter", "la", "le", "du", "di", "da", "dos", "das", "del", "des", "bin", "al", "el", "y", "e", "ibn", "het", "the", "of", "and"}

    def _name_ok(self, lab, sc, txt, piece, a, b):
        """Precision filters on person/username spans (the model tags common nouns such as role words)."""
        txt = txt.strip()
        if sum(c.isalpha() for c in txt) < 2: return False
        if "[" in txt or "]" in txt: return False                                    # already pseudonymised upstream ("[claimant A]")
        rest = [t.strip(".,;:") for t in txt.lower().split() if t.strip(".,;:") not in self._TITLE]
        if all(re.fullmatch(r"x+|[a-z]", t) for t in rest): return False            # "Monsieur X", "M. L." : upstream initials
        if self.stop:
            toks = txt.lower().replace("’", "'").split()
            while toks and toks[0].rstrip("'") in self._ART: toks = toks[1:]
            if " ".join(toks).strip(" .,;:") in self.stop: return False
            if toks and toks[0] == "sehr": return False
        if re.search(r"(?i)(?:file|image|datei|bild|fichier):\s*$", piece[max(0, a - 8):a]) or re.match(r"\.(?:jpe?g|png|svg|gif|pdf|tiff?)\b", piece[b:b + 6], re.I): return False
        if lab == "username":
            if sc < self.thu or ":" in txt: return False
        elif sc < self.thp: return False
        latin = [c for c in txt if c.isalpha() and ord(c) < 0x250]
        if len(latin) >= 0.8 * sum(c.isalpha() for c in txt):
            toks = re.findall(r"[^\W\d_][\w'’.-]*", txt)
            if lab == "person" and not any(t[0].isupper() and t.lower() not in self._PART for t in toks): return False
        return True

    def spans(self, texts, names):
        """texts: list[str]; names: bool (person/username labels on). -> list of [(start, end, kind)] per text."""
        labels = ["address"] + (["person", "username"] if names else [])
        pieces, owner = [], []
        for i, t in enumerate(texts):
            ws = [(m.start(), m.end()) for m in re.finditer(r"\S+", t)]
            if not ws: continue
            k = 0
            while k < len(ws):                                                # windows of <= `words` words and <= MAXC characters
                j = k
                while j < len(ws) and j - k < self.words and ws[j][1] - ws[k][0] <= self.MAXC: j += 1
                j = max(j, k + 1)
                a, b = ws[k][0], min(ws[j - 1][1], ws[k][0] + self.MAXC)
                pieces.append(t[a:b]); owner.append((i, a))
                if j >= len(ws): break
                k = j - 15 if j - k > 30 else j
        out = [[] for _ in texts]
        order = sorted(range(len(pieces)), key=lambda j: len(pieces[j]))
        for b0 in range(0, len(order), 2048):
            idx = order[b0:b0 + 2048]
            res = self.m.inference([pieces[j] for j in idx], labels, threshold=min(self.thp, self.tha, self.thu), batch_size=self.batch)
            for j, ents in zip(idx, res):
                i, off = owner[j]
                for e in ents:
                    lab, sc, txt = e["label"], e["score"], e["text"]
                    if lab == "address":
                        if sc >= self.tha and any(c.isdigit() for c in txt) and len(txt) >= 5: out[i].append((off + e["start"], off + e["end"], "ADDRESS"))
                    elif self._name_ok(lab, sc, txt, pieces[j], e["start"], e["end"]):
                        out[i].append((off + e["start"], off + e["end"], "NAME"))
        return out


def apply_spans(t, spans):
    ph = [(m.start(), m.end()) for m in PLACEHOLDER.finditer(t)]
    sp = sorted(s for s in spans if not any(s[0] < pe and ps < s[1] for ps, pe in ph))
    merged = []
    for a, b, k in sp:
        if merged and a <= merged[-1][1]:
            pa, pb, pk = merged[-1]; merged[-1] = (pa, max(pb, b), pk if pk == k else "NAME" if "NAME" in (pk, k) else pk)
        else: merged.append((a, b, k))
    cnt = collections.Counter(); res, pos = [], 0
    for a, b, k in merged:
        while a < b and t[a] in " ,;:.()-\n": a += 1                      # keep surrounding punctuation outside the mask
        while b > a and t[b - 1] in " ,;:.()-\n": b -= 1
        if a >= b: continue
        res.append(t[pos:a]); res.append(f"[{k}]"); pos = b; cnt[k] += 1
    res.append(t[pos:])
    return "".join(res), dict(cnt)


# ----------------------------------------------------------------------------------------------- row handling
def walk_strings(x, fn):
    if isinstance(x, dict): return {k: walk_strings(v, fn) for k, v in x.items()}
    if isinstance(x, list): return [walk_strings(v, fn) for v in x]
    return fn(x) if isinstance(x, str) else x


def row_texts(row):
    out = []
    walk_strings(row.get("state"), lambda s: out.append(s) or s)
    out.append(row.get("question"))
    for o in row.get("options") or []: out.append(o.get("text"))
    return [t for t in out if isinstance(t, str)]


def needs_ner(t):
    return len(t) >= 4 and any(c.isalpha() for c in t)


def merge_counts(a, b):
    for k, v in b.items(): a[k] = a.get(k, 0) + v


def process_chunk(lines, pol, ner, no_ner):
    """lines: list of raw JSON lines (str). -> (output lines, stats, dropped [(id, reason)])"""
    rows = [json.loads(l) for l in lines]
    stats = collections.defaultdict(lambda: collections.Counter())
    keep_idx, sc = [], []                                                          # sc: indices of self-collected rows
    for i, r in enumerate(rows):
        p = pol.get(r["source"])
        if p is None: raise SystemExit(f"source missing from policy: {r['source']}")
        if p["excluded"]: stats[r["source"]]["licence_excluded"] += 1; continue
        keep_idx.append(i)
        if p["origin"] == "self-collected": sc.append(i)
    rx_cache = {}
    for i in sc:
        names = bool(pol[rows[i]["source"]]["names"])
        for t in row_texts(rows[i]):
            if (t, names) not in rx_cache: rx_cache[(t, names)] = regex_mask(t, names)
    # NER on unique regex-masked texts
    todo = {}
    for (t, names), (m, _) in rx_cache.items():
        if needs_ner(m) and not no_ner: todo[(m, names)] = None
    for names in (True, False):
        keys = [k for k in todo if k[1] == names]
        if not keys: continue
        res = ner.spans([k[0] for k in keys], names)
        for k, sp in zip(keys, res): todo[k] = apply_spans(k[0], sp)
    final = {}
    for (t, names), (m, c) in rx_cache.items():
        f, c2 = todo.get((m, names)) or (m, {})
        cc = dict(c); merge_counts(cc, c2); final[(t, names)] = (f, cc)
    out, dropped = [], []
    sc_set = set(sc)
    for i in keep_idx:
        r = rows[i]
        if i not in sc_set: out.append(lines[i]); continue
        names = bool(pol[r["source"]]["names"]); cnt = {}
        def fn(s):
            f, c = final[(s, names)]; merge_counts(cnt, c); return f
        new = dict(r)
        new["state"] = walk_strings(r.get("state"), fn)
        if isinstance(r.get("question"), str): new["question"] = fn(r["question"])
        if r.get("options"): new["options"] = [dict(o, text=fn(o["text"])) if isinstance(o.get("text"), str) else o for o in r["options"]]
        if new["options"] and len({o["text"] for o in new["options"]}) < len(new["options"]) and len({o["text"] for o in r["options"]}) == len(r["options"]):
            dropped.append((r["id"], r["source"], "options_not_distinct_after_masking")); stats[r["source"]]["dropped_options_collapse"] += 1; continue
        st = stats[r["source"]]; st["rows_self_collected"] += 1
        if new != r:
            st["rows_changed"] += 1
            for k, v in cnt.items(): st["mask_" + k] += v
            out.append(json.dumps(new, ensure_ascii=False))
        else: out.append(lines[i])
    for i in keep_idx: stats[rows[i]["source"]]["rows_out"] += 1
    for _, s, _ in dropped: stats[s]["rows_out"] -= 1
    return out, {k: dict(v) for k, v in stats.items()}, dropped


def run_split(name, path, pol, out_dir, chunk, ner, no_ner, part=None):
    ck = os.path.join(out_dir, "ckpt"); os.makedirs(ck, exist_ok=True)
    buf, ci, total = [], 0, collections.defaultdict(collections.Counter); dropped_all = []; t0 = time.time()

    def flush():
        nonlocal buf, ci
        base = os.path.join(ck, f"{name}_{ci:04d}")
        if part and ci % part[1] != part[0]: buf = []; ci += 1; return
        if not os.path.exists(base + ".json"):
            out, st, dr = process_chunk(buf, pol, ner() if not no_ner else None, no_ner)
            with open(base + ".jsonl", "w", encoding="utf-8") as f:
                for l in out: f.write(l.rstrip("\n") + "\n")
            json.dump({"stats": st, "dropped": dr, "rows_in": len(buf)}, open(base + ".json.tmp", "w"))
            os.replace(base + ".json.tmp", base + ".json")
            print(f"[{name}] chunk {ci} in={len(buf)} out={len(out)} t={time.time() - t0:.0f}s", flush=True)
        buf = []; ci += 1
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for l in f:
            buf.append(l.rstrip("\n"))
            if len(buf) >= chunk: flush()
    if buf: flush()
    if part: return 0, {}, []
    rows_in = 0
    with gzip.open(os.path.join(out_dir, f"{name}.jsonl.gz"), "wt", encoding="utf-8") as g:
        for k in range(ci):
            base = os.path.join(ck, f"{name}_{k:04d}")
            meta = json.load(open(base + ".json")); rows_in += meta["rows_in"]
            for s, st in meta["stats"].items():
                for a, b in st.items(): total[s][a] += b
            dropped_all += meta["dropped"]
            with open(base + ".jsonl", encoding="utf-8") as h:
                for l in h:
                    src = json.loads(l)["source"]
                    if pol[src]["excluded"]:   # excluded after this chunk was cached: apply the current policy
                        total[src]["licence_excluded"] += 1; total[src]["rows_out"] -= 1; continue
                    g.write(l)
    return rows_in, {s: dict(v) for s, v in total.items()}, dropped_all


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True); ap.add_argument("--val", required=True); ap.add_argument("--policy", required=True)
    ap.add_argument("--out-dir", required=True); ap.add_argument("--model", default="urchade/gliner_multi_pii-v1")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--chunk", type=int, default=4000)
    ap.add_argument("--th-person", type=float, default=0.5); ap.add_argument("--th-address", type=float, default=0.5)
    ap.add_argument("--no-ner", action="store_true")
    ap.add_argument("--stop-spans", help="text file, one lowercase span per line, never masked as a name (role words such as 'claimant')")
    ap.add_argument("--part", help="k/N: only process chunks with index %% N == k (run N processes, then rerun without --part to assemble)")
    ap.add_argument("--mem-fraction", type=float, default=None, help="cap on GPU memory share per process")
    a = ap.parse_args()
    pol = json.load(open(a.policy)); os.makedirs(a.out_dir, exist_ok=True)
    part = tuple(int(x) for x in a.part.split("/")) if a.part else None
    holder = {}
    def ner():
        if "n" not in holder:
            stop = [l.strip() for l in open(a.stop_spans, encoding="utf-8")] if a.stop_spans else []
            holder["n"] = NER(a.model, a.device, a.th_person, a.th_address, stop=[x for x in stop if x and not x.startswith("#")], mem_fraction=a.mem_fraction)
        return holder["n"]
    t0 = time.time(); rep = {"model": None if a.no_ner else a.model, "thresholds": {"person": a.th_person, "address": a.th_address}}
    dropped = []
    for name, path in (("val", a.val), ("train", a.train)):
        n_in, st, dr = run_split(name, path, pol, a.out_dir, a.chunk, ner, a.no_ner, part)
        agg = collections.Counter()
        for s in st.values():
            for k, v in s.items(): agg[k] += v
        rep[name] = {"rows_in": n_in, "totals": dict(agg), "per_source": st}
        dropped += [(name,) + tuple(d) for d in dr]
    if part: return
    rep["runtime_s"] = round(time.time() - t0)
    json.dump(rep, open(os.path.join(a.out_dir, "pii_report.json"), "w"), indent=1, ensure_ascii=False)
    with open(os.path.join(a.out_dir, "pii_dropped.tsv"), "w") as f:
        for d in dropped: f.write("\t".join(d) + "\n")
    print(json.dumps({k: rep[k]["totals"] for k in ("train", "val")}, indent=1))


if __name__ == "__main__":
    main()
