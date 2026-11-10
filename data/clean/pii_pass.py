#!/usr/bin/env python3
"""Step 2: mask personal data in the cleaned pool and drop the sources not cleared for redistribution.

    python3 -I pii_pass.py --train clean/train.jsonl.gz --val clean/val.jsonl.gz \\
        --policy pii_policy.json --stop-spans pii_stop_spans.txt --out-dir clean_pii

Each row is handled according to its source's entry in the policy file (see README.md, "The policy file"):

  excluded         the row is dropped (licence terms not cleared for redistribution).
  redistributed    a reformatted copy of a published dataset release: the line is copied byte for byte. These rows
                   already carry the regex masks the pool builder applied when it converted the source.
  self-collected   text the pool builder scraped or extracted itself. Every string in `state` (a plain string, or
                   every string value of a JSON state), in `question` and in each option `text` gets two layers:
                   1. regular expressions: [EMAIL] [PHONE] [IBAN] [ID] [CARD] [IP], [URL] for URLs that carry a
                      personal identifier (plain domain URLs are kept), [ADDRESS] for street addresses, and [NAME]
                      for wiki user references (User:xyz, Benutzer:xyz) when the policy masks names;
                   2. the NER model (GLiNER multilingual PII): address spans that contain a digit become [ADDRESS]
                      in every self-collected source; person and username spans become [NAME] only in the sources
                      whose policy says `"names": true`.
                   If masking makes two option texts of a row identical, the row is dropped (listed in
                   pii_dropped.tsv).

Outputs in --out-dir: train.jsonl.gz and val.jsonl.gz (the original row format, all fields), pii_report.json (counts
per source), pii_dropped.tsv, and ckpt/ (per-chunk work).

Checkpoints: the input is processed in chunks of --chunk rows, each saved under OUT/ckpt, so rerunning the same
command resumes. A saved chunk is reused only if nothing it depends on has changed: its input lines, this script,
the model, the thresholds, the stop spans, --no-ner, and the policy entries (origin, names, excluded) of its sources.

Several processes can share the work: run `--part k/N` for k = 0 .. N-1 (each takes the chunks with index % N == k),
then run once more without --part to assemble the outputs from the checkpoints.

Needs gliner and torch (a GPU is strongly recommended). The NER model is not bit-reproducible across GPUs, so a rerun
on another machine can mask a few names differently.
"""
import argparse
import collections
import gzip
import hashlib
import json
import os
import re
import time

# ----------------------------------------------------------------------------------------------- regex layer
EMAIL = re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9_.-]+\.[A-Za-z]{2,}(?![\w-])")
PHONE = re.compile(r"(?<![\w./-])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?)?\d{2,4}(?:[\s.-]\d{2,4}){1,4}(?![\w./-])")
# forms the grouped pattern above cannot see: +41791234567, +49 170 1234567, 030/12345678, 07911 123456, 555.123.4567,
# and a bare national number (0791234567) when a telephone word precedes it
PHONE_INTL = re.compile(r"(?<![\w.+/-])\+\d{1,3}(?:[\s.-]?\(0\))?(?:[\s./-]?\d){7,13}(?![\w/-])(?!\.\d)")
PHONE_NATL = re.compile(r"(?<![\w./-])0\d{1,5}(?:\s?/\s?|\s)\d{5,9}(?![\w/-])(?!\.\d)|(?<![\w./-])\d{3}\.\d{3}\.\d{4}(?![\w/-])(?!\.\d)")
PHONE_BARE = re.compile(r"(?i)((?:\btel(?:efon|ephone|éphone|\.)?|\bt[ée]l[ée]phone|\bphone|\bmobile?|\bnatel|\bhandy|\bfax|\bcall(?:ed|ing)?|\bwhatsapp|\bportable)"
                        r"[\s:.#-]{0,6}(?:(?:nr|no|n°|number|nummer|numéro)\.?[\s:]{0,3})?)(0\d{8,11})(?!\d)")
_NUM_CTX = re.compile(r"(?i)(?:\bversion|\bver\.?|\bv|\brelease|\bbuild|\bfirmware|§|\bart(?:icle|ikel|\.)?|\bsection|\bpara(?:graph)?\.?|\bchapter|\bkapitel|\bziff(?:er|\.)?|\bnr\.?|\bno\.?)\s*$")
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
    r"(\s*[:#]?\s*)((?-i:[A-Z0-9][A-Z0-9 ./-]{3,22}[A-Z0-9]))(?![A-Za-z0-9])", re.I)
_PROD_CTX = re.compile(r"(?i)(?:NDC|UPC|UDI|GTIN|EAN|SKU|\(01\)|lot|batch|serial|catalog(?:ue)?|model|part|ref|code|barcode|CIG|CUP|CPV|nomenclature|tariff|HTS)\W{0,8}$")
URL = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>\"'\]\[)}{|]+")
URL_PERSONAL = re.compile(
    r"(?i)(?:@|%40)|/(?:users?|profiles?|people|persons?|members?|u|in|pub|author|authors|account|accounts|~[\w.-]+|@[\w.-]+)(?:/|$)|"
    r"(?:Special:Contributions|User(?:_talk)?:|Benutzer(?:in)?(?:_Diskussion)?:|Utilisateur:)|"
    r"[?&;](?:uid|user|userid|user_id|username|email|account|acct|token|key|session|sid|profile|member)=|\d{9,}|/[a-z0-9-]*[a-z]\d[a-z0-9]{10,}")
WIKI_USER = re.compile(r"(?i)((?:\[\[:?)?\b(?:User(?: talk)?|Benutzer(?:in)?(?: Diskussion)?|Utilisateur(?:rice)?|Usuario|Gebruiker)\s*:\s*)"
                       r"(?!\[)([^\s|\]\[)(,;/]+(?: [A-Z][\w.-]+)?)")
_COURT = r"(?:Supreme|High|District|Appeals?|Appellate|Crown|County|Circuit|Federal|Constitutional|European|International|Criminal|Civil|Family|Magistrates'?|Juvenile|Tax|Bankruptcy|Administrative|Labou?r|Commercial|Superior|Municipal|Royal)"
_MONTH = r"(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept?|Oct|Nov|Dec|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"
_NOT_STREET_DE = (r"(?!(?:Park|Arbeits|Sitz|Steh|Stell|Studien|Ausbildungs|Kita|Krippen|Heim|Pflege|Liege|Schlaf|Lager|Start|Tabellen|Listen|Rang|Spiel|Sport|Bau|Stand|Wohn|Abstell|Lande|Flug)platz\b)"
                  r"(?!(?:Aus|Rechts|Um|Rück|Hin|Heim|Dienst|Instanzen|Klage|Beschwerde|Verwaltungs|Zivil|Straf|Schul|Arbeits|Lebens|Bildungs|Lösungs|Mittel|Flucht|Transport|Verkehrs|Wasser|Luft|See|Land|Post|Amts|Gerichts|Versand)weg\b)")
_STREET_W = r"[A-Z][\w'.-]*"
ADDR = [
    # "12 Baker Street": not a court ("2019 Supreme Court") and not a date ("12 June Dr. Meyer")
    re.compile(r"\b\d{1,5}(?:[-/]\d{1,4})?[A-Za-z]?,?\s+(?:(?!" + _MONTH + r"\b)(?!" + _COURT + r"\s+Court\b)" + _STREET_W + r"\s+){1,3}"
               r"(?:Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Drive|Dr|Boulevard|Blvd|Court|Ct|Place|Pl|Way|Terrace|Close|Crescent|Gardens|Square|Highway|Hwy|"
               r"Parkway|Pkwy)\b\.?(?:,?\s+(?:Apt|Unit|Suite|Flat|Ste|#)\.?\s*\w+)?"),
    re.compile(r"\b" + _NOT_STREET_DE + r"[A-ZÄÖÜ][\wäöüß.-]*(?:strasse|straße|str\.|gasse|weg|platz|allee|damm|ufer|quai|promenade)\s+\d{1,4}[a-z]?\b"),
    re.compile(r"\b[A-Z][a-zà-ÿ]+(?:straat|laan|plein|gracht|singel|kade)\s+\d{1,4}[a-z]?\b"),
    # "12 rue de la Paix", "5, rue primeurs"; the words that are also English (place, route, cours) need a capitalised
    # name after them, so "1 place in the race" is not an address
    re.compile(r"\b\d{1,4}(?:\s?(?:bis|ter))?,?\s+(?:(?i:rue|avenue|av\.|boulevard|bd|chemin|impasse|allée|quai)\s+"
               r"(?:(?i:de\s+la|des|du|de)\s+|(?i:de\s+l'|d'))?(?i:[A-ZÉÈ])|(?i:route|place|cours)\s+(?:(?i:de\s+la|des|du|de)\s+|(?i:de\s+l'|d'))?[A-ZÉÈ])"
               r"[\w'-]+(?:\s+(?:(?:de\s+la|des|du|de)\s+|(?:de\s+l'|d'))?[A-ZÉÈ][\w'-]+){0,2}"),
    re.compile(r"\b(?:[Cc]alle|C/|[Aa]venida|Avda\.?|[Pp]aseo|[Pp]laza|[Rr]ua|[Tt]ravessa)\s+(?:de\s+(?:la\s+|los\s+)?)?[A-ZÁÉÍÓÚ][\w'-]+"
               r"(?:\s+[A-ZÁÉÍÓÚ][\w'-]+){0,2},?\s+(?:n[º°.]?\s*)?\d{1,4}"),
    re.compile(r"\bul\.\s+[A-ZŁŚŻŹĆŃÓ][\w.-]+(?:\s+[A-ZŁŚŻŹĆŃÓ][\w.-]+)?\s+\d{1,4}[a-z]?\b"),            # Polish "ul. Name 12"
]
PLACEHOLDER = re.compile(r"\[(?:EMAIL|PHONE|IBAN|ID|CARD|IP|URL|ADDRESS|NAME)\]")


def _luhn(digits):
    """True if the digit string passes the Luhn checksum of payment card numbers."""
    total = 0
    for i, c in enumerate(reversed(digits)):
        n = int(c)
        if i % 2:
            n = n * 2 - 9 if n * 2 > 9 else n * 2
        total += n
    return total % 10 == 0


def _phone_ok(m):
    """Filter for PHONE matches: rejects dates, years, ISBNs, product codes, version numbers and amounts."""
    t = m.group(0)
    d = re.sub(r"\D", "", t)
    if not 9 <= len(d) <= 15:
        return False
    if _ISBN.match(d) and len(d) in (10, 13):
        return False
    if _PROD_CTX.search(m.string[max(0, m.start() - 24):m.start()]):
        return False
    if re.fullmatch(r"(?:(?:19|20)\d\d[\s./-]?){3,}", t):
        return False
    if re.match(r"(?:19|20)\d\d[-./]\d{1,2}[-./]\d{1,2}", t) or m.string[m.end():m.end() + 1] == ":":
        return False
    if re.fullmatch(r"(?:19|20)\d\d[-./]\d{1,2}[-./]\d{1,2}(?:[ T]\d{1,2}[:.]\d\d(?:[:.]\d\d)?)?", t):
        return False
    if re.fullmatch(r"\d{1,2}[-./]\d{1,2}[-./](?:19|20)\d\d", t):
        return False
    if re.fullmatch(r"(?:19|20)\d\d[\s.-](?:19|20)\d\d", t):
        return False
    if re.fullmatch(r"\d{1,4}(?:\.\d{1,4}){2,}", t):
        return False
    # a number written in groups of three ("1 250 000 000 francs", "250 000 000") is an amount, not a telephone number
    if not t.startswith(("+", "0", "(")):
        if re.fullmatch(r"\d{1,3}(?:[\s.]\d{3})+", t):
            return False
        if "-" not in t and (re.search(r"\d[\s.]$", m.string[max(0, m.start() - 2):m.start()])
                             or re.match(r"[\s.]\d", m.string[m.end():m.end() + 2])):
            return False
    if re.fullmatch(r"\d{4}-\d{2}-\d{3}-\d{4}", t):            # NATO stock number
        return False
    return bool(t.startswith("+") or re.search(r"[\s-]", t))


def _phone2_ok(m):
    """Filter for PHONE_INTL and PHONE_NATL matches."""
    d = re.sub(r"\D", "", m.group(0))
    if not 9 <= len(d) <= 15 or _PROD_CTX.search(m.string[max(0, m.start() - 24):m.start()]):
        return False
    # a group inside a longer run of digit groups (a product code such as "UPC 7 08152 32988 7") is not a telephone number
    return not (re.search(r"\d\s$", m.string[max(0, m.start() - 2):m.start()])
                or re.match(r"\s\d", m.string[m.end():m.end() + 2]))


def _id_ok(v):
    """A labelled identifier: at least four digits and six characters, and not a year followed by a word ("account 2015 A...")."""
    return (sum(c.isdigit() for c in v) >= 4 and sum(c.isalnum() for c in v) >= 6
            and not re.fullmatch(r"(?:19|20)\d\d[ ./-]+[A-Z]+", v))


def _url_ok(m):
    """A URL is masked only when its path carries a personal identifier."""
    u = m.group(0).rstrip(".,;:!?")
    path = re.sub(r"^(?i:https?://|www\.)[^/?#]*", "", u)
    return bool(path and path not in ("/",) and URL_PERSONAL.search(path))


def _email_ok(m):
    """Role mailboxes (info@, contact@, ...) and the reserved example.* domains are kept."""
    addr = m.group(0)
    return not (ROLE_LOCAL.match(addr.split("@")[0]) or re.search(r"@(?:[\w-]+\.)*example\.(?:com|org|net)$", addr, re.I))


def _card_ok(m):
    d = re.sub(r"\D", "", m.group(0))
    return (13 <= len(d) <= 19 and _luhn(d) and len(set(d)) > 2 and not _ISBN.match(d)
            and m.string[max(0, m.start() - 1):m.start()] not in ".,"
            and not _PROD_CTX.search(m.string[max(0, m.start() - 24):m.start()]))


def _ip_ok(m):
    ip = m.group(0)
    return (not _PRIVATE_IP.match(ip) and max(map(int, ip.split("."))) >= 32
            and not _NUM_CTX.search(m.string[max(0, m.start() - 12):m.start()]))


def regex_mask(s, names=False):
    """-> (masked text, {kind: count}). The order matters: URLs and e-mail addresses go first so that their parts are
    not matched again by the number patterns."""
    n = collections.Counter()

    def sub(rx, tag, s, ok=lambda m: True):
        def repl(m):
            if not ok(m):
                return m.group(0)
            n[tag] += 1
            return f"[{tag}]"
        return rx.sub(repl, s)

    def url_repl(m):
        if not _url_ok(m):
            return m.group(0)
        n["URL"] += 1
        return "[URL]" + m.group(0)[len(m.group(0).rstrip(".,;:!?")):]      # trailing punctuation stays outside

    def id_label_repl(m):
        if not _id_ok(m.group(3)):
            return m.group(0)
        n["ID"] += 1
        return m.group(1) + m.group(2) + "[ID]"                              # keep the label ("passport no: [ID]")

    def phone_bare_repl(m):
        n["PHONE"] += 1
        return m.group(1) + "[PHONE]"                                        # keep the telephone word

    def wiki_user_repl(m):
        n["NAME"] += 1
        return m.group(1) + "[NAME]"

    s = URL.sub(url_repl, s)
    s = sub(EMAIL, "EMAIL", s, _email_ok)
    s = sub(IBAN, "IBAN", s, lambda m: sum(c.isdigit() for c in m.group(0)) >= 10)
    s = sub(SSN, "ID", s)
    for rx in ID_BARE:
        s = sub(rx, "ID", s)
    s = ID_LABEL.sub(id_label_repl, s)
    s = sub(CARD, "CARD", s, _card_ok)
    s = sub(IPV4, "IP", s, _ip_ok)
    s = sub(PHONE_INTL, "PHONE", s, _phone2_ok)
    s = sub(PHONE, "PHONE", s, _phone_ok)
    s = sub(PHONE_NATL, "PHONE", s, _phone2_ok)
    s = PHONE_BARE.sub(phone_bare_repl, s)
    for rx in ADDR:
        s = sub(rx, "ADDRESS", s)
    if names:
        s = WIKI_USER.sub(wiki_user_repl, s)
    return s, dict(n)


# ----------------------------------------------------------------------------------------------- NER layer
class NER:
    """GLiNER span finder with precision filters for person and username spans."""

    MAXC = 900                      # longest window passed to the model, in characters
    _TITLE = {"monsieur", "madame", "mme", "m", "maître", "me", "docteur", "dr", "mr", "mrs", "ms", "miss", "herr", "frau",
              "prof", "professor", "sir", "mevrouw", "de", "heer", "meneer", "mw", "dhr"}
    _ART = {"the", "a", "an", "der", "die", "das", "den", "dem", "des", "ein", "eine", "einen", "einem", "de", "het", "een",
            "le", "la", "les", "l", "du", "un", "une", "el", "los", "las", "il", "lo", "gli"}
    _PART = {"de", "van", "von", "der", "den", "ten", "ter", "la", "le", "du", "di", "da", "dos", "das", "del", "des", "bin",
             "al", "el", "y", "e", "ibn", "het", "the", "of", "and"}

    def __init__(self, model, device, th_person, th_address, th_user=0.7, batch=24, words=110, stop=(), mem_fraction=None):
        from gliner import GLiNER
        import torch
        if mem_fraction and device.startswith("cuda"):
            torch.cuda.set_per_process_memory_fraction(mem_fraction)
        self.m = GLiNER.from_pretrained(model).to(device)
        self.stop = set(stop)
        self.thp, self.tha, self.thu = th_person, th_address, th_user
        self.batch, self.words = batch, words

    def _name_ok(self, lab, sc, txt, piece, a, b):
        """Precision filters on person and username spans (the model also tags common nouns such as role words)."""
        txt = txt.strip()
        if sum(c.isalpha() for c in txt) < 2:
            return False
        if "[" in txt or "]" in txt:                                        # already pseudonymised upstream ("[claimant A]")
            return False
        rest = [t.strip(".,;:") for t in txt.lower().split() if t.strip(".,;:") not in self._TITLE]
        if all(re.fullmatch(r"x+|[a-z]", t) for t in rest):                 # "Monsieur X", "M. L.": upstream initials
            return False
        if self.stop:
            toks = txt.lower().replace("’", "'").split()
            while toks and toks[0].rstrip("'") in self._ART:
                toks = toks[1:]
            if " ".join(toks).strip(" .,;:") in self.stop:
                return False
            if toks and toks[0] == "sehr":                                  # "Sehr geehrte ..." letter openings
                return False
        if (re.search(r"(?i)(?:file|image|datei|bild|fichier):\s*$", piece[max(0, a - 8):a])
                or re.match(r"\.(?:jpe?g|png|svg|gif|pdf|tiff?)\b", piece[b:b + 6], re.I)):
            return False                                                    # a file name
        if lab == "username":
            if sc < self.thu or ":" in txt:
                return False
        elif sc < self.thp:
            return False
        latin = [c for c in txt if c.isalpha() and ord(c) < 0x250]
        if len(latin) >= 0.8 * sum(c.isalpha() for c in txt):
            # in Latin script a person's name has at least one capitalised word that is not a name particle
            toks = re.findall(r"[^\W\d_][\w'’.-]*", txt)
            if lab == "person" and not any(t[0].isupper() and t.lower() not in self._PART for t in toks):
                return False
        return True

    def _windows(self, texts):
        """Split every text into overlapping windows of at most `words` words and MAXC characters.
        -> (pieces, owner) with owner[j] = (text index, character offset of piece j)."""
        pieces, owner = [], []
        for i, t in enumerate(texts):
            ws = [(m.start(), m.end()) for m in re.finditer(r"\S+", t)]
            if not ws:
                continue
            k = 0
            while k < len(ws):
                j = k
                while j < len(ws) and j - k < self.words and ws[j][1] - ws[k][0] <= self.MAXC:
                    j += 1
                j = max(j, k + 1)
                a, b = ws[k][0], min(ws[j - 1][1], ws[k][0] + self.MAXC)
                pieces.append(t[a:b])
                owner.append((i, a))
                if j >= len(ws):
                    break
                k = j - 15 if j - k > 30 else j                             # 15 words of overlap between long windows
        return pieces, owner

    def spans(self, texts, names):
        """texts: list[str]; names: bool (person and username labels on). -> one list of (start, end, kind) per text."""
        labels = ["address"] + (["person", "username"] if names else [])
        pieces, owner = self._windows(texts)
        out = [[] for _ in texts]
        order = sorted(range(len(pieces)), key=lambda j: len(pieces[j]))   # similar lengths batch together
        for b0 in range(0, len(order), 2048):
            idx = order[b0:b0 + 2048]
            res = self.m.inference([pieces[j] for j in idx], labels, threshold=min(self.thp, self.tha, self.thu),
                                   batch_size=self.batch)
            for j, ents in zip(idx, res):
                i, off = owner[j]
                for e in ents:
                    lab, sc, txt = e["label"], e["score"], e["text"]
                    if lab == "address":
                        if sc >= self.tha and any(c.isdigit() for c in txt) and len(txt) >= 5:
                            out[i].append((off + e["start"], off + e["end"], "ADDRESS"))
                    elif self._name_ok(lab, sc, txt, pieces[j], e["start"], e["end"]):
                        out[i].append((off + e["start"], off + e["end"], "NAME"))
        return out


def apply_spans(t, spans):
    """Replace NER spans with placeholders. Spans that touch an existing placeholder are skipped, overlapping spans
    are merged (NAME wins over ADDRESS), and surrounding punctuation stays outside the mask. -> (text, {kind: count})"""
    ph = [(m.start(), m.end()) for m in PLACEHOLDER.finditer(t)]
    sp = sorted(s for s in spans if not any(s[0] < pe and ps < s[1] for ps, pe in ph))
    merged = []
    for a, b, k in sp:
        if merged and a <= merged[-1][1]:
            pa, pb, pk = merged[-1]
            merged[-1] = (pa, max(pb, b), pk if pk == k else "NAME" if "NAME" in (pk, k) else pk)
        else:
            merged.append((a, b, k))
    cnt = collections.Counter()
    res, pos = [], 0
    for a, b, k in merged:
        while a < b and t[a] in " ,;:.()-\n":
            a += 1
        while b > a and t[b - 1] in " ,;:.()-\n":
            b -= 1
        if a >= b:
            continue
        res.append(t[pos:a])
        res.append(f"[{k}]")
        pos = b
        cnt[k] += 1
    res.append(t[pos:])
    return "".join(res), dict(cnt)


# ----------------------------------------------------------------------------------------------- row handling
def walk_strings(x, fn):
    """Apply fn to every string inside x (nested dicts and lists), keeping the structure."""
    if isinstance(x, dict):
        return {k: walk_strings(v, fn) for k, v in x.items()}
    if isinstance(x, list):
        return [walk_strings(v, fn) for v in x]
    return fn(x) if isinstance(x, str) else x


def row_texts(row):
    """Every string the pass masks: the strings of `state`, the `question`, and each option's `text`."""
    out = []
    walk_strings(row.get("state"), lambda s: out.append(s) or s)
    out.append(row.get("question"))
    for o in row.get("options") or []:
        out.append(o.get("text"))
    return [t for t in out if isinstance(t, str)]


def needs_ner(t):
    return len(t) >= 4 and any(c.isalpha() for c in t)


def merge_counts(a, b):
    for k, v in b.items():
        a[k] = a.get(k, 0) + v


def process_chunk(lines, pol, ner, no_ner):
    """lines: raw JSON lines. -> (output lines, {source: stats}, dropped [(id, source, reason)])"""
    rows = [json.loads(line) for line in lines]
    stats = collections.defaultdict(collections.Counter)
    keep_idx, sc = [], []                                       # sc: indices of the self-collected rows
    for i, r in enumerate(rows):
        p = pol.get(r["source"])
        if p is None:
            raise SystemExit(f"source missing from policy: {r['source']}")
        if p["excluded"]:
            stats[r["source"]]["licence_excluded"] += 1
            continue
        keep_idx.append(i)
        if p["origin"] == "self-collected":
            sc.append(i)

    # regex layer, once per distinct (text, names) pair
    rx_cache = {}
    for i in sc:
        names = bool(pol[rows[i]["source"]]["names"])
        for t in row_texts(rows[i]):
            if (t, names) not in rx_cache:
                rx_cache[(t, names)] = regex_mask(t, names)

    # NER layer, on the distinct regex-masked texts, one model call per label set
    todo = {}
    if not no_ner:
        for (_, names), (m, _) in rx_cache.items():
            if needs_ner(m):
                todo[(m, names)] = None
    for names in (True, False):
        keys = [k for k in todo if k[1] == names]
        if not keys:
            continue
        res = ner.spans([k[0] for k in keys], names)
        for k, sp in zip(keys, res):
            todo[k] = apply_spans(k[0], sp)

    final = {}
    for (t, names), (m, c) in rx_cache.items():
        f, c2 = todo.get((m, names)) or (m, {})
        cc = dict(c)
        merge_counts(cc, c2)
        final[(t, names)] = (f, cc)

    out, dropped = [], []
    sc_set = set(sc)
    for i in keep_idx:
        r = rows[i]
        if i not in sc_set:
            out.append(lines[i])                                # redistributed: byte for byte
            continue
        names = bool(pol[r["source"]]["names"])
        cnt = {}

        def mask(s):
            f, c = final[(s, names)]
            merge_counts(cnt, c)
            return f

        new = dict(r)
        new["state"] = walk_strings(r.get("state"), mask)
        if isinstance(r.get("question"), str):
            new["question"] = mask(r["question"])
        if r.get("options"):
            new["options"] = [dict(o, text=mask(o["text"])) if isinstance(o.get("text"), str) else o for o in r["options"]]
        opts = new.get("options")
        if opts and len({o["text"] for o in opts}) < len(opts) and len({o["text"] for o in r["options"]}) == len(r["options"]):
            dropped.append((r["id"], r["source"], "options_not_distinct_after_masking"))
            stats[r["source"]]["dropped_options_collapse"] += 1
            continue
        st = stats[r["source"]]
        st["rows_self_collected"] += 1
        if new != r:
            st["rows_changed"] += 1
            for k, v in cnt.items():
                st["mask_" + k] += v
            out.append(json.dumps(new, ensure_ascii=False))
        else:
            out.append(lines[i])
    for i in keep_idx:
        stats[rows[i]["source"]]["rows_out"] += 1
    for _, s, _ in dropped:
        stats[s]["rows_out"] -= 1
    return out, {k: dict(v) for k, v in stats.items()}, dropped


# ----------------------------------------------------------------------------------------------- chunks and checkpoints
def chunk_key(lines, pol, fingerprint):
    """What a saved chunk depends on: its input lines, the run's fingerprint and the policy of the sources in it."""
    h = hashlib.blake2b(fingerprint.encode(), digest_size=16)
    srcs = sorted({json.loads(line)["source"] for line in lines})
    h.update(json.dumps([[s, (pol.get(s) or {}).get("origin"), bool((pol.get(s) or {}).get("names")),
                          bool((pol.get(s) or {}).get("excluded"))] for s in srcs]).encode())
    for line in lines:
        h.update(line.encode("utf-8", "surrogatepass"))
        h.update(b"\n")
    return h.hexdigest()


def _load_key(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f).get("key")
    except ValueError:                                          # a half-written checkpoint from an interrupted run
        return None


def run_split(name, path, pol, out_dir, chunk, ner, no_ner, part=None, fingerprint=""):
    """Process one split chunk by chunk (reusing valid checkpoints), then assemble OUT/<name>.jsonl.gz.
    -> (rows in, {source: stats}, dropped). With `part`, only this process's chunks are computed and nothing is assembled."""
    ck = os.path.join(out_dir, "ckpt")
    os.makedirs(ck, exist_ok=True)
    t0 = time.time()

    def do_chunk(ci, buf):
        base = os.path.join(ck, f"{name}_{ci:04d}")
        if part and ci % part[1] != part[0]:
            return
        key = chunk_key(buf, pol, fingerprint)
        if _load_key(base + ".json") == key:
            return
        out, st, dr = process_chunk(buf, pol, None if no_ner else ner(), no_ner)
        with open(base + ".jsonl", "w", encoding="utf-8") as f:
            for line in out:
                f.write(line.rstrip("\n") + "\n")
        with open(base + ".json.tmp", "w") as f:
            json.dump({"key": key, "stats": st, "dropped": dr, "rows_in": len(buf)}, f)
        os.replace(base + ".json.tmp", base + ".json")              # the key is written last: a crash leaves no valid key
        print(f"[{name}] chunk {ci} in={len(buf)} out={len(out)} t={time.time() - t0:.0f}s", flush=True)

    n_chunks, buf = 0, []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            buf.append(line.rstrip("\n"))
            if len(buf) >= chunk:
                do_chunk(n_chunks, buf)
                n_chunks, buf = n_chunks + 1, []
    if buf:
        do_chunk(n_chunks, buf)
        n_chunks += 1
    if part:
        return 0, {}, []

    rows_in, total, dropped_all = 0, collections.defaultdict(collections.Counter), []
    with gzip.open(os.path.join(out_dir, f"{name}.jsonl.gz"), "wt", encoding="utf-8") as g:
        for k in range(n_chunks):
            base = os.path.join(ck, f"{name}_{k:04d}")
            with open(base + ".json") as f:
                meta = json.load(f)
            rows_in += meta["rows_in"]
            for s, st in meta["stats"].items():
                for a, b in st.items():
                    total[s][a] += b
            dropped_all += meta["dropped"]
            with open(base + ".jsonl", encoding="utf-8") as h:
                for line in h:
                    g.write(line)
    return rows_in, {s: dict(v) for s, v in total.items()}, dropped_all


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True, help="input train.jsonl.gz (the output of build_clean.py)")
    ap.add_argument("--val", required=True, help="input val.jsonl.gz")
    ap.add_argument("--policy", required=True, help="pii_policy.json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--stop-spans", help="text file, one lowercase span per line, never masked as a name (role words such as 'claimant')")
    ap.add_argument("--model", default="urchade/gliner_multi_pii-v1", help="GLiNER model id or path")
    ap.add_argument("--device", default="cuda", help="cuda or cpu")
    ap.add_argument("--chunk", type=int, default=4000, help="input rows per checkpoint chunk")
    ap.add_argument("--th-person", type=float, default=0.5, help="NER score threshold for person spans")
    ap.add_argument("--th-address", type=float, default=0.5, help="NER score threshold for address spans")
    ap.add_argument("--no-ner", action="store_true", help="regex layer only (tests)")
    ap.add_argument("--part", help="k/N: only process chunks with index %% N == k (run N processes, then rerun without --part to assemble)")
    ap.add_argument("--mem-fraction", type=float, default=None, help="cap on the GPU memory share per process")
    ap.add_argument("--batch", type=int, default=24, help="NER batch size (lower it on a shared or small GPU)")
    a = ap.parse_args()

    with open(a.policy) as f:
        pol = json.load(f)
    os.makedirs(a.out_dir, exist_ok=True)
    part = tuple(int(x) for x in a.part.split("/")) if a.part else None
    stop_txt = ""
    if a.stop_spans:
        with open(a.stop_spans, encoding="utf-8") as f:
            stop_txt = f.read()
    stop = [x.strip() for x in stop_txt.splitlines()]
    stop = [x for x in stop if x and not x.startswith("#")]

    holder = {}

    def ner():                                                  # loaded on first use: a fully cached rerun needs no GPU
        if "n" not in holder:
            holder["n"] = NER(a.model, a.device, a.th_person, a.th_address, stop=stop, mem_fraction=a.mem_fraction, batch=a.batch)
        return holder["n"]

    with open(__file__, encoding="utf-8") as f:
        script = f.read()
    fingerprint = hashlib.blake2b(json.dumps([script, a.model, a.th_person, a.th_address, stop_txt, a.no_ner]).encode(),
                                  digest_size=16).hexdigest()
    t0 = time.time()
    rep = {"model": None if a.no_ner else a.model, "thresholds": {"person": a.th_person, "address": a.th_address}}
    dropped = []
    for name, path in (("val", a.val), ("train", a.train)):
        n_in, st, dr = run_split(name, path, pol, a.out_dir, a.chunk, ner, a.no_ner, part, fingerprint)
        agg = collections.Counter()
        for s in st.values():
            agg.update(s)
        rep[name] = {"rows_in": n_in, "totals": dict(agg), "per_source": st}
        dropped += [(name,) + tuple(d) for d in dr]
    if part:
        return
    rep["runtime_s"] = round(time.time() - t0)
    with open(os.path.join(a.out_dir, "pii_report.json"), "w") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False)
    with open(os.path.join(a.out_dir, "pii_dropped.tsv"), "w") as f:
        for d in dropped:
            f.write("\t".join(d) + "\n")
    print(json.dumps({k: rep[k]["totals"] for k in ("train", "val")}, indent=1))


if __name__ == "__main__":
    main()
