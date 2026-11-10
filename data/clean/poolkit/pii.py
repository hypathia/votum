"""Stage `mask`: mask personal data and drop the sources not cleared for redistribution.

Each row is handled according to its source's entry in the config:

  "drop": "mask"   the row is dropped.
  redistributed    a reformatted copy of a published dataset release: the line is copied byte for byte. These rows
                   already carry the regex masks the pool builder applied when it converted the source.
  self-collected   text the pool builder scraped or extracted itself. Every string in `state` (a plain string, or
                   every string value of a JSON state), in `question` and in each option `text` gets two layers:
                   1. regular expressions: [EMAIL] [PHONE] [IBAN] [ID] [CARD] [IP], [URL] for URLs that carry a
                      personal identifier (plain domain URLs are kept), [ADDRESS] for street addresses, and [NAME]
                      for wiki user references (User:xyz, Benutzer:xyz) when the config masks names;
                   2. the NER model (GLiNER multilingual PII): address spans that contain a digit become [ADDRESS]
                      in every self-collected source; person and username spans become [NAME] only in the sources
                      whose entry says `"names": true`.
                   If masking makes two option texts of a row identical, the row is dropped (listed in
                   pii_dropped.tsv).

Last, every e-mail address still left in any row, whatever its source, becomes [EMAIL]; the office mailboxes and the
placeholder domains listed in the config are kept. This also catches the addresses that the source datasets' own
masking missed.

The model's work is cached per text. The stage runs in three steps: a parallel pass over the rows that applies the
regular expressions and collects the texts the model has not seen yet; the model on those texts, its raw output going
to the span cache; and a parallel pass that writes the rows. The thresholds, the stop spans and the name filters are
applied when the rows are written, so changing them, the config of a source, or the rows themselves never sends a
cached text through the model again. A rerun with a complete cache needs no GPU.

The cache holds character offsets of personal data: keep it local. The model is not bit-reproducible across GPUs, so a
run elsewhere can mask a few names differently.
"""
import collections
import glob
import hashlib
import importlib.util
import json
import os
import re
import shutil
import time
import warnings

from . import engine

DEFAULTS = {"model": "urchade/gliner_multi_pii-v1", "thresholds": {"person": 0.5, "address": 0.5, "username": 0.7},
            "stop_spans": None, "keep_emails": [], "keep_email_hosts": ["example.com", "example.org", "example.net"]}
G = {}                                              # state shared with the workers (set before a pool is made)

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


# ----------------------------------------------------------------------------------------------- NER layer: windows and filters
WORDS, MAXC, OVERLAP = 110, 900, 15                 # a window: at most WORDS words and MAXC characters; words shared by long windows
LABELS = {True: ["address", "person", "username"], False: ["address"]}      # by "the source masks names"
CODE = {"address": "A", "person": "P", "username": "U"}
_TITLE = {"monsieur", "madame", "mme", "m", "maître", "me", "docteur", "dr", "mr", "mrs", "ms", "miss", "herr", "frau",
          "prof", "professor", "sir", "mevrouw", "de", "heer", "meneer", "mw", "dhr"}
_ART = {"the", "a", "an", "der", "die", "das", "den", "dem", "des", "ein", "eine", "einen", "einem", "de", "het", "een",
        "le", "la", "les", "l", "du", "un", "une", "el", "los", "las", "il", "lo", "gli"}
_PART = {"de", "van", "von", "der", "den", "ten", "ter", "la", "le", "du", "di", "da", "dos", "das", "del", "des", "bin",
         "al", "el", "y", "e", "ibn", "het", "the", "of", "and"}


def windows(t):
    """Split a text into overlapping windows for the model. -> [(start, end)] in characters"""
    ws = [(m.start(), m.end()) for m in re.finditer(r"\S+", t)]
    out, k = [], 0
    while k < len(ws):
        j = k
        while j < len(ws) and j - k < WORDS and ws[j][1] - ws[k][0] <= MAXC:
            j += 1
        j = max(j, k + 1)
        out.append((ws[k][0], min(ws[j - 1][1], ws[k][0] + MAXC)))
        if j >= len(ws):
            break
        k = j - OVERLAP if j - k > 2 * OVERLAP else j
    return out


def needs_ner(t):
    return len(t) >= 4 and any(c.isalpha() for c in t)


class Filters:
    """Turns the model's raw output for a text into the spans to mask: score thresholds, and precision filters on
    person and username spans (the model also tags common nouns such as role words)."""

    def __init__(self, th_person=0.5, th_address=0.5, th_user=0.7, stop=()):
        self.thp, self.tha, self.thu, self.stop = th_person, th_address, th_user, set(stop)

    @property
    def raw_threshold(self):
        return min(self.thp, self.tha, self.thu)

    def name_ok(self, lab, sc, txt, piece, a, b):
        """True if a person or username span of `piece` (at a:b, text txt, score sc) should be masked."""
        txt = txt.strip()
        if sum(c.isalpha() for c in txt) < 2:
            return False
        if "[" in txt or "]" in txt:                                        # already pseudonymised upstream ("[claimant A]")
            return False
        rest = [t.strip(".,;:") for t in txt.lower().split() if t.strip(".,;:") not in _TITLE]
        if all(re.fullmatch(r"x+|[a-z]", t) for t in rest):                 # "Monsieur X", "M. L.": upstream initials
            return False
        if self.stop:
            toks = txt.lower().replace("’", "'").split()
            while toks and toks[0].rstrip("'") in _ART:
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
            if lab == "person" and not any(t[0].isupper() and t.lower() not in _PART for t in toks):
                return False
        return True

    def spans(self, t, raw):
        """t: the text the model saw; raw: its entities [(window index, start, end, label code, score)].
        -> [(start, end, "ADDRESS" or "NAME")]"""
        if not raw:
            return []
        win, out = windows(t), []
        for j, s, e, code, sc in raw:
            txt = t[s:e]
            if code == "A":
                if sc >= self.tha and any(c.isdigit() for c in txt) and len(txt) >= 5:
                    out.append((s, e, "ADDRESS"))
            else:
                a, b = win[j]
                if self.name_ok("username" if code == "U" else "person", sc, txt, t[a:b], s - a, e - a):
                    out.append((s, e, "NAME"))
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


def read_stop_spans(path):
    """A text file with one lowercase span per line that is never masked as a name (role words such as 'claimant')."""
    if not path:
        return []
    with open(path, encoding="utf-8") as f:
        lines = [x.strip() for x in f.read().splitlines()]
    return [x for x in lines if x and not x.startswith("#")]


# ----------------------------------------------------------------------------------------------- the span cache
def text_key(t, names):
    return hashlib.blake2b((b"1" if names else b"0") + t.encode("utf-8", "surrogatepass"), digest_size=16).hexdigest()


class SpanCache:
    """The model's raw output per text, kept in <dir>/spans-<variant>*.jsonl (one JSON line per text, appended as the
    model goes, so an interrupted run loses nothing). The variant names what the output depends on: the model, its
    numerics, the labels, the windows and the lowest threshold. Results of different variants never mix."""

    def __init__(self, directory, variant, part=None):
        self.dir, self.variant = directory, variant
        self.id = hashlib.blake2b(json.dumps(variant, sort_keys=True).encode(), digest_size=6).hexdigest()
        self.d, self.f = {}, None
        self.part = f".part-{part[0]}-of-{part[1]}" if part else ""
        for p in sorted(glob.glob(os.path.join(directory, f"spans-{self.id}*.jsonl"))):
            with open(p, encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except ValueError:              # the last line of an interrupted run
                        continue
                    if "k" in rec:
                        self.d[rec["k"]] = rec["e"]

    def add(self, key, raw):
        if self.f is None:
            os.makedirs(self.dir, exist_ok=True)
            p = os.path.join(self.dir, f"spans-{self.id}{self.part}.jsonl")
            new = not os.path.exists(p) or os.path.getsize(p) == 0
            self.f = open(p, "a", encoding="utf-8")
            if new:
                self.f.write(json.dumps({"variant": self.variant}) + "\n")
        self.d[key] = raw
        self.f.write(json.dumps({"k": key, "e": raw}) + "\n")

    def flush(self):
        if self.f is not None:
            self.f.flush()

    def close(self):
        if self.f is not None:
            self.f.close()
            self.f = None


# ----------------------------------------------------------------------------------------------- the model
def _collate(task):
    """Worker: tokenise one batch of windows on the CPU. -> (tensors for the model, word start and end offsets)"""
    pieces, names = task
    m = G["model"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                 # "sentence truncated": see WORDS
        prep = m.prepare_batch(pieces, LABELS[names])
        batch = m.collate_batch(prep["input_x"], prep["entity_types"], G["collator"])
    return batch, prep["start_token_map"], prep["end_token_map"]


class Model:
    """GLiNER on windows of text. The CPU half of a batch (splitting into words, tokenising, building the span grid) runs
    in worker processes, so the GPU is not left waiting for it.

    precision: "fp32" (what the published dataset was made with) or "fp16" (about twice as fast; a few borderline
    spans change). flash: use FlashDeBERTa's attention (pip install flashdeberta): about twice as fast again."""

    def __init__(self, name, device="cuda", precision="fp32", flash=False, batch=24, threshold=0.5, workers=4):
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        if flash:
            if importlib.util.find_spec("flashdeberta") is None:
                raise engine.PoolError("--flash needs the flashdeberta package: pip install flashdeberta")
            os.environ["USE_FLASHDEBERTA"] = "1"
        else:
            os.environ.pop("USE_FLASHDEBERTA", None)
        import torch
        from gliner import GLiNER
        if precision not in ("fp32", "fp16"):
            raise engine.PoolError("precision must be fp32 or fp16")
        if precision == "fp16" and not device.startswith("cuda"):
            raise engine.PoolError("fp16 needs a GPU")
        self.torch, self.batch, self.th, self.pool = torch, batch, threshold, None
        self.m = GLiNER.from_pretrained(name)
        self.m.eval()
        if workers > 1 and all(hasattr(self.m, a) for a in ("prepare_batch", "collate_batch", "run_batch", "decode_batch", "create_collator")):
            G.update(model=self.m, collator=self.m.create_collator())
            torch.set_num_threads(1)                    # the workers are forked now, before any GPU state exists
            self.pool = engine.Workers(workers)
            torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
        self.m = self.m.to(device)
        if precision == "fp16":
            self.m = self.m.half()

    def close(self):
        if self.pool is not None:
            self.pool.close()
        self.m = None
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def _decode(self, out, batch, smap, emap):
        dec = self.m.decode_batch(out, batch, threshold=self.th, flat_ner=True, multi_label=False)
        return [[(smap[q][sp.start], emap[q][sp.end], sp.entity_type, float(sp.score)) for sp in spans] for q, spans in enumerate(dec)]

    def _run(self, tasks):
        """tasks: (windows, names) batches -> per batch, per window: [(start, end, label, score)], in order."""
        if self.pool is None:
            for pieces, names in tasks:
                res = self.m.inference(pieces, LABELS[names], threshold=self.th, batch_size=len(pieces))
                yield [[(e["start"], e["end"], e["label"], float(e["score"])) for e in ents] for ents in res]
            return
        pending = collections.deque()                   # the next batch is on the GPU while this one is decoded
        for batch, smap, emap in self.pool.imap(_collate, tasks):
            pending.append((self.m.run_batch(batch, threshold=self.th), batch, smap, emap))
            if len(pending) > 1:
                yield self._decode(*pending.popleft())
        while pending:
            yield self._decode(*pending.popleft())

    def raw_spans(self, texts, names, progress=None):
        """-> per text, the model's entities at the lowest threshold: [[window index, start, end, label code, score]].
        Windows of similar length are batched together."""
        pieces = [(b - a, i, j, a, b) for i, t in enumerate(texts) for j, (a, b) in enumerate(windows(t))]
        pieces.sort(key=lambda p: p[0])
        out = [[] for _ in texts]
        tasks = ([[texts[i][a:b] for _, i, _, a, b in pieces[k:k + self.batch]], names] for k in range(0, len(pieces), self.batch))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for n, ents in enumerate(self._run(tasks)):
                for (_, i, j, a, _), es in zip(pieces[n * self.batch:(n + 1) * self.batch], ents):
                    out[i] += [[j, a + s, a + e, CODE[lab], sc] for s, e, lab, sc in es]
                if progress:
                    progress(len(ents))
        for raw in out:
            raw.sort()
        return out


def find_spans(model, filters, texts, names):
    """Without a cache: -> per text, the spans to mask [(start, end, kind)]."""
    return [filters.spans(t, raw) for t, raw in zip(texts, model.raw_spans(texts, names))]


def fill_cache(model, cache, need, wave=40000):
    """Run the model on need = {key: (text, names)} and store its output. Texts go in waves of about `wave` windows, each
    saved when it is done, shortest texts first."""
    items = sorted(((names, len(t), k, t, len(windows(t))) for k, (t, names) in need.items()), key=lambda x: x[:3])
    total = sum(x[4] for x in items)
    t0, done, last = time.time(), [0], [time.time()]

    def progress(n):
        done[0] += n
        if time.time() - last[0] > 30:
            last[0] = time.time()
            rate = done[0] / (time.time() - t0)
            print(f"[ner] {done[0]:,} of {total:,} windows, {rate:.0f}/s, about {(total - done[0]) / rate / 60:.0f} min left", flush=True)

    start = 0
    while start < len(items):
        end, n = start, 0
        while end < len(items) and n < wave and items[end][0] == items[start][0]:
            n += items[end][4]
            end += 1
        raws = model.raw_spans([x[3] for x in items[start:end]], items[start][0], progress)
        for x, raw in zip(items[start:end], raws):
            cache.add(x[2], raw)
        cache.flush()
        start = end
    print(f"[ner] {total:,} windows of {len(need):,} texts in {time.time() - t0:.0f} s", flush=True)


# ----------------------------------------------------------------------------------------------- e-mail sweep
SWEEP_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*\.[A-Za-z]{2,}")


def sweep_emails(x, keep=frozenset(), keep_hosts=()):
    """Mask the e-mail addresses in every string inside x (nested dicts and lists), except the kept ones."""
    def kept(addr):
        addr = addr.lower()
        host = addr.split("@")[1]
        return addr in keep or any(host == d or host.endswith("." + d) for d in keep_hosts)

    return engine.map_strings(x, lambda s: SWEEP_EMAIL.sub(lambda m: m.group(0) if kept(m.group(0)) else "[EMAIL]", s))


# ----------------------------------------------------------------------------------------------- rows
def merge_counts(a, b):
    for k, v in b.items():
        a[k] = a.get(k, 0) + v


def mask_rows(lines, cfg, filters, cache, no_ner=False):
    """lines: raw JSON lines of any sources. cache: {text key: raw model output} or None when no_ner.
    -> (output lines, {source: stats}, dropped [(id, source, reason)], missing {key: (text, names)}).
    When texts are missing from the cache, nothing else is returned: the rows are done again once the model has run."""
    rows = [json.loads(line) for line in lines]
    stats = collections.defaultdict(collections.Counter)
    keep_idx, sc = [], set()
    for i, r in enumerate(rows):
        p = engine.policy(cfg, r["source"], r.get("id"))
        if p.get("drop") == "clean":
            raise engine.PoolError(f"{r.get('id')}: source {r['source']!r} is dropped at the clean stage; run that stage first")
        if p.get("drop"):
            stats[r["source"]]["source_dropped"] += 1
            continue
        keep_idx.append(i)
        if p["origin"] == "self-collected":
            sc.add(i)

    # regex layer, once per distinct (text, names) pair; then the NER layer on the regex-masked text
    final, missing = {}, {}
    for i in sc:
        names = bool(cfg["sources"][rows[i]["source"]].get("names"))
        for t in engine.row_texts(rows[i]):
            if (t, names) in final:
                continue
            m, c = regex_mask(t, names)
            if not no_ner and needs_ner(m):
                k = text_key(m, names)
                raw = cache.get(k)
                if raw is None:
                    missing[k] = (m, names)
                else:
                    m, c2 = apply_spans(m, filters.spans(m, raw))
                    merge_counts(c, c2)
            final[(t, names)] = (m, c)
    if missing:
        return None, None, None, missing

    keep, hosts = G["keep_emails"], G["keep_hosts"]
    out, dropped = [], []
    for i in keep_idx:
        r, line, new = rows[i], lines[i], None
        if i in sc:
            names = bool(cfg["sources"][r["source"]].get("names"))
            cnt = {}

            def mask(s):
                f, c = final[(s, names)]
                merge_counts(cnt, c)
                return f

            new = dict(r)
            new["state"] = engine.map_strings(r.get("state"), mask)
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
            else:
                new = None
        stats[r["source"]]["rows_out"] += 1
        if new is not None or b"@" in line or b"\\u0040" in line:
            cur = new if new is not None else r
            swept = dict(cur, **{k: sweep_emails(cur[k], keep, hosts) for k in ("state", "question", "options") if k in cur})
            if swept != cur:
                stats[r["source"]]["email_sweep_rows"] += 1
                new = swept
        out.append(line if new is None else engine.dump(new))
    return out, {k: dict(v) for k, v in stats.items()}, dropped, {}


def _mask_worker(lines):
    out, stats, dropped, missing = mask_rows(lines, G["cfg"], G["filters"], G["cache"], G["no_ner"])
    if missing:
        return None, len(lines), None, None, missing
    return engine.gz_member(out), len(lines), stats, dropped, None


# ----------------------------------------------------------------------------------------------- the stage
def run(cfg, pool_dir, out_dir, cache_dir=None, workers=None, device="cuda", precision="fp32", flash=False, batch=24,
        no_ner=False, part=None):
    """Run the stage on <pool_dir>/{train,val}.jsonl.gz. Writes the masked splits, pii_report.json and pii_dropped.tsv
    into out_dir. With part = (k, N) only the model runs, on the k-th of N shares of the missing texts (one process
    per GPU; run once more without it to write the rows). -> the report, or None with `part`"""
    t_start = time.time()
    if part and no_ner:
        raise engine.PoolError("--part runs the model; it cannot be combined with --no-ner")
    os.makedirs(out_dir, exist_ok=True)
    opt = dict(DEFAULTS, **cfg.get("mask", {}))
    th = dict(DEFAULTS["thresholds"], **opt.get("thresholds", {}))
    filters = Filters(th["person"], th["address"], th["username"], read_stop_spans(opt["stop_spans"]))
    variant = {"model": opt["model"], "precision": precision, "attention": "flash" if flash else "standard", "labels": LABELS,
               "window": [WORDS, MAXC, OVERLAP], "threshold": filters.raw_threshold}
    cache = None if no_ner else SpanCache(cache_dir or os.path.join(out_dir, "ner_cache"), variant, part)
    n_workers = workers or engine.default_workers()
    parts_dir = os.path.join(out_dir, ".parts")
    shutil.rmtree(parts_dir, ignore_errors=True)
    os.makedirs(parts_dir)
    paths = {name: engine.split_path(pool_dir, name) for name in engine.SPLITS}
    G.update(cfg=cfg, filters=filters, cache=None if no_ner else cache.d, no_ner=no_ner,
             keep_emails=frozenset(a.lower() for a in opt["keep_emails"]), keep_hosts=tuple(opt["keep_email_hosts"]))

    results = {name: {} for name in paths}              # split -> chunk index -> (rows in, stats, dropped)

    def sweep(pending=None):
        """One parallel pass over the chunks (all of them, or the pending ones). A chunk whose texts are all in the
        cache is written to .parts. -> the texts still missing"""
        need = {}
        with engine.Workers(n_workers) as w:
            for name, path in paths.items():
                sent = collections.deque()

                def chunks():
                    for ci, lines in enumerate(engine.chunks(path)):
                        if pending is None or ci in pending[name]:
                            sent.append(ci)
                            yield lines

                for member, n, stats, dropped, missing in w.imap(_mask_worker, chunks()):
                    ci = sent.popleft()
                    if missing:
                        need.update(missing)
                        results[name][ci] = None
                        continue
                    with open(os.path.join(parts_dir, f"{name}-{ci:06d}.gz"), "wb") as f:
                        f.write(member)
                    results[name][ci] = (n, stats, dropped)
        return need

    need = sweep()
    ner = {"texts_missing_from_cache": len(need), "seconds": 0}
    if need:
        if part:
            need = {k: v for k, v in need.items() if int(k, 16) % part[1] == part[0]}
        print(f"[ner] {len(need):,} texts are not in the cache: loading {opt['model']} ({precision}"
              f"{', flash attention' if flash else ''}) on {device}", flush=True)
        t0 = time.time()
        model = Model(opt["model"], device, precision, flash, batch, filters.raw_threshold, workers=min(8, n_workers))
        try:
            fill_cache(model, cache, need)
        finally:
            model.close()
            cache.flush()
        ner["seconds"] = round(time.time() - t0)
    if part:
        cache.close()
        shutil.rmtree(parts_dir, ignore_errors=True)
        return None
    if need:
        left = sweep({name: {ci for ci, r in results[name].items() if r is None} for name in paths})
        if left:
            raise engine.PoolError(f"{len(left)} texts are still missing from the span cache; run the stage once more")
    if cache is not None:
        cache.close()

    rep = {"model": None if no_ner else opt["model"], "numerics": None if no_ner else {"precision": precision, "attention": variant["attention"]},
           "thresholds": th, "ner": ner}
    dropped_all = []
    for name in paths:
        rows_in, total = 0, collections.defaultdict(collections.Counter)
        with engine.GzOut(engine.split_path(out_dir, name)) as g:
            for ci in sorted(results[name]):
                n, stats, dropped = results[name][ci]
                rows_in += n
                for s, st in stats.items():
                    total[s].update(st)
                dropped_all += [(name,) + tuple(d) for d in dropped]
                with open(os.path.join(parts_dir, f"{name}-{ci:06d}.gz"), "rb") as f:
                    g.write(f.read())
        agg = collections.Counter()
        for st in total.values():
            agg.update(st)
        rep[name] = {"rows_in": rows_in, "totals": dict(agg), "per_source": {s: dict(v) for s, v in total.items()}}
    shutil.rmtree(parts_dir, ignore_errors=True)
    rep["runtime_s"] = round(time.time() - t_start)
    with open(os.path.join(out_dir, "pii_report.json"), "w") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False)
    with open(os.path.join(out_dir, "pii_dropped.tsv"), "w") as f:
        for d in dropped_all:
            f.write("\t".join(d) + "\n")
    print(f"mask: {rep['train']['rows_in']:,} -> {rep['train']['totals'].get('rows_out', 0):,} training rows, "
          f"{rep['val']['rows_in']:,} -> {rep['val']['totals'].get('rows_out', 0):,} validation rows, {rep['runtime_s']} s", flush=True)
    return rep
