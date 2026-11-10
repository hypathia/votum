"""Architecture, example and data graphics for the Votum repos (SVG source + 2x PNG), in a product / landing-page style:
cards with depth, one strong red accent, JetBrains Mono for prompts. Figures carry no title of their own: the page that
embeds them has the heading.
Every figure is drawn twice: <name>.png on a light grey backdrop with white elevated cards (light pages, and still legible
where a viewer falls back to it on a dark background), and <name>_dark.png on a transparent canvas with translucent cards.
The PNGs are stored as 8-bit palettes: every frequent flat colour keeps an exact palette entry, so only anti-aliased
edges and shadows are approximated.
Needs rsvg-convert, NumPy and Pillow.   python make_assets.py <out dir>
The example outputs are read from tests/fixtures (the reference scores of the released weights, checked by the GPU test)."""
import io, json, math, subprocess, sys
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "out"); OUT.mkdir(parents=True, exist_ok=True)
SWISS = "#DA291C"                                  # flag, hero cards and "end" chips: the same in both themes
THEMES = {   # RED accent, INK text, MUTED secondary text, SUB strong secondary, NEUTRAL grey marks, GRID tracks and rules,
             # BACK canvas, CARD surface, EDGE card border, SOFT red tint, OFF unread mask cells, PINK removed rows,
             # FAM licence families (dark to light)
    "light": dict(RED=SWISS, INK="#1D1D1F", MUTED="#6E6E73", SUB="#48484A", NEUTRAL="#C7C7CC", GRID="rgba(0,0,0,.08)",
                  BACK="#F5F5F7", CARD="#FFFFFF", EDGE="rgba(0,0,0,.06)", SOFT="rgba(218,41,28,.10)", OFF="rgba(0,0,0,.05)",
                  PINK="rgba(218,41,28,.32)", FAM=["#1D1D1F", "#48484A", "#8E8E93", "#C7C7CC"]),
    "dark": dict(RED="#FF453A", INK="#F5F5F7", MUTED="#A1A1A6", SUB="#C7C7CC", NEUTRAL="#636366", GRID="rgba(255,255,255,.12)",
                 BACK=None, CARD="rgba(255,255,255,.055)", EDGE="rgba(255,255,255,.10)", SOFT="rgba(255,69,58,.16)",
                 OFF="rgba(255,255,255,.04)", PINK="rgba(255,69,58,.38)", FAM=["#F5F5F7", "#C7C7CC", "#8E8E93", "#636366"]),
}
RED = INK = MUTED = SUB = NEUTRAL = GRID = BACK = CARD = EDGE = SOFT = OFF = PINK = FAM = SUFFIX = None


def theme(name):
    """set the palette and the file-name suffix ("" for light, "_dark") used by every drawing function"""
    global RED, INK, MUTED, SUB, NEUTRAL, GRID, BACK, CARD, EDGE, SOFT, OFF, PINK, FAM, SUFFIX
    t = THEMES[name]
    RED, INK, MUTED, SUB, NEUTRAL, GRID = t["RED"], t["INK"], t["MUTED"], t["SUB"], t["NEUTRAL"], t["GRID"]
    BACK, CARD, EDGE, SOFT, OFF, PINK, FAM = t["BACK"], t["CARD"], t["EDGE"], t["SOFT"], t["OFF"], t["PINK"], t["FAM"]
    SUFFIX = "" if name == "light" else "_" + name
FONT = "Inter, 'Helvetica Neue', Helvetica, Arial, sans-serif"
DISPLAY = "'Inter Display', Inter, 'Helvetica Neue', Helvetica, Arial, sans-serif"
MONO = "'JetBrains Mono', 'SF Mono', Menlo, Consolas, 'DejaVu Sans Mono', monospace"
FIX = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
W, PAD, GAP = 1200, 28, 14
WHITE, WHITE2 = "#FFFFFF", "rgba(255,255,255,.78)"


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def T(x, y, s, size=13, fill=None, weight=400, anchor="start", extra=""):
    fill = fill or INK
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" font-weight="{weight}" text-anchor="{anchor}" {extra}>{esc(s)}</text>'


def big(x, y, s, size, fill=None, weight=800, anchor="start"):
    """display figure: Inter Display, tight tracking"""
    return T(x, y, s, size, fill, weight, anchor, f'font-family="{DISPLAY}" letter-spacing="{-size * .025:.2f}"')


def textw(s, size, display=False):
    """approximate advance width of s in Inter (display: Inter Display, tight tracking)"""
    narrow = ".,:;'|!il "; w = sum((.28 if c in narrow else .62 if c.isdigit() else .58 if c.isupper() else .54) for c in s) * size
    return w - (size * .025 * (len(s) - 1) if display else 0)


def eyebrow(x, y, s, fill=None):
    return T(x, y, s.upper(), 10, fill or RED, 700, extra='letter-spacing="1.1"')


def bar(x, y, w, h, fill, r=None):
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{h}" rx="{h / 2 if r is None else r}" fill="{fill}"/>'


def card(x, y, w, h, hero=False, r=14):
    """an elevated surface (light: white with a soft shadow; dark: translucent with a hairline); hero: solid red"""
    if hero: return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r}" fill="{SWISS}"/>'
    sh = ' filter="url(#sh)"' if BACK else ""
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r}" fill="{CARD}"{sh}/>'
            f'<rect x="{x + .5:.1f}" y="{y + .5:.1f}" width="{w - 1:.1f}" height="{h - 1:.1f}" rx="{r - .5}" fill="none" stroke="{EDGE}"/>')


def pill(x, y, s, size=11, fill=None, fg=WHITE, anchor="end"):
    """small rounded label; x is its end (anchor end) or start"""
    w = size * .62 * len(s) + size * 1.1; h = size * 1.55; x0 = x - w if anchor == "end" else x
    return (bar(x0, y - h / 2, w, h, fill or RED) + T(x0 + w / 2, y + size * .36, s, size, fg, 700, "middle"))


def save(name, h, body):
    p = OUT / f"{name}{SUFFIX}.svg"
    back = f'<rect width="{W}" height="{h}" rx="18" fill="{BACK}"/>' if BACK else ""
    defs = ('<defs><filter id="sh" x="-4%" y="-6%" width="108%" height="116%">'
            '<feDropShadow dx="0" dy="1" stdDeviation="1.2" flood-color="#000" flood-opacity=".07"/>'
            '<feDropShadow dx="0" dy="6" stdDeviation="8" flood-color="#000" flood-opacity=".05"/></filter>'
            f'<marker id="ah" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">'
            f'<path d="M0 0 L8 4 L0 8 z" fill="{MUTED}"/></marker></defs>')
    p.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{h}" viewBox="0 0 {W} {h}" font-family="{FONT}">'
                 f'{defs}{back}{body}</svg>')
    png = subprocess.run(["rsvg-convert", "-z", "2", str(p)], check=True, capture_output=True).stdout
    palettize(Image.open(io.BytesIO(png))).save(OUT / f"{name}{SUFFIX}.png", optimize=True)


def palettize(im, n_flat=96):
    """RGBA -> 8-bit palette image: the n_flat most frequent flat colours (a pixel equal to its right and lower
    neighbours: panels, tracks, backgrounds, text cores) get exact entries; the other slots come from an octree palette,
    and every pixel maps to its nearest entry."""
    a = np.asarray(im.convert("RGBA"), dtype=np.uint8); H, W = a.shape[:2]; key = a.reshape(-1, 4).view(np.uint32).ravel()
    same = np.ones((H, W), bool)
    same[:, :-1] &= (a[:, :-1] == a[:, 1:]).all(2); same[:-1, :] &= (a[:-1, :] == a[1:, :]).all(2)
    fk, fc = np.unique(key[same.ravel()], return_counts=True); flat = fk[np.argsort(-fc, kind="stable")][:n_flat]
    rest = Image.fromarray(a).quantize(colors=256 - len(flat), method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE)
    rp = np.asarray(rest.getpalette(rawmode="RGBA")[: 4 * (256 - len(flat))], dtype=np.uint8).reshape(-1, 4)
    pal = np.concatenate([flat.view(np.uint8).reshape(-1, 4), rp]).astype(np.int32)
    uk, inv = np.unique(key, return_inverse=True); uc = uk.view(np.uint8).reshape(-1, 4).astype(np.int32)
    idx = np.empty(len(uk), np.uint8)
    for s in range(0, len(uk), 4096):
        idx[s:s + 4096] = ((uc[s:s + 4096, None, :] - pal[None]) ** 2).sum(2).argmin(1)
    out = Image.fromarray(idx[inv.ravel()].reshape(H, W), "P"); out.putpalette(pal.astype(np.uint8).ravel().tobytes(), rawmode="RGBA")
    return out


def head(kicker, title, sub):
    """figures carry no title of their own: the page that embeds them has the heading and the caption"""
    return ""


TOP = PAD


def tile(x, y, w, h, kicker, value, label, hero=False, unit=""):
    """stat tile: eyebrow, hero number (+ small unit), one-line label"""
    fg, fg2, fgk = (WHITE, WHITE2, WHITE2) if hero else (INK, MUTED, RED)
    b = [card(x, y, w, h, hero), eyebrow(x + 18, y + 26, kicker, fgk), big(x + 17, y + 66, value, 40, fg)]
    if unit: b.append(T(x + 17 + textw(value, 40, True) + 6, y + 66, unit, 16, fg2, 600))
    b.append(T(x + 18, y + h - 16, label, 12, fg2, 500))
    return "".join(b)


def option_rows(x, y, w, opts, row=19, lab_w=0.36, size=12, bh=8, lab_max=0.5):
    """option rows: label, track + bar, value; the first option is the answer (red bar, value in a red pill).
    The label column is lab_w of the width, or wider if the longest label needs it (at most half)."""
    b = []; lw = min(max(w * lab_w, max(textw(lab, size) for lab, _ in opts) + 12), w * lab_max)
    for i, (lab, p) in enumerate(opts):
        win = i == 0; yy = y + i * row
        b.append(T(x, yy + 4, lab, size, INK, 650 if win else 400))
        tw = w - lw - 54
        b.append(bar(x + lw, yy - bh / 2, tw, bh, GRID)); b.append(bar(x + lw, yy - bh / 2, max(tw * p, bh), bh, RED if win else NEUTRAL))
        if win: b.append(pill(x + w, yy, f"{p:.2f}", 11))
        else: b.append(T(x + w - 2, yy + 4, f"{p:.2f}", size, MUTED, 400, "end"))
    return "".join(b)


# ---------------------------------------------------------------------------------------------------------- shared
def fixture():
    """[(request, {qid: answer})] of tests/fixtures: answer = {"ids", "scores", "probs"} of the released weights."""
    reqs = json.loads((FIX / "smoke_requests.json").read_text()); exp = json.loads((FIX / "smoke_expected.json").read_text())
    return [(r, {a["qid"]: a for a in e}) for r, e in zip(reqs, exp)]


def option_labels(q):
    crit = q.get("criteria")
    if q["type"] == "noul": return {"true": "true", "false": "false"}
    if q["type"] == "score": return {str(i): t for i, t in enumerate(crit)}
    return dict(crit) if isinstance(crit, dict) else {str(c): str(c) for c in crit}


def wrap(s, n):
    out, line = [], ""
    for w in s.split():
        if len(line) + len(w) + 1 > n: out.append(line); line = w
        else: line = (line + " " + w).strip()
    return out + [line]


def rich(x, y, parts, size, fill=None, weight=400, anchor="start", family=None):
    """one text line from (text, kind) parts; kind "sub" sets a subscript."""
    fill = fill or INK
    fam = f' font-family="{family}"' if family else ""
    body = "".join(f'<tspan baseline-shift="sub" font-size="{size * .72:.1f}">{esc(t)}</tspan>' if k == "sub" else esc(t) for t, k in parts)
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" font-weight="{weight}" text-anchor="{anchor}"{fam}>{body}</text>'


def step(x, y, n, title):
    """numbered step badge and card title"""
    return (f'<circle cx="{x + 11}" cy="{y - 5}" r="11" fill="{RED}"/>' + T(x + 11, y - 1, str(n), 11.5, WHITE, 800, "middle")
            + T(x + 30, y, title, 14, INK, 700))


def chevron(x, y):
    return f'<path d="M{x - 4} {y - 7} L{x + 3} {y} L{x - 4} {y + 7}" fill="none" stroke="{NEUTRAL}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'


# ---------------------------------------------------------------------------------------------------------- architecture
def architecture():
    req, ans = fixture()[0]
    qs = [("team", req["questions"]["team"]), ("refund", req["questions"]["refund"])]
    b = [head("Architecture", "How a decision is made", "One request, two questions, one forward pass: header and state are read once, every option is scored side by side")]
    c1w, c2w = 432, 278; c3x = PAD + c1w + GAP + c2w + GAP; c3w = W - PAD - c3x
    ctop = TOP; in_top = ctop + 56                       # card top; content top (below the step title)
    mono, lh = 10.5, 13

    # 1 the prompt: blocks in order, with the y of every option row (for the score column)
    ax, aw = PAD + 16, c1w - 32
    blocks, opt_y, q_y = [], [], {}
    srt = sorted(qs[0][1]["criteria"].values())
    hdr = ["Questions in this request:", "1. " + qs[0][1]["instructions"]] + [f"   - {t}" for t in srt] + ["2. " + qs[1][1]["instructions"]]
    CUE = {"noul": " (Answer: the true or the false option.)", "choice": " (Choose exactly one option.)",
           "score": " (Options are levels in increasing order; choose one.)"}          # as in src/votum/prompt.py
    NOUL = {"true": "Yes / true: the proposition holds.", "false": "No / false: the proposition does not hold."}
    state = wrap(req["state"], 56)
    y = in_top
    def block(tag, lines, kind, fill, qi=None):
        nonlocal y
        h = 6 + lh * len(lines); blocks.append(dict(tag=tag, kind=kind, q=qi))
        out = [bar(ax + 26, y, aw - 26, h, fill, 5), T(ax + 10, y + 3 + lh * .8, tag, 10.5, RED if kind == "O" else MUTED, 700, "middle")]
        for k, ln in enumerate(lines):
            out.append(T(ax + 36, y + 3 + lh * (k + .8), ln, mono, INK, 400, extra=f'font-family="{MONO}" xml:space="preserve"'))
        yc = y + 3 + lh * .5; y += h + 2
        return "".join(out), yc
    parts = []
    s_, _ = block("H", hdr, "H", OFF); parts.append(s_)
    s_, _ = block("S", ["State:"] + state, "S", OFF); parts.append(s_)
    for qi, (qid, q) in enumerate(qs):
        a = ans[qid]; labs = option_labels(q)
        s_, yc = block(f"Q{qi + 1}", ["Question:"] + wrap(q["instructions"] + CUE[q["type"]], 56) + ["Options:"], "Q", OFF, qi)
        parts.append(s_); q_y[qi] = yc
        texts = [NOUL[i] for i in a["ids"]] if q["type"] == "noul" else [labs[i] for i in a["ids"]]
        for oid, txt, z in zip(a["ids"], texts, a["scores"]):
            s_, yc = block("o", ["- " + txt], "O", SOFT, qi); parts.append(s_)
            cx = ax + 36 + 6.35 * (len(txt) + 2) + 6
            parts.append(bar(cx, yc - 6, 27, 12, SWISS, 3) + T(cx + 13.5, yc + 3.2, "end", 9, WHITE, 600, "middle"))
            opt_y.append((qi, oid, z, a["probs"][oid], yc))
    a_bottom = y
    cbot = a_bottom + 30; ch = cbot - ctop
    b.append(card(PAD, ctop, c1w, ch) + step(PAD + 16, ctop + 32, 1, "One prompt"))
    b += parts

    # 2 who reads whom
    c2x = PAD + c1w + GAP
    b.append(card(c2x, ctop, c2w, ch) + step(c2x + 16, ctop + 32, 2, "Custom attention mask"))
    n = len(blocks); cs = 16; mx = c2x + 16 + 26 + 34; my = in_top + 16
    def reads(r, c):
        R_, C_ = blocks[r], blocks[c]
        if c == r: return 2
        if c > r: return 0
        if C_["kind"] in ("H", "S"): return 1
        if C_["kind"] == "Q": return 1 if R_["kind"] == "O" and R_["q"] == C_["q"] else 0
        return 0
    for k, bl in enumerate(blocks):
        col = RED if bl["kind"] == "O" else INK
        b.append(T(mx - 6, my + k * cs + cs * .74, bl["tag"], 10, col, 650, "end"))
        b.append(T(mx + k * cs + cs / 2, my - 6, bl["tag"], 10, col, 650, "middle"))
        for c in range(n):
            v = reads(k, c)
            b.append(f'<rect x="{mx + c * cs + .75:.1f}" y="{my + k * cs + .75:.1f}" width="{cs - 1.5:.1f}" height="{cs - 1.5:.1f}" rx="2.5" '
                     f'fill="{(GRID, SUB, RED)[v] if v or c <= k else OFF}"/>')
    ly = my + n * cs + 22; lx = c2x + 20
    b.append(bar(lx, ly - 8, 9, 9, SUB, 2) + T(lx + 14, ly, "row reads column", 10.5, MUTED))
    b.append(bar(lx, ly + 7, 9, 9, RED, 2) + T(lx + 14, ly + 15, "the block itself", 10.5, MUTED))

    # 3 one pass, one score per option: backbone column + scores aligned with the option rows
    b.append(card(c3x, ctop, c3w, ch) + step(c3x + 16, ctop + 32, 3, "One score per option"))
    bx, bw = c3x + 24, 40; by = in_top + 2; lh2 = (a_bottom - 2 - by) / 32
    b.append(bar(bx - 8, by - 8, bw + 16, 32 * lh2 + 14, OFF, 8))
    for L in range(32):
        bid = L in range(3, 32, 4)
        b.append(bar(bx, by + L * lh2 + (lh2 - 4) / 2, bw, 4, RED if bid else NEUTRAL, 2))
    b.append(T(bx + bw / 2, a_bottom + 16, "Apertus-8B", 10.5, MUTED, 600, "middle"))
    px = bx + bw + 36; pr = c3x + c3w - 20
    zx = px + 46
    for qi, oid, z, p, yc in opt_y:
        win = p == max(pp for qq, _, _, pp, _ in opt_y if qq == qi)
        b.append(T(zx + 50, yc + 4, f"{z:+.2f}", 10.5, MUTED, 500, "end", f'font-family="{MONO}"'))
        tx, tw = zx + 64, pr - 52 - (zx + 64)
        b.append(bar(tx, yc - 4, tw, 8, GRID) + bar(tx, yc - 4, max(tw * p, 8), 8, RED if win else NEUTRAL))
        if win: b.append(pill(pr, yc, f"{p:.2f}", 11))
        else: b.append(T(pr - 2, yc + 4, f"{p:.2f}", 11.5, MUTED, 400, "end"))
    for x in (PAD + c1w + GAP / 2, c3x - GAP / 2): b.append(chevron(x, ctop + ch / 2))
    save("architecture", cbot + PAD, "".join(b))


# ---------------------------------------------------------------------------------------------------------- examples
# per fixture request: title, and the question rows (questions in one row are set side by side)
CARDS = [("An email · three questions in one request", [["team"], ["refund", "urgency"]]), ("German · routing", [["team"]]),
         ("A JSON record · arithmetic", [["total"], ["discount"]]), ("French · intent and language", [["action", "lang"]])]


def ex_card(x, y, w, title, state, rows, json_state, h_min=0):
    """one example card; cards of a grid row share a height (h_min) and their question rows line up"""
    b = []; yy = y + 10
    lim = 82 if json_state else 90; lines = wrap(state, lim)
    if len(lines) == 2 and not json_state:
        sent = [k + 2 for k in range(len(state) - 2) if state[k] == "." and state[k + 1] == " "]   # break at a sentence end if it fits
        fits = [k for k in sent if k - 1 <= lim and len(state) - k <= lim]
        lines = [state[:fits[0] - 1], state[fits[0]:]] if fits else wrap(state, math.ceil(len(state) / 2) + 6)
    elif len(lines) > 1: lines = wrap(state, math.ceil(len(state) / len(lines)) + 6)   # balanced lines, no orphan
    y0 = yy
    for ln in lines:
        yy += 16
        b.append(T(x + 30, yy, ln, 10.5 if json_state else 12, INK, 400,
                   extra=f'font-family="{MONO}"' if json_state else 'font-style="italic"'))
    b.append(bar(x + 20, y0 + 4, 3, yy - y0 + 1, RED, 1.5))
    for row in rows:
        n = len(row); gw = (w - 40 - 24 * (n - 1)) / n; yq = yy + 26
        for k, (q, opts) in enumerate(row):
            gx = x + 20 + k * (gw + 24)
            b.append(T(gx, yq, q, 11, MUTED, 600))
            b.append(option_rows(gx, yq + 19, gw, opts, 19, 0.34 if n == 1 else 0.40, 11.5, 8))
        yy = yq + 19 + 19 * (max(len(o) for _, o in row) - 1)
    h = max(yy - y + 20, h_min)
    return card(x, y, w, h) + "".join(b), h


def examples():
    cw = (W - 2 * PAD - GAP) / 2
    E = []
    for (t, layout), (req, ans) in zip(CARDS, fixture()):
        st = req["state"] if isinstance(req["state"], str) else json.dumps(req["state"], ensure_ascii=False)
        qd = {}
        for qid, q in req["questions"].items():
            labs = option_labels(q); pr = ans[qid]["probs"]
            qd[qid] = (q["instructions"], sorted(((labs[i], pr[i]) for i in pr), key=lambda o: -o[1]))
        E.append((t, st, [[qd[q] for q in row] for row in layout], not isinstance(req["state"], str)))
    V = vision_cards()[:2]                                  # the rail incident and the street market
    parts, y = [], TOP
    h = max(ex_card(0, 0, cw, *e)[1] for e in (E[0], E[2]))   # row 1: the support ticket and the JSON order (cards share a height)
    for c, e in enumerate((E[0], E[2])): parts.append(ex_card(PAD + c * (cw + GAP), y, cw, *e, h_min=h)[0])
    y += h + GAP
    h = max(vx_card(0, 0, cw, 99, *e)[1] for e in V)        # row 2: two photos
    for c, e in enumerate(V): parts.append(vx_card(PAD + c * (cw + GAP), y, cw, c, *e, h_min=h)[0])
    y += h + GAP
    save("examples", y - GAP + PAD, head("Examples", "What Votum answers", "Real answers of Votum-8B to four requests. The red bar marks the answer; the numbers are probabilities.") + "".join(parts))


# ---------------------------------------------------------------------------------------------------------- image examples
# per fixtures/vision_requests.json request: the card title
VCARDS = ["A photo · incident triage", "A photo · what is in it", "Aircraft · maintenance log", "Rail-yard camera · operations",
          "Café · order check", "Scanned letter · mail room"]
IMG_W = 196


def thumb(path, w, h_max):
    """a JPEG data URI of the image at 2x its drawn size, and the drawn width and height"""
    import base64
    im = Image.open(path).convert("RGB"); dh = min(h_max, w * im.height / im.width); dw = dh * im.width / im.height
    im = im.resize((round(dw * 2), round(dh * 2)), Image.LANCZOS); b = io.BytesIO(); im.save(b, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode(), dw, dh


def vx_card(x, y, w, k, title, req, ans, h_min=0):
    """one image card: the photo and the state's other fields on the left, the request's questions on the right"""
    b = []
    src = next(v["image"] for v in req["state"].values() if isinstance(v, dict) and "image" in v)
    uri, dw, dh = thumb(FIX / src, IMG_W, 210)
    ix, iy = x + 20, y + 22
    b.append(f'<clipPath id="vc{k}"><rect x="{ix:.1f}" y="{iy:.1f}" width="{dw:.1f}" height="{dh:.1f}" rx="8"/></clipPath>'
             f'<image href="{uri}" x="{ix:.1f}" y="{iy:.1f}" width="{dw:.1f}" height="{dh:.1f}" clip-path="url(#vc{k})" preserveAspectRatio="none"/>'
             f'<rect x="{ix + .5:.1f}" y="{iy + .5:.1f}" width="{dw - 1:.1f}" height="{dh - 1:.1f}" rx="7.5" fill="none" stroke="{EDGE}"/>')
    rest = json.dumps({kk: ("@IMG@" if isinstance(v, dict) and "image" in v else v) for kk, v in req["state"].items()}, ensure_ascii=False)
    yy = iy + dh + 4
    for ln in wrap(rest.replace('"@IMG@"', "<image>"), 32):
        yy += 14; b.append(T(ix, yy, ln, 9.5, MUTED, 400, extra=f'font-family="{MONO}"'))
    qx = ix + IMG_W + 22; qw = x + w - 20 - qx; yq = y + 22
    longest = max(textw(t, 11) for q in req["questions"].values() for t in option_labels(q).values())
    lab = min(0.62, max(0.42, (longest * 1.12 + 18) / qw))      # one label column per card, wide enough for its longest label
    for qid, q in req["questions"].items():
        labs = option_labels(q); pr = ans[qid]["probs"]
        opts = sorted(((labs[i], pr[i]) for i in pr), key=lambda o: -o[1])
        b.append(T(qx, yq + 8, q["instructions"], 11, MUTED, 600))
        b.append(option_rows(qx, yq + 26, qw, opts, 18, lab, 11, 7, lab_max=0.62))
        yq += 26 + 18 * (len(opts) - 1) + 22
    h = max(yq - y + 2, yy - y + 20, h_min)
    return card(x, y, w, h) + "".join(b), h


def vision_cards():
    """[(title, request, {qid: answer})] of tests/fixtures/vision_*.json"""
    reqs = json.loads((FIX / "vision_requests.json").read_text()); exp = json.loads((FIX / "vision_expected.json").read_text())
    return [(t, r, {a["qid"]: a for a in e["answers"]}) for t, r, e in zip(VCARDS, reqs, exp)]


def vision_examples():
    E = vision_cards()
    cw = (W - 2 * PAD - GAP) / 2; parts, y = [], TOP
    for row in range(0, len(E), 2):
        pair = E[row:row + 2]
        h = max(vx_card(0, 0, cw, 99, *e)[1] for e in pair)
        for c, e in enumerate(pair): parts.append(vx_card(PAD + c * (cw + GAP), y, cw, row + c, *e, h_min=h)[0])
        y += h + GAP
    save("examples_vision", y - GAP + PAD, head("Images", "Answers on images",
         "Real answers of Votum-8B. All questions of a request are answered in one pass over the image.") + "".join(parts))


# ---------------------------------------------------------------------------------------------------------- pool
def pool():
    steps = [("All converted rows", 881143), ("13 sources: licence review", -57169), ("LLM-generated sets", -53620),
             ("ContractNLI test split", -1982), ("benchmark overlap (8-gram)", -24315), ("portal / government terms", -20059),
             ("conflicting licence tags", -741), ("Votum Pool", 723257)]
    b = [head("Data", "From training pool to dataset", "Training rows removed at each step, and the licence mix of what remains")]
    tw = (W - 2 * PAD - 3 * GAP) / 4; th = 108
    for i, (k, v, lab) in enumerate([("Training rows", "723,257", "plus 6,087 validation rows"), ("Sources", "181", "164 public datasets, 17 collected"),
                                     ("Languages", "73", "language codes; about 41 % English"), ("Configs", "2", "default · share_alike")]):
        b.append(tile(PAD + i * (tw + GAP), TOP, tw, th, k, v, lab, hero=i == 0))
    cy = TOP + th + GAP; rh = 24; ch = 40 + rh * len(steps) + 76
    b.append(card(PAD, cy, W - 2 * PAD, ch))
    top = cy + 40; x0, x1 = PAD + 270, W - PAD - 130; X = lambda v: x0 + v / 881143 * (x1 - x0); cur = 0
    for i, (lab, v) in enumerate(steps):
        y = top + i * rh; end = i in (0, len(steps) - 1)
        b.append(T(PAD + 22, y + 4, lab, 12.5, INK, 650 if end else 400))
        if end:
            cur = v; b.append(bar(x0, y - 7, X(v) - x0, 14, RED if i else NEUTRAL, 4))
            b.append(pill(X(v) + 62, y, f"{v:,}", 11) if i else T(X(v) + 9, y + 4, f"{v:,}", 12.5, INK, 700))
        else:
            nxt = cur + v; b.append(bar(X(nxt), y - 7, X(cur) - X(nxt), 14, PINK, 4)); b.append(T(X(cur) + 9, y + 4, f"−{-v:,}", 11.5, MUTED, 500)); cur = nxt
    fams = [("CC-BY", 208579, FAM[0]), ("Apache / MIT / AFL", 128234, FAM[1]), ("CC0 / PDDL / PD", 130272, FAM[2]),
            ("other open", 14316, FAM[3]), ("CC-BY-SA / ODbL · share_alike", 241856, RED)]
    y0 = top + rh * len(steps) + 14; tot = sum(v for _, v, _ in fams); x = PAD + 22; ww_all = W - 2 * PAD - 44
    for lab, v, c in fams:
        ww = ww_all * v / tot; b.append(bar(x, y0, ww - 2, 12, c, 3)); x += ww
    x = PAD + 22
    for lab, v, c in fams:
        b.append(bar(x, y0 + 26, 10, 10, c, 2))
        b.append(f'<text x="{x + 15}" y="{y0 + 35}" font-size="11.5" fill="{INK}">{esc(lab)}<tspan dx="6" fill="{MUTED}">{v:,}</tspan></text>')
        x += 15 + textw(lab, 11.5) + 6 + textw(f"{v:,}", 11.5) + 26
    save("pool", cy + ch + PAD, "".join(b))


# ---------------------------------------------------------------------------------------------------- selection
class Sankey:
    """columns of nodes and the flows between them, on one scale (px per unit); kept flows red, removed flows grey"""
    NODE_W, GAP, MIN_H, LABEL_DY = 7, 11, 2, 18

    def __init__(self, y0, y1, total, gaps):
        self.y0, self.y1 = y0, y1; self.scale = (y1 - y0 - gaps * self.GAP) / total; self.nodes, self.links = {}, []

    def column(self, x, items, side):
        """items: (key, value, kept, number, text[, second line]); stacked top to bottom, centred in the band"""
        hs = [max(v * self.scale, self.MIN_H) for _, v, *_ in items]
        y = self.y0 + (self.y1 - self.y0 - sum(hs) - self.GAP * (len(items) - 1)) / 2
        for (key, v, kept, num, text, *rest), h in zip(items, hs):
            self.nodes[key] = dict(x=x, y=y, h=h, v=v, kept=kept, num=num, text=text, line2=rest[0] if rest else None, side=side, out=0.0, inn=0.0)
            y += h + self.GAP

    def link(self, s, t, v):
        self.links.append((s, t, v))

    def draw(self):
        b = []
        for s, t, v in self.links:
            S, Tn = self.nodes[s], self.nodes[t]
            hs = max(v * self.scale, self.MIN_H) * S["h"] / max(S["v"] * self.scale, self.MIN_H)
            ht = max(v * self.scale, self.MIN_H) * Tn["h"] / max(Tn["v"] * self.scale, self.MIN_H)
            ya, yb, xa, xb = S["y"] + S["out"], Tn["y"] + Tn["inn"], S["x"] + self.NODE_W, Tn["x"]; xm = (xa + xb) / 2
            fill, op = (RED, .30) if Tn["kept"] else (NEUTRAL, .38)
            b.append(f'<path d="M{xa:.1f} {ya:.1f} C{xm:.1f} {ya:.1f} {xm:.1f} {yb:.1f} {xb:.1f} {yb:.1f} L{xb:.1f} {yb + ht:.1f} '
                     f'C{xm:.1f} {yb + ht:.1f} {xm:.1f} {ya + hs:.1f} {xa:.1f} {ya + hs:.1f} Z" fill="{fill}" fill-opacity="{op}"/>')
            S["out"] += hs; Tn["inn"] += ht
        for n in self.nodes.values():
            b.append(f'<rect x="{n["x"]:.1f}" y="{n["y"]:.1f}" width="{self.NODE_W}" height="{n["h"]:.1f}" rx="1.5" fill="{RED if n["kept"] else NEUTRAL}"/>')
        for x in sorted({n["x"] for n in self.nodes.values()}):          # labels, nudged apart column by column
            prev = None
            for n in sorted((n for n in self.nodes.values() if n["x"] == x), key=lambda n: n["y"]):
                yc = n["y"] + n["h"] / 2
                if prev is not None and yc < prev + self.LABEL_DY: yc = prev + self.LABEL_DY
                prev = yc + (self.LABEL_DY - 2 if n["line2"] else 0)
                if n["side"] is None: continue
                right = n["side"] == "right"; tx = n["x"] + self.NODE_W + 8 if right else n["x"] - 8; anchor = "start" if right else "end"
                size, w2 = (15, 700) if n["kept"] else (12.5, 400)
                num = f'<tspan font-weight="700" fill="{INK}">{esc(n["num"])}</tspan>'; txt = f'<tspan fill="{INK if w2 == 700 else MUTED}" font-weight="{w2}"> {esc(n["text"])}</tspan>'
                b.append(f'<text x="{tx:.1f}" y="{yc + size * .35:.1f}" font-size="{size}" text-anchor="{anchor}" xml:space="preserve">{num}{txt}</text>')
                if n["line2"]: b.append(T(tx, yc + size * .35 + self.LABEL_DY - 2, n["line2"], 11.5, MUTED, anchor=anchor))
        return "".join(b)


def selection():
    """how the sources and the rows were selected: two Sankey diagrams (numbers: data/README.md and the dataset card)"""
    b = [head("Data", "From 1,289 candidate sources to 729,344 rows", "How sources were selected, and what happened to the rows converted from them")]
    ch = 392; b.append(card(PAD, TOP, W - 2 * PAD, ch)); y0, y1 = TOP + 62, TOP + ch - 44
    b.append(T(PAD + 22, TOP + 32, "Sources", 14, INK, 700)); b.append(T(PAD + 775, TOP + 32, "Rows", 14, INK, 700))
    s = Sankey(y0, y1, 1289, 5); xa, xb, xc = PAD + 172, PAD + 290, PAD + 525
    s.column(xa, [("cand", 1289, True, "1,289", "candidates", "catalogues, hubs, portals")], "left")
    s.column(xb, [("pub", 181, True, "181", "published"), ("late", 36, False, "36", "removed before release"),
                  ("out", 1072, False, "1,072", "not selected")], "right")
    reasons = [("lic", 372, "372", "licence or commercial use"), ("qual", 337, "337", "quality"), ("text", 174, "174", "third-party text rights"),
               ("fake", 84, "84", "not real decisions"), ("crawl", 49, "49", "AI-crawler opt-out"), ("other", 56, "56", "duplicates and other reasons")]
    s.column(xc, [(k, v, False, n, t) for k, v, n, t in reasons], "right")
    for k, v in (("pub", 181), ("late", 36), ("out", 1072)): s.link("cand", k, v)
    for k, v, *_ in reasons: s.link("out", k, v)
    b.append(s.draw())
    r = Sankey(y0, y1, 901098, 2); xr, xs = PAD + 775, PAD + 900
    r.column(xr, [("conv", 901098, True, "901,098", "rows converted")], None)
    r.column(xs, [("pub", 729344, True, "729,344", "published", "723,257 training, 6,087 validation"),
                  ("late", 159015, False, "159,015", "removed before release", "licence review, benchmark overlap"),
                  ("dup", 12739, False, "12,739", "duplicates and other filters")], "right")
    for k, v in (("pub", 729344), ("late", 159015), ("dup", 12739)): r.link("conv", k, v)
    b.append(r.draw())
    b.append(f'<text x="{xr + 3.5:.1f}" y="{y1 + 24:.1f}" font-size="12.5" text-anchor="middle" xml:space="preserve"><tspan font-weight="700" fill="{INK}">901,098</tspan>'
             f'<tspan fill="{MUTED}"> rows converted</tspan></text>')
    save("selection", TOP + ch + PAD, "".join(b))


def flag():
    """assets/flag.svg: the Swiss flag (square; cross arms 1/6 longer than wide), vector and text-free, so it renders the
    same everywhere (a flag emoji needs a colour emoji font and shows as "CH" without one)."""
    (OUT / "flag.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
                                  f'<rect width="32" height="32" rx="3" fill="{SWISS}"/>'
                                  '<path d="M13 6h6v7h7v6h-7v7h-6v-7H6v-6h7z" fill="#fff"/></svg>\n')


flag()
for name in THEMES:
    theme(name); architecture(); examples(); vision_examples(); pool(); selection()
print("written to", OUT)
