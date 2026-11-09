"""Architecture and result graphics for the Votum repos (SVG source + 2x PNG), in a product / landing-page style:
cards with depth, hero numbers, one strong red accent, Inter Display headings, JetBrains Mono for prompts.
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
    """eyebrow, display title, one-line subtitle; content starts at y = 96"""
    return eyebrow(PAD, 36, kicker) + big(PAD, 64, title, 26, INK) + T(PAD, 86, sub, 13, MUTED)


TOP = 104


def tile(x, y, w, h, kicker, value, label, hero=False, unit=""):
    """stat tile: eyebrow, hero number (+ small unit), one-line label"""
    fg, fg2, fgk = (WHITE, WHITE2, WHITE2) if hero else (INK, MUTED, RED)
    b = [card(x, y, w, h, hero), eyebrow(x + 18, y + 26, kicker, fgk), big(x + 17, y + 66, value, 40, fg)]
    if unit: b.append(T(x + 17 + textw(value, 40, True) + 6, y + 66, unit, 16, fg2, 600))
    b.append(T(x + 18, y + h - 16, label, 12, fg2, 500))
    return "".join(b)


def option_rows(x, y, w, opts, row=19, lab_w=0.36, size=12, bh=8):
    """option rows: label, track + bar, value; the first option is the answer (red bar, value in a red pill).
    The label column is lab_w of the width, or wider if the longest label needs it (at most half)."""
    b = []; lw = min(max(w * lab_w, max(textw(lab, size) for lab, _ in opts) + 12), w * .5)
    for i, (lab, p) in enumerate(opts):
        win = i == 0; yy = y + i * row
        b.append(T(x, yy + 4, lab, size, INK, 650 if win else 400))
        tw = w - lw - 54
        b.append(bar(x + lw, yy - bh / 2, tw, bh, GRID)); b.append(bar(x + lw, yy - bh / 2, max(tw * p, bh), bh, RED if win else NEUTRAL))
        if win: b.append(pill(x + w, yy, f"{p:.2f}", 11))
        else: b.append(T(x + w - 2, yy + 4, f"{p:.2f}", size, MUTED, 400, "end"))
    return "".join(b)


# ---------------------------------------------------------------------------------------------------------- state once
def state_once():
    # long_q1 / long_q4 / long_q8: the released weights on vLLM 0.33, one RTX 4090, ~3,200-token states, prefix cache off
    rows = [(1, 3.08, 322.5), (4, 9.69, 391.7), (8, 15.81, 464.5)]
    b = [head("Throughput", "Many questions, one read", "Decisions per second on ~3,200-token documents, one RTX 4090; the questions of a request share one pass over the state")]
    tw = (W - 2 * PAD - 2 * GAP) / 3; th = 150
    for i, (q, v, ms) in enumerate(rows):
        x = PAD + i * (tw + GAP); hero = q == 8; fg, fg2, fgk = (WHITE, WHITE2, WHITE2) if hero else (INK, MUTED, RED)
        b.append(card(x, TOP, tw, th, hero))
        b.append(eyebrow(x + 18, TOP + 26, f"{q} question{'s' if q > 1 else ''} per request", fgk))
        b.append(big(x + 17, TOP + 70, f"{v:.1f}", 44, fg) + T(x + 17 + textw(f"{v:.1f}", 44, True) + 8, TOP + 70, "decisions / s", 14, fg2, 600))
        trk = tw - 36; fill = WHITE if hero else (NEUTRAL if q == 1 else SUB)
        b.append(bar(x + 18, TOP + 90, trk, 8, "rgba(255,255,255,.25)" if hero else GRID) + bar(x + 18, TOP + 90, trk * v / 16, 8, fill))
        sp = f"×{v / rows[0][1]:.1f} throughput" if q > 1 else "baseline"
        b.append(T(x + 18, TOP + th - 18, f"{sp}  ·  {math.floor(ms + .5)} ms median per request", 12, fg2, 500))   # half up, as in the card
    save("state_once", TOP + th + PAD, "".join(b))


# ---------------------------------------------------------------------------------------------------------- results (tiles + routing)
def results():
    tiles = [("Jev Decision Index 0.3", "36.9", "public suite, 140,620 requests"), ("OpenDecisions", "59.4", "Capability"),
             ("JevBench-Open", "55.1", "Capability"), ("Held-out de · fr · it", "+4.4", "points over frozen Apertus")]
    b = [head("Results", "Decision benchmarks, and confidence you can route on", "Accuracy on public sets outside the training pool: all items, and the most confident half of the answers")]
    tw = (W - 2 * PAD - 3 * GAP) / 4; th = 108
    for i, (k, v, lab) in enumerate(tiles):
        b.append(tile(PAD + i * (tw + GAP), TOP, tw, th, k, v, lab, hero=i == 0))
    sets = [("BoolQ", 88.9, 98.2), ("SIB-200 · 12 languages", 82.4, 97.4), ("PubMedQA", 76.1, 89.4), ("Belebele", 61.4, 87.3),
            ("HellaSwag", 71.5, 87.0), ("CLINC-150", 76.4, 86.7), ("BANKING77 · 77-way", 70.4, 85.3), ("TruthfulQA", 64.7, 85.0),
            ("MMLU", 60.4, 82.1), ("Climate-FEVER", 52.6, 71.7), ("WinoGrande", 62.9, 70.5)]
    cy = TOP + th + GAP; rh = 24; ch = 58 + rh * len(sets) + 30
    b.append(card(PAD, cy, W - 2 * PAD, ch))
    ly = cy + 30
    b.append(f'<circle cx="{PAD + 26}" cy="{ly - 4}" r="5" fill="{NEUTRAL}"/>' + T(PAD + 36, ly, "all items", 12, MUTED, 500))
    b.append(f'<circle cx="{PAD + 118}" cy="{ly - 4}" r="5" fill="{RED}"/>' + T(PAD + 128, ly, "most confident half: route these automatically, send the rest to a person", 12, MUTED, 500))
    top = cy + 66; x0, x1 = PAD + 250, W - PAD - 70; X = lambda v: x0 + (v - 50) / 50 * (x1 - x0)
    def spans(a, c):                                   # x ranges of the two value labels of a row
        return [(X(a) - 10 - textw(f"{a:.1f}", 11), X(a) - 8), (X(c) + 8, X(c) + 10 + textw(f"{c:.1f}", 12))]
    for v in (50, 60, 70, 80, 90, 100):
        for i, (lab, a, c) in enumerate(sets):            # the gridline pauses where a label sits on it
            if not any(x0 - 3 <= X(v) <= x1 + 3 for x0, x1 in spans(a, c)):
                b.append(f'<line x1="{X(v):.1f}" y1="{top + i * rh - rh / 2}" x2="{X(v):.1f}" y2="{top + i * rh + rh / 2}" stroke="{GRID}"/>')
        b.append(T(X(v), top + rh * len(sets) + 8, f"{v} %", 10.5, MUTED, 400, "middle"))
    for i, (lab, a, c) in enumerate(sets):
        y = top + i * rh
        b.append(T(PAD + 22, y + 4, lab, 12.5, INK, 500))
        b.append(f'<line x1="{X(a):.1f}" y1="{y}" x2="{X(c):.1f}" y2="{y}" stroke="{RED}" stroke-opacity=".28" stroke-width="4" stroke-linecap="round"/>')
        b.append(f'<circle cx="{X(a):.1f}" cy="{y}" r="5" fill="{NEUTRAL}"/><circle cx="{X(c):.1f}" cy="{y}" r="5.5" fill="{RED}"/>')
        b.append(T(X(c) + 10, y + 4, f"{c:.1f}", 12, INK, 700)); b.append(T(X(a) - 10, y + 4, f"{a:.1f}", 11, MUTED, 400, "end"))
    save("results", cy + ch + PAD, "".join(b))


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
    cbot = a_bottom + 52; ch = cbot - ctop
    b.append(card(PAD, ctop, c1w, ch) + step(PAD + 16, ctop + 32, 1, "The request as one prompt"))
    b += parts
    b.append(T(ax + 26, a_bottom + 16, "The prompt verbatim, system turn not shown. Options share position ids;", 10.5, MUTED))
    b.append(T(ax + 26, a_bottom + 30, "each ends with an end token, where the option is read out.", 10.5, MUTED))

    # 2 who reads whom
    c2x = PAD + c1w + GAP
    b.append(card(c2x, ctop, c2w, ch) + step(c2x + 16, ctop + 32, 2, "Who reads whom"))
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
    b.append(bar(lx, ly + 7, 9, 9, RED, 2) + T(lx + 14, ly + 15, "within a block: causal, or both ways", 10.5, MUTED))
    b.append(T(lx + 14, ly + 29, "in the 8 bidirectional layers", 10.5, MUTED))
    b.append(T(lx, ly + 50, "Options never read each other and nothing", 10.5, MUTED))
    b.append(T(lx, ly + 64, "reads them: each question stands alone.", 10.5, MUTED))
    b.append(T(lx, ly + 84, "Header + state are identical for every", 10.5, MUTED))
    b.append(T(lx, ly + 98, "question, so they are computed once.", 10.5, MUTED))

    # 3 one pass, one score per option: backbone column + scores aligned with the option rows
    b.append(card(c3x, ctop, c3w, ch) + step(c3x + 16, ctop + 32, 3, "One pass, one score per option"))
    bx, bw = c3x + 24, 40; by = in_top + 2; lh2 = (a_bottom - 2 - by) / 32
    b.append(bar(bx - 8, by - 8, bw + 16, 32 * lh2 + 14, OFF, 8))
    for L in range(32):
        bid = L in range(3, 32, 4)
        b.append(bar(bx, by + L * lh2 + (lh2 - 4) / 2, bw, 4, RED if bid else NEUTRAL, 2))
    b.append(T(bx + bw / 2, a_bottom + 16, "Apertus-8B", 10.5, MUTED, 600, "middle"))
    px = bx + bw + 36; pr = c3x + c3w - 20
    fy = in_top + 16 + (q_y[0] - 20 - in_top - 40) / 2       # formula block centred above the first scores
    b.append(rich(px, fy, [("z = ⟨q", ""), ("0", "sub"), (", W", ""), ("o", "sub"), (" · LN(h", ""), ("end", "sub"), (")⟩ / 16", "")], 13, INK, 600))
    b.append(T(px, fy + 18, "h: last-layer state at the option's end token", 10.5, MUTED))
    b.append(rich(px, fy + 33, [("q", ""), ("0", "sub"), (": learned 256-d query · W", ""), ("o", "sub"), (": 256 × 4096", "")], 10.5, MUTED))
    b.append(bar(bx - 8, a_bottom + 27, 9, 9, RED, 2) + T(bx + 6, a_bottom + 35, "bidirectional layers (8 of 32) · LoRA r16 in all 32 · nothing generated", 10.5, MUTED))
    zx = px + 46
    for qi in q_y:
        first = min(yc for qq, _, _, _, yc in opt_y if qq == qi)      # label just above the question's first option row
        b.append(T(px, first - 20, "z  →  softmax over the options", 11, MUTED, 600))
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
CARDS = [("Support ticket · 3 questions, one request", [["team"], ["refund", "urgency"]]), ("German · routing", [["team"]]),
         ("JSON state · arithmetic", [["total"], ["discount"]]), ("French · intent and language", [["action", "lang"]])]


def ex_card(x, y, w, title, state, rows, json_state, h_min=0):
    """one example card; cards of a grid row share a height (h_min) and their question rows line up"""
    b = [eyebrow(x + 20, y + 28, title)]; yy = y + 36
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
    grid = [(E[0], E[2]), (E[1], E[3])]                     # rows of the 2 x 2 grid; cards in a row share a height
    parts, y = [], TOP
    for row in grid:
        h = max(ex_card(0, 0, cw, *e)[1] for e in row)
        for c, e in enumerate(row): parts.append(ex_card(PAD + c * (cw + GAP), y, cw, *e, h_min=h)[0])
        y += h + GAP
    save("examples", y - GAP + PAD, head("Examples", "Real outputs", "Votum-8B as released (served temperatures); the answer carries the red bar") + "".join(parts))


# ---------------------------------------------------------------------------------------------------------- pool
def pool():
    steps = [("Pool used for Votum-8B", 405547), ("13 sources: licence review", -19950), ("bias_in_bios, LLM-generated sets", -27163),
             ("ContractNLI test split", -1982), ("benchmark overlap (8-gram)", -16903), ("portal / government terms", -11419),
             ("conflicting licence tags", -441), ("Votum Pool", 327689)]
    b = [head("Data", "From training pool to dataset", "Training rows removed at each step, and the licence mix of what remains")]
    tw = (W - 2 * PAD - 3 * GAP) / 4; th = 108
    for i, (k, v, lab) in enumerate([("Training rows", "327,689", "plus 2,797 validation rows"), ("Sources", "181", "164 public datasets, 17 collected"),
                                     ("Languages", "73", "language codes; about 38 % English"), ("Configs", "2", "default · share_alike")]):
        b.append(tile(PAD + i * (tw + GAP), TOP, tw, th, k, v, lab, hero=i == 0))
    cy = TOP + th + GAP; rh = 24; ch = 40 + rh * len(steps) + 76
    b.append(card(PAD, cy, W - 2 * PAD, ch))
    top = cy + 40; x0, x1 = PAD + 270, W - PAD - 130; X = lambda v: x0 + v / 405547 * (x1 - x0); cur = 0
    for i, (lab, v) in enumerate(steps):
        y = top + i * rh; end = i in (0, len(steps) - 1)
        b.append(T(PAD + 22, y + 4, lab, 12.5, INK, 650 if end else 400))
        if end:
            cur = v; b.append(bar(x0, y - 7, X(v) - x0, 14, RED if i else NEUTRAL, 4))
            b.append(pill(X(v) + 62, y, f"{v:,}", 11) if i else T(X(v) + 9, y + 4, f"{v:,}", 12.5, INK, 700))
        else:
            nxt = cur + v; b.append(bar(X(nxt), y - 7, X(cur) - X(nxt), 14, PINK, 4)); b.append(T(X(cur) + 9, y + 4, f"−{-v:,}", 11.5, MUTED, 500)); cur = nxt
    fams = [("CC-BY", 98626, FAM[0]), ("Apache / MIT / AFL", 64958, FAM[1]), ("CC0 / PDDL / PD", 56885, FAM[2]),
            ("other open", 3421, FAM[3]), ("CC-BY-SA / ODbL · share_alike", 103799, RED)]
    y0 = top + rh * len(steps) + 14; tot = sum(v for _, v, _ in fams); x = PAD + 22; ww_all = W - 2 * PAD - 44
    for lab, v, c in fams:
        ww = ww_all * v / tot; b.append(bar(x, y0, ww - 2, 12, c, 3)); x += ww
    x = PAD + 22
    for lab, v, c in fams:
        b.append(bar(x, y0 + 26, 10, 10, c, 2))
        b.append(f'<text x="{x + 15}" y="{y0 + 35}" font-size="11.5" fill="{INK}">{esc(lab)}<tspan dx="6" fill="{MUTED}">{v:,}</tspan></text>')
        x += 15 + textw(lab, 11.5) + 6 + textw(f"{v:,}", 11.5) + 26
    save("pool", cy + ch + PAD, "".join(b))


def flag():
    """assets/flag.svg: the Swiss flag (square; cross arms 1/6 longer than wide), vector and text-free, so it renders the
    same everywhere (a flag emoji needs a colour emoji font and shows as "CH" without one)."""
    (OUT / "flag.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
                                  f'<rect width="32" height="32" rx="3" fill="{SWISS}"/>'
                                  '<path d="M13 6h6v7h7v6h-7v7h-6v-7H6v-6h7z" fill="#fff"/></svg>\n')


flag()
for name in THEMES:
    theme(name); results(); state_once(); architecture(); examples(); pool()
print("written to", OUT)
