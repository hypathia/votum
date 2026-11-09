"""Pointer ("span") readout: no letter codes, unbounded option count.

Prompt (Apertus chat template, deliberation disabled; one question per pass):
  system: SPAN_SYSTEM (the wording still mentions a decision marker; it is the training prompt, kept verbatim)
  user:   [header] "State:\n<state>\n\nQuestion:\n<question><type cue>\n\nOptions:\n" + for each option: "- <text>" + OPT_END + "\n"
The prompt ends after the last option's end marker and its line break. OPT_END = <|fim_end|>, an existing special token of
the Apertus vocabulary (code fill-in-the-middle); FIM and reserved special tokens inside data text are neutralised. The
hidden state at each option's OPT_END is the option's representation, scored against a learned constant query:
  score_j = <q0, W_o LN(h(end_j))> / sqrt(r),  r = 256;  softmax over the question's options.
Options are identified by their text only. Training shuffles the option order every step; evaluation uses the row's
canonical order."""
import json, math, re
from dataclasses import dataclass, field
import torch, torch.nn as nn

OPT_END = "<|fim_end|>"
SPAN_SYSTEM = ("Read the state and the question, then judge which of the listed options is correct. "
               "Each option ends with an end marker; the question ends with a decision marker.")
CUE = {"noul": " (Answer: the true or the false option.)", "choice": " (Choose exactly one option.)",
       "score": " (Options are levels in increasing order; choose one.)"}
NOUL_TXT = {"true": "Yes / true: the proposition holds.", "false": "No / false: the proposition does not hold."}
_ESC = re.compile(r"<\|([^|<>\n]{1,64})\|?>")   # every <|name|> token: option end, chat roles, reserved, image / audio
# special tokens without the <|...|> shape, and the strings the tokenizer's normalizer rewrites into control tokens
_ESC2 = re.compile(r"(<(?:/?(?:s|unk|pad|answer|think|thought)|audio|image|channel\||filename|gh_stars|reponame|file_sep|empty_output"
                   r"|issue_\w+|jupyter_\w+|commit_\w+|(?:iban|email|ip)-pii|code_to_intermediate|intermediate_to_code|pr(?:_\w+)?"
                   r"|SPECIAL_\d+)|\[/?(?:INST|AVAILABLE_TOOLS|TOOL_RESULTS|TOOL_CALLS|PREFIX|MIDDLE|SUFFIX))([>\]])")


def clean(s):
    """the same rule as votum.prompt.clean (tests/test_prompt.py compares them): user text never produces a special token."""
    return _ESC2.sub(r"\1 \2", _ESC.sub(lambda m: f"<{m.group(1)}>", s))


def state_text(st):
    return st if isinstance(st, str) else json.dumps(st, ensure_ascii=False, sort_keys=True)


@dataclass
class SpanEnc:
    ids: list
    order: list        # display slot k shows row option order[k]
    ends: list         # token index of OPT_END per display slot
    last: int          # index of the last token (the line break after the last option)
    extra: dict = field(default_factory=dict)


def option_text(o, typ):
    if typ == "noul":
        t = o.get("text"); return t if t else NOUL_TXT.get(o["id"], o["id"])
    return o.get("text") or str(o["id"])


HEADER_TITLE = "Questions in this request:"
HEADER_MAX_OPTS = 32      # a choice question lists its option texts in the header only up to this many (larger label sets: question only)


def header_text(qs, rng=None):
    """Header layout: every question of the request, numbered, before the state; a choice question also
    lists its option texts (no codes; order shuffled with rng so the header carries no option-order signal). qs: rows or
    dicts with question / type / options."""
    lines = [HEADER_TITLE]
    for k, q in enumerate(qs, 1):
        lines.append(f"{k}. {clean(q['question'])}")
        if q["type"] == "choice" and len(q["options"]) <= HEADER_MAX_OPTS:
            o = list(q["options"])
            if rng is not None: rng.shuffle(o)
            lines += [f"   - {clean(option_text(x, 'choice'))}" for x in o]
    return "\n".join(lines) + "\n\n"


def header_siblings(row, cands, max_q=8, rng=None):
    """The other questions of a request for the header, without leaks: a sibling that quotes one of the row's option texts
    (>= 6 characters, case-insensitive; e.g. "Response 2 is preferred. How much better ...") is dropped. Returns the
    question list with the row's own question included (random order with rng, else own first)."""
    opts = [option_text(o, row["type"]).lower() for o in row["options"]]; opts = [t for t in opts if len(t) >= 6]
    sib = [q for q in cands if q["question"] != row["question"] and not any(t in q["question"].lower() for t in opts)]
    if rng is not None: rng.shuffle(sib)
    qs = [row] + sib[:max_q - 1]
    if rng is not None: rng.shuffle(qs)
    return qs


def render_span(tok, row, rng=None, shuffle=False, header=None):
    """header: None, or the header text (header_text) put before "State:"; the state block is then everything before
    extra["qstart"] (bidigate.set_state_ends keeps it from looking ahead at the question block)."""
    opts = row["options"]; n = len(opts); order = list(range(n))
    if shuffle and rng is not None and row["type"] != "score": rng.shuffle(order)
    head = (header or "") + (f"State:\n{clean(state_text(row['state']))}\n\nQuestion:\n{clean(row['question'])}{CUE.get(row['type'], '')}\n\nOptions:\n")
    END = OPT_END
    pieces = [head] + [f"- {clean(option_text(opts[i], row['type']))}{END}\n" for i in order]
    user = "".join(pieces)
    prompt = tok.apply_chat_template([{"role": "system", "content": SPAN_SYSTEM}, {"role": "user", "content": user}],
                                     tokenize=False, add_generation_prompt=False, enable_thinking=False, reasoning_strength="low")
    prompt = prompt[:prompt.rindex(END) + len(END) + 1]   # ends with the last option's end marker + newline
    ids = tok(prompt, add_special_tokens=False).input_ids
    end_id = tok.convert_tokens_to_ids(END)
    ends = [i for i, t in enumerate(ids) if t == end_id]
    if len(ends) != n: raise ValueError(f"{row.get('id')}: {len(ends)} end markers for {n} options")
    hend = None   # header layout: first token of "State:" (system + header before it); None without a header or if it does not tokenize identically
    if header:
        kh = prompt.find("State:\n")
        if kh >= 0:
            preh = tok(prompt[:kh], add_special_tokens=False).input_ids
            if ids[:len(preh)] == preh: hend = len(preh)
    k = prompt.rfind("\n\nQuestion:\n"); qstart = None
    if k >= 0:
        pre = tok(prompt[:k + 2], add_special_tokens=False).input_ids
        if ids[:len(pre)] == pre and len(pre) < ends[0]: qstart = len(pre)
    # first option token (after "Options:\n"), for shared option positions; None if the prefix does not tokenize identically
    k2 = prompt.rfind("\n\nOptions:\n"); ostart = None
    if k2 >= 0:
        pre2 = tok(prompt[:k2 + len("\n\nOptions:\n")], add_special_tokens=False).input_ids
        if ids[:len(pre2)] == pre2 and len(pre2) <= ends[0]: ostart = len(pre2)
    return SpanEnc(ids=ids, order=order, ends=ends, last=len(ids) - 1, extra={"qstart": qstart, "ostart": ostart, "hend": hend})


def option_blocks(e):
    """per-token option-block id (same blocks as shared_positions: "- <text><end>\\n"), -1 elsewhere."""
    n = len(e.ids); out = [-1] * n; P = e.extra.get("ostart")
    if P is None: return out
    s = P
    for k, en in enumerate(e.ends):
        stop = en + 1 if en + 1 < e.last else en
        for t in range(s, stop + 1): out[t] = k
        s = stop + 1
    return out


def shared_positions(e):
    """position ids where every option block starts at the same position (order carries no positional signal):
    prefix 0..P-1; option block k (from its first token through its end marker) P..P+len_k-1; the tail (the final line
    break) continues after the longest block. Falls back to 0..n-1 if the option start is unknown."""
    n = len(e.ids); P = e.extra.get("ostart")
    if P is None: return list(range(n))
    pos = list(range(P)); s = P; L = 0
    for en in e.ends:                                   # block = "- <text><end>\n" (the newline after each end marker is its own token)
        stop = en + 1 if en + 1 < e.last else en        # include the newline, except the prompt's final one
        blen = stop - s + 1; pos += list(range(P, P + blen)); L = max(L, blen); s = stop + 1
    pos += list(range(P + L, P + L + (n - s)))
    return pos


class SpanHead(nn.Module):
    def __init__(self, d, r=256):
        super().__init__()
        # q is not used by the constant-query readout; it is created (and initialised) first so that the random-number
        # sequence and the span_head.pt keys match the released checkpoint
        self.q = nn.Linear(d, r, bias=False); self.o = nn.Linear(d, r, bias=False); self.r = r
        nn.init.normal_(self.q.weight, std=0.02 / math.sqrt(d) * math.sqrt(d) / 8); nn.init.normal_(self.o.weight, std=0.02 / math.sqrt(d) * math.sqrt(d) / 8)
        self.q0 = nn.Parameter(torch.randn(r) * 0.16)   # learned constant query (same scale as W_q LN(h) at init)
        self.norm = nn.LayerNorm(d, elementwise_affine=False)

    def forward(self, h, encs):
        """h [B,T,d] (bf16) -> list of fp32 logit vectors, one per row, in display order."""
        out = []
        for b, e in enumerate(encs):
            hb = self.norm(h[b].float())
            ov = self.o(hb[torch.tensor(e.ends, device=h.device)])
            out.append(ov @ self.q0 / math.sqrt(self.r))
        return out
