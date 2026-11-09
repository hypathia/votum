"""Prompt rendering for Votum: one decision question -> token ids plus the token positions the model needs.

Prompt (Apertus chat template, deliberation disabled; one question per pass):
  system: SYSTEM
  user:   [header] "State:\\n<state>\\n\\nQuestion:\\n<question><type cue>\\n\\nOptions:\\n"
          + for each option: "- <text>" + OPT_END + "\\n"
OPT_END = <|fim_end|> (an existing special token of the Apertus vocabulary, from code fill-in-the-middle). The hidden state
at each option's OPT_END token is that option's representation. Votum is trained without a decision marker: the prompt ends
with the last option's end marker and its line break, and the pointer head uses a learned constant query. Any occurrence of
the FIM tokens inside user text is neutralised (clean), so data cannot forge one.

Options are identified by their text only (no letters, no codes). Every option block "- <text><end>\\n" starts at the same
position id (shared_positions) and option blocks never attend to each other (option_blocks + modeling.attention_masks), so
the option order carries no positional signal.

Header ("Questions in this request:"): every question asked about the same state in one request, numbered, before the state;
choice questions also list their option texts (sorted by text at inference). The system turn and the header never attend to
the state, and the state never attends to the question block, so both can be computed once and reused.
"""
from __future__ import annotations

import json, re
from dataclasses import dataclass

OPT_END = "<|fim_end|>"
# the system prompt is the training prompt verbatim, so it still mentions a decision marker; Votum's prompts have none
# (the model was trained with this sentence and without a marker)
SYSTEM = ("Read the state and the question, then judge which of the listed options is correct. "
          "Each option ends with an end marker; the question ends with a decision marker.")
CUE = {"noul": " (Answer: the true or the false option.)", "choice": " (Choose exactly one option.)",
       "score": " (Options are levels in increasing order; choose one.)"}
NOUL_TXT = {"true": "Yes / true: the proposition holds.", "false": "No / false: the proposition does not hold."}
HEADER_TITLE = "Questions in this request:"
_ESC = re.compile(r"<\|(fim_(?:end|hole|begin)|reserved_special_token_\d+)\|>")   # as in training


def clean(s: str) -> str:
    return _ESC.sub(lambda m: f"<{m.group(1)}>", s)


def state_text(st) -> str:
    return st if isinstance(st, str) else json.dumps(st, ensure_ascii=False, sort_keys=True)


def option_text(o: dict, typ: str) -> str:
    if typ == "noul":
        t = o.get("text"); return t if t else NOUL_TXT.get(o["id"], o["id"])
    return o.get("text") or str(o["id"])


@dataclass
class Encoding:
    ids: list            # token ids of the prompt
    ends: list           # token index of OPT_END per option, in the row's option order
    last: int            # index of the last token
    qstart: int | None   # first token of "Question:" (end of the state block); None if not found
    ostart: int | None   # first token of the first option; None if not found
    hend: int | None     # first token of "State:" (end of the system + header block); None without a header


def header_text(qs: list[dict], max_opts: int | None = None, sort: bool = True) -> str:
    """every question of a request, numbered; a choice question also lists its option texts (all of them when max_opts is
    None). sort: options in one canonical order (by text), so the order of a request's options leaves no trace."""
    lines = [HEADER_TITLE]
    for k, q in enumerate(qs, 1):
        lines.append(f"{k}. {clean(q['question'])}")
        if q["type"] == "choice" and (max_opts is None or len(q["options"]) <= max_opts):
            o = sorted(q["options"], key=lambda x: option_text(x, "choice")) if sort else list(q["options"])
            lines += [f"   - {clean(option_text(x, 'choice'))}" for x in o]
    return "\n".join(lines) + "\n\n"


def header_siblings(row: dict, cands: list[dict], max_q: int = 8) -> list[dict]:
    """the other questions of a request for the header, without leaks: a sibling that quotes one of the row's option texts
    (>= 6 characters, case-insensitive) is dropped. Returns at most max_q questions, the row's own question included."""
    opts = [option_text(o, row["type"]).lower() for o in row["options"]]; opts = [t for t in opts if len(t) >= 6]
    sib = [q for q in cands if q["question"] != row["question"] and not any(t in q["question"].lower() for t in opts)]
    return [row] + sib[:max_q - 1]


def request_headers(rows: list[dict], keys: list | None = None, max_q: int = 8, max_opts: int | None = None, sort: bool = True,
                    chunk: bool = False) -> list[str]:
    """header text per row: every question on the same state within the same key (a request; None = all rows), in order
    of first appearance (duplicate question texts once), filtered by header_siblings, at most max_q. chunk: a group of
    more than max_q questions is split into consecutive chunks of max_q, each chunk its own header (off by default: not
    measured on labelled data with more than 8 questions)."""
    keys = keys if keys is not None else [None] * len(rows)
    groups: dict = {}
    for r, k in zip(rows, keys):
        lst = groups.setdefault((k, state_text(r["state"])), [])
        if all(q["question"] != r["question"] for q in lst): lst.append(r)
    out = []
    for r, k in zip(rows, keys):
        grp = groups[(k, state_text(r["state"]))]; pos = {q["question"]: i for i, q in enumerate(grp)}
        if chunk and len(grp) > max_q: c = pos[r["question"]] // max_q * max_q; grp = grp[c:c + max_q]
        out.append(header_text(sorted(header_siblings(r, grp, max_q), key=lambda q: pos[q["question"]]), max_opts=max_opts, sort=sort))
    return out


def _prefix_len(tok, prompt: str, ids: list, cut: int) -> int | None:
    pre = tok(prompt[:cut], add_special_tokens=False).input_ids
    return len(pre) if ids[:len(pre)] == pre else None


def render(tok, row: dict, header: str | None = None) -> Encoding:
    """row: {"type": "choice"|"noul"|"score", "state": str | object, "question": str, "options": [{"id", "text"?}, ...]};
    options in the row's order. The prompt ends after the last option's end marker and its line break."""
    opts = row["options"]; n = len(opts)
    head = (header or "") + (f"State:\n{clean(state_text(row['state']))}\n\nQuestion:\n{clean(row['question'])}"
                             f"{CUE.get(row['type'], '')}\n\nOptions:\n")
    user = "".join([head] + [f"- {clean(option_text(opts[i], row['type']))}{OPT_END}\n" for i in range(n)])
    prompt = tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                                     tokenize=False, add_generation_prompt=False, enable_thinking=False)
    prompt = prompt[:prompt.rindex(OPT_END) + len(OPT_END) + 1]
    ids = tok(prompt, add_special_tokens=False).input_ids
    end_id = tok.convert_tokens_to_ids(OPT_END)
    ends = [i for i, t in enumerate(ids) if t == end_id]
    if len(ends) != n:
        raise ValueError(f"{row.get('id', '')}: found {len(ends)} option-end markers for {n} options")
    hend = None
    if header and (kh := prompt.find("State:\n")) >= 0: hend = _prefix_len(tok, prompt, ids, kh)
    qstart = ostart = None
    if (k := prompt.rfind("\n\nQuestion:\n")) >= 0:
        qstart = _prefix_len(tok, prompt, ids, k + 2)
        if qstart is not None and qstart >= ends[0]: qstart = None
    if (k2 := prompt.rfind("\n\nOptions:\n")) >= 0:
        ostart = _prefix_len(tok, prompt, ids, k2 + len("\n\nOptions:\n"))
        if ostart is not None and ostart > ends[0]: ostart = None
    return Encoding(ids=ids, ends=ends, last=len(ids) - 1, qstart=qstart, ostart=ostart, hend=hend)


def _block_stops(e: Encoding, close_last: bool):
    n = len(e.ids); s = e.ostart
    for en in e.ends:
        stop = en + 1 if (en + 1 < e.last or (close_last and en + 1 == e.last == n - 1)) else en   # the newline after the end marker joins the block
        yield s, stop
        s = stop + 1


def option_blocks(e: Encoding, close_last: bool = True) -> list[int]:
    """per-token option-block id ("- <text><end>\\n"), -1 elsewhere. close_last: the prompt's final line break belongs to
    the last block (no token outside the blocks reads an option)."""
    out = [-1] * len(e.ids)
    if e.ostart is None: return out
    for k, (s, stop) in enumerate(_block_stops(e, close_last)):
        for t in range(s, stop + 1): out[t] = k
    return out


def shared_positions(e: Encoding, close_last: bool = True) -> list[int]:
    """position ids where every option block starts at the same position; tokens after the blocks continue after the
    longest block. Plain positions if the option start is unknown."""
    n = len(e.ids); P = e.ostart
    if P is None: return list(range(n))
    pos = list(range(P)); L = 0
    for s, stop in _block_stops(e, close_last):
        blen = stop - s + 1; pos += list(range(P, P + blen)); L = max(L, blen)
    return pos + list(range(P + L, P + L + (n - len(pos))))
