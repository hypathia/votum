"""Requests and answers in the TypeSafe / SystemOne shape (the shape of the HTTP API and of the OpenDecisions and JevBench
benchmarks).

request   {"state": str | object, "questions": {qid: {"type": "noul"|"choice"|"score", "instructions": str | object,
                                                       "criteria": ...}}}
          criteria: noul -> optional {"true": str, "false": str}; choice -> {option_id: description} (or a list of ids);
          score -> [level description, ...] from low to high.
answers   noul   {"type": "noul", "noul": P(true)}
          choice {"type": "choice", "choice": id, "confidence": p, "probabilities": {id: p}}
          score  {"type": "score", "score": expected level, "confidence": p, "legend": {level: text}, "probabilities": {level: p}}
Option conversion follows JevBench's semif_direct adapter (github.com/j-geuter/Jeff, Apache-2.0)."""
from __future__ import annotations

import json

from .prompt import state_text

QUESTION_TYPES = ("noul", "choice", "score")


def question_options(q: dict) -> list[dict]:
    """[{"id", "text"}] of one TypeSafe question."""
    t, crit = q["type"], q.get("criteria")
    if t == "noul":
        return [{"id": k, "text": (crit or {}).get(k, f"The proposition is {k}.")} for k in ("true", "false")]
    if t == "choice":
        if isinstance(crit, dict): return [{"id": str(k), "text": v or str(k)} for k, v in crit.items()]
        return [{"id": str(k), "text": str(k)} for k in crit]
    if t == "score":
        return [{"id": str(i), "text": lvl} for i, lvl in enumerate(crit)]
    raise ValueError(f"unknown question type {t!r}")


def as_text(q: dict) -> dict:
    """structured instructions / option descriptions -> text, by the same rule as a structured state."""
    crit = q.get("criteria")
    if isinstance(crit, dict): crit = {k: v if isinstance(v, str) or v is None else state_text(v) for k, v in crit.items()}
    elif isinstance(crit, list): crit = [v if isinstance(v, str) else state_text(v) for v in crit]
    out = {**q, "instructions": state_text(q["instructions"])}
    if "criteria" in q: out["criteria"] = crit
    return out


def to_row(state, question: dict, rid: str = "") -> dict:
    q = as_text(question)
    return {"id": rid, "state": state, "type": q["type"], "question": q["instructions"], "options": question_options(q)}


def _unicode_ok(x) -> bool:
    """False for text that cannot be encoded (a lone surrogate from a "\\ud800" JSON escape), anywhere in x."""
    try: json.dumps(x, ensure_ascii=False).encode("utf-8"); return True
    except (UnicodeEncodeError, TypeError, ValueError): return False


def validate(req, max_questions: int | None = None, max_options: int | None = None) -> str | None:
    """error message for an invalid request, else None. max_questions / max_options: optional caps (the HTTP server sets
    them); the shape checks always apply."""
    if not isinstance(req, dict) or "state" not in req: return "state is required"
    if not _unicode_ok(req["state"]): return "state: not valid Unicode text or JSON"
    qs = req.get("questions")
    if not isinstance(qs, dict) or not qs: return "at least one question is required"
    if max_questions is not None and len(qs) > max_questions: return f"too many questions ({len(qs)}; this server accepts {max_questions} per request)"
    for qid, q in qs.items():
        if not isinstance(q, dict) or q.get("type") not in QUESTION_TYPES: return f"{qid}: type must be noul, choice, or score"
        if not _unicode_ok(q): return f"{qid}: not valid Unicode text or JSON"
        ins = q.get("instructions")
        if not (isinstance(ins, str) and ins.strip()) and not (isinstance(ins, (dict, list)) and ins): return f"{qid}: instructions are required"
        t, crit = q["type"], q.get("criteria")
        if t == "noul":
            if crit is None: continue
            if not isinstance(crit, dict) or set(crit) - {"true", "false"}: return f'{qid}: noul criteria must be an object with the keys "true" and "false" (or absent)'
            continue
        if isinstance(crit, list): ids = [str(k) if isinstance(k, (str, int, float)) and not isinstance(k, bool) else None for k in crit] if t == "choice" else list(range(len(crit)))
        elif isinstance(crit, dict) and t == "choice": ids = [str(k) for k in crit]
        else: return f"{qid}: {'choice criteria must be an object {option_id: text} or a list of option ids' if t == 'choice' else 'score criteria must be a list of levels'}"
        if len(ids) < 2: return f"{qid}: at least two {'options' if t == 'choice' else 'levels'} are required"
        if None in ids: return f"{qid}: option ids must be strings or numbers"
        if len(set(ids)) != len(ids): return f"{qid}: option ids must be unique"
        if max_options is not None and len(ids) > max_options: return f"{qid}: too many options ({len(ids)}; this server accepts {max_options} per question)"
    return None


def decide(request: dict, decide_rows, max_tokens: int) -> dict:
    """one request -> {"answers": {qid: answer}, "usage": {...}}, with decide_rows(rows, keys) of either implementation."""
    if (msg := validate(request)): raise ValueError(msg)
    qs = request["questions"]
    res = decide_rows([to_row(request["state"], q, rid=str(qid)) for qid, q in qs.items()], [0] * len(qs))
    answers, used = {}, 0
    for (qid, q), r in zip(qs.items(), res):
        if r is None: raise ValueError(f"{qid}: prompt longer than {max_tokens} tokens or not renderable")
        answers[qid] = answer(q, r["probs"]); used += r["n_tokens"]
    return {"answers": answers, "usage": {"input_tokens": used, "output_tokens": 0}}


def answer(question: dict, probs: dict[str, float]) -> dict:
    t = question["type"]
    if t == "noul":
        return {"type": "noul", "noul": round(probs["true"], 4)}
    if t == "choice":
        ids = list(probs); best = max(ids, key=probs.__getitem__)
        return {"type": "choice", "choice": best, "confidence": round(probs[best], 4), "probabilities": {i: round(probs[i], 4) for i in ids}}
    levels = [str(i) for i in range(len(question["criteria"]))]
    return {"type": "score", "score": round(sum(i * probs[lvl] for i, lvl in enumerate(levels)), 4),
            "confidence": round(max(probs[lvl] for lvl in levels), 4), "legend": dict(zip(levels, question["criteria"])),
            "probabilities": {lvl: round(probs[lvl], 4) for lvl in levels}}
