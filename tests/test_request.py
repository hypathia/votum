from votum import request as R


def test_options_and_answers():
    q = {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Billing questions", "tech": None}}
    row = R.to_row("state", q, "q1")
    assert row["options"] == [{"id": "billing", "text": "Billing questions"}, {"id": "tech", "text": "tech"}]
    a = R.answer(q, {"billing": 0.8, "tech": 0.2}); assert a["choice"] == "billing" and a["confidence"] == 0.8
    yn = {"type": "noul", "instructions": "Is it urgent?"}
    assert [o["text"] for o in R.to_row("s", yn)["options"]] == ["The proposition is true.", "The proposition is false."]
    assert R.answer(yn, {"true": 0.3, "false": 0.7}) == {"type": "noul", "noul": 0.3}
    sc = {"type": "score", "instructions": "How severe?", "criteria": ["low", "mid", "high"]}
    a = R.answer(sc, {"0": 0.2, "1": 0.3, "2": 0.5}); assert a["score"] == 1.3 and a["legend"] == {"0": "low", "1": "mid", "2": "high"}


def test_validate():
    assert R.validate({"state": "x", "questions": {}}) is not None
    assert R.validate({"state": "x", "questions": {"a": {"type": "choice", "instructions": "?", "criteria": {}}}}) is not None
    assert R.validate({"state": "x", "questions": {"a": {"type": "noul", "instructions": "?"}}}) is None
    assert R.as_text({"type": "noul", "instructions": {"ask": "x"}})["instructions"] == '{"ask": "x"}'


def test_validate_shapes():
    q = lambda **kw: {"state": "x", "questions": {"a": {"type": "choice", "instructions": "?", "criteria": ["a", "b"], **kw}}}
    assert R.validate(q()) is None and R.validate(q(criteria={"a": "A", "b": None})) is None
    for bad in ("abc", 5, ["a"], ["a", "a"], [1, "1"], {"a": "only"}): assert R.validate(q(criteria=bad)) is not None, bad
    assert R.validate(q(type="noul", criteria=["a"])) is not None and R.validate(q(type="noul", criteria={"true": "yes"})) is None
    assert R.validate(q(type="score", criteria=["low"])) is not None and R.validate(q(type="score", criteria=["low", "high"])) is None
    assert R.validate({"state": "\ud800", "questions": q()["questions"]}) is not None          # a lone surrogate cannot be tokenized
    assert R.validate(q(), max_questions=1, max_options=2) is None and R.validate(q(), max_options=1) is not None
