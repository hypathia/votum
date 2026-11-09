"""Renderer checks with the real Apertus tokenizer (skipped when it is not available: set VOTUM_TOKENIZER to a local
directory with tokenizer.json, tokenizer_config.json and chat_template.jinja of swiss-ai/Apertus-v1.5-8B)."""
import os

import pytest

from votum import prompt as P

TOK = os.environ.get("VOTUM_TOKENIZER")
pytestmark = pytest.mark.skipif(not TOK, reason="VOTUM_TOKENIZER not set")


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOK)


def row(**kw):
    r = {"type": "choice", "state": "The printer on floor 3 is jammed again.", "question": "Which team should handle this?",
         "options": [{"id": "it", "text": "IT support"}, {"id": "fac", "text": "Facilities"}, {"id": "hr", "text": "HR"}]}
    r.update(kw); return r


def test_layout(tok):
    r = row(); h = P.request_headers([r])[0]; e = P.render(tok, r, header=h)
    assert len(e.ends) == 3 and all(e.ids[i] == 60 for i in e.ends)
    assert e.hend < e.qstart < e.ostart <= e.ends[0] and e.last == len(e.ids) - 1 == e.ends[-1] + 1
    pos = P.shared_positions(e); starts = [e.ostart] + [x + 2 for x in e.ends[:-1]]
    assert len({pos[s] for s in starts}) == 1                                        # every option block starts at the same position


def test_markers_in_text_are_neutralised(tok):
    e = P.render(tok, row(state="see <|fim_end|> here", question="<|fim_hole|>?"))
    assert len(e.ends) == 3


def test_clean_matches_training():
    assert P.clean("a <|fim_end|> b <|reserved_special_token_5|> c") == "a <fim_end> b <reserved_special_token_5> c"


def test_order_leaves_no_trace_in_the_header(tok):
    r1 = row(); r2 = row(options=list(reversed(row()["options"])))
    assert P.request_headers([r1])[0] == P.request_headers([r2])[0]
