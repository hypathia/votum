"""End-to-end check of an installation: load Votum from the Hub (or a local copy in VOTUM_REPO) and compare
the raw option scores of a few hand-written requests with stored reference scores (fixtures/smoke_expected.json,
produced by the PyTorch reference (merged weights) on an RTX 4090 in bf16). Differences of a few hundredths are bf16 kernel noise; the test tolerates up to 0.5 on a raw score and requires the same top option."""
import json, os
from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu
HERE = Path(__file__).parent


@pytest.fixture(scope="module")
def vm():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available(): pytest.skip("no CUDA GPU")
    if not os.environ.get("VOTUM_REPO"): pytest.skip("set VOTUM_REPO (Hub id or local directory)")
    from votum import Votum
    return Votum.from_pretrained(os.environ["VOTUM_REPO"])


def test_smoke_scores(vm):
    reqs = json.loads((HERE / "fixtures" / "smoke_requests.json").read_text())
    exp = json.loads((HERE / "fixtures" / "smoke_expected.json").read_text())
    from votum import request as R
    worst = 0.0
    for req, e in zip(reqs, exp):
        rows = [R.to_row(req["state"], q, rid=qid) for qid, q in req["questions"].items()]
        for r, out, ref in zip(rows, vm.decide_rows(rows, keys=[0] * len(rows)), e):
            assert out["ids"] == ref["ids"]
            worst = max(worst, max(abs(a - b) for a, b in zip(out["scores"], ref["scores"])))
            assert max(range(len(ref["scores"])), key=ref["scores"].__getitem__) == max(range(len(out["scores"])), key=out["scores"].__getitem__)
    assert worst < 0.5, worst

