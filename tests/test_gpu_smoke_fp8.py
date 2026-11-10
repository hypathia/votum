"""The FP8 variant in PyTorch (VOTUM_REPO_FP8: Hub id or local directory), against the stored full-precision scores of
test_gpu_smoke.py. In a file of its own so that the full-precision model of that test is gone from the GPU first."""
import gc, json, os
from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu
HERE = Path(__file__).parent


def test_smoke_fp8():
    """the FP8 variant in PyTorch (VOTUM_REPO_FP8): its scores stay near the full-precision reference and every clear
    decision (top two more than one point apart) is the same; it holds its weights in FP8, about half the memory."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available(): pytest.skip("no CUDA GPU")
    if not os.environ.get("VOTUM_REPO_FP8"): pytest.skip("set VOTUM_REPO_FP8 (Hub id or local directory)")
    from votum import Votum, request as R
    gc.collect(); torch.cuda.empty_cache(); vm = Votum.from_pretrained(os.environ["VOTUM_REPO_FP8"])
    assert torch.cuda.memory_allocated() < 10 * 2**30
    reqs = json.loads((HERE / "fixtures" / "smoke_requests.json").read_text())
    exp = json.loads((HERE / "fixtures" / "smoke_expected.json").read_text())
    for req, e in zip(reqs, exp):
        rows = [R.to_row(req["state"], q, rid=qid) for qid, q in req["questions"].items()]
        for out, ref in zip(vm.decide_rows(rows, keys=[0] * len(rows)), e):
            top = sorted(ref["scores"], reverse=True)
            assert max(abs(a - b) for a, b in zip(out["scores"], ref["scores"])) < 1.5
            if top[0] - top[1] > 1.0:
                assert max(range(len(top)), key=ref["scores"].__getitem__) == max(range(len(top)), key=out["scores"].__getitem__)


def test_vision_fp8():
    """image requests on the FP8 variant: scores near the full-precision reference, the same clear decisions"""
    torch = pytest.importorskip("torch"); pytest.importorskip("torchvision")
    if not torch.cuda.is_available(): pytest.skip("no CUDA GPU")
    if not os.environ.get("VOTUM_REPO_FP8"): pytest.skip("set VOTUM_REPO_FP8 (Hub id or local directory)")
    from votum import Votum, request as R
    from test_gpu_smoke import with_paths
    gc.collect(); torch.cuda.empty_cache(); vm = Votum.from_pretrained(os.environ["VOTUM_REPO_FP8"])
    reqs = json.loads((HERE / "fixtures" / "vision_requests.json").read_text())
    exp = json.loads((HERE / "fixtures" / "vision_expected.json").read_text())
    for req, e in zip(reqs, exp):
        rows = R.request_rows(dict(req, state=with_paths(req["state"], HERE / "fixtures")), vm.images.encode)
        for out, ref in zip(vm.decide_rows(rows, keys=[0] * len(rows)), e["answers"]):
            top = sorted(ref["scores"], reverse=True)
            assert max(abs(a - b) for a, b in zip(out["scores"], ref["scores"])) < 1.5
            if top[0] - top[1] > 1.0:
                assert max(range(len(top)), key=ref["scores"].__getitem__) == max(range(len(top)), key=out["scores"].__getitem__)
