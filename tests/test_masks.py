"""The reference attention masks and positions (votum.modeling / votum.prompt) equal the vLLM layout (votum.vllm.isolate)
on random single-question layouts. CPU only; no weights, no vLLM."""
import random

import numpy as np
import pytest
import torch

from votum import prompt as P
from votum.config import VotumConfig
from votum.modeling import attention_masks
from votum.vllm import isolate

END, OPT_HDR, Q_HDR = 60, [10140, 1877], [25229, 1877]


def make(rng):
    """random token ids laid out as [system+header][State..][Question:..][Options:\\n][- text END \\n]..."""
    H = rng.randint(3, 12); S = H + rng.randint(3, 20)
    ids = [rng.randint(100, 9000) for _ in range(H)] + [rng.randint(100, 9000) for _ in range(S - H - 2)]
    ids += Q_HDR; qstart = S                                        # "Question", ":\n" open the question block
    ids += [rng.randint(100, 9000) for _ in range(rng.randint(2, 9))] + OPT_HDR; ostart = len(ids)
    ends = []
    for _ in range(rng.randint(2, 6)):
        ids += [rng.randint(100, 9000) for _ in range(rng.randint(1, 7))] + [END]; ends.append(len(ids) - 1); ids.append(1010)
    return P.Encoding(ids=ids, ends=ends, last=len(ids) - 1, qstart=qstart - 2, ostart=ostart, hend=H)


@pytest.mark.parametrize("seed", range(20))
def test_reference_equals_vllm_layout(seed):
    rng = random.Random(seed); e = make(rng); cfg = VotumConfig()
    prompts, _, _, _ = isolate.build_prompts([e], False, 8192, None)
    (ids, lay), = prompts; key = isolate.register(ids, lay)
    try:
        st = isolate.batch_layout(ids, [0, len(ids)], END, True, OPT_HDR, Q_HDR, cfg.close_last)
    finally:
        isolate.unregister([key])
    assert st["pos"].tolist() == P.shared_positions(e, cfg.close_last)
    stt = {k: (torch.from_numpy(v) if isinstance(v, np.ndarray) else v) for k, v in st.items()}
    c, b = attention_masks([e], cfg, len(ids), "cpu")
    assert torch.equal(c[0, 0], isolate.dense_mask(stt, False))
    assert torch.equal(b[0, 0], isolate.dense_mask(stt, True))


def test_mask_rules():
    rng = random.Random(0); e = make(rng); cfg = VotumConfig(); T = len(e.ids)
    c, b = attention_masks([e], cfg, T, "cpu"); c, b = c[0, 0], b[0, 0]
    blk = torch.tensor(P.option_blocks(e, True)); i = torch.arange(T)
    assert not (c & (i[None] > i[:, None])).any()                                    # causal layers never look ahead
    cross = (blk[:, None] >= 0) & (blk[None] >= 0) & (blk[:, None] != blk[None])
    assert not (b & cross).any() and not (c & cross).any()                           # options never see each other
    assert not b[(blk < 0)][:, blk >= 0].any()                                       # nothing outside the options reads them
    assert not b[:e.qstart, e.qstart:].any()                                          # the state never reads the question
    assert not b[:e.hend, e.hend:].any()                                              # system + header never read the state
    assert (blk[e.ends[-1] + 1] == len(e.ends) - 1)                                  # close_last: the final line break is in the last block


def test_padding_rows_are_finite():
    rng = random.Random(1); e1, e2 = make(rng), make(rng); T = max(len(e1.ids), len(e2.ids)) + 3
    c, b = attention_masks([e1, e2], VotumConfig(), T, "cpu")
    assert c.any(-1).all() and b.any(-1).all()                                       # every query row sees at least one key
