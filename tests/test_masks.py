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


def test_same_token_prompts_never_share_an_engine_call():
    """two prompts with the same token ids and different layouts (their header or state came from the cache) must not be
    registered together: the registry is keyed by the ids, and the second layout would replace the first."""
    a, b, c = [5, 6, 7, 8], [5, 6, 7, 8], [9, 9, 9]
    lay = [{"H": 3, "h": b"H1"}, {"H": 5, "h": b"H2"}, {"H": 0, "h": None}]
    prompts = [a, c, b, a]; layouts = [lay[0], lay[2], lay[1], lay[0]]
    ws = isolate.waves(prompts)
    assert sorted(j for w in ws for j in w) == [0, 1, 2, 3] and ws[0] == [0, 1] and ws[1] == [2] and ws[2] == [3]
    for w in ws:                                             # what the client does per call
        assert len({isolate.digest(prompts[j]) for j in w}) == len(w)
        keys = [isolate.register(prompts[j], layouts[j]) for j in w]
        for j in w: assert isolate.lookup(prompts[j]) is layouts[j]
        isolate.unregister(keys)
    assert isolate.lookup(a) is None and isolate.waves([]) == [] and isolate.waves([a]) == [[0]]


def test_engine_calls_are_bounded_in_tokens():
    """a call holds at most the token budget; a prompt longer than the budget goes alone; order is kept; 0 = no bound."""
    prompts = [[i] * n for i, n in enumerate([5, 4, 3, 20, 2, 2])]
    assert isolate.waves(prompts, 10) == [[0, 1], [2], [3], [4, 5]]
    assert isolate.waves(prompts, 0) == [list(range(6))] and isolate.waves(prompts, 10 ** 9) == [list(range(6))]
    same = [[7, 7, 7], [7, 7, 7], [8] * 9]                      # the repeat still goes to a call of its own
    assert isolate.waves(same, 10) == [[0], [2], [1]]


@pytest.mark.parametrize("bidir", [False, True])
def test_block_mask_built_in_chunks_equals_the_whole(bidir, monkeypatch):
    """the block tables of a step do not depend on how many query rows are evaluated at a time."""
    rng = random.Random(3); ids, qsl = [], [0]
    for _ in range(24):
        ids += make(rng).ids; qsl.append(len(ids))
    st = isolate.batch_layout(ids, qsl, END, True, OPT_HDR, Q_HDR, VotumConfig().close_last)
    try:
        monkeypatch.setattr(isolate, "MASK_CHUNK", 1 << 30); whole = isolate.StepMasks(st, "cpu").block_mask(bidir)
        monkeypatch.setattr(isolate, "MASK_CHUNK", isolate.PAD); parts = isolate.StepMasks(st, "cpu").block_mask(bidir)
    except Exception as e:   # no compiler for the compiled mask on this machine
        pytest.skip(f"create_block_mask not available on CPU here: {type(e).__name__}")
    assert len(ids) > 3 * isolate.PAD and parts.seq_lengths == whole.seq_lengths
    for name in ("kv_num_blocks", "kv_indices", "full_kv_num_blocks", "full_kv_indices"):
        a, b = getattr(whole, name), getattr(parts, name)
        assert (a is None) == (b is None) and (a is None or torch.equal(a, b)), name
