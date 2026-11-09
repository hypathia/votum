"""FlexAttention path for the BidiGate attention layouts (opt-in: train.py --attn flex).

Every layout is ONE mask rule, `mask_rule` below, evaluated as a FlexAttention mask_mod(b, h, q, kv) over per-batch tensors:
  lens[b]      real length of row b (keys at or after it are padding)
  blk[b, t]    option-block id of token t (-1 = not an option token: prefix, tail, marker, padding)
  S[b]         state end (header layout: span extra["qstart"]); padded length when the row has none
  H[b]         header end (--header-blind: span extra["hend"]); none when unset
and per-layer static flags: kind (causal | bidirectional), and the global flags of the batch (option blocks set,
isolate, state ends set).
  past  = kv <= q
  cross = blk[q] >= 0 & blk[kv] >= 0 & blk[q] != blk[kv]                     (two different option blocks)
  causal layer:         real & past [& ~cross if option blocks are set]
  bidirectional layer:  real & (past [& ~cross if isolate]
                                | ~past [& ~cross & ~(blk[q] < 0 & blk[kv] >= 0) if isolate]
                                        [& ~(q < S & kv >= S) if state ends are set]
                                        [& ~(q < H & kv >= H) if header ends are set])
This is bidigate.BidiGate.mask_for with beta in {0, -1e9}: beta = 0 (bidirectional layer) adds 0 to the visible future keys,
beta = -1e9 (causal layer) is exactly the causal rule (every real query row keeps its own key, so exp(-1e9 - max) == 0 in
the softmax). Any other beta is not a mask and is refused (assert).

One BlockMask per layer KIND per batch (causal, bidirectional), shared by every layer of that kind and by the
gradient-checkpoint recomputation; rebuilt when the batch changes (identity of the padding / blocks / state-end objects
set through bidigate, with references held so an id can never be reused). T is padded up to a multiple of PAD (128, the
block size) inside the attention call; padded keys are masked by `real`, padded query rows are fully
masked (flex returns 0 there) and sliced off. flex_attention is torch.compile'd once (dynamic shapes); GQA via enable_gqa.

Mechanism: enable(model, gate) (1) registers the attention function "votum_flex" in transformers' AttentionInterface and
points every attention module at it through a per-module shallow copy of the config; (2) replaces the
gate instance's mask_for so its existing forward pre-hook passes the BlockMask instead of the dense float [B,1,T,T] mask.
Nothing changes unless enable() is called."""
import copy
import torch, torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, BlockMask
import bidigate

PAD = 128
_FLEX = {"fn": None}


def _flex_fn():
    if _FLEX["fn"] is None:
        import torch._dynamo.config as dc   # dynamic shapes still specialise B == 1 and the two mask kinds (fwd + bwd): about
        dc.recompile_limit = max(dc.recompile_limit, 64)   # 12 graphs; past the default limit of 8 dynamo falls back to eager flex
        _FLEX["fn"] = torch.compile(flex_attention, dynamic=True)
    return _FLEX["fn"]


def mask_rule(lens, blk, S, bidir, blocks=False, isolate=False, state_ends=False, H=None):
    """-> mask_mod(b, h, q, kv) (True = key visible). See the module docstring."""
    def mask_mod(b, h, q, kv):
        real = kv < lens[b]
        past = kv <= q
        if blocks or isolate:
            bq = blk[b, q]; bk = blk[b, kv]
            cross = (bq >= 0) & (bk >= 0) & (bq != bk)
        if not bidir:
            ok = past & ~cross if blocks else past
            return real & ok
        fut = ~past
        if blocks and isolate:
            past = past & ~cross
            fut = fut & ~cross & ~((bq < 0) & (bk >= 0))
        if state_ends: fut = fut & ~((q < S[b]) & (kv >= S[b]))
        if H is not None: fut = fut & ~((q < H[b]) & (kv >= H[b]))   # header blind to the state (bidigate.set_header_ends)
        return real & (past | fut)
    return mask_mod


class FlexMasks:
    """Per-batch BlockMasks for one BidiGate. mask_for(layer, T, dtype, device) has BidiGate.mask_for's signature."""

    def __init__(self, gate):
        self.gate = gate; self._key = None; self._refs = None; self._t = None; self._bm = {}
        b = gate.betas().detach().float().cpu()
        for li in range(len(b)):
            assert float(b[li]) == 0.0 or float(b[li]) <= -1e8, \
                f"flexmask supports beta 0 (bidirectional) or <= -1e8 (causal) only; layer {li} has beta {float(b[li])}"

    def _kind(self, layer):
        return float(self.gate.betas()[layer]) == 0.0   # True: bidirectional layer

    def _batch(self, T, device):
        am, blocks, ends, iso = bidigate._PAD["am"], bidigate._OPT["blocks"], bidigate._OPT["state_ends"], bool(bidigate._OPT["isolate"]); hends = bidigate._OPT.get("header_ends")
        key = (id(am), id(blocks), id(ends), id(hends), iso, T, str(device))
        if key == self._key: return
        self._key = key; self._refs = (am, blocks, ends, hends); self._bm = {}
        Tp = -(-T // PAD) * PAD
        B = am.shape[0] if am is not None else 1
        lens = am[:, :T].long().sum(1).to(device) if am is not None else torch.full((1,), T, dtype=torch.long, device=device)
        if am is not None:   # right padding only (model.pad_batch): real tokens are exactly the first lens[b]
            assert bool((am[:, :T].bool() == (torch.arange(T, device=am.device)[None] < lens.to(am.device)[:, None])).all()), "flexmask expects right padding"
        blk = torch.full((B, Tp), -1, dtype=torch.long)
        if blocks is not None:
            for bi, ids in enumerate(blocks[:B]):
                n = min(len(ids), T); blk[bi, :n] = torch.tensor(ids[:n], dtype=torch.long)
        S = torch.full((B,), Tp, dtype=torch.long)
        if ends is not None:
            for bi, x in enumerate(list(ends)[:B]):
                if x is not None: S[bi] = int(x)
        Hh = None
        if hends is not None:
            Hh = torch.full((B,), Tp, dtype=torch.long)
            for bi, x in enumerate(list(hends)[:B]):
                if x is not None: Hh[bi] = int(x)
            Hh = Hh.to(device)
        self._t = dict(lens=lens, blk=blk.to(device), S=S.to(device), B=B, Tp=Tp, blocks=blocks is not None, isolate=iso,
                       state_ends=ends is not None, H=Hh)

    def block_mask(self, layer, T, device):
        self._batch(T, device); kind = self._kind(layer)
        if kind not in self._bm:
            t = self._t
            mm = mask_rule(t["lens"], t["blk"], t["S"], bidir=kind, blocks=t["blocks"], isolate=t["isolate"],
                           state_ends=t["state_ends"], H=t["H"])
            self._bm[kind] = create_block_mask(mm, t["B"], None, t["Tp"], t["Tp"], device=device, BLOCK_SIZE=128)
        return self._bm[kind]

    def mask_for(self, layer, T, dtype, device):
        return self.block_mask(layer, T, device)


def votum_flex_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
    """transformers attention-interface function: [B,H,T,D] q / [B,Hkv,T,D] k, v + BlockMask -> ([B,T,H,D], None)."""
    if not isinstance(attention_mask, BlockMask):   # not a gated call: plain sdpa
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        return sdpa_attention_forward(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kwargs)
    assert dropout == 0.0 or not module.training, "flexmask: attention dropout is not supported"
    T = query.shape[2]; Tp = attention_mask.seq_lengths[0]
    dt = query.dtype; key = key.to(dt); value = value.to(dt)
    if Tp != T:
        pad = (0, 0, 0, Tp - T); query, key, value = F.pad(query, pad), F.pad(key, pad), F.pad(value, pad)
    out = _flex_fn()(query, key, value, block_mask=attention_mask, scale=scaling, enable_gqa=query.shape[1] != key.shape[1])
    return out[:, :, :T].transpose(1, 2).contiguous(), None


def register():
    from transformers import AttentionInterface
    AttentionInterface.register("votum_flex", votum_flex_attention)


def enable(model, gate):
    """switch the gate's hooked attention layers of `model` to FlexAttention with the same mask rule; -> FlexMasks."""
    import model as M
    assert gate is not None, "flexmask needs a BidiGate (--bidir-layers)"
    register(); fm = FlexMasks(gate)
    for mod in M.attn_modules(model):
        mod.config = copy.copy(mod.config); mod.config._attn_implementation_internal = "votum_flex"
    gate.mask_for = fm.mask_for
    print(f"FLEX attention: {len(M.attn_modules(model))} layers, pad {PAD}", flush=True)
    return fm
