"""Per-layer attention layout gate (interleaved bidirectional layers): which layers also attend to LATER tokens of the prompt.

Every hooked attention layer l has a bias beta_l added to the attention scores of future keys (padding keys stay masked;
past keys get 0, as in the causal model):
    mask_l[i, j] = 0        if j <= i and j is a real token
                 = beta_l   if j >  i and j is a real token
                 = -inf     if j is padding
The trainer uses only beta_l = 0 (bidirectional layer) and beta_l = -1e9 (causal layer: future keys get zero weight). beta is
a frozen parameter so that it travels with the checkpoint (gate.pt); betas() clamps it to <= 0.
Implemented as a forward pre-hook on each attention module that replaces its `attention_mask` kwarg; the sdpa path then
uses the float mask (is_causal is ignored when a mask is given). The 2D padding mask of the current batch is set by
`set_padding(am)` (model.hidden does it every forward).
  gate = BidiGate(n_layers, init=-1e9).to(dev); gate.install(model, attn_modules)     # once
  ... forward as usual ...; gate.state_dict() / load_state_dict() travel with the checkpoint (gate.pt)."""
import torch, torch.nn as nn

_PAD = {"am": None}
_OPT = {"blocks": None, "isolate": False, "state_ends": None, "header_ends": None}   # per-batch option-block ids (lists of per-token ids, -1 = not an option), or None; isolate: see set_isolate


def set_padding(am):
    _PAD["am"] = am


def set_option_blocks(blocks):
    """option blocks of the current batch: in the CAUSAL layers a token of one option block cannot see another option's
    block (combined with shared option positions this removes every trace of option order); bidirectional layers keep
    full visibility. None disables."""
    _OPT["blocks"] = blocks


def set_isolate(flag):
    """--opt-isolate: option blocks are masked from each other in EVERY hooked layer (bidirectional ones included) and the
    prefix (state / question) never attends to option tokens, so each option's states depend on the prefix and itself
    only: the prefix can be encoded once and options scored independently."""
    _OPT["isolate"] = bool(flag)


def set_state_ends(ends):
    """--header: per-row token index where the state block ends (span extra["qstart"]: header + state before it, question +
    options + marker from it). Tokens before it never attend to tokens at or after it, in every hooked layer, so the
    state block's states are identical for every question of a request (cacheable once per request).
    None disables; a None entry leaves that row unrestricted."""
    _OPT["state_ends"] = ends


def set_header_ends(ends):
    """--header-blind: per row, the first token after the header (span extra["hend"]); tokens
    before it (system + header) never attend to tokens at/after it, so system + header depend only on themselves and are
    cacheable across requests with the same question set. None disables; a None entry leaves that row unrestricted."""
    _OPT["header_ends"] = ends


class BidiGate(nn.Module):
    def __init__(self, n_layers, init=-1e9):
        super().__init__()
        self.beta = nn.Parameter(torch.full((n_layers,), float(init)))
        # not used; kept so the saved gate.pt keeps the keys of the released checkpoint (all layers hooked, no window)
        self.register_buffer("only", torch.ones(n_layers, dtype=torch.bool))
        self.register_buffer("window", torch.zeros(n_layers, dtype=torch.long))

    def betas(self):
        return self.beta.clamp(max=0.0)

    @staticmethod
    def _base(am, T, device):
        """bool [B,1,T,T]: (past-or-self & key real), (future & key real). Rebuilt every call: caching on the padding
        tensor's address is unsafe (a later micro-batch can reuse the address with a different padding pattern)."""
        i = torch.arange(T, device=device)
        past = (i[None, :] <= i[:, None])[None, None]
        real = am[:, None, None, :T].bool()
        return past & real, (~past) & real

    def mask_for(self, layer, T, dtype, device):
        am = _PAD["am"]
        if am is None: am = torch.ones(1, T, dtype=torch.long, device=device)
        past, fut = self._base(am, T, device)
        neg = torch.finfo(dtype).min
        if _OPT["blocks"] is not None and (_OPT["isolate"] or float(self.beta[layer]) < -1e8):     # causal layer (or every layer under isolate): no cross-option visibility
            blk = torch.full((past.shape[0], T), -1, dtype=torch.long, device=device)
            for bi, ids in enumerate(_OPT["blocks"]):
                n = min(len(ids), T); blk[bi, :n] = torch.tensor(ids[:n], device=device)
            cross = ((blk[:, :, None] >= 0) & (blk[:, None, :] >= 0) & (blk[:, :, None] != blk[:, None, :]))[:, None]
            past = past & ~cross
            if _OPT["isolate"]:   # also in the bidirectional layers, and the prefix never looks ahead at option tokens
                fut = fut & ~cross & ~((blk[:, :, None] < 0) & (blk[:, None, :] >= 0))[:, None]
        if _OPT["state_ends"] is not None:   # header layout: the state block never looks ahead past its end
            S = torch.tensor([T if x is None else int(x) for x in _OPT["state_ends"]], device=device)[:fut.shape[0]]
            i = torch.arange(T, device=device)
            fut = fut & ~((i[None, :, None] < S[:, None, None]) & (i[None, None, :] >= S[:, None, None]))[:, None]
        if _OPT["header_ends"] is not None:   # header blind to the state: same cut at the header end
            Hh = torch.tensor([T if x is None else int(x) for x in _OPT["header_ends"]], device=device)[:fut.shape[0]]
            i = torch.arange(T, device=device)
            fut = fut & ~((i[None, :, None] < Hh[:, None, None]) & (i[None, None, :] >= Hh[:, None, None]))[:, None]
        b = self.betas()[layer].to(dtype)
        m = torch.full(past.shape, neg, dtype=dtype, device=device)
        m = torch.where(past, torch.zeros((), dtype=dtype, device=device), m)
        m = torch.where(fut, b, m)
        return m

    def install(self, model, attn_modules):
        gate = self
        for li, mod in enumerate(attn_modules):
            def pre(module, args, kwargs, li=li):
                hs = kwargs.get("hidden_states", args[0] if args else None)
                T = hs.shape[1]
                kwargs["attention_mask"] = gate.mask_for(li, T, hs.dtype, hs.device)
                if len(args) > 2:   # attention_mask passed positionally (hidden_states, position_embeddings, attention_mask, ...)
                    args = args[:2] + (kwargs.pop("attention_mask"),) + args[3:]
                return args, kwargs
            mod.register_forward_pre_hook(pre, with_kwargs=True)
            mod.is_causal = False
        return self
