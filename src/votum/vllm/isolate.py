"""Attention layout of Votum on vLLM: packing, per-step BlockMasks, FlexAttention, and an exact prefix cache.

Attention semantics (= votum.modeling.attention_masks, the training-time rule):
  causal layers         real & past & ~cross
  bidirectional layers  real & (past & ~cross | future & ~cross & ~(non-option query -> option key) & ~(state query -> question block key)
                                                              & ~(header query -> state key))
  cross = query and key in two different option blocks ("- <text><end>\\n"); header block = system turn + request header;
  state block = header block + state (tokens before the row's qstart); question block = question + options.
Packing ("state once"): the state block's states depend on the state block only, so every question that shares it (same
header and state; the client groups them) is sent as ONE prompt  [header][state][question block 1]...[question block J].
Each question block sees the state block and itself, never another question block, and keeps the position ids of its own
single-question rendering (the question block starts at position S; shared option positions inside it). This is exactly
the single-pass computation of every question, with the state computed once.
The layout of each prompt (S, question-block starts, option starts) is registered by the client under a digest of the
prompt's token ids (same process: VLLM_ENABLE_V1_MULTIPROCESSING=0) and looked up here per scheduled prompt; an
unregistered prompt falls back to a token-pattern parse (warned once).
Kernel: one compiled FlexAttention call per layer over the step's packed tokens, with one 128x128 BlockMask per layer kind
(causal / bidirectional) per step built from per-token (prompt, question block, option block) ids; tiles across prompts and
across question blocks are skipped. KV-cache writes are skipped (a one-step pooling model never reads them: prefix caching
and chunked prefill stay off)."""
from __future__ import annotations

import hashlib, logging, os, threading
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)
PAD = 128
_REG: "OrderedDict[bytes, dict]" = OrderedDict()
_REG_LOCK = threading.Lock()
_REG_MAX = 65536
_WARNED = set()


def digest(ids) -> bytes:
    return hashlib.blake2b(np.asarray(ids, dtype=np.int32).tobytes(), digest_size=16).digest()


def register(ids, layout: dict) -> bytes:
    """remember the layout of a prompt (see build_prompts) under a digest of its token ids."""
    k = digest(ids)
    with _REG_LOCK:
        _REG[k] = layout; _REG.move_to_end(k)
        while len(_REG) > _REG_MAX: _REG.popitem(last=False)
    return k


# An engine call is scheduled by vLLM as steps of up to max_num_batched_tokens tokens, which cannot be set below the context
# limit. The attention working memory of a step grows with its token count, so the calls themselves are bounded here: a
# call holds at most STEP_TOKENS tokens (a longer prompt goes alone). VOTUM_STEP_TOKENS=0 turns the bound off.
STEP_TOKENS = int(os.environ.get("VOTUM_STEP_TOKENS", "16384"))
# create_block_mask materialises a dense bool [queries, keys] tensor: quadratic in the step (16 GiB at 131,072 tokens).
# The block mask is built MASK_CHUNK query rows at a time instead (same block tables, [MASK_CHUNK, keys] bytes at a time).
MASK_CHUNK = max(PAD, int(os.environ.get("VOTUM_MASK_CHUNK", "4096")) // PAD * PAD)


def waves(id_lists, step_tokens: int | None = None) -> list[list[int]]:
    """indices of the prompts, split into engine calls so that no call holds two prompts with the same token ids (the
    registry is keyed by them; the k-th repeat of a prompt goes to call k) and no call holds more than step_tokens tokens
    (default STEP_TOKENS; a prompt longer than that goes alone). Usually a single call."""
    seen: dict = {}; out: list[list[int]] = []
    for j, ids in enumerate(id_lists):
        d = digest(ids); k = seen.get(d, 0); seen[d] = k + 1
        if k == len(out): out.append([])
        out[k].append(j)
    budget = STEP_TOKENS if step_tokens is None else step_tokens
    if budget <= 0: return out
    split: list[list[int]] = []
    for w in out:
        cur: list[int] = []; tot = 0
        for j in w:
            n = len(id_lists[j])
            if cur and tot + n > budget: split.append(cur); cur, tot = [], 0
            cur.append(j); tot += n
        if cur: split.append(cur)
    return split


def unregister(keys) -> None:
    with _REG_LOCK:
        for k in keys: _REG.pop(k, None)


def lookup(ids) -> dict | None:
    with _REG_LOCK:
        return _REG.get(digest(ids))


def _find_last(ids, pat, lo, hi):
    h = len(pat)
    for i in range(hi - h, lo - 1, -1):
        if ids[i:i + h] == pat: return i
    return None


def parse_layout(ids, opt_end_id, opt_hdr, q_hdr) -> dict | None:
    """best-effort layout of a single-question prompt from its token ids alone (used only when the client did not register
    the prompt): the state ends at the last "Question:\\n", the options start after the last "Options:\\n" before the first
    end marker."""
    ends = [i for i, t in enumerate(ids) if t == opt_end_id]
    if not ends: return None
    h = _find_last(ids, opt_hdr, 0, ends[0]); P = None if h is None else h + len(opt_hdr)
    qs = _find_last(ids, q_hdr, 0, h if h is not None else ends[0]) if q_hdr else None
    S = qs if qs is not None else 0
    return {"H": 0, "h": None, "groups": [{"start": 0, "S": S, "s": None, "qb": [(S, P)]}]}


# ---------------------------------------------------------------------------------------------------- prefix cache
GEOM = {}            # set by the model: n_layers, kv heads, head dim, dtype, device (the client sizes cache entries with it)
CACHE_BASE = 1 << 24  # ids of cached segments in the mask arrays (in-step ids stay below)


class Entry:
    __slots__ = ("key", "n", "k", "v", "ready", "pins", "bytes", "kind")

    def __init__(self, key, n, kind):
        L, Hkv, D, dt, dev = GEOM["layers"], GEOM["kv_heads"], GEOM["head_dim"], GEOM["dtype"], GEOM["device"]
        self.key, self.n, self.kind = key, n, kind
        self.k = torch.empty((L, n, Hkv, D), dtype=dt, device=dev); self.v = torch.empty_like(self.k)
        self.ready, self.pins, self.bytes = False, 1, 2 * self.k.numel() * self.k.element_size()


class PrefixCache:
    """Persistent cross-request cache of the KV (post-RoPE, every layer: 128 KB per token on Apertus-8B) of exactly reusable
    prefix segments: the system + header block (depends only on itself) and the state block
    (depends only on the header block and itself). Two LRU pools by bytes: headers (`budget_mb`) and states (`state_mb`,
    0 = off); entries referenced or being written by an in-flight call are pinned. Admission: a header block on first sight
    (small, shared by every request with the same question set); a state block on its second sight (unique states would only
    churn the pool)."""

    def __init__(self, budget_mb: float, state_mb: float = 0.0):
        self.budget = {0: int(budget_mb * 2 ** 20), 1: int(state_mb * 2 ** 20)}; self.used_k = {0: 0, 1: 0}
        self.entries: "OrderedDict[bytes, Entry]" = OrderedDict()
        self.seen: "OrderedDict[bytes, None]" = OrderedDict()

    def get(self, key):
        e = self.entries.get(key)
        if e is None or not e.ready: return None
        self.entries.move_to_end(key); e.pins += 1; return e

    def sighting(self, key) -> int:
        n = 1 if key in self.seen else 0
        self.seen[key] = None; self.seen.move_to_end(key)
        while len(self.seen) > 200000: self.seen.popitem(last=False)
        return n + 1

    def reserve(self, key, n, kind):
        if key in self.entries or not GEOM: return None
        cap = self.budget[kind]
        need = 2 * GEOM["layers"] * n * GEOM["kv_heads"] * GEOM["head_dim"] * torch.empty((), dtype=GEOM["dtype"]).element_size()
        if need > cap: return None
        for k in list(self.entries):
            if self.used_k[kind] + need <= cap: break
            e = self.entries[k]
            if e.kind == kind and e.pins == 0: self.used_k[kind] -= e.bytes; del self.entries[k]
        if self.used_k[kind] + need > cap: return None
        e = Entry(key, n, kind); self.entries[key] = e; self.used_k[kind] += e.bytes; return e

    def finish(self, used, saved, ok=True):
        for e in used: e.pins -= 1
        for e in saved:
            e.pins -= 1
            if ok: e.ready = True
            else: self.used_k[e.kind] -= e.bytes; self.entries.pop(e.key, None)


CACHE: "PrefixCache | None" = None


def build_prompts(encs, pack: bool, max_len: int, cache: "PrefixCache | None" = None):
    """encs: prompt.Encoding or None per question -> (prompts [(ids, layout)], members [row indices per prompt], used
    entries, saved entries). Exact reuse levels:
      (1) questions with identical state-block tokens share one state block (state once per (header, state));
      (2) groups with identical system + header tokens share one header block in the same prompt (header once per
          question set);
      (3) with a PrefixCache: a header / state block computed by an earlier call is read from the cache (its tokens are
          not sent), and new ones are saved for later calls.
    layout (prompt coordinates): {"H": header length in the single rendering (0 = none), "h": cached header key or
    None (header tokens [0, H) in the prompt), "h_save": key or None, "groups": [{"start", "S" (single-rendering qstart),
    "s": cached state key or None (state tokens [start, start + S - H) in the prompt), "s_save", "qb": [(start, ostart)]}]}."""
    prompts, members, buckets = [], [], OrderedDict()
    used, saved = [], []
    for i, e in enumerate(encs):
        if e is None: continue
        S = e.qstart; H = e.hend
        if H is not None and S is not None and H > S: H = None
        if S is None or not pack:   # own prompt
            Hh = H or 0; S0 = S if S is not None else Hh
            prompts.append((list(e.ids), {"H": Hh, "h": None, "groups": [{"start": Hh, "S": S0, "s": None, "qb": [(S0, e.ostart)]}]}))
            members.append([i]); continue
        Hh = H or 0
        buckets.setdefault(tuple(e.ids[:Hh]), OrderedDict()).setdefault(tuple(e.ids[Hh:S]), []).append(i)
    for Hids, groups in buckets.items():
        Hn = len(Hids); hkey = b"H" + digest(Hids) if Hn else None
        hent = cache.get(hkey) if (cache is not None and hkey) else None
        if hent is not None: used.append(hent)
        hsave = None
        if cache is not None and hkey and hent is None:
            hsave = cache.reserve(hkey, Hn, 0)
            if hsave is not None: saved.append(hsave)
        cur = lay = mem = None

        def flush():
            if mem: prompts.append((cur, lay)); members.append(mem)

        def new_prompt(first):
            nonlocal cur, lay, mem
            if hent is not None: cur, lay = [], {"H": Hn, "h": hkey, "groups": []}
            else: cur, lay = list(Hids), {"H": Hn, "h": None, "h_save": hkey if (first and hsave is not None) else None, "groups": []}
            mem = []

        new_prompt(True)
        for Sids, idx in groups.items():
            S = Hn + len(Sids); skey = b"S" + digest(Hids + Sids) if cache is not None else None
            sent = cache.get(skey) if cache is not None else None
            if sent is not None: used.append(sent)
            ssave = None
            if cache is not None and sent is None and cache.budget[1] > 0 and cache.sighting(skey) >= 2:
                ssave = cache.reserve(skey, len(Sids), 1)
                if ssave is not None: saved.append(ssave)
            g = None; s_saved = False
            for i in idx:
                e = encs[i]; q = list(e.ids[S:]); o = e.ostart
                st_len = 0 if (g is not None or sent is not None) else len(Sids)
                if mem and len(cur) + st_len + len(q) > max_len:
                    flush(); new_prompt(False); g = None; st_len = 0 if sent is not None else len(Sids)
                if g is None:
                    g = {"start": len(cur), "S": S, "s": skey if sent is not None else None, "s_save": None, "qb": []}
                    if sent is None:
                        if ssave is not None and not s_saved: g["s_save"] = skey; s_saved = True   # saved once, by its first copy
                        cur += list(Sids)
                    lay["groups"].append(g)
                g["qb"].append((len(cur), None if o is None else len(cur) + o - S)); cur += q; mem.append(i)
        flush()
    return prompts, members, used, saved


def batch_layout(ids_all: list[int], qsl: list[int], opt_end_id: int, shared_positions: bool, opt_hdr: list[int], q_hdr: list[int],
                 close_last: bool = True):
    """per-token arrays of the step (current tokens first, then the KV of referenced cached segments):
    kind (0 header block, 1 state block, 2 question block), hid / gid / qb (ids of the token's header block, state group,
    question block), blk (option block, -1 none), ordk (order for past/future; -1 = cached, always in the past); positions of
    the current tokens; the cached entries to append (in order) and the saves [(entry, first token, n)]. close_last: the
    line break after a question's last end marker belongs to its last option block."""
    N = qsl[-1]
    pos = np.empty(N, dtype=np.int64); kind = np.full(N, 2, dtype=np.int32); hid = np.empty(N, dtype=np.int32); gid = np.empty(N, dtype=np.int32)
    qb = np.full(N, -1, dtype=np.int32); blk = np.full(N, -1, dtype=np.int32)
    nh = ng = nq = nb = 0; refs: "OrderedDict[bytes, int]" = OrderedDict(); saves = []
    cache = CACHE
    for p, (a, e) in enumerate(zip(qsl[:-1], qsl[1:])):
        ids = ids_all[a:e]; n = e - a
        lay = lookup(ids)
        if lay is None:
            lay = parse_layout(ids, opt_end_id, opt_hdr, q_hdr)
            if "parse" not in _WARNED:
                _WARNED.add("parse"); log.warning("votum.vllm.isolate: prompt layout not registered by the client; parsed from tokens")
        if lay is None:   # not a decision prompt: plain positions, one block, no options
            pos[a:e] = np.arange(n); hid[a:e] = nh; gid[a:e] = ng; qb[a:e] = nq; nh += 1; ng += 1; nq += 1; continue
        H = int(lay["H"])
        if lay.get("h") is not None:
            k = lay["h"]; refs.setdefault(k, 0); h_id = CACHE_BASE + list(refs).index(k)
        else:
            h_id = nh; nh += 1
            if H:
                pos[a:a + H] = np.arange(H); kind[a:a + H] = 0; hid[a:a + H] = h_id; gid[a:a + H] = -2 - h_id
                if lay.get("h_save") is not None and cache is not None and lay["h_save"] in cache.entries: saves.append((cache.entries[lay["h_save"]], a, H))
        groups = lay["groups"]
        for gi, g in enumerate(groups):
            gs = int(g["start"]); ge = int(groups[gi + 1]["start"]) if gi + 1 < len(groups) else n; S = int(g["S"])
            if g.get("s") is not None:
                k = g["s"]; refs.setdefault(k, 1); g_id = CACHE_BASE + list(refs).index(k)
            else:
                g_id = ng; ng += 1; L = S - H
                if L:
                    pos[a + gs:a + gs + L] = np.arange(H, S); kind[a + gs:a + gs + L] = 1; hid[a + gs:a + gs + L] = h_id; gid[a + gs:a + gs + L] = g_id
                    if g.get("s_save") is not None and cache is not None and g["s_save"] in cache.entries: saves.append((cache.entries[g["s_save"]], a + gs, L))
            starts = [int(x) for x, _ in g["qb"]] + [ge]
            for j, (s, P) in enumerate(g["qb"]):
                s, t_end = int(s), starts[j + 1]
                hid[a + s:a + t_end] = h_id; gid[a + s:a + t_end] = g_id; qb[a + s:a + t_end] = nq; nq += 1
                base = S - s                                   # single-rendering position of packed token t = t + base
                pos[a + s:a + t_end] = np.arange(s, t_end) + base
                if P is None: continue                         # option start unknown: no option blocks, plain positions
                P = int(P); seg = ids[s:t_end]
                ends = [s + i for i, t in enumerate(seg) if t == opt_end_id]; last = t_end - 1
                c = P; Lmax = 0
                for en in ends:
                    stop = en + 1 if (en + 1 < last or (close_last and en + 1 == last)) else en
                    blen = stop - c + 1
                    if shared_positions: pos[a + c:a + stop + 1] = np.arange(blen) + P + base
                    blk[a + c:a + stop + 1] = nb
                    nb += 1; Lmax = max(Lmax, blen); c = stop + 1
                if shared_positions: pos[a + c:a + t_end] = np.arange(t_end - c) + P + base + Lmax
    ref_entries = [cache.entries[k] for k in refs] if refs else []
    ordk = np.arange(N, dtype=np.int32)
    if ref_entries:
        ck, ch, cg = [], [], []
        for j, (k, kd) in enumerate(refs.items()):
            n = ref_entries[j].n; ck.append(np.full(n, kd, np.int32))
            ch.append(np.full(n, CACHE_BASE + j if kd == 0 else -(1 << 30), np.int32)); cg.append(np.full(n, CACHE_BASE + j if kd == 1 else -(1 << 30), np.int32))
        C = sum(x.n for x in ref_entries)
        kind = np.concatenate([kind] + ck); hid = np.concatenate([hid] + ch); gid = np.concatenate([gid] + cg)
        qb = np.concatenate([qb, np.full(C, -1, np.int32)]); blk = np.concatenate([blk, np.full(C, -1, np.int32)]); ordk = np.concatenate([ordk, np.full(C, -1, np.int32)])
    return dict(pos=pos, kind=kind, hid=hid, gid=gid, qb=qb, blk=blk, ordk=ordk, refs=ref_entries, saves=saves, N=N)


def mask_rule(kind, hid, gid, qb, blk, ordk, bidir: bool, qa=None):
    """FlexAttention mask_mod over the step (queries = current tokens, keys = current tokens + cached segments).
    Structure: a header-block key is visible to every token of its prompt's header; a state key to its group (state and
    question blocks); a question-block key to its own question block. Within that: causal layers past & ~cross;
    bidirectional layers ~cross & (past | ~(non-option query -> option key)). Cached keys are always in the past.
    qa: query-side (hid, gid, qb, blk), padded past the step's tokens with ids that match nothing. The returned closures
    capture tensors only (no Python flags), so the compiled mask and flex kernels are reused across steps."""
    qh, qg, qq, qk = (hid, gid, qb, blk) if qa is None else qa
    if bidir:
        def mask_bidir(b, h, q, kv):
            kk = kind[kv]; bq = qk[q]; bk = blk[kv]
            ok = ((kk == 0) & (hid[kv] == qh[q])) | ((kk == 1) & (gid[kv] == qg[q])) | ((kk == 2) & (qb[kv] == qq[q]))
            cross = (bq >= 0) & (bk >= 0) & (bq != bk)
            return ok & ~cross & ((ordk[kv] <= q) | ~((bq < 0) & (bk >= 0)))
        return mask_bidir

    def mask_causal(b, h, q, kv):
        kk = kind[kv]; bq = qk[q]; bk = blk[kv]
        ok = ((kk == 0) & (hid[kv] == qh[q])) | ((kk == 1) & (gid[kv] == qg[q])) | ((kk == 2) & (qb[kv] == qq[q]))
        cross = (bq >= 0) & (bk >= 0) & (bq != bk)
        return ok & (ordk[kv] <= q) & ~cross
    return mask_causal


def dense_mask(st, bidir):
    """bool [N, N + C] reference of mask_rule (tests); st = batch_layout(...) with torch tensors."""
    N = st["N"]; NK = st["kind"].shape[0]; q = torch.arange(N)[:, None]; kv = torch.arange(NK)[None, :]
    return mask_rule(st["kind"], st["hid"], st["gid"], st["qb"], st["blk"], st["ordk"], bidir)(None, None, q, kv)


_FLEX = {"fn": None}
# fixed flex kernel config: the sm89 default (bitwise equal outputs and the same speed, measured), pinned because with
# appended cached keys (KV length != Q length) inductor's autotune can pick a config over the 99 KB shared-memory limit
KERNEL_OPTIONS = {"BLOCK_M": 128, "BLOCK_N": 64, "num_warps": 8, "num_stages": 2}


def flex_fn():
    if _FLEX["fn"] is None:
        from torch.nn.attention.flex_attention import flex_attention
        import torch._dynamo.config as dc
        dc.recompile_limit = max(dc.recompile_limit, 64); dc.accumulated_recompile_limit = max(getattr(dc, "accumulated_recompile_limit", 256), 512)
        _FLEX["fn"] = torch.compile(flex_attention, dynamic=True)
    return _FLEX["fn"]


class StepMasks:
    """BlockMasks of one scheduled step (built lazily, once per layer kind; create_block_mask compiled). No padding: flex
    takes views of vLLM's [N, heads, dim] tensors; keys of cached segments are appended after the step's tokens."""

    def __init__(self, st, device):
        self.N = st["N"]; self.NK = int(st["kind"].shape[0])
        # fewer than 128 queries: pad the queries to 128 (padded rows see no key, output dropped) so flex always runs its
        # prefill kernel; its decoding kernel (< 128 queries) has no valid config on sm89 once cached keys are appended
        self.NQ = max(self.N, PAD)
        t = lambda x: torch.from_numpy(np.ascontiguousarray(x, dtype=np.int32)).to(device, non_blocking=True)
        self.a = {k: t(st[k]) for k in ("kind", "hid", "gid", "qb", "blk", "ordk")}
        pad = self.NQ - self.N   # query-side arrays always built (padded rows: ids that match nothing, no option block)
        self.qa = tuple(t(np.concatenate([st[k][:self.N], np.full(pad, -(1 << 29) - 7, np.int32)])) for k in ("hid", "gid", "qb"))
        self.qa = self.qa + (t(np.concatenate([st["blk"][:self.N], np.full(pad, -1, np.int32)])),)
        # the per-token arrays are captured by the mask closure; their lengths change every step, so mark them dynamic or
        # torch (2.14) specialises the compiled mask and flex kernels on each length and falls back to eager after 64
        for v in list(self.a.values()) + list(self.qa): torch._dynamo.mark_dynamic(v, 0)
        self.refs, self.saves = st["refs"], st["saves"]
        self._bm = {}

    def block_mask(self, bidir: bool):
        if bidir not in self._bm:
            from torch.nn.attention.flex_attention import BlockMask, create_block_mask
            x = self.a; mm = mask_rule(x["kind"], x["hid"], x["gid"], x["qb"], x["blk"], x["ordk"], bidir, self.qa); dev = x["kind"].device
            if self.NQ <= MASK_CHUNK:
                self._bm[bidir] = create_block_mask(mm, 1, None, self.NQ, self.NK, device=dev, BLOCK_SIZE=PAD, _compile=True)
            else:   # MASK_CHUNK query rows at a time (a multiple of the block size), block tables joined along the query axis
                parts = []
                for q0 in range(0, self.NQ, MASK_CHUNK):
                    off = torch.tensor(q0, device=dev, dtype=torch.int32)
                    parts.append(create_block_mask(lambda b, h, q, kv, off=off: mm(b, h, q + off, kv), 1, None, min(MASK_CHUNK, self.NQ - q0),
                                                   self.NK, device=dev, BLOCK_SIZE=PAD, _compile=True))
                cat = lambda name: None if getattr(parts[0], name) is None else torch.cat([getattr(p, name) for p in parts], dim=2)
                self._bm[bidir] = BlockMask.from_kv_blocks(cat("kv_num_blocks"), cat("kv_indices"), cat("full_kv_num_blocks"), cat("full_kv_indices"),
                                                           BLOCK_SIZE=PAD, mask_mod=mm, seq_lengths=(self.NQ, self.NK))
        return self._bm[bidir]


def flex_forward(orig, impl, owner, bidir: bool, layer_idx: int = 0):
    """replacement for an attention impl's forward: FlexAttention over the step's packed tokens (plus the KV of cached
    segments) with the layout's mask; saves the KV of segments marked for the prefix cache. Without a step layout
    (profiling / dummy runs) the stock forward runs."""
    def forward(layer, query, key, value, kv_cache, attn_metadata, output=None, output_scale=None, output_block_scale=None):
        m = owner._step
        if attn_metadata is None or m is None:
            return orig(layer, query, key, value, kv_cache, attn_metadata, output, output_scale, output_block_scale)
        N = m.N
        for ent, st, n in m.saves:
            ent.k[layer_idx].copy_(key[st:st + n]); ent.v[layer_idx].copy_(value[st:st + n])
        k, v = key[:N], value[:N]
        if m.refs:
            k = torch.cat([k] + [ent.k[layer_idx] for ent in m.refs]); v = torch.cat([v] + [ent.v[layer_idx] for ent in m.refs])
        q = query[:N].transpose(0, 1).unsqueeze(0); k = k.transpose(0, 1).unsqueeze(0); v = v.transpose(0, 1).unsqueeze(0)
        if m.NQ != N: q = F.pad(q, (0, 0, 0, m.NQ - N))
        o = flex_fn()(q, k, v, block_mask=m.block_mask(bidir), scale=impl.scale, enable_gqa=q.shape[1] != k.shape[1], kernel_options=KERNEL_OPTIONS)
        output[:N].copy_(o[0, :, :N].transpose(0, 1).view_as(output[:N]))
        return output
    return forward


def pointer_scores(h: torch.Tensor, ids: torch.Tensor, opt_end_id: int, q0: torch.Tensor, o_w: torch.Tensor, rank: int):
    """pointer scores score_j = <q0, W_o LN h(end_j)> / sqrt(r) for every option end of the prompt, in prompt order (the
    client splits them by question block)."""
    import math
    ends = torch.nonzero(ids == opt_end_id).flatten()
    if ends.numel() == 0: return torch.empty(0, dtype=torch.float32, device=h.device)
    hn = F.layer_norm(h[ends].float(), (h.shape[-1],))
    return F.linear(hn, o_w.float()) @ q0.float() / math.sqrt(rank)
