"""One LoRA run on Apertus-v1.5-8B with the span (pointer) readout; resumable from the last full state.

Recipe (common.RECIPE): LoRA r16 a32 dropout 0 on q/k/v/o/up/down, AdamW wd 0, grad clip 1; 65,536 padded tokens per step
in micro-batches of <= 8,192 padded tokens; bf16 autocast; gradient checkpointing. Loss per question: cross-entropy of the
softmax over the question's options (SpanHead scores) against the pool's target distribution.
Loss normalisation (--loss-norm): "constant" (default for new runs) divides each step's summed loss by a fixed n = the
mean questions per step of the data plan (estimated over its first 500 steps), so every question weighs the same;
"step" (kept when resuming a run that used it) divides by that step's own question count.
Presentation: options shuffled every step (span.render_span), JSON states optionally preceded by a header of sibling questions.
Schedule: linear warm-up `warmup` steps, constant, cosine to 0 over the last `final_decay` steps of `steps`. A run can be
stopped at --until N (state_N.pt kept) and resumed (--resume) up to `steps`.
Outputs (runs/<tag>/): log.jsonl, state_<step>.pt (full: LoRA + optimizer + RNG; every save_every steps, last two kept,
.keep states kept), ckpt_<step>/ (light: adapter + head + train_meta.json; the newest only, rotating), projection.json after 20 steps."""
import argparse, hashlib, itertools, json, math, os, random, shutil, time
from pathlib import Path
import common as C
import numpy as np, torch, torch.nn.functional as F
import span as SP
import model as M


def hkey(*xs):
    """stable integer hash used to seed per-row, per-step randomness."""
    return int(hashlib.sha1("\x1f".join(map(str, xs)).encode()).hexdigest()[:12], 16)


def lr_mult(s, warmup, decay_start, decay_len):
    w = min(1.0, (s + 1) / warmup)
    d = 1.0 if s < decay_start else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - decay_start + 1) / decay_len)))
    return w * d


class Plan:
    """deterministic step plan over the training rows: step -> list of micro-batches (lists of row indices)."""
    def __init__(self, ntok, seed, tokens_per_step, micro_tokens, margin=16):
        self.ntok = [n + margin for n in ntok]; self.seed = seed; self.tps = tokens_per_step; self.mt = micro_tokens
        self.n_mb = max(1, tokens_per_step // micro_tokens)

    def __iter__(self):
        N = len(self.ntok); epoch = 0; rng = np.random.default_rng(self.seed); queue = list(rng.permutation(N)); left = []
        mean = float(np.mean(self.ntok)); want = int(self.tps / mean * 1.6) + 16
        while True:
            while len(queue) < want:
                epoch += 1; queue += list(np.random.default_rng(self.seed + 1000 * epoch).permutation(N))
            buf = left + queue[:want - len(left)]; queue = queue[want - len(left):]
            buf = [i for i in buf if self.ntok[i] <= self.mt]
            order = sorted(buf, key=lambda i: -self.ntok[i]); mbs = []; used = set()
            cur, mx = [], 0
            for i in order:
                if len(mbs) >= self.n_mb: break
                n = self.ntok[i]
                if cur and max(mx, n) * (len(cur) + 1) > self.mt:
                    mbs.append(cur); cur, mx = [], 0
                    if len(mbs) >= self.n_mb: break
                cur.append(i); mx = max(mx, n)
            if cur and len(mbs) < self.n_mb: mbs.append(cur)
            for mb in mbs: used.update(mb)
            left = [i for i in buf if i not in used]
            yield mbs


class FifoPlan(Plan):
    """every row once per pass, in the pass's random order: a step takes the next rows of the queue for as long as they
    pack into its micro-batches (sorted by length only inside the step). Plan fills each step with the longest rows of a
    look-ahead window and carries the shortest over, which is fine over hundreds of steps of a large pool but leaves
    the short rows of a small set under-trained; finetune.py uses this plan."""
    def _pack(self, rows):
        mbs, cur, mx = [], [], 0
        for i in sorted(rows, key=lambda i: -self.ntok[i]):
            n = self.ntok[i]
            if cur and max(mx, n) * (len(cur) + 1) > self.mt: mbs.append(cur); cur, mx = [], 0
            cur.append(i); mx = max(mx, n)
        return mbs + ([cur] if cur else [])

    def __iter__(self):
        ok = [i for i in range(len(self.ntok)) if self.ntok[i] <= self.mt]
        if not ok: raise SystemExit(f"no training row fits a micro-batch of {self.mt} tokens (rows need their length plus 16): use a larger --micro-tokens or shorter rows")
        epoch = 0; queue: list = []; budget = self.n_mb * self.mt
        while True:
            take, tok = [], 0
            while True:
                if not queue:
                    queue = [ok[j] for j in np.random.default_rng(self.seed + 1000 * epoch).permutation(len(ok))]; epoch += 1
                i = queue[0]; n = self.ntok[i]
                if take and (tok + n > budget or (tok + n > budget // 2 and len(self._pack(take + [i])) > self.n_mb)): break
                take.append(queue.pop(0)); tok += n
            yield self._pack(take)


class Trainer:
    def __init__(self, a):
        self.a = a; R_ = C.RECIPE; self.R = R_; self.dev = "cuda"
        self.repo, path = C.backbone_path(a.backbone); self.tok = C.load_tokenizer(path)
        rows = list(C.iter_jsonl(a.train))
        if a.tok_field != "n_tok":   # pack the plan with another token count (e.g. the span-rendered length)
            assert all(a.tok_field in r for r in rows[:1000]), f"{a.tok_field} missing from {a.train}"
            for r in rows: r["n_tok"] = r[a.tok_field]
        SP.SERVE_LAYOUT = bool(a.serve_layout)
        self.hdr_groups = None
        if a.header != "none":
            SP.HEADER_MAX_OPTS = a.header_max_opts   # header: the questions of the same state (whole --train file) are listed before the state
            g = {}
            for r in rows:
                st = r["state"] if isinstance(r["state"], str) else json.dumps(r["state"], ensure_ascii=False, sort_keys=True)
                k = hashlib.md5(st.encode()).hexdigest(); r["_sk"] = k; lst = g.setdefault(k, [])
                if all(q["question"] != r["question"] for q in lst): lst.append(dict(question=r["question"], type=r["type"], options=r["options"]))
            self.hdr_groups = g
        # the sorted plan of a long run reads the leading rows it will consume; the fifo plan walks every row
        n = a.n_rows or (len(rows) if a.plan == "fifo" else C.train_subset_size([r["n_tok"] for r in rows], a.steps, a.tokens_per_step)); self.rows = rows[:n]
        print(f"[{a.tag}] {len(self.rows)} training rows", flush=True)
        self.model = M.load_backbone(path, a.quant)
        self.model = M.add_lora(self.model, a.seed, r=a.lora_r, alpha=2 * a.lora_r if a.lora_r else None)
        self.model.base_model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.gate = None
        self.span = SP.SpanHead(M.d_model(self.model)).to(self.dev)
        if a.bidir_layers:   # layers 3, 7, ..., 31 fully bidirectional over the prompt, the rest exactly causal
            import bidigate
            n = len(M.attn_modules(self.model)); self.gate = bidigate.BidiGate(n, init=-1e9).to(self.dev)
            bidir = list(range(3, n, 4))
            with torch.no_grad():
                for l in bidir: self.gate.beta[l] = 0.0
            self.gate.beta.requires_grad_(False); self.gate.install(self.model, M.attn_modules(self.model))
            if a.opt_isolate: a.opt_block = True; bidigate.set_isolate(True); print(f"[{a.tag}] OPT-ISOLATE: option blocks masked in every layer, prefix blind to options", flush=True)
            print(f"[{a.tag}] interleaved bidirectional layers: {bidir}", flush=True)
        if a.attn == "flex":   # FlexAttention path (flexmask.py): same mask rule, fused kernel skipping fully masked tiles
            assert self.gate is not None, "--attn flex needs --bidir-layers"
            import flexmask; flexmask.enable(self.model, self.gate)
        torch.manual_seed(a.seed + 17)
        lora = [p for n, p in self.model.named_parameters() if p.requires_grad]
        print(f"[{a.tag}] trainable LoRA parameters: {sum(p.numel() for p in lora):,}", flush=True)
        groups = [dict(params=lora, lr=a.lr, base_lr=a.lr, name="lora"),
                  dict(params=list(self.span.parameters()), lr=a.lr_span, base_lr=a.lr_span, name="span", weight_decay=0.0)]
        self.opt = torch.optim.AdamW(groups, betas=R_["betas"], weight_decay=R_["weight_decay"])
        self.plan = (FifoPlan if a.plan == "fifo" else Plan)([r["n_tok"] for r in self.rows], a.seed, a.tokens_per_step, a.plan_micro_tokens or a.micro_tokens)
        self.step = 0; self.out = C.RUNS / a.tag; self.out.mkdir(parents=True, exist_ok=True)
        if not a.no_save:                                             # fail now, not after hours of training (states ~0.6 GB, checkpoints ~0.2 GB)
            try: C.disk_guard(self.out, need_gb=2.0)
            except RuntimeError as e: raise SystemExit(str(e))

    # ------------------------------------------------------------------ helpers
    def encode(self, row, step):
        return SP.render_span(self.tok, row, random.Random(hkey(self.a.seed, step, row["id"])), shuffle=True,
                              header=self.header_for(row, step))

    def header_for(self, row, step):
        """--header own: the row's own question only; group: every distinct question on the row's state (own always in,
        at most --header-max-q, random order per step)."""
        if self.hdr_groups is None: return None
        rng = random.Random(hkey(self.a.seed, step, row["id"], "header")); own = dict(question=row["question"], type=row["type"], options=row["options"])
        qs = [own] if self.a.header == "own" else SP.header_siblings(own, self.hdr_groups.get(row.get("_sk"), []), self.a.header_max_q, rng)
        return SP.header_text(qs, rng)

    def forward_mb(self, rows, step, pre=None):
        encs = pre if pre is not None else [self.encode(r, step) for r in rows]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if self.a.opt_block:
                import bidigate; bidigate.set_option_blocks([SP.option_blocks(e) for e in encs])
            if self.a.header != "none":
                import bidigate; bidigate.set_state_ends([e.extra.get("qstart") for e in encs])
                if self.a.header_blind: bidigate.set_header_ends([e.extra.get("hend") for e in encs])
            h = M.hidden(self.model, [e.ids for e in encs], positions=[SP.shared_positions(e) for e in encs] if self.a.shared_positions else None)
        zs = self.span(h, encs); K = max(len(e.order) for e in encs)
        z = torch.full((len(encs), K), -1e9, device=self.dev)
        for j, zz in enumerate(zs): z[j, :len(zz)] = zz
        return z, encs

    def targets(self, rows, encs):
        tg = []
        for r, e in zip(rows, encs):
            t = np.array(r["target"], dtype=np.float64)
            tg.append([t[i] for i in e.order])
        return tg

    def set_lr(self, s, decay_start, decay_len):
        m = lr_mult(s, self.R["warmup"], decay_start, decay_len)
        if self.a.rewarm: m *= min(1.0, (s - self.start_step + 1) / self.a.rewarm)   # --rewarm: linear re-warm-up after a resume / --init-from
        for g in self.opt.param_groups: g["lr"] = g["base_lr"] * m
        return m

    # ------------------------------------------------------------------ state
    def lora_state(self):
        from peft import get_peft_model_state_dict
        return {k: v.detach().cpu() for k, v in get_peft_model_state_dict(self.model).items()}

    def save_state(self, step, keep=False):
        n = sum(p.numel() for g in self.opt.param_groups for p in g["params"]); C.disk_guard(self.out, need_gb=n * 12 * 1.1 / 2**30)   # fp32 weights + 2 Adam moments
        st = dict(step=step, lora=self.lora_state(), opt=self.opt.state_dict(), gate=self.gate.state_dict() if self.gate is not None else None, span=self.span.state_dict(),
                  rng=dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(), py=random.getstate(), np=np.random.get_state()),
                  args=vars(self.a), recipe=self.R)
        p = self.out / f"state_{step}.pt"; torch.save(st, str(p) + ".tmp"); os.replace(str(p) + ".tmp", p)
        if keep: (self.out / f"state_{step}.keep").touch()
        olds = sorted(self.out.glob("state_*.pt"), key=lambda x: int(x.stem.split("_")[1]))
        for o in olds[:-2]:
            if not (self.out / f"{o.stem}.keep").exists(): o.unlink()

    def load_state(self, path):
        from peft import set_peft_model_state_dict
        st = torch.load(path, map_location="cpu", weights_only=False)
        self._resumed_args = st.get("args") or {}
        set_peft_model_state_dict(self.model, {k: v.to(self.dev) for k, v in st["lora"].items()})
        if self.gate is not None and st.get("gate") is not None:
            miss, unexp = self.gate.load_state_dict(st["gate"], strict=False)   # older states lack the newer buffers ("only", "window")
            assert not unexp and set(miss) <= {"only", "window"}, (miss, unexp)
        if st.get("span") is not None: self.span.load_state_dict(st["span"])
        self.opt.load_state_dict(st["opt"])
        for g in self.opt.param_groups: g["base_lr"] = {"span": self.a.lr_span}.get(g.get("name"), self.a.lr)
        torch.set_rng_state(st["rng"]["torch"]); torch.cuda.set_rng_state(st["rng"]["cuda"]); random.setstate(st["rng"]["py"]); np.random.set_state(st["rng"]["np"])
        self.step = st["step"]; print(f"[{self.a.tag}] resumed from {path} at step {self.step}", flush=True)

    def save_light(self, name):
        d = self.out / name; C.disk_guard(self.out, need_gb=0.5); d.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(d / "adapter"))
        if self.gate is not None: torch.save(self.gate.state_dict(), d / "gate.pt")
        torch.save(self.span.state_dict(), d / "span_head.pt")
        if self.a.shared_positions: C.jdump(dict(positions="every option block starts at the same position"), d / "shared_positions.json")
        if self.a.header != "none": C.jdump(dict(mode=self.a.header, max_q=self.a.header_max_q, max_opts=SP.HEADER_MAX_OPTS, blind=bool(self.a.header_blind), mask="state block (header + state) never attends to the question block" + ("; system + header never attend to the state (cacheable across requests)" if self.a.header_blind else "")), d / "header.json")
        if self.a.opt_isolate: C.jdump(dict(mask="option blocks masked from each other in every layer; prefix blind to option tokens"), d / "opt_isolate.json")
        C.jdump(dict(step=self.step, tag=self.a.tag, seed=self.a.seed, lr=self.a.lr, backbone=self.repo, bidir_layers=self.a.bidir_layers, recipe=self.R, args=vars(self.a)), d / "train_meta.json")
        print(f"[{self.a.tag}] wrote {d}", flush=True)

    # ------------------------------------------------------------------ loop
    def run(self):
        a = self.a; R_ = self.R
        states = sorted(self.out.glob("state_*.pt"), key=lambda x: int(x.stem.split("_")[1]))
        if a.resume and states: self.load_state(states[-1])
        elif a.init_weights:   # continue from a light checkpoint (adapter / span head / gate only): fresh optimizer, plan resumed at --init-step
            from peft import set_peft_model_state_dict; from safetensors.torch import load_file
            ck = Path(a.init_weights)
            if (ck / "adapter" / "adapter_model.safetensors").exists():
                res = set_peft_model_state_dict(self.model, {k: v.to(self.dev) for k, v in load_file(str(ck / "adapter" / "adapter_model.safetensors")).items()})
                assert not getattr(res, "unexpected_keys", None), res.unexpected_keys
            else: print(f"[{a.tag}] no adapter in {ck}: the LoRA starts from zero on the given backbone", flush=True)
            self.span.load_state_dict(torch.load(ck / "span_head.pt", map_location=self.dev))
            if self.gate is not None and (ck / "gate.pt").exists():
                miss, unexp = self.gate.load_state_dict(torch.load(ck / "gate.pt", map_location=self.dev), strict=False); assert not unexp, unexp
            self.step = a.init_step; print(f"[{a.tag}] weights from {ck} (no optimizer state); data plan resumed at step {self.step}; re-warm-up {a.rewarm} steps", flush=True)
        elif a.init_from:   # continue another run: its LoRA / head / optimizer / RNG state and step, this run's --steps plan from that step
            self.load_state(a.init_from); print(f"[{a.tag}] initialised from {a.init_from} (step {self.step}); re-warm-up {a.rewarm} steps", flush=True)
        end = min(a.steps, a.until) if a.until else a.steps; decay_start, decay_len = a.steps - a.final_decay, a.final_decay
        if a.decay_at is not None:                            # decay branch of a constant-LR run: same --steps (= same data order),
            decay_start, decay_len, end = a.decay_at, a.final_decay, a.decay_at + a.final_decay   # cosine to 0 over decay_at..+final_decay
        dj = self.out / "DECAYING.json"
        if dj.exists():   # a DECAY request was already honoured before a restart: keep its schedule
            x = json.load(open(dj)); decay_start, decay_len, end = x["decay_from"], x["decay_len"], x["end"]; print(f"[{a.tag}] decay branch in force: cosine {decay_start}->{end}", flush=True)
        if self.step >= end: print(f"[{a.tag}] already at step {self.step} >= {end}: nothing to do", flush=True); return
        if a.loss_norm is None:                               # new runs: constant; resumed runs keep what they used (old states: step)
            ra = getattr(self, "_resumed_args", None)
            a.loss_norm = (ra.get("loss_norm") or "step") if ra is not None else "constant"
            if a.loss_norm_n is None and ra: a.loss_norm_n = ra.get("loss_norm_n")
        if a.loss_norm == "constant" and not a.loss_norm_n:
            a.loss_norm_n = float(np.mean([sum(len(mb) for mb in mbs) for mbs in itertools.islice(iter(self.plan), 500)]))
        print(f"[{a.tag}] loss normalisation: {a.loss_norm}" + (f" (n = {a.loss_norm_n:.1f} questions/step)" if a.loss_norm == "constant" else ""), flush=True)
        plan = iter(self.plan)
        for _ in range(self.step - a.plan_origin): next(plan)  # deterministic replay of the data order (from --plan-origin)
        log = open(self.out / "log.jsonl", "a")
        torch.cuda.reset_peak_memory_stats(); t_run = time.time(); tok_run = 0; self.start_step = self.step
        self.model.train()
        while self.step < end:
            s = self.step; mbs = next(plan); lrm = self.set_lr(s, decay_start, decay_len)
            nq = sum(len(mb) for mb in mbs); t0 = time.time(); agg = dict(ce=0.0, acc=0.0, pad_tok=0, tok=0)
            if not nq: raise SystemExit(f"[{a.tag}] step {s + 1} has no rows: every row is longer than a micro-batch ({a.plan_micro_tokens or a.micro_tokens} tokens, less a 16-token margin)")
            pre = {}
            if a.repack:   # same rows as the plan's step; regroup into micro-batches by the RENDERED length (less padding)
                flat = [i for mb in mbs for i in mb]; pre = {i: self.encode(self.rows[i], s) for i in flat}
                order = sorted(flat, key=lambda i: -len(pre[i].ids)); mbs, cur, mx = [], [], 0
                for i in order:
                    n = len(pre[i].ids)
                    if cur and max(mx, n) * (len(cur) + 1) > a.micro_tokens: mbs.append(cur); cur, mx = [], 0
                    cur.append(i); mx = max(mx, n)
                if cur: mbs.append(cur)
            for mb in mbs:
                rows = [self.rows[i] for i in mb]
                z, encs = self.forward_mb(rows, s, [pre[i] for i in mb] if pre else None)
                tg = self.targets(rows, encs); loss = 0.0
                for j, e in enumerate(encs):
                    K = len(e.order)
                    lp = F.log_softmax(z[j, :K].float(), -1); t = torch.tensor(tg[j], device=self.dev, dtype=torch.float32)
                    ce = -(t * lp).sum(); agg["ce"] += ce.item(); agg["acc"] += float(int(lp.argmax()) == int(t.argmax()))
                    loss = loss + ce
                (loss / (nq if a.loss_norm == "step" else a.loss_norm_n)).backward()
                agg["tok"] += sum(len(e.ids) for e in encs); agg["pad_tok"] += max(len(e.ids) for e in encs) * len(encs)
            params = [p for g in self.opt.param_groups for p in g["params"]]
            clip = a.grad_clip if a.grad_clip is not None else R_["grad_clip"]
            if a.clip_scale == "sqrt" and a.loss_norm == "constant":   # gradient norm grows ~ sqrt(questions in step) under the fixed divisor:
                clip = clip * math.sqrt(nq / a.loss_norm_n)          # scale the threshold so it only catches outliers, for any step size
            gn = torch.nn.utils.clip_grad_norm_(params, clip)
            self.opt.step(); self.opt.zero_grad(set_to_none=True); self.step += 1
            torch.cuda.synchronize(); dt = time.time() - t0; tok_run += agg["pad_tok"]
            rec = dict(step=self.step, lr=a.lr * lrm, lr_mult=round(lrm, 5), n_q=nq, n_mb=len(mbs), ce=agg["ce"] / nq,
                       acc=agg["acc"] / nq, tokens=agg["tok"], padded=agg["pad_tok"], grad_norm=float(gn), clip=round(float(clip), 4), sec=round(dt, 2),
                       padded_tok_s=round(agg["pad_tok"] / dt), peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                       beta=[round(x, 3) for x in self.gate.betas().tolist()] if self.gate is not None else None)
            log.write(json.dumps(rec) + "\n"); log.flush()
            if self.step % a.print_every == 0 or self.step == end:
                print(f"[{a.tag}] step {self.step}/{end} ce {rec['ce']:.4f} acc {rec['acc']:.3f} q {nq} lr {rec['lr']:.2e} "
                      f"{dt:.1f}s {rec['padded_tok_s']} padded tok/s peak {rec['peak_gib']} GiB", flush=True)
            if not a.no_save and self.step % a.save_every == 0 and self.step < end:
                self.save_state(self.step, keep=bool(a.keep_every) and self.step % a.keep_every == 0)
                for old in self.out.glob("ckpt_*"):
                    if old.is_dir(): shutil.rmtree(old)
                self.save_light(f"ckpt_{self.step}")
            if (self.out / "DECAY").exists() and not dj.exists():   # a DECAY marker file requests: decay to 0 over the next --final-decay steps, then finish
                decay_start, decay_len, end = self.step, a.final_decay, min(a.steps, self.step + a.final_decay)
                C.jdump(dict(decay_from=decay_start, decay_len=decay_len, end=end), dj); print(f"[{a.tag}] DECAY requested at step {self.step}: cosine to 0 by step {end}", flush=True)
            if (self.out / "STOP").exists():   # a STOP marker file requests: save, write STOPPED.json, exit 0
                if not a.no_save: self.save_light(f"ckpt_{self.step}"); self.save_state(self.step, keep=True)
                C.jdump(dict(stopped_at=self.step, planned=end), self.out / "STOPPED.json"); print(f"[{a.tag}] STOP requested: stopped at step {self.step}", flush=True); log.close(); return
            if self.step - self.start_step == 20:   # measured rate -> projection of the remaining run time
                sec = float(np.mean([json.loads(l)["sec"] for l in open(self.out / "log.jsonl")][-15:]))
                C.jdump(dict(sec_per_step=sec, steps_left_this_run=end - self.step, steps_planned_total=a.steps_planned_total,
                             hours_left_all=(a.steps_planned_total - self.step) * sec / 3600 if a.steps_planned_total else None, at_step=self.step, tag=a.tag), self.out / "projection.json")
        if not a.no_save:
            for old in self.out.glob("ckpt_*"):
                if old.is_dir() and old.name != f"ckpt_{self.step}": shutil.rmtree(old)
            self.save_light(f"ckpt_{self.step}"); self.save_state(self.step, keep=True)
        print(f"[{a.tag}] done: {self.step} steps, {tok_run / max(1e-9, time.time() - t_run):.0f} padded tok/s over the run", flush=True)


def main():
    ap = build_parser(); a = ap.parse_args()
    a.tag = a.tag or f"lr{a.lr:g}"; C.RECIPE["warmup"] = a.warmup
    assert a.header == "none" or a.bidir_layers, "--header needs --bidir-layers"
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cuda.enable_cudnn_sdp(False)
    if a.deterministic:   # bitwise-reproducible steps (math SDPA, deterministic kernels), e.g. for a resume == direct check
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8"); torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_flash_sdp(False)
    Trainer(a).run()


def build_parser():
    R_ = C.RECIPE; ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, required=True); ap.add_argument("--seed", type=int, default=R_["seed"]); ap.add_argument("--tag", default=None)
    ap.add_argument("--train", default=str(C.DATA / "pool/train.jsonl.gz"))
    ap.add_argument("--backbone", default=None); ap.add_argument("--quant", default="bf16", choices=["bf16", "nf4"])
    ap.add_argument("--steps", type=int, default=R_["total_steps"]); ap.add_argument("--until", type=int, default=None, help="stop after this step")
    ap.add_argument("--final-decay", type=int, default=R_["final_decay"]); ap.add_argument("--warmup", type=int, default=R_["warmup"])
    ap.add_argument("--tokens-per-step", type=int, default=R_["tokens_per_step"]); ap.add_argument("--micro-tokens", type=int, default=R_["micro_tokens"])
    ap.add_argument("--plan-micro-tokens", type=int, default=None, help="micro-batch size the data plan is built with (default --micro-tokens); set to the parent run's value to resume its exact data order with smaller runtime micro-batches (needs --repack)")
    ap.add_argument("--plan", choices=["sorted", "fifo"], default="sorted", help="data plan: sorted (the released run: each step takes the longest rows of a look-ahead window) or fifo (every row once per pass, in random order; what finetune.py uses)")
    ap.add_argument("--n-rows", type=int, default=None); ap.add_argument("--print-every", type=int, default=5); ap.add_argument("--save-every", type=int, default=R_["save_every"])
    ap.add_argument("--resume", action="store_true"); ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--deterministic", action="store_true"); ap.add_argument("--lora-r", type=int, default=None)
    ap.add_argument("--steps-planned-total", type=int, default=None, help="for projection.json: all steps the whole job still has to run")
    ap.add_argument("--lr-span", type=float, default=1e-3)
    ap.add_argument("--repack", action="store_true", help="regroup each step's rows into micro-batches by rendered length (same rows, less padding)")
    ap.add_argument("--decay-at", type=int, default=None, help="decay branch: cosine to 0 over [decay_at, decay_at + final_decay), keeping --steps (and so the data order) of the parent run")
    ap.add_argument("--keep-every", type=int, default=0, help="keep the full state every N steps")
    ap.add_argument("--opt-block", action="store_true", help="in causal layers, option blocks cannot see each other (bidigate.set_option_blocks)")
    ap.add_argument("--header", default="none", choices=["none", "own", "group"], help="layout header: the request's questions listed before the state, which never attends to the question block (bidigate.set_state_ends). own: the row's question only; group: every question on the same state in --train. Needs --bidir-layers")
    ap.add_argument("--header-max-q", type=int, default=8)
    ap.add_argument("--serve-layout", action="store_true", help="train in the layout the released code serves: header options sorted by text (not shuffled) and the final line break inside the last option block (votum_config.json: header.sort, close_last)")
    ap.add_argument("--clip-scale", default="fixed", choices=["fixed", "sqrt"], help="gradient-clip threshold: fixed (the recipe's grad_clip, default) or sqrt: grad_clip x sqrt(questions in step / loss_norm_n), with --loss-norm constant")
    ap.add_argument("--grad-clip", type=float, default=None, help="override the recipe's grad_clip (1.0)")
    ap.add_argument("--header-blind", action="store_true", help="system + header never attend to the state or anything after it (bidigate.set_header_ends): the header prefix is cacheable across requests with the same question set")
    ap.add_argument("--header-max-opts", type=int, default=32, help="a choice question lists its option texts in the header up to this many (999 = all)")
    ap.add_argument("--init-from", default=None, help="state_<step>.pt of another run to continue from (used only when this tag has no state of its own)")
    ap.add_argument("--init-weights", default=None, help="light checkpoint dir (adapter/, span_head.pt, gate.pt) to start from with a FRESH optimizer (when the optimizer state is gone); used only when this tag has no state of its own")
    ap.add_argument("--init-step", type=int, default=0, help="with --init-weights: the step to resume the data plan / schedule at")
    ap.add_argument("--plan-origin", type=int, default=0, help="the step at which this run's data plan starts (default 0). With --init-from and a new --train file, set it to the step of the state you continue from, so the new data is read from its beginning")
    ap.add_argument("--rewarm", type=int, default=0, help="linear learning-rate re-warm-up over this many steps from the resume / --init-from step (0 = off)")
    ap.add_argument("--opt-isolate", action="store_true", help="full option isolation: option blocks masked from each other in EVERY layer and the prefix blind to option tokens (bidigate.set_isolate); implies --opt-block")
    ap.add_argument("--shared-positions", action="store_true", help="every option block gets the same position ids (span.shared_positions)")
    ap.add_argument("--bidir-layers", action="store_true", help="interleaved bidirectional attention: layers 3, 7, ..., 31 bidirectional, the rest causal")
    ap.add_argument("--loss-norm", choices=["constant", "step"], default=None,
                    help="divide each step's summed loss by a constant (mean questions/step: every question weighs the same; default for new runs) or by the step's own question count (default when resuming a run that used it)")
    ap.add_argument("--loss-norm-n", type=float, default=None, help="the constant for --loss-norm constant (default: estimated from the data plan)")
    ap.add_argument("--tok-field", default="n_tok", help="row field used as the token count by the data plan (e.g. n_tok_span for the span-rendered length)")
    ap.add_argument("--attn", choices=["sdpa", "flex"], default="sdpa", help="attention kernel of the gated layers: sdpa with the dense float mask (default) or FlexAttention with the same mask rule (flexmask.py)")
    return ap


if __name__ == "__main__":
    main()
