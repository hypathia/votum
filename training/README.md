# Votum-8B training code

This is the trainer that produced Votum-8B from `swiss-ai/Apertus-v1.5-8B`. It is research code. Code for experiments the
released run did not use (alternative readouts, the decision-marker layout, auxiliary losses, other attention layouts)
was removed, so every flag in `train.py --help` works; the command below reproduces the released run.

| File | Role |
|---|---|
| `train.py` | trainer |
| `common.py` | paths, environment, recipe constants |
| `span.py` | prompt rendering, option-block layout and the pointer (span) readout head |
| `model.py` | backbone loading and LoRA |
| `bidigate.py`, `flexmask.py` | attention layout (bidirectional layers, option isolation) on FlexAttention |

## Environment

- Python 3.12
- torch 2.9.0 (cu128 build: `pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128`)
- the swiss-ai transformers fork at commit `3797303dda74844e3d1f8977ff5518bb91f818b4`
  (`https://github.com/swiss-ai/transformers/archive/3797303dda74844e3d1f8977ff5518bb91f818b4.tar.gz`), which has the
  Apertus 1.5 model class (the `votum` package itself runs on mainline transformers; only this trainer needs the fork)
- peft 0.21.2

Paths are read from environment variables: `VOTUM_DATA` (training data, default `data/` here), `VOTUM_RUNS`
(checkpoints, logs and a tokenizer cache, default `runs/` here), `VOTUM_BACKBONE` (default `swiss-ai/Apertus-v1.5-8B`,
read from the Hugging Face cache in `HF_HOME`) and `VOTUM_DISK_FLOOR_GB` (default 30).

## Data

The training data is not included; `../data/` lists its sources and licences. `train.py` reads a gzipped JSONL file
(`--train`); the released run read the pool v7 training file (not released; its cleaned successor is the Votum-Pool dataset).
Without `--train`, `train.py` reads `$VOTUM_DATA/pool/train.jsonl.gz`.

## Reproducing the released run

The command below is reconstructed from the `args` block of `train_meta.json` in the released checkpoint's run directory
(`ckpt_600/train_meta.json`). It lists the flags whose recorded value differs from the default, plus the data path
(`<pool v7 train.jsonl.gz>` stands for the pool v7 training file). Three flags are renamed or gone: the run's
`--startlux` is `--bidir-layers` here; `--no-marker` and `--no-kl` are no longer flags because the no-marker layout and
the absence of a KL term are now the only behaviour. `--tag` only names the output directory (`runs/<tag>/`).

```
python train.py \
  --lr 1e-4 --seed 2 --tag votum-8b \
  --train <pool v7 train.jsonl.gz> \
  --steps 1500 --steps-planned-total 1500 \
  --micro-tokens 16384 --plan-micro-tokens 8192 \
  --resume --keep-every 100 --tok-field n_tok_span \
  --opt-block --opt-isolate --shared-positions --bidir-layers --attn flex \
  --header group --header-blind --header-max-opts 999 \
  --loss-norm constant --loss-norm-n 92.81
```

Recorded values that equal the defaults and are therefore omitted: `--warmup 25`, `--final-decay 100`,
`--tokens-per-step 65536`, `--save-every 25`, `--print-every 5`, `--quant bf16`, `--lr-span 1e-3` and
`--header-max-q 8`. The recorded `train_meta.json` also lists flags of removed experiments (`--readout`, `--base`,
`--lr-gate`, `--loss`, `--mj-*`, `--gold-drop*`, `--bidi-gate`, ...) at their defaults; this trainer no longer has them.
`--steps-planned-total` only feeds the cost projection file and does not change training. `--loss-norm-n 92.81` is the value the run recorded
(otherwise estimated from the data plan).

### Schedule as recorded

- seed 2, peak learning rate 1e-4 (span head 1e-3), 25 warm-up steps, then constant
- `--steps 1500` was the planned length, so the data order is that of a 1500-step plan; the run was cut short by a decay
  request, and it stopped at step 600
- cosine decay to 0 over the final 100 steps (steps 500 to 600)
- 65,536 tokens per optimizer step (micro-batches of 8,192 tokens in the data plan)
- hardware: one RTX PRO 6000 (96 GB), about 7 hours (the step times in `log.jsonl` sum to 7.05 h, including 15 superseded steps)

### How the decay was triggered

`train.py` checks, after each step, for an empty marker file named `DECAY` in the run directory. When it sees one it
writes `DECAYING.json` (`decay_from` = the current step, `decay_len` = `--final-decay`, `end` = `decay_from + decay_len`)
and runs a cosine decay to 0 from that step. The released run's `DECAYING.json` reads
`{"decay_from": 500, "decay_len": 100, "end": 600}`. `log.jsonl` shows `lr_mult` 1.0 through step 500, 0.99975 at step 501,
0.5 at step 550 and 0.0 at step 600. So the decay was requested by hand when the run reached step 500; it was not part of
the command line. The `DECAY` file itself was not kept.

An earlier decay had been requested at step 197 (`DECAYING.json` with 197 to 297, kept in `_decay_197/`). That attempt was
abandoned after step 215: the run was resumed from its step-200 state with the file moved aside and constant LR. As a
result `log.jsonl` has 615 lines for 600 steps: steps 201 to 215 appear twice, first with the abandoned decayed LR
(`lr_mult` 0.996 down to 0.922) and then with constant LR. The resumed steps 201 onward (constant LR) are the ones that
produced the released weights, assuming the resume loaded the step-200 state (state is saved every 25 steps). `projection.json` (written 20 steps after the resume, at step 220) belongs to the resumed
segment.

### Notes on `train_meta.json`

The `recipe` block in the released `train_meta.json` is an older, longer copy of `common.RECIPE` (it shows `total_steps`
400 and `seed` 1, which the run did not use); the `args` block is authoritative. Its LoRA targets include `gate_proj`,
which Apertus does not have (its MLP is `up_proj` and `down_proj` only), so PEFT ignored it; the training run's
`adapter_config.json` also lists it, the published one does not. The adapter covers `q_proj`, `k_proj`, `v_proj`,
`o_proj`, `up_proj` and `down_proj` (64 tensors each); LoRA rank 16, alpha 32, dropout 0.
