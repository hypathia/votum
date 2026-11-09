# Votum-8B training code

This is the trainer that produced Votum-8B from `swiss-ai/Apertus-v1.5-8B`, plus a small kit to fine-tune Votum-8B on
your own decision data. It is research code. Code for experiments the released run did not use (alternative readouts,
the decision-marker layout, auxiliary losses, other attention layouts) was removed, so every flag in `train.py --help`
works; the command under "Reproducing the released run" reproduces the released run.

| File | Role |
|---|---|
| `prepare_data.py` | your rows or the Votum Pool → the trainer's input (validated, token counts, train/val split, stats) |
| `finetune.py` | fine-tune from the released adapter and head with the released layout; picks the micro-batch for your GPU |
| `calibrate.py` | refit the three temperatures on your validation rows and write them into the exported checkpoint |
| `train.py` | trainer |
| `common.py` | paths, environment, recipe constants |
| `span.py` | prompt rendering, option-block layout and the pointer (span) readout head |
| `model.py` | backbone loading and LoRA |
| `bidigate.py`, `flexmask.py` | attention layout (bidirectional layers, option isolation) on FlexAttention |

## Environment

Fine-tuning from the released checkpoint (the default `finetune.py` mode) needs only `pip install votum pyarrow`
(mainline transformers, PEFT, torch 2.5 or newer for FlexAttention; the same environment serves the model with the
`vllm` extra). The environment below is for `--mode adapter` and for reproducing the released run from the Apertus
base:

- Python 3.12
- torch 2.9.0 (cu128 build: `pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128`)
- the swiss-ai transformers fork at commit `3797303dda74844e3d1f8977ff5518bb91f818b4`
  (`https://github.com/swiss-ai/transformers/archive/3797303dda74844e3d1f8977ff5518bb91f818b4.tar.gz`), which has the
  Apertus 1.5 model class (the `votum` package itself runs on mainline transformers; only this trainer needs the fork)
- peft 0.21.2

Paths are read from environment variables: `VOTUM_DATA` (training data, default `data/` here), `VOTUM_RUNS`
(checkpoints, logs and a tokenizer cache, default `runs/` here), `VOTUM_BACKBONE` (default `swiss-ai/Apertus-v1.5-8B`,
read from the Hugging Face cache in `HF_HOME`, pinned revision preferred), `VOTUM_TOKENIZER`, `VOTUM_REPO` (default
for `finetune.py --from`) and `VOTUM_DISK_FLOOR_GB` (default 10).

## Fine-tune on your own data

Four commands: prepare the rows, train, export, calibrate. Everything runs in the environment `pip install votum`
gives you (mainline transformers, PEFT, torch 2.5+ with FlexAttention; add `pyarrow` for the Pool options), on one GPU,
starting from the released checkpoint. No Apertus download and no transformers fork are needed for this path.

Before you start: `pip install votum pyarrow` (add `pytest` to run the tests); accept the Apertus policy on the
Votum-8B model page and `hf auth login` (as in the Quickstart); have an NVIDIA GPU with at least 19.5 GiB free; have
about 45 GB of disk (16 GB model download, 16 GB export, 12 GB free where the runs go); run the commands from this
`training/` directory. Download the model once into a local directory and use that directory everywhere below, as the
tokenizer, as `--from` and as the model to evaluate (a Hub id works too, but downloads a second copy into the cache):

```bash
hf download {{ORG}}/Votum-8B --local-dir ./Votum-8B
```

Paths come from `VOTUM_RUNS` (default `training/runs`), `VOTUM_TOKENIZER` and `VOTUM_REPO` (default for `--from`).

**Hardware, first.** The backbone trains in bf16 with LoRA and gradient checkpointing; 4-bit (`train.py --quant nf4`)
is untested with this kit and there is no CPU path. The free GPU memory decides the micro-batch (`--micro-tokens`, the
padded tokens of one forward pass), and the micro-batch caps the prompt length you can train on: the data plan skips
rows within 16 tokens of, or longer than, one micro-batch.

| GPU memory | Micro-batch | Peak measured | Trains rows up to | Speed |
|---|---|---|---|---|
| 24 GB (RTX 4090), nothing else on the card | 3,072 tokens | 18.5 GiB (4,096: out of memory) | 3,072 tokens (99.1 % of the Votum Pool) | 1,250 padded tokens/s: about 52 s per 65k-token step |
| 24 GB that also drives a display (about 20 GiB free) | 2,048 tokens | 17.6 GiB | 2,048 tokens (98 % of the Pool) | 1,400 padded tokens/s: about 47 s per step |
| 32 GB | 4,096 tokens | not measured (between the 24 GB and 48 GB points) | 4,096 tokens (99.7 % of the Pool) | |
| 48 GB | 8,192 tokens | not measured directly; 16,384 peaks at 46.9 GiB | 8,192 tokens | |
| 80–96 GB (released run) | 16,384 tokens | 46.9 GiB | all rows | 1,785 padded tokens/s: 41 s per step |

The automatic choice uses the free memory at start: at least 70 GiB gives 16,384 tokens; 44 gives 8,192; 30 gives
4,096; 21 gives 3,072; 19.5 to 21 gives 2,048; below 19.5 GiB free there is no supported configuration. `finetune.py
--probe` runs two steps on your card (in its own `runs/<tag>_probe/`) and prints the peak, so measure before a long run.
Expect roughly 80 minutes per 100 steps on a 24 GB card and 70 minutes on a 96 GB card; the released run was 600 steps.
The first step compiles the FlexAttention kernels.

**1. Prepare the data.** Three inputs are accepted:

```bash
# a classification table: one text column, one label column, the same question for every row
python prepare_data.py --csv tickets.csv --text-col text --label-col team --tokenizer ./Votum-8B --val-frac 0.2 \
    --question "Which team should handle this ticket?" --labels "billing=Billing and payments;tech=Technical support;sales=Sales" --out data/tickets
# rows in the Votum Pool schema (any mix of choice, yes/no and score questions, several questions per state)
python prepare_data.py --rows my_rows.jsonl --val-frac 0.05 --tokenizer ./Votum-8B --out data/mine     # or --val my_val.jsonl (JSONL, same schema)
# the published pool
python prepare_data.py --pool {{ORG}}/Votum-Pool --config default --tokenizer ./Votum-8B --out data/pool   # 223,890 rows, ~4 minutes; needs pyarrow
```

Rows in the [Votum Pool](https://huggingface.co/datasets/{{ORG}}/Votum-Pool) schema are one JSON object per line: `id`,
`type` (`choice`, `noul` for yes/no, `score` for ordered levels), `state` (text or a JSON object), `question`, `options`
(`[{"id", "text"}]`; yes/no rows have ids `true` and `false`; score levels low to high), `target` (one probability per
option, in option order; one-hot for a single label) and, if several rows share a state, `group` (rows of one group stay
on the same side of the split). The preparer validates every row, computes each row's prompt length with the request
header the trainer uses, drops rows over 8,128 tokens (the 8,192-token context minus a margin), splits by group and writes `train.jsonl.gz`,
`val.jsonl.gz` and `stats.json`. The tokenizer is read from `--tokenizer <dir>` (a local Votum-8B or Apertus directory),
else `$VOTUM_TOKENIZER`, else the Apertus snapshot in `HF_HOME`. `--val` takes a JSONL file in the Pool schema (not a
CSV); without it `--val-frac` splits by group: use 0.2 or more for a set of a few hundred rows, so that the
calibration in step 4 has enough rows (the default 0.05 suits thousands of rows). A test file for step 4 is any such
JSONL with `type`, `options` and `target`; `--rows test_rows.jsonl --val-frac 1.0 --out data/test` produces one with
token counts, or write it directly.

**2. Train.** `finetune.py` takes the released checkpoint as the backbone (its adapter is already folded into the
weights), puts a new LoRA on it starting from zero, initialises the pointer head from the released one, and runs
`train.py` with the released layout (option isolation, shared positions, bidirectional layers 3, 7, …, 31, request
headers, FlexAttention):

```bash
python finetune.py --from ./Votum-8B --data data/tickets --tag tickets --probe    # two steps: peak memory and step time
python finetune.py --from ./Votum-8B --data data/tickets --tag tickets            # one pass over the rows
```

Defaults: one epoch (`--epochs`, or `--steps`) but never fewer than 40 optimizer steps (`--min-steps`); the step size
adapts to the data (`--tokens-per-step auto`: 65,536 padded tokens for 2M+ training tokens, 32,768 for 500k+, else
16,384), and the real number of passes is printed. Learning rate 5e-5 for the LoRA and 5e-4 for the head (untuned
starting points), warm-up and a cosine decay sized to the run, options shuffled every step, headers of up to 8
questions, LoRA rank 16. The run writes `$VOTUM_RUNS/<tag>/` (`log.jsonl` with training loss, accuracy and peak
memory per step; `ckpt_<step>/` with the adapter and head; `state_<step>.pt` for resuming). `--dry-run` prints the
`train.py` command and writes nothing; `--resume` continues a stopped or crashed run (it removes a leftover `STOP`
file); `--train-arg=--flag=value` (the `=` form) passes any other trainer flag.

What matters for a small set is the number of updates, not the number of passes: on a 90-row set, 10 steps of 65k
tokens left held-out accuracy at chance while 40 steps of 16k tokens over the same rows reached 90 %. The defaults
above encode that; for a few thousand rows, run several epochs (`--epochs 5`) and check the held-out accuracy that
`calibrate.py` prints in step 4. Mixing your rows with a slice of the Pool keeps the general behaviour (below).

`--mode adapter` is the other way: continue the released adapter itself on the Apertus base, as the released run was
trained. It needs `swiss-ai/Apertus-v1.5-8B` in `HF_HOME` (gated on the Apertus policy) and the training fork from the
Environment section, in its own environment. Use it if you want one adapter relative to Apertus; the default mode gives
an adapter relative to Votum-8B.

**3. Export.** The checkpoint becomes a Votum repository directory (merged bf16 weights, the `adapter/` form relative to
the checkpoint it was trained on, tokenizer, `config.json`, `votum_config.json`), which `Votum.from_pretrained` and
`votum serve` load like the release:

```bash
python ../scripts/export_hf.py --ckpt $VOTUM_RUNS/tickets/ckpt_<step> --out /path/to/Votum-8B-tickets                 # 16 GB on disk (checked first)
python ../scripts/export_hf.py --ckpt $VOTUM_RUNS/tickets/ckpt_<step> --out /path/to/Votum-8B-tickets --adapter-only  # 160 MB: adapter/ only
```

(`finetune.py` prints the exact `--ckpt` path when it finishes; `runs/` is relative to `training/` unless `VOTUM_RUNS`
is set.)

The adapter-only form loads with `Votum.from_pretrained(dir, exact=True)`: it applies the LoRA to the checkpoint it
was trained on, whose absolute path it records, so it is not portable to another machine. `calibrate.py --exact`
works on it (it writes into `adapter/votum_config.json`); serving needs the merged form.

**4. Calibrate and check.** The probabilities are `softmax(scores / T)` with one temperature per question type. After
training the parent's temperatures no longer match, so refit them on your validation rows (minimum cross-entropy on a
grid, as for the release) and write them into the exported checkpoint; the same pass reports accuracy, cross-entropy
and calibration error per type:

```bash
python calibrate.py --model /path/to/Votum-8B-tickets --val data/tickets/val.jsonl.gz      # --dry-run to only report
```

Types with fewer than 50 scored rows (`--min-rows`), or whose fit lands on the edge of the grid (rows too few or too
easy), keep the parent's temperature; `--force` writes such a fit anyway. To measure on a held-out test file (same
format as the validation rows) without writing anything, use the same script as an evaluator, before and after; in
this mode it always reports accuracy, cross-entropy and calibration error, whatever the row count:

```bash
python calibrate.py --model ./Votum-8B --val data/test/val.jsonl.gz --eval-only                 # before
python calibrate.py --model /path/to/Votum-8B-tickets --val data/test/val.jsonl.gz --eval-only  # after
```

Then use the directory exactly like the release: `votum decide --repo /path/to/Votum-8B-tickets`,
`votum serve --model ...`, or `Votum.from_pretrained(...)`. The merged export is itself a valid `--from` for a further
fine-tune in the default mode.

**Keeping the general behaviour.** A narrow fine-tune moves the model towards your rows and away from everything
else. `--mix-pool` adds a random sample of the public pool (config `default`; needs `pyarrow`) to your training file,
shuffled in; 1 to 3 pool rows per row of yours is a reasonable range:

```bash
python prepare_data.py --csv tickets.csv --text-col text --label-col team --question "..." --tokenizer ./Votum-8B \
    --mix-pool 5000 --out data/tickets_mixed          # --pool-dir <local dataset dir> instead of the Hub download
```

Check a few questions outside your domain before and after.

### Worked example: BANKING77

Intent classification with 77 options, a set the released model has never seen (it is one of the out-of-pool
evaluations on the model card). 2,000 of the 10,003 public training rows, converted with `--csv` as above, 200 of them
held out for calibration; accuracy on 1,000 test rows before and after. One RTX 4090, 2,048-token micro-batches.

| | Released Votum-8B | After the fine-tune |
|---|---|---|
| Accuracy on 1,000 held-out test rows (77-way) | 72.4 % | **82.0 %** |
| Expected calibration error on those rows, with the checkpoint's temperatures | 0.286 | 0.035 |
| Choice temperature | 1.516 (released) | 0.99 (refit on 192 validation rows, accuracy there 81.8 %) |
| Training | — | 70 steps of 32 rows, 25 s each, 30 minutes; peak 16.7 GiB; about 1.2 passes over the 1,808 rows |

A 1,100-token row fits a 2,048-token micro-batch alone, so each step carried 32 rows (35k tokens), not the nominal
65k: `--epochs 2` gave 70 steps and about 1.2 passes. `finetune.py` estimates from how many rows fit, so the command
prints the honest count; pass `--steps` or `--tokens-per-step` to change it.

The commands were the four above with `--val-frac 0.1`, `--epochs 2`, `--tokens-per-step 65536` and `--micro-tokens
2048` (the 4090 also drives a display); `examples/banking77.sh`
in this directory downloads the public data and runs the whole example. Expect your own numbers to vary with the amount
of data and the difficulty of the labels; the point is the shape of the result, a large gain on the target task from a
short run.

### Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `CUDA out of memory` in the first step | The micro-batch is too large for the free memory (a display or another process on the card counts). `--probe` first; pass a smaller `--micro-tokens` (2,048 fits a 24 GB card with about 19.5 GiB free). |
| `disk guard: ... has X GiB free, need 2.0 GiB with a 10 GiB floor` at start | Fewer than 12 GB free where the run directory lives (`VOTUM_RUNS`). Free space, point `VOTUM_RUNS` elsewhere, or lower `VOTUM_DISK_FLOOR_GB`. The export later needs another 16 GB (`--adapter-only` avoids it). |
| The first step takes minutes | FlexAttention compiles its kernels once per shape class; the tokenizer also loads slowly the first time (then cached under `runs/cache`). Normal. |
| `the LoRA starts from zero on the given backbone` | Expected in the default mode: the released adapter is already inside the checkpoint's weights. |
| Rows missing from training, fewer questions per step than expected | Rows within 16 tokens of, or longer than, `--micro-tokens` are skipped by the data plan. `stats.json` has the token percentiles; use a larger micro-batch or shorter states. |
| The run crashed or was stopped | `finetune.py ... --tag <same tag> --resume` continues from the last `state_<step>.pt` (saved every `--save-every` steps). Without `--resume`, the same tag starts over. |
| `temperature hit the grid edge` from `calibrate.py` | The validation rows are too few or too easy for a stable fit. Use more rows (a few hundred per type) or keep the parent's temperatures (`--min-rows` above the count). |
| Training loss near 0 after a few steps | The set is small and the 40-step minimum repeats it many times (the pass count is printed). That is usually fine for a narrow task; check the held-out accuracy, and add `--mix-pool` to keep the general behaviour. |
| Held-out accuracy did not improve | Too few optimizer updates: with `--steps` below 40 or a huge `--tokens-per-step`, a small set gets a handful of updates. Use the defaults, or more steps with a smaller `--tokens-per-step`. |
| `this checkpoint is the Apertus 1.5 multimodal release, which needs the swiss-ai transformers fork ...` | You are in `--mode adapter` without the training fork. Use the default mode, or install the Environment above in its own venv. |

**Not covered.** Changing the layout or the LoRA rank, 4-bit training, multi-GPU, prompts longer than your micro-batch,
and quantizing the result (`scripts/quantize_fp8.py` works on any exported directory, but measure parity as on the
Votum-8B-FP8 card before relying on it).

## Data

The training data of the released run is not included; `../data/` lists its sources and licences. `train.py` reads a
gzipped JSONL file (`--train`) in the Pool schema with a token-count field, which `prepare_data.py` produces from the
published Votum Pool or from your rows. The released run read the pool v7 training file (not released; its cleaned
successor is the Votum-Pool dataset). Without `--train`, `train.py` reads `$VOTUM_DATA/pool/train.jsonl.gz`.

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
