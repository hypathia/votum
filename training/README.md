# Fine-tuning Votum-8B

This guide fine-tunes Votum-8B on your own labelled decisions: prepare the data, train a LoRA adapter on top of the
released model, export a merged checkpoint and refit its calibration temperatures.

You need labelled examples (a few hundred already help, a few thousand are better), one NVIDIA GPU with 24 GB of
memory, and about an hour. In our reference run, 2,000 BANKING77 examples took the model from 72 % to 86 % accuracy
on 77 intents in 49 minutes.

The second half of the page, [Reproducing the original training](#reproducing-the-original-training), is not needed
for fine-tuning.

## What you need

An NVIDIA GPU with at least 20 GB of memory free, and roughly 50 GB of free disk space. The model
is 16 GB to download, the fine-tuned copy you produce is another 16 GB, and the rest is working space.

Install the package, log in to Hugging Face, and download the model once. The model is gated: accept the Apertus
Acceptable Use Policy on the Votum-8B model page first.

```bash
pip install "votum @ git+https://github.com/hypathia/votum" pyarrow
hf auth login
cd training
hf download votum-ai/Votum-8B --local-dir ./Votum-8B
```

Everything below is run from this `training` folder, and `./Votum-8B` is the folder you just downloaded.

## The four steps

Fine-tuning is four commands: convert your examples into training rows, train, export a checkpoint, and refit the
calibration temperatures.

### Step 1: Prepare your examples

The simplest input is a CSV with a text column and a label column. Suppose `tickets.csv` has a `text` column with
support messages and a `team` column with the team that handled each one. Pass the two columns, the question to ask
about every text, and a description for each label:

```bash
python prepare_data.py --csv tickets.csv --text-col text --label-col team \
    --question "Which team should handle this ticket?" \
    --labels "billing=Billing and payments;tech=Technical support;sales=Sales" \
    --val-frac 0.2 --tokenizer ./Votum-8B --out data/tickets
```

The descriptions matter: the model reads them, so "Billing and payments" carries more than a bare code like "B2".

The script validates the rows, holds out a validation split (`--val-frac 0.2` suits a few hundred examples; with
thousands, 0.05 is plenty), and writes both to `data/tickets`. It also reports the token lengths, which matter in the
next step.

Rows are not limited to single-label classification: yes/no questions, graded scales and several questions about
the same text are supported; the section [Other kinds of examples](#other-kinds-of-examples) shows the
format for those.

### Step 2: Train

Run a probe first. It takes under two minutes and reports whether the job fits your GPU and how long the real run
will take:

```bash
python finetune.py --from ./Votum-8B --data data/tickets --tag tickets --probe
```

Then start the real run:

```bash
python finetune.py --from ./Votum-8B --data data/tickets --tag tickets
```

`--tag` names the run; its files go to `runs/tickets`. The script derives the settings from the size of your data
and the free GPU memory, prints them, and logs progress every few seconds. A few hundred examples take around ten
minutes; a few thousand take up to an hour. If the run is interrupted, the same command with `--resume` continues
it.

Memory grows with prompt length. On a 24 GB card the limit is about 2,000 tokens per prompt (state, question and
options together). Longer rows are skipped during training, and the script warns when a noticeable share of yours is
affected. Larger cards handle longer prompts.

When training ends, the script prints the exact command for the next step.

### Step 3: Export the checkpoint

Training produces a LoRA adapter. This step merges it into the base weights and writes a complete checkpoint, used
exactly like the released one:

```bash
python ../scripts/export_hf.py --ckpt runs/tickets/ckpt_<number> --out ./Votum-8B-tickets
```

Replace `ckpt_<number>` with the folder name the training script printed. The result takes 16 GB.

### Step 4: Calibrate and measure

Votum's probabilities are calibrated: of the answers given with 80 % confidence, about 80 % should be right.
Fine-tuning shifts the scores, so the last step refits the per-type temperatures on the validation split from step 1:

```bash
python calibrate.py --model ./Votum-8B-tickets --val data/tickets/val.jsonl.gz
```

It prints the accuracy on those rows and writes the temperatures into the checkpoint. With too few validation rows
it keeps the released temperatures.

The checkpoint is used like the original, for instance with the server (which needs the
`votum[vllm]` install described in the main README):

```bash
votum serve --model ./Votum-8B-tickets
```

To measure the gain, evaluate the original and the fine-tuned model on the same rows; `--eval-only` writes nothing:

```bash
python calibrate.py --model ./Votum-8B         --val data/tickets/val.jsonl.gz --eval-only
python calibrate.py --model ./Votum-8B-tickets --val data/tickets/val.jsonl.gz --eval-only
```

## A complete example

`examples/banking77.sh` runs the four steps end to end on 2,000 examples of BANKING77 (banking customer messages,
77 intents):

```bash
bash examples/banking77.sh ./Votum-8B ./Votum-8B-banking77
```

Results on one RTX 4090, measured on 1,000 held-out messages:

| | Original Votum-8B | After fine-tuning |
|---|---|---|
| Accuracy | 72.4 % | **85.6 %** |
| ECE (expected calibration error; 0 is perfect) | 0.29 | 0.04 |
| Time | | 49 minutes of training, 57 minutes for the whole example |

The original model was under-confident on this task; the fine-tuned one is more accurate and better calibrated.
Results depend on the number of examples and on how separable the labels are.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `CUDA out of memory` when training starts | The GPU does not have enough free memory, often because another program is using it. Close other GPU programs and try again, or add `--micro-tokens 2048` to process less text at once. |
| `CUDA out of memory` in step 4 | The same cause. Run the command again, or add `--batch 2`. |
| `disk guard: ... GiB free` | The disk is nearly full. Free some space; training wants at least 12 GB spare, and step 3 needs 17 GB. |
| `runs/tickets already holds a run` | You used this run name before. Add `--resume` to continue that run, `--fresh` to delete it and start again, or choose another `--tag`. |
| The first step of training takes a few minutes | Normal. The attention kernel (FlexAttention) is compiled once; later steps are fast. |
| The message `the LoRA starts from zero` | Normal. A new adapter is initialised at zero on top of the merged weights. |
| Training accuracy reaches 100 % quickly | Normal for a small set seen many times. What counts is the validation accuracy that step 4 reports. |
| The fine-tuned model is no better than the original | Usually too few examples, or labels that are inconsistent. Check a sample of your data by hand; more examples help more than longer training. |
| `temperature kept at ...` in step 4 | Too few validation rows for a stable fit, so the released temperature is kept. Nothing to fix. |

## Going further

### Keeping general performance

A model fine-tuned on one task can lose some accuracy on others. If one model has to serve many kinds of decisions,
mix general rows into the training file. `--mix-pool` adds a random sample of the public Votum Pool:

```bash
python prepare_data.py --csv tickets.csv --text-col text --label-col team \
    --question "Which team should handle this ticket?" --tokenizer ./Votum-8B \
    --mix-pool 5000 --out data/tickets_mixed
```

One to three pool rows per row of your own is a sensible range.

### Other kinds of examples

A CSV with one answer per text covers the most common case. For anything else, write your examples as a JSON Lines
file, one example per line, and pass it with `--rows my_examples.jsonl` instead of the `--csv` options. Each example
has these fields:

| Field | Meaning |
|---|---|
| `id` | A unique id. |
| `type` | `choice` for picking one option, `noul` for a yes/no question, `score` for a graded scale. |
| `state` | The context: a string or a JSON object. |
| `question` | The question. |
| `options` | The possible answers, each with an `id` and a `text`. Yes/no questions use the ids `true` and `false`. A scale lists its levels from lowest to highest. |
| `target` | A probability per option, in the same order, summing to 1: one-hot for a single gold label, soft (0.7 and 0.3) when annotators disagree. |
| `group` | Optional. Give the same value to examples about the same text, for instance three questions about one ticket. They stay in the same split and are trained as one multi-question request. |

For example, a yes/no question:

```json
{"id": "t17-refund", "type": "noul", "state": "I was charged twice in October.", "question": "Does the customer ask for money back?",
 "options": [{"id": "true"}, {"id": "false"}], "target": [1, 0], "group": "t17"}
```

### Controlling the training

The defaults need no tuning, and the script prints what it picked. To override them:

`--epochs 3` makes three passes over the data. More passes help with few examples, up to a point; watch the
validation accuracy, not the training accuracy. `--steps` sets the number of
update steps directly.

For small data, the script takes many small steps instead of a few large ones, and never fewer than 40. This
matters: with 90 examples, 10 large steps gave no gain, while 40 small steps over the same examples reached 90 %
accuracy.

A larger GPU is used automatically for longer prompts. `--probe` shows the memory actually used on your card.

`python finetune.py --help` lists every option.

### Adapter-only export

Step 3 writes a complete 16 GB checkpoint. To keep only the adapter, add `--adapter-only`: a 160 MB folder. It can
be evaluated with `calibrate.py --exact --eval-only`, but it needs the original model folder on the same machine and
cannot be served. For anything you deploy, use the full export.

### FP8 for 16 GB cards

The 16 GB checkpoint needs a 24 GB card. For a 16 GB card, quantize it to FP8 (about 9 GB) after step 4. It takes
about a minute and needs no GPU:

```bash
votum quantize --model ./Votum-8B-tickets --out ./Votum-8B-tickets-FP8
```

Use it like any other checkpoint (`votum serve --model ./Votum-8B-tickets-FP8`). It gives the same answer as the
full model on most rows, but not all: evaluate both with `calibrate.py --eval-only` as above before you
rely on it.

### One known difference between training and serving

Training and serving treat the very last line break of the prompt slightly differently: serving counts it as part of
the last option, training does not. The released model was trained and is served this way and its accuracy did not
move beyond noise, but the effect has not been measured separately on fine-tuned models. If you compare a fine-tuned
model against the original, measure both with the same tool, as step 4 does.

## Reproducing the original training

How the released Votum-8B was trained from Apertus, for reproduction or audit.

The released model was trained for 600 steps, about seven hours, on a single 96 GB GPU. The training data was
drawn from the public [Votum Pool](https://huggingface.co/datasets/votum-ai/Votum-Pool), with large sources cut to a
limit, and also held some rows the published pool leaves out; that exact file is not published. The published pool marks the rows the released model was trained on (column `in_votum_8b`),
and `prepare_data.py --pool` converts the pool for training. The command was:

```
python train.py \
  --lr 1e-4 --seed 2 --tag votum-8b \
  --train <training file> \
  --steps 1500 --steps-planned-total 1500 \
  --micro-tokens 16384 --plan-micro-tokens 8192 \
  --resume --keep-every 100 --tok-field n_tok_span \
  --opt-block --opt-isolate --shared-positions --bidir-layers --attn flex \
  --header group --header-blind --header-max-opts 999 \
  --loss-norm constant --loss-norm-n 92.81
```

The run was planned for 1,500 steps and stopped at 600: a decay to zero learning rate over 100 steps was triggered at
step 500.

This path needs its own environment, because the original Apertus checkpoint requires the Swiss AI fork of
transformers.

<details>
<summary><b>Software environment for the original training</b></summary>

- Python 3.12
- torch 2.9.0 (`pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128`)
- the Swiss AI version of transformers at commit `3797303dda74844e3d1f8977ff5518bb91f818b4`
  (`https://github.com/swiss-ai/transformers/archive/3797303dda74844e3d1f8977ff5518bb91f818b4.tar.gz`)
- peft 0.21.2

The trainer finds its files through environment variables: `VOTUM_RUNS` for run folders (default `runs` here),
`VOTUM_DATA` for training data (default `data` here), `VOTUM_BACKBONE` for the base model (default
`swiss-ai/Apertus-v1.5-8B`, read from the Hugging Face cache in `HF_HOME`), `VOTUM_TOKENIZER`, `VOTUM_REPO` (the
default for `finetune.py --from`) and `VOTUM_DISK_FLOOR_GB` (how much disk to keep free, default 10).

The same environment lets `finetune.py --mode adapter` continue the released training directly on the Apertus model
instead of on Votum-8B, which yields an adapter relative to Apertus rather than to Votum-8B. The default mode exists
because it needs none of this setup.

</details>

<details>
<summary><b>Detailed record of the released run</b></summary>

The command above was reconstructed from the settings saved with the released model (`train_meta.json`, block
`args`). It shows every setting that differed from the trainer's defaults.

Some settings in that saved file no longer exist in this code. The trainer once supported experimental variants that
the released run did not use; they were removed so that every remaining option works. In the saved file they appear
at their default values and can be ignored. One option was renamed: the saved `--startlux` is `--bidir-layers` here.

Settings that equal the defaults and are therefore not in the command: 25 warm-up steps, a wind-down of 100 steps,
65,536 tokens per step, a learning rate of 1e-3 for the pointer head, and up to 8 questions per request.

How much text was processed at once was set by `--plan-micro-tokens 8192`. The value `--micro-tokens 16384` is in
the saved settings but had no effect in this run. Peak GPU memory was 46.9 GiB.

The wind-down was requested by hand. The trainer checks after every step for an empty file named `DECAY` in the run
folder and, when it finds one, lowers the learning rate to zero over the next 100 steps. In the released run this
happened at step 500, so training ended at step 600. The log shows the learning rate at full value through step
500, at half by step 550 and at zero by step 600.

There was one false start. A wind-down was first requested at step 197 and cancelled at step 215; training then
continued from the state saved at step 200 with the full learning rate. Because of this the log has 615 lines for
600 steps, with steps 201 to 215 recorded twice. The second set is the one that led to the released model.

Two details of the saved file can mislead. Its `recipe` block is an outdated copy of the defaults and shows 400
steps and seed 1, neither of which was used; the `args` block is the accurate one. And its list of adapted layers
names `gate_proj`, a layer Apertus does not have, so it had no effect. The adapter covers the attention and
feed-forward projections of all 32 layers, with rank 16.

</details>

## Files in this folder

| File | What it does |
|---|---|
| `prepare_data.py` | Converts your examples into training and validation rows. |
| `finetune.py` | Fine-tunes a Votum model on them. |
| `calibrate.py` | Refits the temperatures of a fine-tuned model, or evaluates any model on labelled rows. |
| `examples/banking77.sh` | The end-to-end example. |
| `train.py` and the other `.py` files | The trainer and its parts, driven by `finetune.py`. |
