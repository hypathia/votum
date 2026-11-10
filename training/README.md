# Fine-tuning Votum-8B

Votum-8B answers decision questions out of the box: you give it a text, a question and a list of options, and it
returns a probability for each option. Often that is enough. When it is not, because your categories are unusual,
your wording is specific to your company, or you simply need more accuracy, you can teach it with examples of your
own. That is fine-tuning, and this page shows how to do it.

You need labelled examples (a few hundred already help, a few thousand are better), one NVIDIA GPU with 24 GB of
memory, and about an hour. You do not need to know how the model works inside. In our reference run, 2,000 examples
of banking customer messages took the model from 72 % to 86 % accuracy on 77 possible intents in 49 minutes.

The second half of the page is for a different reader: someone who wants to reproduce how the released model itself
was trained. If you only want to fine-tune, you can stop before it.

## What you need

A computer with an NVIDIA GPU that has at least 20 GB of memory free, and roughly 50 GB of free disk space. The model
is 16 GB to download, the fine-tuned copy you produce is another 16 GB, and the rest is working space.

Install the package, sign in to Hugging Face, and download the model once. Votum-8B is built on the Apertus model, so
you first accept the Apertus usage policy on the Votum-8B model page; the download works after that.

```bash
pip install "votum @ git+https://github.com/hypathia/votum" pyarrow
hf auth login
cd training
hf download {{ORG}}/Votum-8B --local-dir ./Votum-8B
```

Everything below is run from this `training` folder, and `./Votum-8B` is the folder you just downloaded.

## The four steps

Fine-tuning is four commands. You turn your examples into a training file, train, save the result as a model folder,
and adjust how confident the new model is. Each step is explained below with the command to run.

### Step 1: Prepare your examples

The easiest starting point is a spreadsheet saved as CSV, with one column holding the text and one holding the
correct answer. Suppose `tickets.csv` has a `text` column with support messages and a `team` column saying which team
handled each one. You tell the tool which columns to use, what question the model should answer about every text, and
a readable name for each answer:

```bash
python prepare_data.py --csv tickets.csv --text-col text --label-col team \
    --question "Which team should handle this ticket?" \
    --labels "billing=Billing and payments;tech=Technical support;sales=Sales" \
    --val-frac 0.2 --tokenizer ./Votum-8B --out data/tickets
```

The readable names matter. The model reads them, so "Billing and payments" teaches it more than a bare code like
"B2" would.

The tool checks your examples, sets one part aside for testing the model later (`--val-frac 0.2` keeps a fifth, a
good choice for a few hundred examples; with thousands, a twentieth is plenty), and writes the result to
`data/tickets`. It also tells you how long your texts are, which matters in the next step.

Your examples do not have to be single-answer categories. Votum also handles yes/no questions, graded scales, and
several questions about the same text; the section [Other kinds of examples](#other-kinds-of-examples) shows the
format for those.

### Step 2: Train

Do a quick trial first. It runs for under two minutes and tells you whether the job fits your GPU and how long the
real run will take:

```bash
python finetune.py --from ./Votum-8B --data data/tickets --tag tickets --probe
```

Then start the real run:

```bash
python finetune.py --from ./Votum-8B --data data/tickets --tag tickets
```

`--tag` is just a name for this run; its files go to `runs/tickets`. The script decides the settings for you from the
size of your data and the memory on your GPU, prints what it chose, and shows the training progress every few
seconds. A few hundred examples take around ten minutes; a few thousand take up to an hour. If the run is
interrupted, the same command with `--resume` added picks up where it stopped.

One limit to know about: the longer a text, the more GPU memory it needs. On a 24 GB card, texts up to roughly 1,300
words are fine (the limit is about 2,000 tokens, a token being three quarters of a word, and it includes the question
and the options). Longer examples are skipped during training, and the script warns you when a noticeable share of
yours is affected. Larger cards handle longer texts.

When training ends, the script prints the exact command for the next step.

### Step 3: Save the model

Training produces a small file of adjustments. This step combines it with the original model into a complete model
folder that you use exactly like the released one:

```bash
python ../scripts/export_hf.py --ckpt runs/tickets/ckpt_<number> --out ./Votum-8B-tickets
```

Replace `ckpt_<number>` with the folder name the training script printed. The result takes 16 GB.

### Step 4: Calibrate and measure

Votum does not only pick an answer, it says how sure it is, and those probabilities are meant to be trustworthy: of
all the answers it gives with 80 % confidence, about 80 % should be right. Training shifts that balance, so the last
step restores it using the examples that were set aside in step 1:

```bash
python calibrate.py --model ./Votum-8B-tickets --val data/tickets/val.jsonl.gz
```

It prints the accuracy on those examples and saves the adjustment into the model folder. If you had only a handful
of test examples, it leaves the confidence settings as they were, which is the safe choice.

Your model is ready. Use it wherever you used the original, for instance with the server (which needs the
`votum[vllm]` install described in the main README):

```bash
votum serve --model ./Votum-8B-tickets
```

To see how much the fine-tuning helped, compare the original and the new model on the same examples. The option
`--eval-only` makes the tool measure without changing anything:

```bash
python calibrate.py --model ./Votum-8B         --val data/tickets/val.jsonl.gz --eval-only
python calibrate.py --model ./Votum-8B-tickets --val data/tickets/val.jsonl.gz --eval-only
```

## A complete example

To try the whole process before using your own data, run the example that ships with the code. It downloads
BANKING77, a public collection of banking customer messages labelled with one of 77 intents such as "card arrival"
or "declined card payment", and runs the four steps on 2,000 of them:

```bash
bash examples/banking77.sh ./Votum-8B ./Votum-8B-banking77
```

Here is what it produced on a single RTX 4090, measured on 1,000 messages the model never saw during training:

| | Original Votum-8B | After fine-tuning |
|---|---|---|
| Correct answers | 72.4 % | **85.6 %** |
| Calibration error (0 is perfect) | 0.29 | 0.04 |
| Time | | 49 minutes of training, 57 minutes for the whole example |

Calibration error is the average gap between how confident the model says it is and how often it is right. The
original model was unsure of itself on this task; the fine-tuned and calibrated one is both more accurate and more
honest about its confidence.

Your results will depend on how many examples you have and how hard your categories are to tell apart. The pattern
to expect is the one above: a clear gain on your task from a short run.

## If something goes wrong

| What you see | What it means and what to do |
|---|---|
| `CUDA out of memory` when training starts | The GPU does not have enough free memory, often because another program is using it. Close other GPU programs and try again, or add `--micro-tokens 2048` to process less text at once. |
| `CUDA out of memory` in step 4 | The same cause. Run the command again, or add `--batch 2`. |
| `disk guard: ... GiB free` | The disk is nearly full. Free some space; training wants at least 12 GB spare, and step 3 needs 17 GB. |
| `runs/tickets already holds a run` | You used this run name before. Add `--resume` to continue that run, `--fresh` to delete it and start again, or choose another `--tag`. |
| The first step of training takes a few minutes | Normal. The software prepares itself for your GPU once; later steps are fast. |
| The message `the LoRA starts from zero` | Normal. It only says that the new adjustments start empty. |
| Accuracy on the training examples reaches 100 % quickly | Normal for a small set, which the model sees many times. What counts is the accuracy on the examples set aside, which step 4 reports. |
| The fine-tuned model is no better than the original | Usually too few examples, or labels that are inconsistent. Check a sample of your data by hand; more examples help more than longer training. |
| `temperature kept at ...` in step 4 | There were too few test examples to adjust the confidence reliably, so it was left unchanged. Nothing to fix. |

## Going further

### Keeping the model good at everything else

A model trained only on support tickets gets better at support tickets and may get slightly worse at unrelated
questions. If you plan to use one model for many kinds of decisions, mix some general examples into your training
file. The data the original model learned from is public, and one option adds a random sample of it:

```bash
python prepare_data.py --csv tickets.csv --text-col text --label-col team \
    --question "Which team should handle this ticket?" --tokenizer ./Votum-8B \
    --mix-pool 5000 --out data/tickets_mixed
```

Adding between one and three general examples for each of your own is a sensible range.

### Other kinds of examples

A CSV with one answer per text covers the most common case. For anything else, write your examples as a JSON Lines
file, one example per line, and pass it with `--rows my_examples.jsonl` instead of the `--csv` options. Each example
has these fields:

| Field | Meaning |
|---|---|
| `id` | Any unique name for the example. |
| `type` | `choice` for picking one option, `noul` for a yes/no question, `score` for a graded scale. |
| `state` | The text the question is about. It can also be a structured record, such as an order with its items. |
| `question` | The question. |
| `options` | The possible answers, each with an `id` and a `text`. Yes/no questions use the ids `true` and `false`. A scale lists its levels from lowest to highest. |
| `target` | How correct each option is, in the same order, as numbers that add up to 1. For a single correct answer that is a 1 for it and 0 for the others. If annotators disagreed, you can express that: 0.7 and 0.3. |
| `group` | Optional. Give the same value to examples about the same text, for instance three questions about one ticket. They are then kept together, and the model learns to answer them in one go, as it does when you send several questions in one request. |

For example, a yes/no question:

```json
{"id": "t17-refund", "type": "noul", "state": "I was charged twice in October.", "question": "Does the customer ask for money back?",
 "options": [{"id": "true"}, {"id": "false"}], "target": [1, 0], "group": "t17"}
```

### Controlling the training

The defaults are chosen to work without tuning, and the script prints what it picked. If you want to steer it:

`--epochs 3` goes over your examples three times instead of once. More passes help when you have few examples, up to
a point; watch the accuracy on the set-aside examples rather than the training accuracy. `--steps` sets the number of
update steps directly.

For small data, the script deliberately takes many small steps instead of a few large ones, and never fewer than
forty. We found this matters a great deal: with ninety examples, ten large steps taught the model nothing, while
forty small steps over the same examples reached 90 % accuracy.

If you have a larger GPU, the script uses it automatically to handle longer texts. The trial run with `--probe`
shows the memory actually used on your card.

`python finetune.py --help` lists every option.

### A smaller file instead of a full model

Step 3 writes a complete 16 GB model. If you only want to keep the trained adjustments, add `--adapter-only` and you
get a 160 MB folder instead. It can be measured with `calibrate.py --exact --eval-only`, but it only works on the
computer that still has the original model folder, and it cannot be served. For anything you deploy, use the full
export.

### One known difference between training and use

Training and serving treat the very last line break of the prompt slightly differently: serving counts it as part of
the last option, training does not. The released model was trained and is served this way and its accuracy did not
move beyond noise, but the effect has not been measured separately on fine-tuned models. If you compare a fine-tuned
model against the original, measure both with the same tool, as step 4 does.

## Reproducing the original training

This part is for readers who want to reproduce how the released Votum-8B was trained from the Apertus model, or
audit it. It is not needed for fine-tuning.

The released model was trained for 600 steps, about seven hours, on a single 96 GB GPU. The training data was an
earlier version of the public [Votum Pool](https://huggingface.co/datasets/{{ORG}}/Votum-Pool) dataset; that exact
file is not published, but the cleaned dataset that replaced it is, and `prepare_data.py --pool` converts it for
training. The command was:

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

The run was planned for 1,500 steps and stopped at 600: at step 500 we asked it to wind down, which it does by
lowering the learning rate to zero over the next hundred steps.

This path needs its own software environment, separate from the one used above, because Apertus in its original
form requires a special version of the transformers library.

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
instead of on Votum-8B, which yields adjustments relative to Apertus rather than to Votum-8B. The default mode exists
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
65,536 tokens per step, a learning rate of 1e-3 for the output layer, and up to 8 questions per request.

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
| `prepare_data.py` | Turns your examples into training and test files. |
| `finetune.py` | Fine-tunes a Votum model on them. |
| `calibrate.py` | Adjusts the confidence of a fine-tuned model, or measures any model on labelled examples. |
| `examples/banking77.sh` | The complete example. |
| `train.py` and the other `.py` files | The underlying trainer and its parts, which `finetune.py` drives for you. |
