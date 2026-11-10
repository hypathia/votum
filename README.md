<h1 align="center"><img src="assets/flag.svg" width="28" height="28" alt="Swiss flag"> votum</h1>

<p align="center"><b>Code for the Votum decision models: run them, serve them, fine-tune them.</b></p>

<p align="center"><sub>Independent project, built on Apertus-1.5-8B (Swiss AI Initiative) · not affiliated</sub></p>

<p align="center">
  <a href="https://huggingface.co/votum-ai/Votum-8B"><img alt="model" src="https://img.shields.io/badge/model-Votum--8B-DA291C?logo=huggingface&logoColor=white"></a>
  <a href="https://huggingface.co/datasets/votum-ai/Votum-Pool"><img alt="dataset" src="https://img.shields.io/badge/data-Votum--Pool-DA291C?logo=huggingface&logoColor=white"></a>
  <a href="https://huggingface.co/swiss-ai/Apertus-v1.5-8B"><img alt="base" src="https://img.shields.io/badge/base-Apertus--1.5--8B-1D1D1F"></a>
  <img alt="python" src="https://img.shields.io/badge/python-3.11%2B-1D1D1F">
  <img alt="license" src="https://img.shields.io/badge/license-Apache--2.0-1D1D1F">
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> ·
  <a href="#serve">Serve</a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="#reproduce">Reproduce and fine-tune</a> ·
  <a href="#acknowledgements">Acknowledgements</a> ·
  <a href="https://huggingface.co/votum-ai/Votum-8B">Model card</a>
</p>

**Votum** turns <img src="assets/flag.svg" width="14" height="14" alt="Swiss flag"> [Apertus-1.5-8B](https://huggingface.co/swiss-ai/Apertus-v1.5-8B)
into a decision model. You give it some context (an email, a document, a JSON record), a question about it and the
possible answers. It gives back a probability for each answer.

It does not write text. It reads the prompt once and scores the answers, so a decision takes about 40 ms on an
RTX 4090, the result is always one of your options, and the probabilities always sum to one.

<picture><source media="(prefers-color-scheme: dark)" srcset="assets/architecture_dark.png"><img src="assets/architecture.png" alt="The request as one prompt, the attention mask, the backbone and the pointer head"></picture>

<picture><source media="(prefers-color-scheme: dark)" srcset="assets/examples_dark.png"><img src="assets/examples.png" alt="Real outputs of Votum-8B"></picture>

## Quickstart

```bash
pip install "votum @ git+https://github.com/hypathia/votum"            # to use the model from Python
pip install "votum[vllm] @ git+https://github.com/hypathia/votum"      # the same, plus the HTTP server
```

You need Python 3.11 or newer and a CUDA GPU with 24 GB, or 16 GB with the [FP8 variant](#on-a-16-gb-card). It also
runs on a CPU, slowly.

Before the first download, accept the Apertus Acceptable Use Policy on
[votum-ai/Votum-8B](https://huggingface.co/votum-ai/Votum-8B) and run `hf auth login`. That repository holds everything
the model needs; there is no base model to fetch.

```python
from votum import Votum

vm = Votum.from_pretrained("votum-ai/Votum-8B")     # 16 GB of weights; fits a 24 GB GPU

out = vm.decide({
    "state": {"from": "anna@example.com", "subject": "Invoice 4411", "body": "I was charged twice in October."},
    "questions": {
        "team":   {"type": "choice", "instructions": "Which team handles this?",
                   "criteria": {"billing": "Billing and payments", "tech": "Technical support", "sales": "Sales"}},
        "refund": {"type": "noul",   "instructions": "Does the customer ask for money back?"},
        "tone":   {"type": "score",  "instructions": "How upset is the customer?",
                   "criteria": ["calm", "annoyed", "angry"]},
    },
})
out["answers"]["team"]     # {"type": "choice", "choice": "billing", "confidence": ..., "probabilities": {...}}
```

The context is called the `state`; it can be a string or any JSON object. A request asks up to 8 questions about one
state, and they are answered together. There are three kinds of question:

| `type` | Use it for | You give | You get back |
|---|---|---|---|
| `choice` | picking one of several options | `criteria`: the options, as `{id: description}` or a list of ids | `choice` (the most likely option), `confidence`, and `probabilities` for every option |
| `noul` | a yes/no question | nothing more (optionally a description of what yes and no mean) | `noul`: the probability of yes |
| `score` | a rating on a scale | `criteria`: the levels, lowest first | `score` (the expected level, e.g. 1.3), `confidence`, and `probabilities` for every level |

Every question also takes `instructions`, the question itself. The names `noul`, `instructions` and `criteria` come
from the request format of the [Jev and Jeff](#acknowledgements) decision models, which Votum follows so that
existing clients work unchanged.

A question whose prompt is longer than 8,192 tokens is refused, not cut short: an error in Python, HTTP 413 on the
server.

## Serve

```bash
pip install "votum[vllm] @ git+https://github.com/hypathia/votum"
VOTUM_API_KEY=... votum serve --model votum-ai/Votum-8B --host 0.0.0.0 --port 8011
```

`POST /v1/systemone` takes the same request as `vm.decide` and returns the same answers. Requests that arrive at the
same time are processed together. Start-up takes about a minute.

### On a 16 GB card

Use [votum-ai/Votum-8B-FP8](https://huggingface.co/votum-ai/Votum-8B-FP8): the same model with its weights stored in
8 bits instead of 16. It takes a little over half the memory (9.2 GB) and gives the same answer as the full model
96–99 % of the time. Everything above works unchanged with its name:

```bash
votum serve --model votum-ai/Votum-8B-FP8 --port 8011
```

```python
vm = Votum.from_pretrained("votum-ai/Votum-8B-FP8")     # 8.5 GB on the GPU
```

The server reserves the weights plus 3 GB of GPU memory, so about 12 GB for this variant and 18 GB for the full one;
`--gpu-memory-gb` changes that. To make an FP8 copy of your own fine-tuned checkpoint, run
`votum quantize --model <dir> --out <dir>` (about a minute, no GPU needed).

**How fast is it?** A decision costs one read of the prompt. Here it is next to asking the same Apertus model to write
its answer as text, on one RTX 4090:

| How the answer is produced | Typical latency | Tokens written | Requests per second under load |
|---|---:|---:|---:|
| **Votum: a decision** | **39 ms** | **0** | **23.5** |
| Text: the choice as JSON (structured output) | 206 ms | 8 | 21.0 |
| Text: the choice and its probabilities as JSON (structured output) | 1.1 s | 54 | 10.9 |
| Text: reasoning first, then the JSON | 10.5 s | 370 | 1.9 |

Questions about the same state share one pass over it, so asking several at once is cheaper than asking them one by
one: on long documents the server makes 3.1 decisions per second with one question per request, and 15.8 with eight.
The [model card](https://huggingface.co/votum-ai/Votum-8B#performance) has the full measurements.

<details>
<summary><b>Server settings</b></summary>

All settings are environment variables:

| Variable | Default | Effect |
|---|---|---|
| `VOTUM_API_KEY` | unset | when set, requests need `Authorization: Bearer <key>` |
| `VOTUM_PREFIX_CACHE_MB` | 512 | memory for reusing the question list of a request seen before |
| `VOTUM_STATE_CACHE_MB` | 0 (off) | memory for reusing a state seen before |
| `VOTUM_MAX_BODY_MB` | 8 | largest request body; beyond it HTTP 413 |
| `VOTUM_MAX_QUESTIONS` | 64 | questions per request; beyond it HTTP 400 |
| `VOTUM_MAX_OPTIONS` | 2000 | options per question; beyond it HTTP 400 |
| `VOTUM_MAX_QUEUE` | 4096 | questions waiting for the GPU; beyond it HTTP 503 |
| `VOTUM_HEADER_CHUNK` | 0 (off) | `1` splits requests of more than 8 questions into separate headers (not measured on labelled data) |

vLLM's own prefix cache and chunked prefill stay off, because some of Votum's layers read the prompt in both
directions and need all of it at once. The two caches above take their place, and they return exactly what a fresh
computation would.

</details>

## How it works

The diagram at the top follows one request through the model.

**One prompt.** A request becomes a single prompt: a header that lists its questions, then the state, then each
question with its options. The model was trained on this layout, and the package builds it for you.

**Who reads whom.** The model is told which parts of the prompt may look at which. The header never reads the state,
the state never reads a question, and the options never see each other. Because of that, the header and the state
are the same for every question of the request and are computed once. Every option also starts at the same position,
so the order in which you list the options carries no signal.

**One score per option.** Each option ends with a marker token. A small extra layer (the "pointer head") reads what
the model computed at that marker and turns it into a score. The scores of a question's options are then normalised
into probabilities.

**Limits.** A question's prompt can hold 8,192 tokens, which is how far the model was trained and tested. Any number
of options fits within that, though accuracy falls as their number grows. A request can ask up to 8 questions. The
model card's Limitations section has the details.

<details>
<summary><b>Weights and implementation</b></summary>

- **Adapter and merged weights.** Votum is a LoRA adapter (a small set of weight changes) and a head on top of
  Apertus. The published checkpoint has the adapter already folded into the Apertus weights (bf16), so it is one
  download and loads like any 8B model. The unmerged adapter is in the model repo's `adapter/` folder; load it with
  `Votum.from_pretrained(repo, exact=True)` or build a serving checkpoint with `votum build --exact`. The two forms
  agree on about 98 % of answers. `votum build` rebuilds the shipped weights byte for byte from the official Apertus
  snapshot and the adapter.
- **Layers and head.** Eight of the 32 layers (3, 7, …, 31) attend in both directions; the other 24 stay causal. The
  head scores an option as ⟨q₀, W_o · LN(h)⟩ / 16, where h is the final hidden state at the option's end token.
- **The attention layout is rules, not weights.** The rules live in `votum_config.json`. `src/votum/modeling.py`
  applies them as dense masks and `src/votum/vllm/isolate.py` as FlexAttention; `tests/test_masks.py` checks that the
  two produce the same mask and positions for single-question prompts.
- **Two paths.** The PyTorch reference runs one pass per question. The server packs a request's questions into one
  prompt, each at the positions it would have alone; the two agree on 97 to 99 % of answers (parity numbers are on the
  model card).

</details>

## Repository

```
src/votum/          the model in PyTorch · prompt builder · request format · checkpoint builder · FP8 · command line
src/votum/vllm/     the vLLM 0.33 server: model, attention rules and caches, client, HTTP API
training/           fine-tuning kit (prepare_data · finetune · calibrate) and the trainer that produced Votum-8B
data/               source list with licences · cleaning and personal-data removal
scripts/            export a checkpoint to the Hub layout · smoke fixture · README graphics
tests/              attention rules, prompt builder, request format (CPU) · end-to-end test (GPU)
```

## Reproduce

**Fine-tune on your own data.** [`training/README.md`](training/README.md) is a step-by-step guide: prepare your rows,
train from the released checkpoint, export, calibrate. A 24 GB card trains rows up to 3,072 tokens, or 2,048 if it
also drives a display.

**Reproduce the release.**

1. **Data.** The cleaned pool is [votum-ai/Votum-Pool](https://huggingface.co/datasets/votum-ai/Votum-Pool); the pipeline
   that builds it is in `data/`.
2. **Training.** `training/README.md` has the exact command of the released run: one 96 GB GPU, about 7 hours.
3. **Export.** `python scripts/export_hf.py --ckpt <run>/ckpt_600 --out <dir>` reproduces the published model
   repository byte for byte.

The released checkpoint was trained before the pool was cleaned. Its known training-data issues are listed on the
model card, and a re-train on the published pool is planned.

## Tests

```bash
pip install -e ".[test]"
pytest                                                     # CPU
VOTUM_TOKENIZER=<dir with the Apertus tokenizer> pytest    # + renderer
VOTUM_REPO=votum-ai/Votum-8B pytest -m gpu                   # + end-to-end smoke test
```

## Acknowledgements

Votum builds on other people's work:

- **[Apertus](https://huggingface.co/swiss-ai/Apertus-v1.5-8B)**, by the Swiss AI Initiative (EPFL, ETH Zurich, CSCS):
  the open model underneath Votum.
- **Jev**: the original decision models. Votum uses their request format (`choice`, `noul`, `score`) and is measured on
  their public benchmarks, [JevBench](https://github.com/fstandhartinger/jevbench) and the Jev Decision Index.
- **[Jeff](https://github.com/j-geuter/Jeff)**, by Jonathan Geuter, Andrew Cheng and David Alvarez-Melis
  ([Releasing the Inner Jev](https://arxiv.org/abs/2610.17205)): the open recipe for turning a language model into a
  decision model. Our evaluation pipeline was checked against their published numbers, and the code that reads
  requests is derived from theirs (see `NOTICE`).
- **[JevPertus](https://huggingface.co/jaluus/jevpertus-v1.5-8B)** ([code](https://github.com/Jaluus/JevPertus)): an
  earlier decision model on the same Apertus-1.5-8B backbone, also a LoRA adapter with a pointer head.

## License

Apache-2.0 (`LICENSE`, `NOTICE`). The base model carries the
[Apertus Acceptable Use Policy](https://github.com/swiss-ai/apertus-legal/blob/main/apertus_1.5/USAGE_POLICY.pdf), which
binds anyone who uses Votum. Votum is an independent project, not affiliated with or endorsed by the Swiss AI
Initiative.
