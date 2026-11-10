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
  <a href="#images">Images</a> ·
  <a href="#serve">Serve</a> ·
  <a href="#speed">Speed</a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="#fine-tune">Fine-tune</a> ·
  <a href="#acknowledgements">Acknowledgements</a> ·
  <a href="https://huggingface.co/votum-ai/Votum-8B">Model card</a>
</p>

**Votum** turns <img src="assets/flag.svg" width="14" height="14" alt="Swiss flag"> [Apertus-1.5-8B](https://huggingface.co/swiss-ai/Apertus-v1.5-8B)
into a decision model. You give it some context (an email, a document, a JSON record, a photo), a question about it and
the possible answers. It gives back a probability for each answer.

It does not write text. It reads the prompt once and scores the answers, so a decision takes about 40 ms on an
RTX 4090, the result is always one of your options, and the probabilities always sum to one.

<picture><source media="(prefers-color-scheme: dark)" srcset="assets/architecture_dark.png"><img src="assets/architecture.png" alt="The request as one prompt, the attention mask, the backbone and the pointer head"></picture>

<picture><source media="(prefers-color-scheme: dark)" srcset="assets/examples_dark.png"><img src="assets/examples.png" alt="Real outputs of Votum-8B"></picture>

## Quickstart

```bash
pip install "votum @ git+https://github.com/hypathia/votum"            # to use the model from Python
pip install "votum[vllm] @ git+https://github.com/hypathia/votum"      # the same, plus the HTTP server
```

You need Python 3.11 or newer and a CUDA GPU. Which model to use depends on the card:

| GPU memory | Model | Notes |
|---|---|---|
| 24 GB or more | [votum-ai/Votum-8B](https://huggingface.co/votum-ai/Votum-8B) | the full model: 16 GB of weights; the server reserves about 18 GB |
| 16 GB | [votum-ai/Votum-8B-FP8](https://huggingface.co/votum-ai/Votum-8B-FP8) | the same model with weights stored in 8 bits instead of 16: 9.2 GB; the server reserves about 12 GB; gives the same answer as the full model 96–99 % of the time |
| 24 GB, for fine-tuning | votum-ai/Votum-8B | trains on examples up to 3,072 tokens, or 2,048 if the card also drives a display ([guide](training/README.md)) |
| none | either | runs on a CPU, slowly |

The two models are used the same way; wherever `votum-ai/Votum-8B` appears below, the FP8 name works too.

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

## Images

The state can also hold images. Put one anywhere in it as `{"image": ...}`:

```python
out = vm.decide({
    "state": {"photo": {"image": "wreck.jpg"}, "line": "mountain branch, km 41"},
    "questions": {
        "incident": {"type": "choice", "instructions": "What happened?",
                     "criteria": {"derailment": "A locomotive derailed", "collision": "Two trains collided",
                                  "fire": "A fire", "none": "Nothing, normal operation"}},
        "people":   {"type": "noul",   "instructions": "Are people at the scene?"},
        "severity": {"type": "score",  "instructions": "How severe is the damage?",
                     "criteria": ["minor", "serious", "severe"]},
    },
})
```

<picture><source media="(prefers-color-scheme: dark)" srcset="assets/examples_vision_dark.png"><img src="assets/examples_vision.png" alt="Real outputs of Votum-8B on images"></picture>

This works with the released model as it is. Apertus 1.5 reads an image as tokens of its own vocabulary, one token per
16 × 16 pixel patch, and Votum keeps that vocabulary. An image is scaled to at most 1,024 tokens, about 512 × 512
pixels (`vm.decide(request, max_image_codes=...)`, or `--max-image-codes` on the server). On the server, all questions
of a request share one pass over the image: the three questions above take one prompt of about 1,200 tokens and about
130 ms on an RTX 4090.

In Python an image can be a file path, bytes, a PIL image or base64. The server takes base64 or a data URL
(`"data:image/png;base64,..."`) and never opens files or URLs named in a request; a request holds at most 8 images.
Images need two more packages, `pip install "votum[vision]"`, and the first image downloads the image tokenizer
(0.9 GB) from the [Apertus repository](https://huggingface.co/swiss-ai/Apertus-v1.5-8B), whose usage policy you accept
there as well. Nothing is downloaded or loaded until the first image; `vision="eager"` (`--vision eager` on the server)
loads it at start-up instead, and `vision="off"` refuses images. The tokenizer adds about 2 GB of GPU memory;
`--image-device cpu` keeps it off the GPU, at a few seconds per image.

**How well it works.** Votum-8B was never trained on images, so take this as a preview. Accuracy on held-out test sets,
next to the Apertus model it is built on, answering with the letter of an option:

| Test set (chance) | Votum-8B | Votum-8B-FP8 | Apertus-1.5-8B |
|---|---:|---:|---:|
| A-OKVQA: questions about photos, 4 options (25 %) | 68.3 % | 68.5 % | 76.0 % |
| POPE: is this object in the photo? (50 %) | 80.4 % | 79.4 % | 82.8 % |
| CIFAR-10: what does the photo show? 10 classes (10 %) | 73.1 % | 73.5 % | 71.4 % |
| RVL-CDIP: which of 16 document types? (6 %) | 25.5 % | | 29.5 % |

Everyday photos work well. Reading the text in a scanned document does not yet: the payment reminder above is
classified correctly, but with little confidence. The figure shows 6 of the 12 image requests written for this page;
across all 12, 29 of the 36 answers are right. In a test run, 1.5 hours of fine-tuning on images mixed with text raised
A-OKVQA to 79 %, CIFAR-10 to 83 % and the document types to 64 %, with the text benchmarks unchanged; image support in
the training kit is not released yet.

## Serve

```bash
pip install "votum[vllm] @ git+https://github.com/hypathia/votum"
VOTUM_API_KEY=... votum serve --model votum-ai/Votum-8B --host 0.0.0.0 --port 8011
```

`POST /v1/systemone` takes the same request as `vm.decide` and returns the same answers. Requests that arrive at the
same time are processed together. Start-up takes about a minute.

The server reserves the model's weights plus 3 GB of GPU memory; `--gpu-memory-gb` changes that.

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

## Speed

All numbers below are for the full Votum-8B (16-bit weights) on one NVIDIA RTX 4090 (24 GB), running on vLLM.

**A short prompt, one question.** The requests are items of the OpenDecisions benchmark: one question per request,
about 445 prompt tokens each. Votum is compared with the same Apertus model writing its answer as text.

| How the answer is produced | Prompt | Tokens written | Latency, one request at a time (median) | Requests per second, 256 sent at once |
|---|---|---:|---:|---:|
| **Votum: a decision** | 1 question, ~445 tokens | **0** | **39 ms** | **23.5** |
| Text: the choice as JSON (structured output) | 1 question, ~445 tokens | 8 | 206 ms | 21.0 |
| Text: the choice and its probabilities as JSON (structured output) | 1 question, ~445 tokens | 54 | 1.1 s | 10.9 |
| Text: reasoning first, then the JSON | 1 question, ~445 tokens | 370 | 10.5 s | 1.9 (64 sent at once) |

The Votum row is the released model, with latency taken over 50 requests. The three text rows were measured earlier
on the same card, with an older vLLM and a different sample of 256 OpenDecisions items; the reasoning row is free text
capped at 512 new tokens.

**A long document, several questions.** Each request carries a state of about 3,000 tokens and 1, 4 or 8 questions
with up to 8 options each. Questions about the same state share one pass over it, so asking several at once is
cheaper than asking them one by one.

| Questions per request | Prompt tokens per question | Latency per request, one at a time (median) | Decisions per second, all requests sent at once |
|---:|---:|---:|---:|
| 1 | 3,042 | 323 ms | 3.1 (64 requests) |
| 4 | 3,190 | 392 ms | 9.7 (32 requests) |
| 8 | 3,330 | 465 ms | 15.8 (32 requests) |

Released model, no reuse of results between requests. The [model card](https://huggingface.co/votum-ai/Votum-8B#performance) has more workloads.

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

## Fine-tune

To teach Votum your own task, follow [`training/README.md`](training/README.md). It is a step-by-step guide: prepare
your examples, train from the released model, export the result and calibrate its probabilities. One 24 GB GPU is
enough, and a fine-tuned model can be shrunk to FP8 afterwards with `votum quantize --model <dir> --out <dir>` (about
a minute, no GPU needed).

## Reproduce the release

The same guide has the exact command of the released training run, which took about 7 hours on one 96 GB GPU. The
training data is published as [votum-ai/Votum-Pool](https://huggingface.co/datasets/votum-ai/Votum-Pool), and the
pipeline that builds it is in `data/`. After training,
`python scripts/export_hf.py --ckpt <run>/ckpt_600 --out <dir>` reproduces the published model repository byte for
byte.

The released model was trained before the pool was cleaned. Its known training-data issues are listed on the model
card, and a re-train on the published pool is planned.

## Repository

```
src/votum/          the model in PyTorch · prompt builder · request format · images · checkpoint builder · FP8 · command line
src/votum/vllm/     the vLLM server: model, attention rules and caches, client, HTTP API
training/           fine-tuning kit (prepare_data · finetune · calibrate) and the trainer that produced Votum-8B
data/               source list with licences · cleaning and personal-data removal
scripts/            export a checkpoint to the Hub layout · smoke fixture · README graphics
tests/              attention rules, prompt builder, request format (CPU) · end-to-end test (GPU)
```

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
