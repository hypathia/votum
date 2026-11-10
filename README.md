<h1 align="center"><img src="assets/flag.svg" width="28" height="28" alt=""> votum</h1>

<p align="center"><b>A decision model on a fully open LLM backbone.</b></p>

<p align="center"><sub>Independent project, built on Apertus-1.5-8B (Swiss AI Initiative) · not affiliated</sub></p>

<p align="center">
  <a href="https://huggingface.co/votum-ai/Votum-8B"><img alt="Model: Votum-8B on Hugging Face" src="https://img.shields.io/badge/model-Votum--8B-DA291C?logo=huggingface&logoColor=white"></a>
  <a href="https://huggingface.co/datasets/votum-ai/Votum-Pool"><img alt="Training data: Votum-Pool on Hugging Face" src="https://img.shields.io/badge/data-Votum--Pool-DA291C?logo=huggingface&logoColor=white"></a>
  <img alt="Python 3.11 or newer" src="https://img.shields.io/badge/python-3.11%2B-1D1D1F">
  <img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-1D1D1F">
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> ·
  <a href="#ask-a-question">Ask a question</a> ·
  <a href="#images">Images</a> ·
  <a href="#run-it-as-a-server">Server</a> ·
  <a href="#speed">Speed</a> ·
  <a href="#fine-tune-it">Fine-tune</a> ·
  <a href="#documentation">Documentation</a>
</p>

**Votum** turns <img src="assets/flag.svg" width="14" height="14" alt="Swiss flag"> [Apertus-1.5-8B](https://huggingface.co/swiss-ai/Apertus-v1.5-8B)
into a decision model. You give it some context (an email, a document, a JSON record, a photo), a question about it and
the possible answers. It gives back a probability for each answer.

It does not write text. It reads the prompt once and scores the answers, so a decision takes about 40 ms on an
RTX 4090, the result is always one of your options, and the probabilities always sum to one.

<picture><source media="(prefers-color-scheme: dark)" srcset="assets/examples_dark.png"><img src="assets/examples.png" alt="Four requests answered by Votum-8B. An email about a double charge: team is Billing and payments (0.96), the customer asks for money back (0.90), urgency is medium (0.41). A JSON order record: the total is 65.0 (0.73), no discount was applied (0.91). A photo of a derailed locomotive: a locomotive derailed (0.46), people are at the scene (0.76), the damage is serious (0.47). A photo of a market in front of a church: a street market (0.76), fruit or vegetables are for sale (0.79), the large building is a church (0.79)."></picture>

The open weights and the open dataset are on Hugging Face: [Votum-8B](https://huggingface.co/votum-ai/Votum-8B),
[Votum-8B-FP8](https://huggingface.co/votum-ai/Votum-8B-FP8) (smaller) and the training data,
[Votum-Pool](https://huggingface.co/datasets/votum-ai/Votum-Pool). This repository holds the code.

## Quickstart

You need Python 3.11 or newer and an NVIDIA graphics card (GPU). Pick the model that fits it:

| Your GPU has | Use | Good to know |
|---|---|---|
| 24 GB or more | [votum-ai/Votum-8B](https://huggingface.co/votum-ai/Votum-8B) | the full model, a 16 GB download |
| 16 GB | [votum-ai/Votum-8B-FP8](https://huggingface.co/votum-ai/Votum-8B-FP8) | a compressed copy, a 9 GB download; it gives the same answer as the full model 96 to 99 % of the time |
| no GPU | either | it runs on a processor, slowly |

**1. Install the package.**

```bash
pip install "votum @ git+https://github.com/hypathia/votum"
```

**2. Get access to the model.** The download needs nothing but this one repository, but it is gated, because it
contains Apertus weights: open [votum-ai/Votum-8B](https://huggingface.co/votum-ai/Votum-8B), accept the Apertus
Acceptable Use Policy there once, then log in on your machine:

```bash
hf auth login
```

**3. Ask.** This is the email from the figure above, with its three questions:

```python
from votum import Votum

# On a 16 GB GPU, use "votum-ai/Votum-8B-FP8" instead.
vm = Votum.from_pretrained("votum-ai/Votum-8B")  # downloads the model the first time

request = {
    "state": "Hi, I was charged twice for my October invoice (INV-4411). Could you refund one of the two payments?",
    "questions": {
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this ticket?",
            "criteria": {
                "billing": "Billing and payments",
                "tech": "Technical support",
                "sales": "Sales",
                "hr": "Human resources",
            },
        },
        "refund": {
            "type": "noul",
            "instructions": "Does the customer ask for money back?",
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this ticket?",
            "criteria": ["low", "medium", "high"],
        },
    },
}

out = vm.decide(request)

print(out["answers"])
```

```python
{
    "team": {
        "type": "choice",
        "choice": "billing",
        "confidence": 0.9619,
        "probabilities": {
            "billing": 0.9619,
            "tech": 0.0079,
            "sales": 0.0243,
            "hr": 0.0059,
        },
    },
    "refund": {
        "type": "noul",
        "noul": 0.8994,
    },
    "urgency": {
        "type": "score",
        "score": 0.9621,
        "confidence": 0.4133,
        "legend": {
            "0": "low",
            "1": "medium",
            "2": "high",
        },
        "probabilities": {
            "0": 0.3123,
            "1": 0.4133,
            "2": 0.2744,
        },
    },
}
```

Read it as: send the ticket to billing (96 % sure); the customer very likely wants a refund (90 %); the urgency is
around medium, and the model is not sure (41 %). That is why Votum returns probabilities and not just a label: your
program can act on the confident answers and pass the others to a person.

## Ask a question

A request has two parts:

- **`state`**: the context to decide on. A string, or any JSON object.
- **`questions`**: up to 8 questions about that state, each under a name you choose. They are answered together.

Every question has `instructions` (the question itself, in plain words) and a `type`:

| `type` | Use it for | What you add | What you get back |
|---|---|---|---|
| `choice` | picking one of several options | `criteria`: the options, as `{name: description}` or a list of names | `choice`: the most likely option. `confidence`: its probability. `probabilities`: one per option |
| `noul` | a yes/no question | nothing | `noul`: the probability that the answer is yes |
| `score` | a rating on a scale | `criteria`: the levels, lowest first | `score`: the expected level, counting from 0 (0.96 is close to level 1, the second one). `confidence` and `probabilities` per level, and `legend`: which number is which level |

Three things help the model:

- **Describe the options.** `{"billing": "Billing and payments"}` works better than a bare `"billing"`.
- **Ask in the language of your text, or in English.** Votum was built for English, German, French and Italian.
- **Keep it within 8,192 tokens** (roughly 30,000 characters) for the state, one question and its options. A longer
  request is refused with an error; it is never silently cut short.

The words `noul`, `instructions` and `criteria` come from the request format of the [Jev and Jeff](#acknowledgements)
decision models. Votum follows it so that programs written for those models work unchanged.

## Images

A state can hold images. Put one anywhere in it as `{"image": ...}`:

```python
request = {
    "state": {
        "photo": {
            "image": "wreck.jpg",
        },
        "line": "mountain branch, km 41",
    },
    "questions": {
        "incident": {
            "type": "choice",
            "instructions": "What happened?",
            "criteria": {
                "derailment": "A locomotive derailed",
                "collision": "Two trains collided",
                "fire": "A fire",
                "none": "Nothing, normal operation",
            },
        },
    },
}

out = vm.decide(request)
```

This needs two more packages: `pip install "votum[vision] @ git+https://github.com/hypathia/votum"`.

Treat images as a preview. Votum-8B was never trained on them: everyday photos work well, text inside scanned
documents does not yet. [docs/images.md](docs/images.md) has more examples, the accuracy we measured and the settings.

## Run it as a server

When several programs need the model, or you have many decisions to make, run it as an HTTP server:

```bash
pip install "votum[vllm] @ git+https://github.com/hypathia/votum"
votum serve --model votum-ai/Votum-8B --port 8011
```

```bash
curl -s localhost:8011/v1/systemone -H 'content-type: application/json' -d '{
  "state": "The printer on floor 3 is jammed again.",
  "questions": {"team": {"type": "choice", "instructions": "Which team?",
                         "criteria": {"it": "IT support", "fac": "Facilities"}}}}'
```

The server takes the same request as `vm.decide` and returns the same answers. The address and the format are those
of Jev's System One API, so a client written for Jev can point at it. It is also the fast way to run Votum: requests
that arrive together are processed together. [docs/serving.md](docs/serving.md) covers the API key, memory
and the other settings.

## Speed

Measured on one NVIDIA RTX 4090 (24 GB) with the full Votum-8B, running as a server:

| What is asked | Size of the prompt | Time for one request | Decisions/s, many requests at once |
|---|---|---:|---:|
| 1 question about a short text | about 450 tokens | 39 ms | 23.5 |
| 1 question about a long document | about 3,000 tokens | 323 ms | 3.1 |
| 8 questions about a long document | about 3,300 tokens | 465 ms | 15.8 |

Asking several questions about the same state at once is cheaper than asking them one by one, because the state is
read only once. With eight questions per request, the long document gives 5 times as many decisions per second as
with one.

For comparison, the same Apertus model *writing* the answer and its probabilities as JSON takes 1.1 s for the short
text, about 27 times longer. [docs/benchmarks.md](docs/benchmarks.md) says exactly how each number was measured.

## How good are the answers?

Votum-8B is a small model, so expect solid everyday judgement, not expert knowledge. Some reference points, on public
test sets it was not trained on:

| Task | Correct | Correct among the half it is most sure about |
|---|---:|---:|
| Yes/no questions about a passage (BoolQ) | 88.9 % | 98.2 % |
| Topic of a text, in 12 languages (SIB-200) | 82.4 % | 97.4 % |
| Customer intent among 77 options (BANKING77) | 70.4 % | 85.3 % |
| School and university exam questions (MMLU) | 60.4 % | 82.1 % |

The last column is why the probabilities matter: when Votum says it is sure, it is right far more often. The
[model card](https://huggingface.co/votum-ai/Votum-8B#evaluation) has the full evaluation, and an honest list of
[limitations](https://huggingface.co/votum-ai/Votum-8B#limitations).

## How it works

Votum is Apertus-1.5-8B with two additions: a small set of weight changes (a LoRA adapter) and a small extra layer
that scores answers.

1. **Your request becomes one prompt:** the list of questions, then the state, then each question with its options.
2. **The model reads it once.** Rules decide which parts of the prompt may look at which. In particular, the options
   never see each other, so the order in which you list them does not change the answer.
3. **Each option gets a score,** read off the model at the end of that option. The scores of a question are turned
   into probabilities that sum to one.

Nothing is generated at any point, which is where the speed comes from.
[docs/internals.md](docs/internals.md) has the diagram and the details.

## Fine-tune it

To teach Votum your own task, follow [training/README.md](training/README.md). It is a step-by-step guide: prepare
your examples, train from the released model, export the result and calibrate its probabilities. One 24 GB GPU is
enough.

## Documentation

| Page | What is in it |
|---|---|
| [docs/serving.md](docs/serving.md) | the HTTP server: API, API key, memory, settings |
| [docs/images.md](docs/images.md) | images in requests: set-up, examples, accuracy |
| [docs/benchmarks.md](docs/benchmarks.md) | speed measurements and how they were made |
| [docs/internals.md](docs/internals.md) | how the model works inside, the repository layout, tests, reproducing the release |
| [training/README.md](training/README.md) | fine-tuning on your own data |
| [Votum-8B model card](https://huggingface.co/votum-ai/Votum-8B) | evaluation, training data, limitations |
| [Votum-8B-FP8 model card](https://huggingface.co/votum-ai/Votum-8B-FP8) | the compressed model and what it trades away |
| [Votum-Pool dataset card](https://huggingface.co/datasets/votum-ai/Votum-Pool) | the training data and its sources |

## Acknowledgements

Votum builds on other people's work:

- **[Apertus](https://huggingface.co/swiss-ai/Apertus-v1.5-8B)**, by the Swiss AI Initiative (EPFL, ETH Zurich, CSCS):
  the open model underneath Votum.
- **Jev**: the original decision models. Votum uses their request format (`choice`, `noul`, `score`).
- **[Jeff](https://github.com/j-geuter/Jeff)**, by Jonathan Geuter, Andrew Cheng and David Alvarez-Melis
  ([Releasing the Inner Jev](https://arxiv.org/abs/2610.17205)): the open recipe for turning a language model into a
  decision model. Our evaluation was checked against their published numbers, and the code that reads requests is
  derived from theirs (see `NOTICE`).
- **[JevPertus](https://huggingface.co/jaluus/jevpertus-v1.5-8B)** ([code](https://github.com/Jaluus/JevPertus)): an
  earlier decision model built on the same Apertus-1.5-8B.
- **[JevBench](https://github.com/fstandhartinger/jevbench)** and the **Jev Decision Index**: independent public
  benchmarks for decision models, on which Votum is measured.

## License

Apache-2.0 (`LICENSE`, `NOTICE`). The Apertus model underneath carries the
[Apertus Acceptable Use Policy](https://github.com/swiss-ai/apertus-legal/blob/main/apertus_1.5/USAGE_POLICY.pdf),
which applies to anyone who uses Votum. Votum is an independent project, not affiliated with or endorsed by the Swiss
AI Initiative.
