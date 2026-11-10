# Speed measurements

How fast Votum answers, and exactly how each number was measured. Accuracy results are on the
[model card](https://huggingface.co/votum-ai/Votum-8B#evaluation).

**Setup for every number on this page:** the full Votum-8B (16-bit weights) on one NVIDIA RTX 4090 (24 GB), served
with `votum serve`.

## A short prompt, one question

The requests are items of the OpenDecisions benchmark: one question per request, about 445 prompt tokens each. Votum
is compared with Apertus-1.5-8B generating its answer.

| Model and output | Prompt | Tokens written | Latency, one request at a time (median) | Requests per second, 256 sent at once |
|---|---|---:|---:|---:|
| **Votum-8B: single-pass decision** | 1 question, ~445 tokens | **0** | **39 ms** | **23.5** |
| Apertus-1.5-8B: structured output with just the choice | 1 question, ~445 tokens | 8 | 206 ms | 21.0 |
| Apertus-1.5-8B: structured output with the choice and its probabilities | 1 question, ~445 tokens | 54 | 1.1 s | 10.9 |
| Apertus-1.5-8B: reasoning, then the answer (up to 512 tokens) | 1 question, ~445 tokens | 370 | 10.5 s | 1.9 (64 sent at once) |

The Votum row is the released model on vLLM 0.33, with latency taken over 50 requests. The three Apertus rows were
measured earlier on the same card, with vLLM 0.12 and a different sample of 256 OpenDecisions items, so the comparison
is close but not exactly like for like. The reasoning row is free text capped at 512 new tokens.

## A long document, several questions

Each request carries a state of about 3,000 tokens and 1, 4 or 8 questions with up to 8 options each. Questions about
the same state share one pass over it, so asking several at once is cheaper than asking them one by one.

| Questions per request | Prompt tokens per question | Latency per request, one at a time (median) | Decisions per second, all requests sent at once |
|---:|---:|---:|---:|
| 1 | 3,042 | 323 ms | 3.1 (64 requests) |
| 4 | 3,190 | 392 ms | 9.7 (32 requests) |
| 8 | 3,330 | 465 ms | 15.8 (32 requests) |

Released model, vLLM 0.33, latency over 32 single requests, no reuse of results between requests.

The [model card](https://huggingface.co/votum-ai/Votum-8B#speed) has more workloads, and compares the PyTorch
and server implementations.
