# Serving Votum over HTTP

`votum serve` starts an HTTP server built on [vLLM](https://github.com/vllm-project/vllm). It takes the same requests
as `vm.decide` and batches concurrent ones.

```bash
pip install "votum[vllm] @ git+https://github.com/hypathia/votum"
VOTUM_API_KEY=... votum serve --model votum-ai/Votum-8B --host 0.0.0.0 --port 8011
```

Start-up takes about a minute.

## The API

`POST /v1/systemone` takes a request and returns its answers:

```bash
curl -s localhost:8011/v1/systemone -H 'content-type: application/json' -H "Authorization: Bearer $VOTUM_API_KEY" -d '{
  "model": "votum-8b",
  "state": "The printer on floor 3 is jammed again.",
  "questions": {"team": {"type": "choice", "instructions": "Which team?",
                         "criteria": {"it": "IT support", "fac": "Facilities"}}}}'
```

The body is the request described in the [README](../README.md#ask-a-question): a `state` and up to 8 `questions`.
The response has one entry per question under `answers`, in the same form `vm.decide` returns. Its
`usage.output_tokens` is always 0, because the model writes nothing.

| Situation | Response |
|---|---|
| a question's prompt is longer than 8,192 tokens | HTTP 413; the request is refused, never cut short |
| the body is larger than `VOTUM_MAX_BODY_MB` | HTTP 413 |
| too many questions or options (see below) | HTTP 400 |
| too many questions waiting for the GPU | HTTP 503 |

## GPU memory

The server reserves the model's weights plus 3 GB: about 18 GB for Votum-8B and about 12 GB for Votum-8B-FP8.
`--gpu-memory-gb` changes that. Votum-8B-FP8 on the server needs an NVIDIA GPU of the RTX 30 series or newer.

## Settings

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

Image settings (`--vision`, `--max-image-codes`, `--image-device`) are described in [images.md](images.md).
