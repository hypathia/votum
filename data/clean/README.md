# From converted rows to the Votum Pool dataset

`pool.py` turns the converted rows of the pool (without the per-source limit that was applied for
training the released checkpoint) into the published dataset `votum-ai/Votum-Pool`, which is also the input of the
planned re-train. It has three stages and an audit.

| Stage | In one line | Writes |
|---|---|---|
| `clean` | removes sources and rows that should not be in the pool | `clean/` |
| `mask` | masks contact details and identifiers; e-mail addresses and phone numbers in every row | `clean_pii/` |
| `export` | writes the dataset files | `dataset/` |
| `audit` | checks a 500-row sample independently | `audit/` |

| Stage | Training rows |
|---|---:|
| all converted rows | 881,143 |
| after `clean` | 744,057 (188 sources) |
| after `mask` | 723,257 (181 sources) |
| with the 11 sources added later | 942,671 (192 sources) |

The converted rows themselves are not public (see `../README.md`). The pipeline is shared to show exactly what was
done, and so that you can run the same stages on your own rows: nothing in the code is specific to this pool. What
happens to each source, and every setting, is in one file, `pool.json`.

## Requirements

Python 3.11 or newer and `pip install -r requirements.txt` (numpy and pyarrow; gliner with torch only if you turn the
NER model on). Everything runs on the CPU.

## Run it

All stages in one go:

```bash
python -I pool.py run --pool <dir with train.jsonl.gz and val.jsonl.gz> --work <work dir> --bench <benchmark jsonl files ...>
```

Or one stage at a time:

```bash
python -I pool.py clean  --pool <pool> --out work/clean --bench <benchmark jsonl files ...>
python -I pool.py mask   --pool work/clean --out work/clean_pii
python -I pool.py export --pool work/clean_pii --out work/dataset [--flag in_votum_8b=ids.txt.gz]
python -I pool.py audit sample --pool work/clean_pii --audit-dir work/audit
python -I pool.py audit regex  --audit-dir work/audit
python -I pool.py audit report --audit-dir work/audit
```

`python -I pool.py <stage> --help` lists a stage's options. `run --from mask` starts at a later stage.

## Speed

Every stage reads its input once, spreads the work over all CPU cores and writes rows in input order, so the result
does not depend on the number of workers. On the whole pool (888,000 rows, 2.4 GB of text) with 26 workers:

| Stage | Time |
|---|---:|
| `clean` | 30 s |
| `mask` | 18 s |
| `export` | 10 s |
| all stages and the audit sample | 71 s |

### The NER model (off for this pool)

`mask` can also mask names and addresses with a NER model (`"ner": true` in the `mask` section). The published
dataset does not: tested on this pool, the model masked mostly role words ("Ehefrau", "Wykonawca"), companies and
public figures, while the names it was meant for are published openly by the sources anyway.

With the model on, its output is cached per text, in `work/ner_cache`, so only texts it has not seen go through it.
A first run on one RTX 4090 takes 32 min for the 203,000 distinct texts of the self-collected sources, or 8 min with
`--precision fp16 --flash` (`pip install flashdeberta`), which changes a few borderline spans. With several GPUs, run
`mask --part k/N --device cuda:k` once per GPU, then `mask` once more to write the rows. The cache holds offsets of
personal data; keep it local.

## The row format

One JSON object per line, gzip-compressed. `clean` and `mask` read these fields and carry every other field through
unchanged; `export` keeps only the fields listed here.

| Field | Meaning |
|---|---|
| `id` | unique row id |
| `source` | source name; it selects the row's entry in `pool.json` |
| `state` | the context: a string, or a JSON object whose string values are the text |
| `question` | the question |
| `options` | list of `{"id", "text"}` |
| `target`, `soft`, `type`, `kind`, `group`, `lang`, `license`, `license_detail` | carried into the dataset by `export` |

Benchmark files for `clean` are JSON lines too. A record's text is read from `record.state`, `row.state` or `state`;
records without state text fall back to their `questions.*.instructions`.

## The config file

`pool.json` has one entry per source under `sources`:

| Key | Meaning |
|---|---|
| `drop` | the source is removed: `"clean"` (before anything else) or `"mask"` (not cleared for redistribution; removed after the benchmark comparison). `drop_reason` says why |
| `origin` | `redistributed` (a copy of a published dataset, left as its authors released it apart from the two sweeps) or `self-collected` (masked) |
| `names` | `true`: person names are masked too, when the NER model is on (false everywhere in this pool) |
| `config` | the dataset config the rows go to (`default` or `share_alike`) |
| `license`, `upstream`, `description` | the licence tags, the upstream release and one line on what the source is, copied to `SOURCES.csv` |
| `license_fix` | optional `[[text in license_detail, corrected tag], ...]`, applied by `export` |
| `reason` | one line on why the source is handled this way |

Every source in the data needs an entry. The other sections hold the settings of the stages:

| Section | Keys |
|---|---|
| `clean` | `drop_rows` (rows to drop by source, field and value), `must_drop_patterns` (a safety net: the run stops if a source name contains one of them and is not dropped), `decontaminate` (gram lengths and stop-list thresholds), and optionally `benchmarks` and `info_labels` |
| `mask` | `ner` (the NER model on or off), `keep_emails` and `keep_email_hosts` (addresses that are not masked); for the model: `model`, `thresholds`, `stop_spans` (a file of role words the model wrongly tags as names: Claimant, Antragsteller, she, ...) |
| `export` | `configs`, `license_guard` (licence tags a config must not contain), and optionally `flags` |

## What each stage does

**`clean`** drops three things:

- whole sources: the 13 removed by the licence review, and the LLM-generated sets with unverified terms;
- the ContractNLI test split;
- rows that overlap an evaluation benchmark. A row goes if its state shares a word 8-gram with any benchmark item. A
  short stop-list of contract and chat boilerplate is exempt. Scripts written without spaces (Chinese, Japanese,
  Thai, ...) are compared as 16-character grams instead. The row's question is checked too, but a match counts only if
  at most 5 rows share it, which separates a quoted benchmark item from a question template.

Kept rows are copied byte for byte. `REPORT.md` and `REPORT.json` give the counts per rule, source and benchmark;
`dropped.tsv.gz` lists every dropped row with its rule.

**`mask`** does four things:

- drops the sources whose terms were not cleared for redistribution: six portal and government sources, and one
  source with conflicting licence tags;
- leaves rows from already-published datasets as their authors released them, apart from the two sweeps below;
- masks the 17 sources the builder collected itself with regular expressions: e-mail addresses, phone numbers, IBANs,
  card and ID numbers, IP addresses, URLs that carry a personal identifier, and street addresses. Names stay;
- sweeps every row of every source: e-mail addresses become `[EMAIL]`, except seven office mailboxes of Bernese and
  Zurich courts and cantonal services (each checked by hand) and the reserved `example.com/org/net` addresses; phone
  numbers become `[PHONE]` when their form cannot be mistaken for a case number or a code (international `+41 31 ...`,
  North American `(415) 555-0134`, or right after a word such as "phone", "Tel." or "call").

The stage is safe to run on its own output: a second pass changes nothing.

**`export`** writes one directory of parquet files per dataset config (`default`, `share_alike`) and `SOURCES.csv`. It
also corrects a row's licence tag where the config says the pool's own tag was wrong (`license_fix`; the two
parliamentary sources that had been tagged MIT now carry their portals' real terms), and it stops if a share-alike row
would land in the `default` config. `--flag NAME=FILE` adds a boolean column that is true for the row ids listed in
the file; the published dataset has `in_votum_8b`, the rows the released checkpoint was trained on.

**`audit`** draws 500 masked rows and looks for what `mask` missed, with the stage's own patterns and looser patterns
of its own. When names are masked, `audit spacy` and `audit gliner` also look for missed names. Its hit files contain
residual personal data; keep them local.

## The code

| File | Contents |
|---|---|
| `pool.py` | the command line |
| `pool.json`, `pii_stop_spans.txt` | the config of this pool |
| `poolkit/engine.py` | reading and writing rows in parallel, the config |
| `poolkit/decontam.py` | `clean` |
| `poolkit/pii.py` | `mask`: the regular expressions, the sweeps, the model and its span cache |
| `poolkit/export.py` | `export` |
| `poolkit/audit.py` | `audit` |

`tests/test_data_clean.py` runs every stage on small synthetic inputs on the CPU (`pytest tests/test_data_clean.py`;
needs numpy and pyarrow, not gliner). On the real pool, a run from the converted rows reproduces the published files
byte for byte.
