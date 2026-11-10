# From the training pool to the Votum Pool dataset

These scripts turn the converted rows of the pool ("pool v7", without the per-source limit that was applied for
training the released checkpoint) into the published dataset `votum-ai/Votum-Pool`, which is also the input of the
planned re-train. There are four build steps and one audit.

| Step | Script | In one line | Writes |
|---|---|---|---|
| 1 | `build_clean.py` | removes sources and rows that should not be in the pool | `clean/` |
| 2 | `pii_pass.py` | masks personal data in the sources the builder collected itself | `clean_pii/` |
| 3 | `email_sweep.py` | masks the e-mail addresses still left anywhere | `clean_pii/`, in place |
| 4 | `build_hf.py` | writes the dataset files | `Votum-Pool/` |
| 5 | `audit.py` | checks a 500-row sample independently | `audit/` |

| Stage | Training rows |
|---|---:|
| pool v7, all converted rows | 881,143 |
| after step 1 | 744,057 (188 sources) |
| after step 2 | 723,257 (181 sources) |

The pool v7 files themselves are not public (see `../README.md`), so the scripts are shared to show exactly what was
done, and to let you apply the same steps to your own data in the same row format.

## Requirements

Python 3.11 or newer and `pip install -r requirements.txt` (numpy, pyarrow, gliner with torch). Step 2 runs a NER
model: on a GPU it takes about 40 minutes for the whole pool. Steps 1, 3 and 4 run on the CPU in a few minutes.

## Run it

All steps in one go:

```bash
./run_all.sh <dir with train.jsonl.gz and val.jsonl.gz> <work dir> <benchmark jsonl files ...>
```

Or one step at a time:

```bash
python -I build_clean.py --train <pool>/train.jsonl.gz --val <pool>/val.jsonl.gz --out-dir clean \
    --bench <benchmark jsonl files ...>
python -I pii_pass.py --train clean/train.jsonl.gz --val clean/val.jsonl.gz --policy pii_policy.json \
    --stop-spans pii_stop_spans.txt --out-dir clean_pii
python -I email_sweep.py --pool-dir clean_pii
python -I build_hf.py --pool-dir clean_pii --policy pii_policy.json --out-dir Votum-Pool [--trained-ids ids.txt.gz]
python -I audit.py sample --audit-dir audit --pool-dir clean_pii --policy pii_policy.json
python -I audit.py regex  --audit-dir audit          # then spacy and gliner (optional), then:
python -I audit.py report --audit-dir audit --policy pii_policy.json
```

Every script explains its options with `--help`. Step 2 saves its work in chunks, so an interrupted run resumes
where it stopped; `--part k/N` splits it over N processes (`run_all.sh` does this with `NER_PARTS=N`).

## The row format

One JSON object per line, gzip-compressed. Steps 1 to 3 read these fields and carry every other field through
unchanged; step 4 keeps only the fields listed here.

| Field | Meaning |
|---|---|
| `id` | unique row id |
| `source` | source name; it selects the row's entry in `pii_policy.json` |
| `state` | the context: a string, or a JSON object whose string values are the text |
| `question` | the question |
| `options` | list of `{"id", "text"}` |
| `target`, `soft`, `type`, `kind`, `group`, `lang`, `license`, `license_detail` | carried into the dataset by step 4 |
| `upstream_split` | only on ContractNLI rows; step 1 drops its `test` split |

Benchmark files for step 1 are JSON lines too. A record's text is read from `record.state`, `row.state` or `state`;
records without state text fall back to their `questions.*.instructions`.

## The policy file

`pii_policy.json` has one entry per source:

| Key | Meaning |
|---|---|
| `excluded` | `true`: every row of the source is dropped in step 2 (`exclude_reason` says why) |
| `origin` | `redistributed` (a copy of a published dataset, left as its authors released it) or `self-collected` (masked in step 2) |
| `names` | `true`: person names are masked too (sources about private people) |
| `config` | `default` or `share_alike`: the dataset config the rows go to |
| `license`, `upstream` | the licence tags and the upstream release, copied to `SOURCES.csv` |
| `license_fix` | optional `[[text in license_detail, corrected tag], ...]`, applied in step 4 |
| `reason` | one line on why the source is handled this way |
| `description` | one line on what the source is, copied to `SOURCES.csv` (published sources only) |
| `train_rows`, `val_rows` | row counts at the time of the review, for reference |

`pii_stop_spans.txt` lists role words the NER model wrongly tags as names (Claimant, Antragsteller, she, ...). They
are never masked.

## What each step does

**1. `build_clean.py`** drops three things:

- whole sources: the 13 removed by the licence review, bias_in_bios, and the LLM-generated sets with unverified terms;
- the ContractNLI test split;
- rows that overlap an evaluation benchmark. A row goes if its state shares a word 8-gram with any benchmark item. A
  short stop-list of contract and chat boilerplate is exempt. Scripts written without spaces (Chinese, Japanese,
  Thai, ...) are compared as 16-character grams instead. The row's question is checked too, but a match counts only if
  at most 5 rows share it, which separates a quoted benchmark item from a question template.

Kept rows are copied byte for byte. `REPORT.md` and `REPORT.json` give the counts per rule, source and benchmark;
`dropped.tsv.gz` lists every dropped row with its rule.

**2. `pii_pass.py`** follows `pii_policy.json`. It does three things:

- drops the sources whose terms were not cleared for redistribution: six portal and government sources, and one
  source with conflicting licence tags;
- leaves rows from already-published datasets as their authors released them;
- masks the 17 sources the builder collected itself: structured identifiers by regular expression, names and
  addresses with the NER model `urchade/gliner_multi_pii-v1`. Names are masked only in the sources about private
  people.

A saved chunk is reused only when nothing it depends on has changed: its rows, the script, the model and thresholds,
the stop spans, and the policy of its sources. The NER model is not bit-reproducible across GPUs, so a rerun
elsewhere can mask a few names differently.

**3. `email_sweep.py`** replaces every e-mail address with `[EMAIL]`, in place. It keeps seven office mailboxes of
Bernese and Zurich courts and cantonal services (listed in the script, each checked by hand) and the reserved
`example.com/org/net` addresses. It changes 36 rows; a second run changes none.

**4. `build_hf.py`** writes the two parquet configs (`default`, `share_alike`) and `SOURCES.csv`. It also corrects a
row's licence tag where the policy says the pool's own tag was wrong (`license_fix`; the two parliamentary sources
that had been tagged MIT now carry their portals' real terms), and it stops if a share-alike row would land in the
`default` config. With `--trained-ids` (a file with one row id per line) it adds the column `in_votum_8b`, true for
the rows the released checkpoint was trained on.

**5. `audit.py`** draws 500 masked rows and looks for what the pass missed, with the pass's own patterns, looser
patterns of its own, spaCy on English rows and GLiNER at a lower threshold on the others. Its hit files contain
residual personal data; keep them local.

## Checking the scripts

`tests/test_data_clean.py` runs every step on small synthetic inputs on the CPU (`pytest tests/test_data_clean.py`;
needs numpy and pyarrow, not gliner). On the real pool, steps 1, 3 and 4 reproduce the published files byte for byte.
