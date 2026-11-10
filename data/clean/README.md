# From the training pool to the Votum Pool dataset

These scripts turn the pool the released checkpoint was trained on ("pool v7") into the published dataset
`{{ORG}}/Votum-Pool`, which is also the input of the planned re-train. There are four build steps and one audit.

| Step | Script | In one line |
|---|---|---|
| 1 | `build_clean.py` | removes sources and rows that should not be in the pool |
| 2 | `pii_pass.py` | masks personal data in the sources the builder collected itself |
| 3 | `email_sweep.py` | masks the e-mail addresses still left anywhere |
| 4 | `build_hf.py` | writes the dataset files |
| 5 | `audit.py` | checks a 500-row sample independently |

| Stage | Training rows |
|---|---:|
| pool v7 | 405,547 |
| after step 1 | 339,524 (188 sources) |
| after step 2 | 327,664 (181 sources) |

## Run it

```bash
python -I build_clean.py --train <pool v7 train.jsonl.gz> --val <pool v7 val.jsonl.gz> --out-dir clean \
    --bench <benchmark jsonl files ...>
python -I pii_pass.py --train clean/train.jsonl.gz --val clean/val.jsonl.gz --policy pii_policy.json \
    --stop-spans pii_stop_spans.txt --out-dir clean_pii                       # GPU recommended (gliner, sentencepiece)
python -I email_sweep.py --pool-dir clean_pii
python -I build_hf.py --pool-dir clean_pii --policy pii_policy.json --out-dir Votum-Pool   # pyarrow
```

## What each step does

**1. `build_clean.py`** drops three things:

- whole sources: the 13 removed by the licence review, bias_in_bios, and the LLM-generated sets with unverified terms;
- the ContractNLI test split;
- rows that overlap an evaluation benchmark. A row goes if its state shares a word 8-gram with any benchmark item. A
  short stop-list of contract and chat boilerplate is exempt. Scripts written without spaces (Chinese, Japanese,
  Thai, ...) are compared as 16-character grams instead. The row's question is checked too, but a match counts only if
  at most 5 rows share it, which separates a quoted benchmark item from a question template.

**2. `pii_pass.py`** follows `pii_policy.json`, which gives each source's origin, config and name-masking decision
with a one-line reason. It does three things:

- drops the sources whose terms were not cleared for redistribution: six portal and government sources, and one
  source with conflicting licence tags;
- leaves rows from already-published datasets as their authors released them;
- masks the 17 sources the builder collected itself: structured identifiers by regular expression, names and
  addresses with the NER model `urchade/gliner_multi_pii-v1`. Names are masked only in the sources about private
  people. `pii_stop_spans.txt` lists role words the model wrongly tags as names (Claimant, she, ...).

The pass saves its work in chunks and reuses a chunk only when nothing it depends on has changed: its rows, the
script, the model and thresholds, the stop spans, and the policy of its sources. The NER model runs on the GPU and is
not bit-reproducible across machines, so a rerun elsewhere can mask a few names differently.

**3. `email_sweep.py`** replaces every e-mail address with `[EMAIL]`, in place. It keeps seven office mailboxes of
Bernese and Zurich courts and cantonal services (listed in the script, each checked by hand) and the reserved
`example.com/org/net` addresses. It changes 19 rows.

**4. `build_hf.py`** writes the two parquet configs (`default`, `share_alike`) and `SOURCES.csv`. It also corrects a
row's licence tag where the policy says the pool's own tag was wrong (`license_fix`; the two parliamentary sources
that had been tagged MIT now carry their portals' real terms), and it stops if a share-alike row would land in the
`default` config.

**5. `audit.py`** draws 500 masked rows and looks for what the pass missed. Run its subcommands in this order:
`sample`, `regex`, `spacy`, `gliner`, `report`. Its hit files contain residual personal data; keep them local.
