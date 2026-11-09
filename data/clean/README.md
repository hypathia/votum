# Cleaned pool and the Votum Pool dataset

Five steps turn the pool the released checkpoint was trained on (pool v7) into the published dataset
`{{ORG}}/Votum-Pool`, which is also the input of the planned re-train.

| Step | Script | What it does |
|---|---|---|
| 1 | `build_clean.py` | drops sources (the 13 from the licence review, bias_in_bios, LLM-generated sets with unverified terms) and the ContractNLI test split; then document-level decontamination, dropping any row whose state shares a word 8-gram with an evaluation benchmark (a short stop-list of contract and chat boilerplate is exempt) |
| 2 | `pii_pass.py` | applies `pii_policy.json` (see below) |
| 3 | `email_sweep.py` | every e-mail address outside public court and cantonal domains (`*.be.ch`, `*.zh.ch`) and the reserved `example.com/org/net` becomes `[EMAIL]`, in place (19 rows) |
| 4 | `build_hf.py` | writes the two parquet configs (`default`, `share_alike`) and `SOURCES.csv` |
| 5 | `audit.py` | independent check of a 500-row sample; subcommands `sample`, `regex`, `spacy`, `gliner`, `report` |

`pii_pass.py` does three things, driven by `pii_policy.json`:
- drops sources whose terms were not cleared for redistribution: six portal and government sources, and one source
  with conflicting licence tags;
- leaves rows from already-published datasets as their authors released them;
- masks the 17 sources the builder collected itself: structured identifiers by regex, and names and addresses with
  `urchade/gliner_multi_pii-v1`. Names are masked only in the sources about private people.

The policy file gives each source's origin, config and name-masking decision, with a one-line reason.
`pii_stop_spans.txt` lists role words the NER model wrongly tags as names (Claimant, she, ...).

```bash
python -I build_clean.py --train <pool v7 train.jsonl.gz> --val <pool v7 val.jsonl.gz> --out-dir clean \
    --bench <benchmark jsonl files ...>
python -I pii_pass.py --train clean/train.jsonl.gz --val clean/val.jsonl.gz --policy pii_policy.json \
    --stop-spans pii_stop_spans.txt --out-dir clean_pii                       # GPU recommended (gliner, sentencepiece)
python -I email_sweep.py --pool-dir clean_pii
python -I build_hf.py --pool-dir clean_pii --policy pii_policy.json --out-dir Votum-Pool   # pyarrow
```

| Stage | Training rows |
|---|---:|
| pool v7 | 405,547 |
| after step 1 | 339,549 (188 sources) |
| after step 2 | 327,689 (181 sources) |
