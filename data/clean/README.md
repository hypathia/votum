# Cleaned pool and the Votum Pool dataset

Five steps turn the pool the released checkpoint was trained on (pool v7) into the published dataset
`{{ORG}}/Votum-Pool`, which is also the input of the planned re-train.

| Step | Script | What it does |
|---|---|---|
| 1 | `build_clean.py` | drops sources (the 13 from the licence review, bias_in_bios, LLM-generated sets with unverified terms) and the ContractNLI test split; then document-level decontamination, dropping any row whose state shares a word 8-gram with an evaluation benchmark (a short stop-list of contract and chat boilerplate is exempt). Text in scripts written without spaces (Chinese, Japanese, Thai, ...) is also compared as 16-character grams, and a row's question is checked too, where a match counts only if at most 5 rows share it (a quoted benchmark item, not a question template) |
| 2 | `pii_pass.py` | applies `pii_policy.json` (see below) |
| 3 | `email_sweep.py` | every e-mail address becomes `[EMAIL]`, in place, except seven office mailboxes of Bernese and Zurich courts and cantonal services (listed in the script, each checked by hand) and the reserved `example.com/org/net` (SWEEPROWS rows changed) |
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

`pii_pass.py` keeps per-chunk checkpoints and reuses one only when nothing it depends on has changed (the chunk's rows,
the script, the model and thresholds, the stop spans, the policy of the chunk's sources). The NER model runs on the GPU
and is not bit-reproducible across machines, so a rerun elsewhere can mask a few names differently.

`build_hf.py` corrects a row's licence tag where the policy says the pool's own tag was wrong (`license_fix`: the two
parliamentary sources that had been tagged MIT carry their portals' real terms), and stops if a share-alike row would
land in the `default` config.

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
