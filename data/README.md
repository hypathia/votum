# Training data

Votum-8B was trained on "pool v7": public decision data converted to one row format. The rows are not in this repository; the
cleaned pool is published as the dataset [{{ORG}}/Votum-Pool](https://huggingface.co/datasets/{{ORG}}/Votum-Pool).

| Path | Contents |
|---|---|
| `SOURCES.csv` | every source: licence tags, training rows, whether the second licence review removed it, rows in the cleaned pool |
| `clean/` | the scripts that turn pool v7 into the cleaned pool used for the re-train |

## The pool as trained

| Measure | Value |
|---|---|
| Training rows | 405,547 |
| Validation rows | 3,373 (whole groups; no group appears in both splits) |
| Sources | 217 |
| Question types | 295,876 choice, 62,172 yes/no, 47,499 score |
| Tokens | 238.5M (mean 588 per row, max 8,190) |
| Languages | 73 language codes; English is 41 % of rows (21 % of rows carry no language code) |

Rows by licence:

| Licence family | Rows | Share |
|---|---:|---:|
| CC-BY | 144,536 | 35.6 % |
| Apache-2.0 / MIT / AFL-3.0 | 98,753 | 24.4 % |
| CC0 / PDDL / public domain | 76,912 | 19.0 % |
| CC-BY-SA (share-alike) | 69,636 | 17.2 % |
| Government / portal terms permitting reuse | 11,820 | 2.9 % |
| ODC-By / ODbL / CDLA-Permissive / Etalab / OGL-UK | 3,890 | 1.0 % |

Notes:
- The licence tags come from each source's documentation and were not re-verified for this release.
- Wikipedia-derived rows (101,874) carry the attribution "Wikipedia contributors, CC-BY-SA".

## Known issues in the released checkpoint's training data

- **ContractNLI test split:** 1,982 rows are in training, and 283 items of the Jev Decision Index test suite were seen
  in training.
- **Sources removed after training:** a licence review removed 13 sources (19,950 rows). They supplied 2,697 of the
  55,784 questions seen in training (4.8 %).
- **bias_in_bios:** `decider/bias_in_bios` (2,350 rows) is in training although the build excluded it on paper.
- **LLM-generated content with unverified terms:**
  - ExpertQA (5,958 rows);
  - HelpSteer2/3 (about 9k);
  - hh-rlhf red-team (3.6k; its card advises against fine-tuning on it);
  - Aegis 2.0.
- **Benchmark overlap** with the Jev Decision Index suite: SATA-Bench 300, HoVer 195, BRIGHT 83, ANLI 38.
- **Personal data:** PII masking was regex-only, so names remain. A pilot found 82 missed identifiers per 500 rows.
- **Decontamination** was 13-gram on state + question, so it misses benchmarks that share documents with training
  sources.

## The cleaned pool

`clean/build_clean.py` removes three things:
- the sources listed above, together with the 13 sources dropped by the licence review;
- the ContractNLI test split;
- every row whose state shares a word 8-gram with any benchmark item (document-level decontamination).

The result is 339,549 training rows from 188 sources. The PII pass then drops seven more sources, which leaves the
327,689 training rows from 181 sources that are published (see `clean/README.md`).

The code that converted each source into rows is not released: it depends on raw downloads, a per-source specification
whose review notes quote personal data from source rows, and internal review records. `SOURCES.csv` and the published
dataset record its result. Source names keep the prefix of the conversion batch they were added in (`tasksource/`,
`lab/`, `expand/`, …); the prefix carries no other meaning.
