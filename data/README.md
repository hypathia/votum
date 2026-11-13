# Training data

Votum-8B was trained on a pool of public decision data converted to one row format. The rows themselves are not in
this repository. A cleaned and larger version of the pool is published as the dataset
[votum-ai/Votum-Pool](https://huggingface.co/datasets/votum-ai/Votum-Pool).

| Path | Contents |
|---|---|
| `SOURCES.csv` | every source: licence tags, training rows, whether the second licence review removed it, rows in the cleaned pool |
| `clean/` | the pipeline that turns the converted rows into the cleaned pool used for the re-train, and into the published dataset |

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
| CC-BY | 145,251 | 35.8 % |
| Apache-2.0 / MIT / AFL-3.0 | 96,013 | 23.7 % |
| CC0 / PDDL / public domain | 77,253 | 19.0 % |
| CC-BY-SA (share-alike) | 69,636 | 17.2 % |
| Government / portal terms permitting reuse | 13,504 | 3.3 % |
| ODC-By / ODbL / CDLA-Permissive / Etalab / OGL-UK | 3,890 | 1.0 % |

The licence tags come from each source's documentation and were not re-verified for this release. Wikipedia-derived
rows (101,874) carry the attribution "Wikipedia contributors, CC-BY-SA".

## Known issues in the released checkpoint's training data

A review after training found the following. The cleaned pool removes or addresses each of them.

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
- **Benchmark overlap** with the Jev Decision Index suite, as items with a row in the pool (of them, seen in training):
  SATA-Bench 314 (300), HoVer 195 (17), BRIGHT 83 (33), ANLI 38 (2).
- **Personal data:** PII masking was regex-only, so names remain. A pilot found 82 missed identifiers per 500 rows.
- **Decontamination** was 13-gram on state + question, so it misses benchmarks that share documents with training
  sources.

## The cleaned pool

For training, each source was cut to a limit so that no single source dominates. The cleaned pool has no
such limit: it starts from all the rows the converters produced for those sources, 881,143 training rows after
duplicates are removed. Every row of the training pool is among them, unchanged.

The `clean` stage of `clean/pool.py` removes three things:

- the sources listed above, together with the 13 sources dropped by the licence review;
- the ContractNLI test split;
- every row that overlaps a benchmark: its state shares a word 8-gram with a benchmark item (16-character grams for
  scripts written without spaces), or its question quotes one.

That leaves 744,057 training rows from 188 sources. The `mask` stage then drops seven more sources and masks contact
details and identifiers (names stay), which gives the published 723,257 training rows from 181 sources; 328,075 of them were in the training pool (column
`in_votum_8b` of the dataset). `clean/README.md` describes each stage.

**Sources added later.** Eleven sources were added to the dataset after this build (219,414 training rows): ShARC,
NTSB aviation accident reports, Transport Canada and RappelConso recalls, ABCD, MTOP, NatCS, written questions of the
French National Assembly, the Dolly task categories, Polis deliberations and public-domain verse. They went through
the same `clean` and `mask` stages with their entries in `clean/pool.json`, and through a second benchmark comparison
on the full text of each row (state, question and options). The published dataset has 942,671 training rows from 192
sources. None of the added rows was in the training pool of the released checkpoint; the dataset card lists them under
"Sources added later".

**What is not released.** The code that converted each source into rows stays private. It depends on raw downloads,
on a per-source specification whose review notes quote personal data from source rows, and on internal review
records. `SOURCES.csv` and the published dataset record its result.

**Source names.** A name keeps the prefix of the conversion batch it was added in (`tasksource/`, `lab/`, `expand/`,
…). The prefix carries no other meaning.
