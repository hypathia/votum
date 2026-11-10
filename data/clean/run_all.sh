#!/usr/bin/env bash
# Run the whole pipeline of README.md: steps 1-4, then the regex part of the audit.
#
#   ./run_all.sh <pool dir> <work dir> <benchmark file> [<benchmark file> ...]
#
# <pool dir> holds the pool's train.jsonl.gz and val.jsonl.gz. Everything is written under <work dir>:
#   clean/        step 1 (build_clean.py)      clean_pii/   steps 2 and 3 (pii_pass.py, email_sweep.py)
#   Votum-Pool/   step 4 (build_hf.py)         audit/       step 5 (audit.py sample, regex, report)
# Rerunning resumes: step 2 reuses its saved chunks when nothing they depend on has changed.
#
# Environment (all optional):
#   PYTHON       interpreter (default python3)
#   INFO_LABELS  benchmark labels that are only measured, never used to drop rows (build_clean.py --info-labels)
#   NER_PARTS    run step 2 as this many parallel processes on the GPU (default 1)
#   PII_ARGS     extra pii_pass.py arguments, e.g. "--batch 8" on a small GPU or "--no-ner" for a dry run
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
POOL="${1:?usage: run_all.sh <pool dir> <work dir> <benchmark file> [...]}"
WORK="${2:?usage: run_all.sh <pool dir> <work dir> <benchmark file> [...]}"
shift 2
[ $# -ge 1 ] || { echo "run_all.sh: give at least one benchmark file" >&2; exit 2; }
PY="${PYTHON:-python3}"
PARTS="${NER_PARTS:-1}"
read -r -a INFO <<< "${INFO_LABELS:-}"
read -r -a PII_EXTRA <<< "${PII_ARGS:-}"
mkdir -p "$WORK"

echo "== 1/5 build_clean"
"$PY" -I "$HERE/build_clean.py" --train "$POOL/train.jsonl.gz" --val "$POOL/val.jsonl.gz" --out-dir "$WORK/clean" \
    --bench "$@" ${INFO[@]+--info-labels "${INFO[@]}"}

echo "== 2/5 pii_pass"
PII=("$PY" -I "$HERE/pii_pass.py" --train "$WORK/clean/train.jsonl.gz" --val "$WORK/clean/val.jsonl.gz"
     --policy "$HERE/pii_policy.json" --stop-spans "$HERE/pii_stop_spans.txt" --out-dir "$WORK/clean_pii" ${PII_EXTRA[@]+"${PII_EXTRA[@]}"})
if [ "$PARTS" -gt 1 ]; then
    pids=()
    for k in $(seq 0 $((PARTS - 1))); do
        "${PII[@]}" --part "$k/$PARTS" > "$WORK/pii_part_$k.log" 2>&1 &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || { echo "a pii_pass part failed; see $WORK/pii_part_*.log" >&2; exit 1; }; done
fi
"${PII[@]}"                                           # assembles the outputs (and computes any chunk still missing)

echo "== 3/5 email_sweep"
"$PY" -I "$HERE/email_sweep.py" --pool-dir "$WORK/clean_pii"

echo "== 4/5 build_hf"
"$PY" -I "$HERE/build_hf.py" --pool-dir "$WORK/clean_pii" --policy "$HERE/pii_policy.json" --out-dir "$WORK/Votum-Pool"

echo "== 5/5 audit (regex check; run 'audit.py spacy' and 'audit.py gliner' by hand for the name checks)"
"$PY" -I "$HERE/audit.py" sample --audit-dir "$WORK/audit" --pool-dir "$WORK/clean_pii" --policy "$HERE/pii_policy.json"
"$PY" -I "$HERE/audit.py" regex --audit-dir "$WORK/audit"
"$PY" -I "$HERE/audit.py" report --audit-dir "$WORK/audit" --policy "$HERE/pii_policy.json"
echo "done: dataset in $WORK/Votum-Pool (add the dataset card README.md by hand)"
