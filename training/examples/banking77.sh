#!/bin/bash
# Worked example from training/README.md: fine-tune Votum-8B on 2,000 BANKING77 training rows (77 intents) and measure
# accuracy on 1,000 held-out test rows before and after. Run from the training/ directory, in the environment that has
# votum installed, with a CUDA GPU. Needs: pyarrow, huggingface_hub, 17 GB of free disk for the merged export, and the
# model (MODEL below: a Hub id or a local directory). About an hour on a 24 GB card: 228 steps of 32k tokens.
#
#     bash examples/banking77.sh [MODEL] [OUT_DIR]
set -euo pipefail
DEF="{{ORG}}/Votum-8B"; MODEL=${1:-${VOTUM_REPO:-$DEF}}; OUT=${2:-./Votum-8B-banking77}; W=./examples/banking77_work; mkdir -p "$W"
PY=${PYTHON:-python}

echo "== data: PolyAI/banking77 from the Hub -> CSV (2,000 training rows) and a 1,000-row test file"
$PY - "$W" <<'EOF'
import csv, gzip, json, random, sys
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
W = sys.argv[1]; d = snapshot_download("PolyAI/banking77", repo_type="dataset", allow_patterns=["data/*.parquet", "README.md"])
import re
names = {}
for line in open(f"{d}/README.md"):                        # the label names, by label index, from the dataset card: '0': activate_my_card
    m = re.match(r"\s*'(\d+)':\s*(\S+)\s*$", line)
    if m: names[int(m[1])] = m[2]
    elif names and line.strip() and not line.startswith(" "): break
names = [names[i] for i in range(77)] if len(names) >= 77 else None
assert names, "could not read the 77 label names from the dataset card"
tr = pq.read_table(f"{d}/data/train-00000-of-00001.parquet").to_pylist(); te = pq.read_table(f"{d}/data/test-00000-of-00001.parquet").to_pylist()
rng = random.Random(0); sub = rng.sample(tr, 2000); ts = rng.sample(te, 1000)
with open(f"{W}/train.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["text", "label"]); w.writeheader()
    for r in sub: w.writerow({"text": r["text"], "label": str(r["label"])})
opts = [{"id": str(i), "text": n.replace("_", " ")} for i, n in enumerate(names)]
with gzip.open(f"{W}/test.jsonl.gz", "wt") as f:
    for i, r in enumerate(ts):
        f.write(json.dumps({"id": f"b77test/{i}", "group": f"b77test/{i}", "type": "choice", "state": r["text"], "question": "What is the customer's intent?",
                            "options": opts, "target": [1.0 if o["id"] == str(r["label"]) else 0.0 for o in opts]}) + "\n")
open(f"{W}/labels.txt", "w").write(";".join(f"{o['id']}={o['text']}" for o in opts))
print("train.csv 2000 rows, test.jsonl.gz 1000 rows, 77 labels")
EOF

echo "== prepare"
$PY prepare_data.py --csv "$W/train.csv" --text-col text --label-col label --question "What is the customer's intent?" \
    --labels "$(cat "$W/labels.txt")" --val-frac 0.1 --tokenizer "$MODEL" --out "$W/data"
echo "== before: accuracy of $MODEL on the 1,000 test rows"
$PY calibrate.py --model "$MODEL" --val "$W/test.jsonl.gz" --eval-only --batch 4
echo "== train (2 epochs)"
$PY finetune.py --from "$MODEL" --data "$W/data" --tag banking77 --epochs 2 --micro-tokens 2048
CK=$(ls -d "${VOTUM_RUNS:-runs}"/banking77/ckpt_* | sort -t_ -k2 -n | tail -1)
echo "== export $CK -> $OUT"
$PY ../scripts/export_hf.py --ckpt "$CK" --out "$OUT"
echo "== calibrate on the validation split"
$PY calibrate.py --model "$OUT" --val "$W/data/val.jsonl.gz" --batch 4
echo "== after: accuracy of $OUT on the same 1,000 test rows"
$PY calibrate.py --model "$OUT" --val "$W/test.jsonl.gz" --eval-only --batch 4
