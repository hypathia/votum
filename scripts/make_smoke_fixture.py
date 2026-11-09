"""Write tests/fixtures/smoke_expected.json with the reference implementation (run once per release).
    python scripts/make_smoke_fixture.py --repo <Votum repo or dir>"""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from votum import Votum, request as R
ap = argparse.ArgumentParser(); ap.add_argument("--repo", required=True); a = ap.parse_args()
vm = Votum.from_pretrained(a.repo)
reqs = json.loads((ROOT / "tests/fixtures/smoke_requests.json").read_text()); out = []
for req in reqs:
    rows = [R.to_row(req["state"], q, rid=qid) for qid, q in req["questions"].items()]
    out.append([{"qid": r["id"], "ids": o["ids"], "scores": [round(x, 4) for x in o["scores"]], "probs": {k: round(v, 4) for k, v in o["probs"].items()}}
                for r, o in zip(rows, vm.decide_rows(rows, keys=[0] * len(rows)))])
(ROOT / "tests/fixtures/smoke_expected.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
for req, o in zip(reqs, out):
    for x in o: print(x["qid"], max(x["probs"], key=x["probs"].get), round(max(x["probs"].values()), 3))
