"""Write tests/fixtures/smoke_expected.json and vision_expected.json with the reference implementation (run once per release).
    python scripts/make_smoke_fixture.py --repo <Votum repo or dir>
vision_expected.json holds, per image request, the option scores of every question and the image's codes, so the GPU test
checks the image tokenizer as well as the model."""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from votum import Votum, request as R
FIX = ROOT / "tests" / "fixtures"


def with_paths(x, base: Path):
    """image values of the fixtures are paths relative to tests/fixtures"""
    if R.is_image(x): return {"image": str(base / x["image"])}
    if isinstance(x, dict): return {k: with_paths(v, base) for k, v in x.items()}
    if isinstance(x, list): return [with_paths(v, base) for v in x]
    return x


def answers(vm, rows):
    return [{"qid": r["id"], "ids": o["ids"], "scores": [round(x, 4) for x in o["scores"]], "probs": {k: round(v, 4) for k, v in o["probs"].items()}}
            for r, o in zip(rows, vm.decide_rows(rows, keys=[0] * len(rows)))]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--repo", required=True); a = ap.parse_args()
    vm = Votum.from_pretrained(a.repo)
    reqs = json.loads((FIX / "smoke_requests.json").read_text())
    out = [answers(vm, [R.to_row(req["state"], q, rid=qid) for qid, q in req["questions"].items()]) for req in reqs]
    (FIX / "smoke_expected.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    vout = []
    for req in json.loads((FIX / "vision_requests.json").read_text()):
        rows = R.request_rows(dict(req, state=with_paths(req["state"], FIX)), vm.images.encode)
        vout.append({"images": [{"h": im["h"], "w": im["w"], "codes": im["codes"]} for im in rows[0]["images"]],
                     "answers": answers(vm, rows)})
    (FIX / "vision_expected.json").write_text(json.dumps(vout, ensure_ascii=False) + "\n")
    for o in out + [v["answers"] for v in vout]:
        for x in o: print(x["qid"], max(x["probs"], key=x["probs"].get), round(max(x["probs"].values()), 3))
