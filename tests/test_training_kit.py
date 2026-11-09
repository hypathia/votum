"""CPU tests of the fine-tuning kit's data handling (no tokenizer, no model): prepare_data's validation, splitting and
CSV conversion, and export_hf's LoRA key normalisation."""
import csv, importlib.util, sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "training")); sys.path.insert(0, str(ROOT / "src"))
    spec.loader.exec_module(mod); return mod


@pytest.fixture(scope="module")
def pd():
    return _load("prepare_data", ROOT / "training" / "prepare_data.py")


def row(**kw):
    r = dict(id="r/1", type="choice", state="Hi", question="Which team?", options=[{"id": "a", "text": "A"}, {"id": "b", "text": "B"}], target=[1, 0])
    r.update(kw); return r


def test_check_normalises_and_defaults(pd):
    r = pd.check(row(target=[2, 2]), "t")
    assert r["target"] == [0.5, 0.5] and r["group"] == "r/1" and r["soft"] is True


@pytest.mark.parametrize("bad, msg", [
    (dict(type="multi"), "type must be one of"),
    (dict(options=[{"id": "a"}]), "at least two options"),
    (dict(options=[{"id": "a"}, {"id": "a"}]), "option ids repeat"),
    (dict(type="noul", options=[{"id": "yes"}, {"id": "no"}]), "options true and false"),
    (dict(target=[1]), "one probability per option"),
    (dict(target=[0, 0]), "not all zero"),
    (dict(question=" "), "empty question"),
])
def test_check_rejects(pd, bad, msg):
    with pytest.raises(SystemExit) as e: pd.check(row(**bad), "t")
    assert msg in str(e.value)


def test_split_keeps_groups_together(pd):
    rows = [row(id=f"g{i // 3}/{i}", group=f"g{i // 3}") for i in range(300)]
    tr, va = pd.split_rows(rows, 0.2, seed=0)
    assert 30 < len(va) < 90 and len(tr) + len(va) == 300
    assert not ({r["group"] for r in tr} & {r["group"] for r in va})
    assert pd.split_rows(rows, 0.2, seed=0)[1] == va          # deterministic


def test_csv_rows(pd, tmp_path):
    p = tmp_path / "t.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["text", "team"]); w.writeheader()
        w.writerow({"text": "double charge", "team": "billing"}); w.writerow({"text": "printer jammed", "team": "tech"})
    rows = list(pd.csv_rows(p, "text", "team", "Which team?", "billing=Billing;tech=Tech;sales=Sales", None))
    assert len(rows) == 2 and [o["id"] for o in rows[0]["options"]] == ["billing", "tech", "sales"]
    assert rows[0]["target"] == [1.0, 0.0, 0.0] and rows[1]["target"] == [0.0, 1.0, 0.0] and rows[0]["question"] == "Which team?"
    rows = list(pd.csv_rows(p, "text", "team", "Which team?", None, None))          # labels from the file
    assert [o["id"] for o in rows[0]["options"]] == ["billing", "tech"]
    with pytest.raises(SystemExit): list(pd.csv_rows(p, "text", "team", "Q", "billing=Billing", None))   # unknown label tech
    with pytest.raises(SystemExit): list(pd.csv_rows(p, "txt", "team", "Q", None, None))                   # missing column


def test_stats(pd):
    rows = [dict(row(), n_tok_span=n, group=f"g{n}", lang="en") for n in (100, 200, 300, 400)]
    st = pd.stats(rows, 65536)
    assert st["rows"] == 4 and st["tokens_total"] == 1000 and st["tokens_max"] == 400 and st["steps_per_epoch"] == 1


def test_lora_key_normalisation():
    sys.path.insert(0, str(ROOT / "src"))
    from votum.build import LORA
    plain = "base_model.model.layers.3.self_attn.q_proj.lora_A.weight"           # plain PEFT wrapper on the bare text stack
    canon = plain.replace("base_model.model.layers.", "base_model.model.model.layers.", 1)
    assert LORA.fullmatch(plain) is None and LORA.fullmatch(canon) is not None and LORA.fullmatch(canon)[1] == "3"
