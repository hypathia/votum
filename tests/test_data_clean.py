"""CPU tests of the dataset pipeline in data/clean on small synthetic inputs (no GPU, no NER model, no network):
build_clean's three rules, pii_pass's regex layer, span merging, row handling and checkpoints, email_sweep, build_hf
and the audit's regex check."""
import gzip
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

CLEAN = Path(__file__).resolve().parents[1] / "data" / "clean"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, CLEAN / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(script, *args):
    return subprocess.run([sys.executable, "-I", str(CLEAN / script), *map(str, args)], capture_output=True, text=True, check=True)


def _write_jsonl_gz(path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_jsonl_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def row(id, source="src/a", state="A short neutral state.", question="Which one?", options=("yes", "no"), **kw):
    r = {"id": id, "source": source, "license": "CC-BY-4.0", "license_detail": "", "type": "choice", "state": state,
         "question": question, "options": [{"id": o, "text": o} for o in options], "target": [1.0] + [0.0] * (len(options) - 1),
         "soft": False, "group": id, "kind": "all"}
    r.update(kw)
    return r


@pytest.fixture(scope="module")
def pii():
    return _load("pii_pass")


# ------------------------------------------------------------------------------------------------ build_clean.py
BENCH_TEXT = "the quick brown fox jumps over the lazy sleeping dog near the river bank"
CJK_TEXT = "这是一个用于测试的中文句子它的长度足够构成十六个字符的片段"
TEMPLATE_Q = "Is the following statement about the quick brown fox jumps over the lazy sleeping dog true?"


def _pool_and_bench(tmp_path, extra_train=()):
    train = [
        row("keep/1", state="Nothing in common with any benchmark item at all here."),
        row("r1/1", source="expand/expertqa"),
        row("r2/1", source="ContractNLI", upstream_split="test"),
        row("r2/2", source="ContractNLI", upstream_split="train"),
        row("r3/state", state={"text": "Intro. The QUICK brown fox -- jumps over the lazy, sleeping dog! End."}),
        row("r3/cjk", state="前言" + CJK_TEXT[:20] + "后记"),
        row("r3/question", question="Read: " + BENCH_TEXT),
    ]
    # a question template shared by many rows is not an item quoted from a benchmark
    train += [row(f"tmpl/{i}", source=f"src/t{i % 2}", question=TEMPLATE_Q) for i in range(6)]
    # boilerplate shared by many rows of several sources: goes on the stop-list, so these rows stay
    boiler = "this agreement shall be governed by the laws of the state of new york"
    train += [row(f"boiler/{i}", source=f"src/b{i % 3}", state=f"Clause {i}. {boiler}.") for i in range(6)]
    train += list(extra_train)
    val = [row("val/1"), row("val/2", state="We saw that " + BENCH_TEXT)]
    bench = [{"state": BENCH_TEXT}, {"record": {"state": CJK_TEXT}}, {"row": {"state": "Contract: " + boiler + " and more."}},
             {"state": "", "questions": {"q": {"instructions": "Pick an answer: a totally different long benchmark question text here"}}}]
    _write_jsonl_gz(tmp_path / "train.jsonl.gz", train)
    _write_jsonl_gz(tmp_path / "val.jsonl.gz", val)
    with open(tmp_path / "bench.jsonl", "w") as f:
        for b in bench:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")


def test_build_clean_rules(tmp_path):
    pytest.importorskip("numpy")
    _pool_and_bench(tmp_path)
    out = tmp_path / "clean"
    _run("build_clean.py", "--train", tmp_path / "train.jsonl.gz", "--val", tmp_path / "val.jsonl.gz", "--out-dir", out,
         "--bench", tmp_path / "bench.jsonl", "--workers", 2, "--stop-min-rows", 3, "--stop-min-sources", 2)
    kept = {r["id"] for r in _read_jsonl_gz(out / "train.jsonl.gz")}
    assert {"keep/1", "r2/2"} <= kept
    assert {f"tmpl/{i}" for i in range(6)} <= kept
    assert {f"boiler/{i}" for i in range(6)} <= kept
    assert not kept & {"r1/1", "r2/1", "r3/state", "r3/cjk", "r3/question"}
    assert [r["id"] for r in _read_jsonl_gz(out / "val.jsonl.gz")] == ["val/1"]
    with gzip.open(out / "dropped.tsv.gz", "rt") as f:
        rules = {line.split("\t")[2] or line.split("\t")[3]: line.split("\t")[4] for line in list(f)[1:]}
    assert rules["expand/expertqa"].startswith("R1:") and rules["ContractNLI"].startswith("R2:")
    assert rules["r3/state"] == rules["r3/cjk"] == rules["r3/question"] == "R3:benchmark 8-gram"
    rep = json.load(open(out / "REPORT.json"))
    assert rep["summary"]["train"]["R3_rows_dropped_for_the_question_only"] == 1
    assert rep["stoplist"]["size"] >= 1
    # kept lines are copied byte for byte
    with gzip.open(tmp_path / "train.jsonl.gz", "rt") as f:
        src = {json.loads(line)["id"]: line for line in f}
    with gzip.open(out / "train.jsonl.gz", "rt") as f:
        assert all(src[json.loads(line)["id"]] == line for line in f)


def test_build_clean_stops_on_unlisted_pattern_match(tmp_path):
    pytest.importorskip("numpy")
    _pool_and_bench(tmp_path, extra_train=[row("x/1", source="lab/helpsteer2_new")])
    p = subprocess.run([sys.executable, "-I", str(CLEAN / "build_clean.py"), "--train", str(tmp_path / "train.jsonl.gz"),
                        "--val", str(tmp_path / "val.jsonl.gz"), "--out-dir", str(tmp_path / "o"), "--bench", str(tmp_path / "bench.jsonl")],
                       capture_output=True, text=True)
    assert p.returncode != 0 and "lab/helpsteer2_new" in p.stderr


# ------------------------------------------------------------------------------------------------ pii_pass.py: regex layer
@pytest.mark.parametrize("text, expected", [
    ("write to jane.doe@gmail.com now", "write to [EMAIL] now"),
    ("mail info@court.ch", "mail info@court.ch"),                                  # role mailbox
    ("see a.b@example.com", "see a.b@example.com"),                                # reserved domain
    ("call +41 79 123 45 67 today", "call [PHONE] today"),
    ("Tel. 0791234567", "Tel. [PHONE]"),
    ("CHF 1 250 000 000 paid", "CHF 1 250 000 000 paid"),                          # an amount
    ("on 2019-05-12 at", "on 2019-05-12 at"),                                      # a date
    ("version 10.2.33.4", "version 10.2.33.4"),                                    # not an IP address
    ("ISBN 978-3-16-148410-0", "ISBN 978-3-16-148410-0"),
    ("IBAN CH93 0076 2011 6238 5295 7", "IBAN [IBAN]"),
    ("AHV 756.1234.5678.97", "AHV [ID]"),
    ("passport no: X1234567", "passport no: [ID]"),
    ("server 85.12.34.56 logged", "server [IP] logged"),
    ("lives at 12 Baker Street, London", "lives at [ADDRESS], London"),
    ("the 2019 Supreme Court ruling", "the 2019 Supreme Court ruling"),
    ("Bahnhofstrasse 12, Zürich", "[ADDRESS], Zürich"),
    ("Spielplatz 42", "Spielplatz 42"),
    ("profile https://example.org/users/jdoe.", "profile [URL]."),
    ("https://www.admin.ch/gov/de/start.html", "https://www.admin.ch/gov/de/start.html"),
])
def test_regex_mask(pii, text, expected):
    assert pii.regex_mask(text)[0] == expected


def test_wiki_user_names_only_when_policy_masks_names(pii):
    assert pii.regex_mask("see User:Foobar", names=False)[0] == "see User:Foobar"
    assert pii.regex_mask("see User:Foobar", names=True) == ("see User:[NAME]", {"NAME": 1})
    # the pattern is case-insensitive, so one following word is masked with the user name (over-masks, never under-masks)
    assert pii.regex_mask("see User:Foobar here", names=True)[0] == "see User:[NAME]"


def test_apply_spans_merges_and_respects_placeholders(pii):
    t = "Dear Jean Dupont, [EMAIL] wrote from 3 Rue X."
    out, cnt = pii.apply_spans(t, [(5, 9, "NAME"), (9, 16, "NAME"), (18, 25, "NAME"), (36, 45, "ADDRESS")])
    assert out == "Dear [NAME], [EMAIL] wrote from [ADDRESS]." and cnt == {"NAME": 1, "ADDRESS": 1}


def test_name_filter(pii):
    ner = pii.NER.__new__(pii.NER)
    ner.stop, ner.thp, ner.thu = {"claimant"}, 0.5, 0.7
    ok = {t: ner._name_ok("person", 0.9, t, t, 0, len(t)) for t in ["the Claimant", "Monsieur X", "van der", "[claimant A]", "Jean Dupont"]}
    assert ok == {"the Claimant": False, "Monsieur X": False, "van der": False, "[claimant A]": False, "Jean Dupont": True}
    assert not ner._name_ok("person", 0.4, "Jean Dupont", "Jean Dupont", 0, 11)                 # below the threshold


# ------------------------------------------------------------------------------------------------ pii_pass.py: rows
POLICY = {
    "pub": {"excluded": False, "origin": "redistributed", "names": False, "config": "default", "license": "CC-BY-4.0"},
    "scraped": {"excluded": False, "origin": "self-collected", "names": True, "config": "default", "license": "CC0-1.0"},
    "portal": {"excluded": True, "origin": "redistributed", "names": False, "config": "default", "license": "portal terms"},
}


class FakeNER:
    """Tags every occurrence of 'Anna Muster' as a person."""

    def spans(self, texts, names):
        out = []
        for t in texts:
            i = t.find("Anna Muster")
            out.append([(i, i + 11, "NAME")] if names and i >= 0 else [])
        return out


def test_process_chunk(pii):
    rows = [
        row("p/1", source="pub", state="Anna Muster wrote to x@gmail.com"),                          # copied as is
        row("s/1", source="scraped", state={"body": "Anna Muster, Tel. 0791234567"}),
        row("s/2", source="scraped", options=("Anna Muster", "Peter Muster")),                        # still distinct
        row("s/3", source="scraped", options=("Anna Muster", "[NAME]")),                              # collapses
        row("x/1", source="portal"),
    ]
    lines = [json.dumps(r, ensure_ascii=False) for r in rows]
    out, stats, dropped = pii.process_chunk(lines, POLICY, FakeNER(), no_ner=False)
    assert out[0] == lines[0]
    s1 = json.loads(out[1])
    assert s1["state"] == {"body": "[NAME], Tel. [PHONE]"}
    assert [o["text"] for o in json.loads(out[2])["options"]] == ["[NAME]", "Peter Muster"]
    assert len(out) == 3 and dropped == [("s/3", "scraped", "options_not_distinct_after_masking")]
    assert stats["portal"] == {"licence_excluded": 1}
    assert stats["scraped"]["rows_changed"] == 2 and stats["scraped"]["mask_NAME"] == 2 and stats["scraped"]["rows_out"] == 2


def test_pii_pass_checkpoints(tmp_path):
    rows = [row(f"s/{i}", source="scraped", state=f"Call +41 79 123 45 6{i} please") for i in range(5)] + [row("p/1", source="pub")]
    _write_jsonl_gz(tmp_path / "train.jsonl.gz", rows)
    _write_jsonl_gz(tmp_path / "val.jsonl.gz", rows[:1])
    pol = tmp_path / "policy.json"
    pol.write_text(json.dumps(POLICY))
    args = ["pii_pass.py", "--train", tmp_path / "train.jsonl.gz", "--val", tmp_path / "val.jsonl.gz", "--policy", pol,
            "--out-dir", tmp_path / "out", "--no-ner", "--chunk", 2]
    first = _run(*args).stdout
    assert first.count("chunk") == 4                                     # val 1 chunk, train 3 chunks
    assert {r["state"] for r in _read_jsonl_gz(tmp_path / "out" / "train.jsonl.gz")[:5]} == {"Call [PHONE] please"}
    assert "chunk" not in _run(*args).stdout                              # everything reused
    POLICY2 = dict(POLICY, pub=dict(POLICY["pub"], excluded=True))
    pol.write_text(json.dumps(POLICY2))
    assert _run(*args).stdout.count("chunk") == 1                        # only the chunk holding the `pub` row
    assert [r["id"] for r in _read_jsonl_gz(tmp_path / "out" / "train.jsonl.gz")] == [f"s/{i}" for i in range(5)]


# ------------------------------------------------------------------------------------------------ email_sweep.py
def test_email_sweep(tmp_path):
    es = _load("email_sweep")
    assert es.sweep({"a": ["mail jane@uni.edu", "obergericht-straf.bern@justice.be.ch", "x@sub.example.org"]}) == \
        {"a": ["mail [EMAIL]", "obergericht-straf.bern@justice.be.ch", "x@sub.example.org"]}
    d = tmp_path / "pool"
    d.mkdir()
    _write_jsonl_gz(d / "train.jsonl.gz", [row("a", state="write jane@uni.edu"), row("b")])
    _write_jsonl_gz(d / "val.jsonl.gz", [row("c")])
    assert es.sweep_file(d / "train.jsonl.gz") == 1
    assert es.sweep_file(d / "train.jsonl.gz") == 0                       # idempotent
    assert _read_jsonl_gz(d / "train.jsonl.gz")[0]["state"] == "write [EMAIL]"


# ------------------------------------------------------------------------------------------------ build_hf.py
def _hf_inputs(tmp_path, policy):
    d = tmp_path / "pool"
    d.mkdir()
    _write_jsonl_gz(d / "train.jsonl.gz", [row("a/1", source="pub", n_tok=12), row("s/1", source="sa", license="CC-BY-SA-4.0",
                                                                                     state={"k": 1, "b": "x"}),
                                            row("f/1", source="fixme", license="MIT", license_detail="Kanton Thurgau")])
    _write_jsonl_gz(d / "val.jsonl.gz", [row("a/2", source="pub")])
    (tmp_path / "policy.json").write_text(json.dumps(policy))
    return d


HF_POLICY = {
    "pub": {"excluded": False, "origin": "redistributed", "names": False, "config": "default", "license": "CC-BY-4.0", "upstream": "u"},
    "sa": {"excluded": False, "origin": "self-collected", "names": True, "config": "share_alike", "license": "CC-BY-SA-4.0"},
    "fixme": {"excluded": False, "origin": "self-collected", "names": False, "config": "default", "license": "CC0-1.0",
              "license_fix": [["Thurgau", "CC0-1.0"], ["", "unknown"]]},
}


def test_build_hf(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    d = _hf_inputs(tmp_path, HF_POLICY)
    out = tmp_path / "hf"
    _run("build_hf.py", "--pool-dir", d, "--policy", tmp_path / "policy.json", "--out-dir", out)
    default = pq.read_table(out / "data/default/train-00000-of-00001.parquet").to_pylist()
    assert [r["id"] for r in default] == ["a/1", "f/1"] and default[1]["license"] == "CC0-1.0"
    assert "n_tok" not in default[0]
    sa = pq.read_table(out / "data/share_alike/train-00000-of-00001.parquet").to_pylist()
    assert sa[0]["state"] == '{"b": "x", "k": 1}'
    assert pq.read_table(out / "data/default/validation-00000-of-00001.parquet").num_rows == 1
    assert (out / "SOURCES.csv").read_text().splitlines()[1:] == [
        "fixme,default,self-collected,CC0-1.0,1,0,no,,", "pub,default,redistributed,CC-BY-4.0,1,1,no,u,",
        "sa,share_alike,self-collected,CC-BY-SA-4.0,1,0,yes,,"]          # the last column is the description (none here)
    assert json.load(open(d / "build_hf_stats.json"))["dropped_fields"] == {"n_tok": 1}


@pytest.mark.parametrize("change, message", [
    ({"sa": dict(HF_POLICY["sa"], config="default")}, "share-alike"),
    ({"pub": dict(HF_POLICY["pub"], excluded=True)}, "excluded"),
])
def test_build_hf_guards(tmp_path, change, message):
    pytest.importorskip("pyarrow")
    d = _hf_inputs(tmp_path, dict(HF_POLICY, **change))
    p = subprocess.run([sys.executable, "-I", str(CLEAN / "build_hf.py"), "--pool-dir", str(d), "--policy", str(tmp_path / "policy.json"),
                        "--out-dir", str(tmp_path / "hf")], capture_output=True, text=True)
    assert p.returncode != 0 and message in p.stderr


# ------------------------------------------------------------------------------------------------ audit.py
def test_audit_sample_regex_report(tmp_path):
    d = tmp_path / "pool"
    d.mkdir()
    pol = {"sc": dict(POLICY["scraped"]), "sc2": dict(POLICY["scraped"], config="share_alike"), "pub": POLICY["pub"]}
    (tmp_path / "policy.json").write_text(json.dumps(pol))
    _write_jsonl_gz(d / "train.jsonl.gz", [row("m/1", source="sc", state="reach me on +41 79 123 45 67"),
                                           row("m/2", source="sc2"), row("p/1", source="pub", state="+41 79 123 45 67")])
    _write_jsonl_gz(d / "val.jsonl.gz", [])
    au = tmp_path / "audit"
    _run("audit.py", "sample", "--audit-dir", au, "--pool-dir", d, "--policy", tmp_path / "policy.json", "--n", 2)
    assert sorted(json.loads(line)["id"] for line in open(au / "sample.jsonl")) == ["m/1", "m/2"]
    _run("audit.py", "regex", "--audit-dir", au)
    hits = json.load(open(au / "hits_regex.json"))
    assert {(h[0], h[1]) for h in hits} == {("m/1", "phone"), ("m/1", "phone (loose)")}
    rep = _run("audit.py", "report", "--audit-dir", au, "--policy", tmp_path / "policy.json").stdout
    assert "default {'phone / regex': 2, 'phone (loose) / regex': 1}" in rep     # two of the pass's phone patterns match
