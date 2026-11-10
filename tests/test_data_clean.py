"""CPU tests of the pool pipeline in data/clean on small synthetic inputs (no GPU, no NER model, no network): the
engine, the three rules of the clean stage, the regex layer, span filters and cache of the mask stage, the e-mail
sweep, the export and the audit's regex check."""
import gzip
import json
import subprocess
import sys
from pathlib import Path

import pytest

CLEAN = Path(__file__).resolve().parents[1] / "data" / "clean"
sys.path.insert(0, str(CLEAN))

from poolkit import audit, engine, pii  # noqa: E402


def _pool(*args):
    """Run pool.py the way the README does (python -I)."""
    return subprocess.run([sys.executable, "-I", str(CLEAN / "pool.py"), *map(str, args)], capture_output=True, text=True)


def _ok(*args):
    p = _pool(*args)
    assert p.returncode == 0, p.stderr
    return p.stdout


def _write_jsonl_gz(path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_jsonl_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _write_pool(d, train, val=()):
    d.mkdir(parents=True, exist_ok=True)
    _write_jsonl_gz(d / "train.jsonl.gz", train)
    _write_jsonl_gz(d / "val.jsonl.gz", list(val))
    return d


def _config(path, sources, **sections):
    path.write_text(json.dumps(dict(sections, sources=sources)))
    return path


def row(id, source="src/a", state="A short neutral state.", question="Which one?", options=("yes", "no"), **kw):
    r = {"id": id, "source": source, "license": "CC-BY-4.0", "license_detail": "", "type": "choice", "state": state,
         "question": question, "options": [{"id": o, "text": o} for o in options], "target": [1.0] + [0.0] * (len(options) - 1),
         "soft": False, "group": id, "kind": "all"}
    r.update(kw)
    return r


def pub(**kw):
    return dict({"origin": "redistributed", "names": False, "config": "default", "license": "CC-BY-4.0"}, **kw)


# ------------------------------------------------------------------------------------------------ engine
def _double(lines):
    return [ln * 2 for ln in lines]


def test_engine_reads_in_order_and_writes_members(tmp_path):
    lines = [f"{i}\n".encode() for i in range(25)]
    with engine.GzOut(tmp_path / "a.jsonl.gz") as g:
        for i in range(0, 25, 10):
            g.write(engine.gz_member(lines[i:i + 10]))
        g.write(engine.gz_member([]))
    assert list(engine.lines_of(tmp_path / "a.jsonl.gz")) == lines                     # gzip members read as one file
    assert [len(c) for c in engine.chunks(tmp_path / "a.jsonl.gz", 10)] == [10, 10, 5]
    (tmp_path / "plain.jsonl").write_bytes(b"a\nb")                                    # no newline at the end
    assert list(engine.lines_of(tmp_path / "plain.jsonl")) == [b"a\n", b"b\n"]
    with engine.GzOut(tmp_path / "empty.jsonl.gz"):
        pass
    assert gzip.open(tmp_path / "empty.jsonl.gz").read() == b""
    for n in (1, 3):
        with engine.Workers(n) as w:
            assert list(w.imap(_double, engine.chunks(tmp_path / "a.jsonl.gz", 4))) == [_double(lines[i:i + 4]) for i in range(0, 25, 4)]


def test_config_is_checked(tmp_path):
    for bad in ({"s": {"origin": "found", "config": "default"}}, {"s": {"drop": "later", "drop_reason": "x"}}, {"s": {"drop": "clean"}}):
        with pytest.raises(engine.PoolError):
            engine.load_config(_config(tmp_path / "c.json", bad))
    cfg = engine.load_config(_config(tmp_path / "c.json", {"a": pub(), "b": {"drop": "mask", "drop_reason": "terms"}}))
    assert engine.dropped_at(cfg, "mask") == {"b": "terms"} and engine.dropped_at(cfg, "clean") == {}
    with pytest.raises(engine.PoolError):
        engine.policy(cfg, "unknown")


# ------------------------------------------------------------------------------------------------ clean
BENCH_TEXT = "the quick brown fox jumps over the lazy sleeping dog near the river bank"
CJK_TEXT = "这是一个用于测试的中文句子它的长度足够构成十六个字符的片段"
TEMPLATE_Q = "Is the following statement about the quick brown fox jumps over the lazy sleeping dog true?"
CLEAN_SOURCES = dict({"src/a": pub(), "ContractNLI": pub(), "expand/expertqa": {"drop": "clean", "drop_reason": "expertqa (LLM-generated)"}},
                     **{f"src/{k}{i}": pub() for k in "tb" for i in range(3)})
CLEAN_SECTION = {"drop_rows": [{"source": "ContractNLI", "field": "upstream_split", "equals": "test", "reason": "ContractNLI test split"}],
                 "must_drop_patterns": ["expertqa", "helpsteer2"], "decontaminate": {"stop_min_rows": 3, "stop_min_sources": 2}}


def _pool_and_bench(tmp_path, extra_train=(), extra_sources=None):
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
    _write_pool(tmp_path / "pool", train, val)
    with open(tmp_path / "bench.jsonl", "w") as f:
        for b in bench:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")
    return _config(tmp_path / "pool.json", dict(CLEAN_SOURCES, **(extra_sources or {})), clean=CLEAN_SECTION)


def test_clean_rules(tmp_path):
    pytest.importorskip("numpy")
    cfg = _pool_and_bench(tmp_path)
    out = tmp_path / "clean"
    _ok("clean", "--config", cfg, "--pool", tmp_path / "pool", "--out", out, "--bench", tmp_path / "bench.jsonl", "--workers", 2)
    kept = {r["id"] for r in _read_jsonl_gz(out / "train.jsonl.gz")}
    assert {"keep/1", "r2/2"} <= kept
    assert {f"tmpl/{i}" for i in range(6)} <= kept
    assert {f"boiler/{i}" for i in range(6)} <= kept
    assert not kept & {"r1/1", "r2/1", "r3/state", "r3/cjk", "r3/question"}
    assert [r["id"] for r in _read_jsonl_gz(out / "val.jsonl.gz")] == ["val/1"]
    with gzip.open(out / "dropped.tsv.gz", "rt") as f:
        rules = {line.split("\t")[2] or line.split("\t")[3]: line.split("\t")[4] for line in list(f)[1:]}
    assert rules["expand/expertqa"] == "R1:expertqa (LLM-generated)" and rules["ContractNLI"] == "R2:ContractNLI test split"
    assert rules["r3/state"] == rules["r3/cjk"] == rules["r3/question"] == "R3:benchmark overlap"
    rep = json.load(open(out / "REPORT.json"))
    assert rep["summary"]["train"]["rows_removed_by_rule"] == {"R1_sources": 1, "R2_row_filters": 1, "R3_benchmark_overlap": 3}
    assert rep["summary"]["train"]["R3_rows_dropped_for_the_question_only"] == 1
    assert rep["stoplist"]["size"] >= 1 and "governed by the laws" in (out / "stoplist.tsv").read_text()
    # kept lines are copied byte for byte
    with gzip.open(tmp_path / "pool" / "train.jsonl.gz", "rt") as f:
        src = {json.loads(line)["id"]: line for line in f}
    with gzip.open(out / "train.jsonl.gz", "rt") as f:
        assert all(src[json.loads(line)["id"]] == line for line in f)
    # the number of workers does not change the output
    _ok("clean", "--config", cfg, "--pool", tmp_path / "pool", "--out", tmp_path / "clean1", "--bench", tmp_path / "bench.jsonl", "--workers", 1)
    assert (tmp_path / "clean1" / "train.jsonl.gz").read_bytes() == (out / "train.jsonl.gz").read_bytes()
    # the plain rule also drops the boilerplate rows; --report-only writes no rows
    _ok("clean", "--config", cfg, "--pool", tmp_path / "pool", "--out", tmp_path / "plain", "--bench", tmp_path / "bench.jsonl",
        "--no-stoplist", "--report-only")
    plain = json.load(open(tmp_path / "plain" / "REPORT.json"))["summary"]["train"]
    assert plain["rows_removed_by_rule"]["R3_benchmark_overlap"] == 9 and not (tmp_path / "plain" / "train.jsonl.gz").exists()


def test_clean_stops_on_a_source_the_patterns_say_must_go(tmp_path):
    pytest.importorskip("numpy")
    cfg = _pool_and_bench(tmp_path, extra_train=[row("x/1", source="lab/helpsteer2_new")], extra_sources={"lab/helpsteer2_new": pub()})
    p = _pool("clean", "--config", cfg, "--pool", tmp_path / "pool", "--out", tmp_path / "o", "--bench", tmp_path / "bench.jsonl")
    assert p.returncode != 0 and "lab/helpsteer2_new" in p.stderr and "Traceback" not in p.stderr


# ------------------------------------------------------------------------------------------------ mask: regex layer
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
def test_regex_mask(text, expected):
    assert pii.regex_mask(text)[0] == expected


def test_wiki_user_names_only_when_the_source_masks_names():
    assert pii.regex_mask("see User:Foobar", names=False)[0] == "see User:Foobar"
    assert pii.regex_mask("see User:Foobar", names=True) == ("see User:[NAME]", {"NAME": 1})
    # the pattern is case-insensitive, so one following word is masked with the user name (over-masks, never under-masks)
    assert pii.regex_mask("see User:Foobar here", names=True)[0] == "see User:[NAME]"


# ------------------------------------------------------------------------------------------------ mask: spans
def test_apply_spans_merges_and_respects_placeholders():
    t = "Dear Jean Dupont, [EMAIL] wrote from 3 Rue X."
    out, cnt = pii.apply_spans(t, [(5, 9, "NAME"), (9, 16, "NAME"), (18, 25, "NAME"), (36, 45, "ADDRESS")])
    assert out == "Dear [NAME], [EMAIL] wrote from [ADDRESS]." and cnt == {"NAME": 1, "ADDRESS": 1}


def test_name_filter():
    f = pii.Filters(0.5, 0.5, 0.7, stop={"claimant"})
    ok = {t: f.name_ok("person", 0.9, t, t, 0, len(t)) for t in ["the Claimant", "Monsieur X", "van der", "[claimant A]", "Jean Dupont"]}
    assert ok == {"the Claimant": False, "Monsieur X": False, "van der": False, "[claimant A]": False, "Jean Dupont": True}
    assert not f.name_ok("person", 0.4, "Jean Dupont", "Jean Dupont", 0, 11)                    # below the threshold
    assert f.name_ok("username", 0.8, "jdupont77", "by jdupont77", 3, 12) and not f.name_ok("username", 0.6, "jdupont77", "by jdupont77", 3, 12)


def test_windows_cover_the_text_and_overlap():
    assert pii.windows("") == [] and pii.windows("   ") == []
    assert pii.windows("  Anna Muster  ") == [(2, 13)]
    t = " ".join(f"w{i}" for i in range(250))
    win = pii.windows(t)
    assert len(win) == 3 and win[0][0] == 0 and win[-1][1] == len(t)
    assert all(len(t[a:b].split()) <= pii.WORDS and b - a <= pii.MAXC for a, b in win)
    assert all(t[b0:b1].split()[:pii.OVERLAP] == t[a0:a1].split()[-pii.OVERLAP:] for (a0, a1), (b0, b1) in zip(win, win[1:]))
    assert all(b - a <= pii.MAXC for a, b in pii.windows("x" * 2000 + " y"))            # one very long word is cut


def test_filters_turn_raw_model_output_into_spans():
    t = "Anna Muster lives at Postbox 44 with the claimant"
    raw = [[0, 0, 11, "P", 0.9], [0, 21, 31, "A", 0.8], [0, 41, 49, "P", 0.95], [0, 12, 17, "A", 0.9], [0, 0, 4, "U", 0.6]]
    assert pii.Filters(stop={"claimant"}).spans(t, raw) == [(0, 11, "NAME"), (21, 31, "ADDRESS")]
    assert pii.Filters(th_person=0.95, th_address=0.85).spans(t, raw) == []
    assert pii.Filters(th_person=0.95, th_user=0.5).spans(t, raw) == [(21, 31, "ADDRESS"), (0, 4, "NAME")]
    assert pii.Filters().spans(t, []) == []


def test_span_cache_survives_an_interrupted_run(tmp_path):
    v = {"model": "m", "precision": "fp32"}
    c = pii.SpanCache(tmp_path, v)
    c.add("k1", [[0, 0, 4, "P", 0.9]])
    c.add("k2", [])
    c.close()
    path = next(tmp_path.glob("spans-*.jsonl"))
    with open(path, "a") as f:
        f.write('{"k": "k3", "e": [[0, 1')                                                  # a torn last line
    assert pii.SpanCache(tmp_path, v).d == {"k1": [[0, 0, 4, "P", 0.9]], "k2": []}
    assert pii.SpanCache(tmp_path, dict(v, precision="fp16")).d == {}                       # another variant: nothing shared
    part = pii.SpanCache(tmp_path, v, part=(1, 2))
    part.add("k4", [])
    part.close()
    assert set(pii.SpanCache(tmp_path, v).d) == {"k1", "k2", "k4"} and len(list(tmp_path.glob("spans-*.jsonl"))) == 2


# ------------------------------------------------------------------------------------------------ mask: rows
MASK_SOURCES = {
    "pub": pub(),
    "scraped": {"origin": "self-collected", "names": True, "config": "default", "license": "CC0-1.0"},
    "records": {"origin": "self-collected", "names": False, "config": "default", "license": "CC0-1.0"},
    "portal": {"drop": "mask", "drop_reason": "portal terms"},
    "gone": {"drop": "clean", "drop_reason": "licence review"},
}
MASK_SECTION = {"keep_emails": ["Registry@court.example.ch"], "keep_email_hosts": ["example.org"]}


class FakeModel:
    """Tags every 'Anna Muster' as a person and every 'Postbox 44' as an address, like pii.Model.raw_spans."""
    calls = []

    def __init__(self, *args, **kw):
        pass

    def raw_spans(self, texts, names, progress=None):
        FakeModel.calls += texts
        out = []
        for t in texts:
            ents = []
            for needle, code in (("Anna Muster", "P"), ("Postbox 44", "A")):
                i = t.find(needle)
                if i >= 0 and (names or code == "A"):
                    ents.append([0, i, i + len(needle), code, 0.9])
            out.append(ents)
        return out

    def close(self):
        pass


def _raw_cache(texts, names=True):
    return {pii.text_key(t, names): raw for t, raw in zip(texts, FakeModel().raw_spans(texts, names))}


@pytest.fixture
def mask_state():
    cfg = {"sources": MASK_SOURCES}
    pii.G.update(keep_emails=frozenset({"registry@court.example.ch"}), keep_hosts=("example.org",))
    return cfg, pii.Filters()


def test_mask_rows(mask_state):
    cfg, filters = mask_state
    rows = [
        row("p/1", source="pub", state="Anna Muster wrote to x@gmail.com"),                          # only the e-mail sweep
        row("p/2", source="pub", state="Anna Muster called 0791234567"),                             # copied as is
        row("s/1", source="scraped", state={"body": "Anna Muster, Tel. 0791234567"}),
        row("s/2", source="scraped", options=("Anna Muster", "Peter Muster")),                        # still distinct
        row("s/3", source="scraped", options=("Anna Muster", "[NAME]")),                              # collapses
        row("s/4", source="scraped", state="ask registry@court.example.ch or a@b.example.org or c@d.ch"),
        row("r/1", source="records", state="Anna Muster moved to Postbox 44"),                        # addresses only
        row("x/1", source="portal"),
    ]
    lines = [engine.dump(r) for r in rows]
    texts = ["[NAME], Tel. [PHONE]", "Anna Muster, Tel. [PHONE]", "Anna Muster", "Peter Muster", "[NAME]", "A short neutral state.",
             "Which one?", "yes", "no", "ask registry@court.example.ch or a@b.example.org or [EMAIL]"]
    cache = dict(_raw_cache(texts), **_raw_cache(["Anna Muster moved to Postbox 44", "Which one?", "yes", "no"], names=False))
    out, stats, dropped, missing = pii.mask_rows(lines, cfg, filters, cache)
    assert not missing and len(out) == 6
    assert json.loads(out[0])["state"] == "Anna Muster wrote to [EMAIL]" and out[1] == lines[1]
    assert json.loads(out[2])["state"] == {"body": "[NAME], Tel. [PHONE]"}
    assert [o["text"] for o in json.loads(out[3])["options"]] == ["[NAME]", "Peter Muster"]
    assert json.loads(out[4])["state"] == "ask registry@court.example.ch or a@b.example.org or [EMAIL]"
    assert json.loads(out[5])["state"] == "Anna Muster moved to [ADDRESS]"
    assert dropped == [("s/3", "scraped", "options_not_distinct_after_masking")]
    assert stats["portal"] == {"source_dropped": 1} and stats["pub"] == {"rows_out": 2, "email_sweep_rows": 1}
    assert stats["scraped"]["rows_changed"] == 3 and stats["scraped"]["mask_NAME"] == 2 and stats["scraped"]["rows_out"] == 3
    assert stats["records"] == {"rows_self_collected": 1, "rows_changed": 1, "mask_ADDRESS": 1, "rows_out": 1}
    # a text the cache does not hold: the rows are not written yet, the text is asked for
    del cache[pii.text_key("Anna Muster", True)]
    out, _, _, missing = pii.mask_rows(lines, cfg, filters, cache)
    assert out is None and missing == {pii.text_key("Anna Muster", True): ("Anna Muster", True)}
    # without the model: the regex layer only
    out, stats, _, missing = pii.mask_rows(lines, cfg, filters, None, no_ner=True)
    assert not missing and json.loads(out[2])["state"] == {"body": "Anna Muster, Tel. [PHONE]"}
    with pytest.raises(engine.PoolError):
        pii.mask_rows([engine.dump(row("g/1", source="gone"))], cfg, filters, {})
    with pytest.raises(engine.PoolError):
        pii.mask_rows([engine.dump(row("u/1", source="unknown"))], cfg, filters, {})


def test_sweep_emails():
    assert pii.sweep_emails({"a": ["mail jane@uni.edu", "Office@Court.ch", "x@sub.example.org"]}, frozenset({"office@court.ch"}),
                            ("example.org",)) == {"a": ["mail [EMAIL]", "Office@Court.ch", "x@sub.example.org"]}


def test_mask_stage_uses_the_model_once_per_text(tmp_path, monkeypatch):
    monkeypatch.setattr(pii, "Model", FakeModel)
    FakeModel.calls = []
    train = [row(f"s/{i}", source="scraped", state=f"Report {i}: Anna Muster of Postbox 44 called +41 79 123 45 6{i}") for i in range(5)]
    train += [row("p/1", source="pub", state="jane@uni.edu"), row("x/1", source="portal")]
    pool = _write_pool(tmp_path / "pool", train, [row("v/1", source="scraped", state="Anna Muster again")])
    cfg = engine.load_config(_config(tmp_path / "pool.json", MASK_SOURCES, mask=MASK_SECTION))
    rep = pii.run(cfg, pool, tmp_path / "out", tmp_path / "cache", workers=1)
    got = _read_jsonl_gz(tmp_path / "out" / "train.jsonl.gz")
    assert [r["id"] for r in got] == [f"s/{i}" for i in range(5)] + ["p/1"]
    assert got[0]["state"] == "Report 0: [NAME] of [ADDRESS] called [PHONE]" and got[5]["state"] == "[EMAIL]"
    assert _read_jsonl_gz(tmp_path / "out" / "val.jsonl.gz")[0]["state"] == "[NAME] again"
    assert rep["train"]["totals"] == {"rows_self_collected": 5, "rows_changed": 5, "mask_PHONE": 5, "mask_NAME": 5, "mask_ADDRESS": 5,
                                      "rows_out": 6, "email_sweep_rows": 1, "source_dropped": 1}
    assert len(FakeModel.calls) == len(set(FakeModel.calls)) == 7 and rep["ner"]["texts_missing_from_cache"] == 7     # "yes" and "no" are too short
    assert not (tmp_path / "out" / ".parts").exists()
    first = (tmp_path / "out" / "train.jsonl.gz").read_bytes()

    def boom(*a, **k):
        raise AssertionError("the model must not be loaded when every text is cached")

    # the same rows, a stricter threshold, another worker count: no model, and only the filters change the output
    monkeypatch.setattr(pii, "Model", boom)
    assert pii.run(cfg, pool, tmp_path / "out2", tmp_path / "cache", workers=2)["ner"]["texts_missing_from_cache"] == 0
    assert (tmp_path / "out2" / "train.jsonl.gz").read_bytes() == first
    cfg["mask"]["thresholds"] = {"person": 0.95, "address": 0.5, "username": 0.95}
    pii.run(cfg, pool, tmp_path / "out3", tmp_path / "cache", workers=1)
    assert _read_jsonl_gz(tmp_path / "out3" / "train.jsonl.gz")[0]["state"] == "Report 0: Anna Muster of [ADDRESS] called [PHONE]"
    # one new row: only its text goes through the model
    cfg["mask"].pop("thresholds")
    monkeypatch.setattr(pii, "Model", FakeModel)
    FakeModel.calls = []
    _write_pool(pool, train + [row("s/9", source="scraped", state="Anna Muster, a new report")], [])
    pii.run(cfg, pool, tmp_path / "out4", tmp_path / "cache", workers=1)
    assert FakeModel.calls == ["Anna Muster, a new report"]
    # the stage is idempotent: its own output passes through unchanged
    pii.run(cfg, tmp_path / "out", tmp_path / "again", tmp_path / "cache", workers=1)
    assert gzip.open(tmp_path / "again" / "train.jsonl.gz").read() == gzip.open(tmp_path / "out" / "train.jsonl.gz").read()


def test_mask_stage_in_parts(tmp_path, monkeypatch):
    monkeypatch.setattr(pii, "Model", FakeModel)
    FakeModel.calls = []
    pool = _write_pool(tmp_path / "pool", [row(f"s/{i}", source="scraped", state=f"Anna Muster, report number {i}") for i in range(8)])
    cfg = engine.load_config(_config(tmp_path / "pool.json", MASK_SOURCES))
    for k in range(2):
        assert pii.run(cfg, pool, tmp_path / "out", tmp_path / "cache", workers=1, part=(k, 2)) is None
    assert sorted(FakeModel.calls) == sorted(set(FakeModel.calls)) and len(FakeModel.calls) == 9        # 8 states and the question
    assert not (tmp_path / "out" / "train.jsonl.gz").exists()
    FakeModel.calls = []
    pii.run(cfg, pool, tmp_path / "out", tmp_path / "cache", workers=1)
    assert FakeModel.calls == [] and _read_jsonl_gz(tmp_path / "out" / "train.jsonl.gz")[3]["state"] == "[NAME], report number 3"


def test_mask_cli_without_the_model(tmp_path):
    pool = _write_pool(tmp_path / "pool", [row(f"s/{i}", source="scraped", state=f"Call +41 79 123 45 6{i} please") for i in range(5)]
                       + [row("p/1", source="pub")], [row("v/1", source="pub")])
    cfg = _config(tmp_path / "pool.json", MASK_SOURCES)
    _ok("mask", "--config", cfg, "--pool", pool, "--out", tmp_path / "out", "--no-ner", "--workers", 2)
    assert {r["state"] for r in _read_jsonl_gz(tmp_path / "out" / "train.jsonl.gz")[:5]} == {"Call [PHONE] please"}
    rep = json.load(open(tmp_path / "out" / "pii_report.json"))
    assert rep["model"] is None and rep["train"]["totals"]["mask_PHONE"] == 5 and rep["val"]["rows_in"] == 1
    assert not (tmp_path / "ner_cache").exists()


# ------------------------------------------------------------------------------------------------ export
EXPORT_SOURCES = {
    "pub": pub(upstream="u"),
    "sa": {"origin": "self-collected", "names": True, "config": "share_alike", "license": "CC-BY-SA-4.0", "description": "what it is"},
    "fixme": {"origin": "self-collected", "names": False, "config": "default", "license": "CC0-1.0",
              "license_fix": [["Thurgau", "CC0-1.0"], ["", "unknown"]]},
}
EXPORT_SECTION = {"configs": ["default", "share_alike"], "license_guard": {"default": ["share-alike", "cc-by-sa"]}}


def _export_inputs(tmp_path, sources=EXPORT_SOURCES):
    d = _write_pool(tmp_path / "pool", [row("a/1", source="pub", n_tok=12), row("s/1", source="sa", license="CC-BY-SA-4.0", state={"k": 1, "b": "x"}),
                                        row("f/1", source="fixme", license="MIT", license_detail="Kanton Thurgau")], [row("a/2", source="pub")])
    return d, _config(tmp_path / "pool.json", sources, export=EXPORT_SECTION)


def test_export(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    d, cfg = _export_inputs(tmp_path)
    out = tmp_path / "hf"
    _ok("export", "--config", cfg, "--pool", d, "--out", out)
    default = pq.read_table(out / "data/default/train-00000-of-00001.parquet").to_pylist()
    assert [r["id"] for r in default] == ["a/1", "f/1"] and default[1]["license"] == "CC0-1.0"
    assert "n_tok" not in default[0]
    sa = pq.read_table(out / "data/share_alike/train-00000-of-00001.parquet").to_pylist()
    assert sa[0]["state"] == '{"b": "x", "k": 1}'
    assert pq.read_table(out / "data/default/validation-00000-of-00001.parquet").num_rows == 1
    assert not list((out / "data/share_alike").glob("validation-*"))
    assert (out / "SOURCES.csv").read_text().splitlines()[1:] == [
        "fixme,default,self-collected,CC0-1.0,1,0,no,,", "pub,default,redistributed,CC-BY-4.0,1,1,no,u,",
        "sa,share_alike,self-collected,CC-BY-SA-4.0,1,0,yes,,what it is"]
    stats = json.load(open(d / "export_stats.json"))
    assert stats["dropped_fields"] == {"n_tok": 1} and stats["rows"] == {"default/train": 2, "share_alike/train": 1, "default/validation": 1}


def test_export_flags_and_shards(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    d, cfg = _export_inputs(tmp_path)
    ids = tmp_path / "ids.txt.gz"
    with gzip.open(ids, "wt") as f:
        f.write("a/1\ns/1\n")
    out = tmp_path / "hf"
    _ok("export", "--config", cfg, "--pool", d, "--out", out, "--flag", f"in_votum_8b={ids}")
    default = pq.read_table(out / "data/default/train-00000-of-00001.parquet").to_pylist()
    assert [(r["id"], r["in_votum_8b"]) for r in default] == [("a/1", True), ("f/1", False)]
    assert pq.read_table(out / "data/default/validation-00000-of-00001.parquet").to_pylist()[0]["in_votum_8b"] is False
    assert json.load(open(d / "export_stats.json"))["in_votum_8b"]["default/train"] == 1
    # many rows: shards of the asked size, the same files whatever the number of workers
    big = _write_pool(tmp_path / "big", [row(f"a/{i}", source="pub", state="word " * 400 + str(i)) for i in range(12000)])
    for name, workers in (("w1", 1), ("w3", 3)):
        _ok("export", "--config", cfg, "--pool", big, "--out", tmp_path / name, "--shard-mb", 6, "--workers", workers)
    files = sorted(p.name for p in (tmp_path / "w1" / "data/default").iterdir())
    assert files == ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet"]
    assert all((tmp_path / "w1" / "data/default" / f).read_bytes() == (tmp_path / "w3" / "data/default" / f).read_bytes() for f in files)
    t = pq.ParquetFile(tmp_path / "w1" / "data/default" / files[0])
    assert t.metadata.row_group(0).num_rows == 2000
    assert [r["id"] for f in files for r in pq.read_table(tmp_path / "w1" / "data/default" / f, columns=["id"]).to_pylist()] == [f"a/{i}" for i in range(12000)]


@pytest.mark.parametrize("change, message", [
    ({"sa": dict(EXPORT_SOURCES["sa"], config="default")}, "not allowed in the config `default`"),
    ({"pub": {"drop": "mask", "drop_reason": "terms"}}, "dropped by the config"),
    ({"sa": dict(EXPORT_SOURCES["sa"], config="extra")}, "export.configs"),
])
def test_export_guards(tmp_path, change, message):
    pytest.importorskip("pyarrow")
    d, cfg = _export_inputs(tmp_path, dict(EXPORT_SOURCES, **change))
    p = _pool("export", "--config", cfg, "--pool", d, "--out", tmp_path / "hf")
    assert p.returncode != 0 and message in p.stderr and "Traceback" not in p.stderr


# ------------------------------------------------------------------------------------------------ audit
def test_audit_sample_regex_report(tmp_path, capsys):
    sc = MASK_SOURCES["scraped"]
    cfg = _config(tmp_path / "pool.json", {"sc": sc, "sc2": dict(sc, config="share_alike"), "pub": pub()}, export=EXPORT_SECTION)
    d = _write_pool(tmp_path / "pool", [row("m/1", source="sc", state="reach me on +41 79 123 45 67"), row("m/2", source="sc2"),
                                        row("p/1", source="pub", state="+41 79 123 45 67")])
    au = tmp_path / "audit"
    _ok("audit", "sample", "--config", cfg, "--audit-dir", au, "--pool", d, "--n", 2)
    assert sorted(json.loads(line)["id"] for line in open(au / "sample.jsonl")) == ["m/1", "m/2"]
    _ok("audit", "regex", "--config", cfg, "--audit-dir", au)
    hits = json.load(open(au / "hits_regex.json"))
    assert {(h[0], h[1]) for h in hits} == {("m/1", "phone"), ("m/1", "phone (loose)")}
    rep = _ok("audit", "report", "--config", cfg, "--audit-dir", au)
    assert "default {'phone / regex': 2, 'phone (loose) / regex': 1}" in rep     # two of the stage's phone patterns match
    # the sample depends on the pool, the seed and n only
    many = _write_pool(tmp_path / "many", [row(f"m/{i}", source="sc" if i % 2 else "sc2") for i in range(60)], [row("v/1", source="sc")])
    picks = []
    for workers in (1, 3):
        audit.sample(engine.load_config(cfg), many, tmp_path / f"au{workers}", n=10, seed=7, workers=workers)
        picks.append([json.loads(line)["id"] for line in open(tmp_path / f"au{workers}" / "sample.jsonl")])
    capsys.readouterr()
    assert picks[0] == picks[1] and len(set(picks[0])) == 10


# ------------------------------------------------------------------------------------------------ the whole pipeline
def test_run_all_stages(tmp_path):
    pytest.importorskip("numpy")
    pq = pytest.importorskip("pyarrow.parquet")
    sources = dict(CLEAN_SOURCES, scraped=MASK_SOURCES["scraped"], portal=MASK_SOURCES["portal"])
    _pool_and_bench(tmp_path, extra_train=[row("s/1", source="scraped", state="Tel. 0791234567"), row("x/1", source="portal")])
    cfg = _config(tmp_path / "pool.json", sources, clean=dict(CLEAN_SECTION, benchmarks=["bench.jsonl"]), export=EXPORT_SECTION)
    out = _ok("run", "--config", cfg, "--pool", tmp_path / "pool", "--work", tmp_path / "work", "--no-ner", "--workers", 2)
    assert [line for line in out.splitlines() if line.startswith("==")] == ["== clean", "== mask", "== export", "== audit"]
    rows = pq.read_table(tmp_path / "work/dataset/data/default/train-00000-of-00001.parquet").to_pylist()
    ids = [r["id"] for r in rows]
    assert "s/1" in ids and not {"x/1", "r1/1", "r2/1", "r3/state"} & set(ids)
    assert rows[ids.index("s/1")]["state"] == "Tel. [PHONE]"
    assert (tmp_path / "work/audit/sample.jsonl").exists()
    # start again at a later stage
    assert "== clean" not in _ok("run", "--config", cfg, "--pool", tmp_path / "pool", "--work", tmp_path / "work", "--no-ner", "--from", "export")
