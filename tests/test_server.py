"""HTTP server checks on the CPU with a stand-in model: limits, authentication, error isolation."""
import pytest

pytest.importorskip("fastapi"); pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from votum.vllm.server import build_app


class FakeVM:
    max_model_len = 8192

    def __init__(self): self.fail = False

    def decide_rows(self, rows, keys=None):
        if self.fail: raise RuntimeError("secret engine detail")
        out = []
        for r in rows:
            ids = [str(o["id"]) for o in r["options"]]
            out.append(None if "TOO LONG" in str(r["state"]) else {"ids": ids, "scores": [0.0] * len(ids), "probs": {i: 1 / len(ids) for i in ids}, "n_tokens": 7})
        return out


def req(state="s", **q):
    return {"state": state, "questions": {"a": {"type": "choice", "instructions": "Which?", "criteria": ["x", "y"], **q}}}


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("VOTUM_MAX_BODY_MB", "1"); monkeypatch.setenv("VOTUM_MAX_QUESTIONS", "3"); monkeypatch.setenv("VOTUM_MAX_OPTIONS", "4")
    vm = FakeVM(); c = TestClient(build_app(vm, "votum-8b", "k3y", 1.0, 16)); c.vm = vm
    c.headers.update({"authorization": "Bearer k3y"}); return c


def test_ok_and_auth(client):
    r = client.post("/v1/systemone", json=req()); assert r.status_code == 200 and r.json()["answers"]["a"]["probabilities"] == {"x": 0.5, "y": 0.5}
    assert client.post("/v1/systemone", json=req(), headers={"authorization": "Bearer nope"}).status_code == 401
    assert client.post("/v1/systemone", json=req(), headers={"authorization": ""}).status_code == 401


def test_limits(client):
    assert client.post("/v1/systemone", content=b"{" + b" " * (1024 * 1024 + 1)).status_code == 413          # body
    many = {"state": "s", "questions": {str(i): {"type": "noul", "instructions": "?"} for i in range(4)}}
    assert client.post("/v1/systemone", json=many).status_code == 400                                         # questions
    assert client.post("/v1/systemone", json=req(criteria=list("abcde"))).status_code == 400                  # options
    assert client.post("/v1/systemone", json=req(state="TOO LONG")).status_code == 413                        # prompt length


@pytest.mark.parametrize("crit", ["abc", 5, ["a"], ["a", "a"], [1, "1"], [["a"], "b"], {"only": "one"}])
def test_bad_choice_criteria_are_400(client, crit):
    assert client.post("/v1/systemone", json=req(criteria=crit)).status_code == 400


def test_bad_shapes_are_400(client):
    assert client.post("/v1/systemone", json=req(type="noul", criteria=["a"])).status_code == 400
    assert client.post("/v1/systemone", json=req(type="score", criteria={"a": "b"})).status_code == 400
    assert client.post("/v1/systemone", json=req(type="score", criteria=["only"])).status_code == 400
    assert client.post("/v1/systemone", content=b'{"state": "\\ud800", "questions": {"a": {"type": "noul", "instructions": "?"}}}').status_code == 400
    assert client.post("/v1/systemone", content=b"not json").status_code == 400


def test_engine_errors_are_not_echoed(client):
    client.vm.fail = True
    r = client.post("/v1/systemone", json=req()); assert r.status_code == 500 and "secret" not in r.text
