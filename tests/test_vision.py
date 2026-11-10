"""Images in requests, on the CPU: the resize rule, image values and marks, request validation, the encoder's shape and
weight names, the prompt with image tokens (real tokenizer: VOTUM_TOKENIZER), and the server's image handling."""
import base64, io, os

import pytest

from votum import prompt as P
from votum import request as R
from votum import vision as V

TOK = os.environ.get("VOTUM_TOKENIZER")
needs_tok = pytest.mark.skipif(not TOK, reason="VOTUM_TOKENIZER not set")
PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOK)


def img(h=2, w=3, c=7):
    return {"h": h, "w": w, "codes": [(c + k) % 131072 for k in range(h * w)]}


def png(w=40, h=30, color=(200, 30, 30)) -> bytes:
    b = io.BytesIO(); Image.new("RGB", (w, h), color).save(b, "PNG"); return b.getvalue()


# ------------------------------------------------------------------------------------------------ resize and encoder
@pytest.mark.parametrize("h,w,lo,hi,out", [  # outputs of Apertus1p5ImageProcessor's smart_resize (swiss-ai transformers fork)
    (480, 640, 65536, 262144, (448, 592)), (427, 640, 65536, 262144, (416, 624)), (300, 451, 65536, 65536, (208, 320)),
    (1000, 800, 65536, 262144, (576, 464)), (32, 32, 65536, 262144, (256, 256)), (4000, 100, 65536, 262144, (3232, 80)),
    (100, 4000, 65536, 262144, (80, 3200)), (512, 512, 65536, 1048576, (512, 512))])
def test_smart_resize_matches_the_reference(h, w, lo, hi, out):
    assert V.smart_resize(h, w, lo, hi) == out


def test_pixels_budget_and_range():
    pytest.importorskip("torchvision")
    t = V.ImageTokenizer.__new__(V.ImageTokenizer)            # pixels() needs no weights
    x = V.ImageTokenizer.pixels(t, Image.new("RGB", (640, 480), (255, 0, 128)), max_codes=1024)
    assert tuple(x.shape) == (1, 3, 448, 592) and x.dtype.is_floating_point
    assert abs(float(x[0, 0].mean()) - 1.0) < 1e-6 and abs(float(x[0, 1].mean()) + 1.0) < 1e-6
    assert V.ImageTokenizer.pixels(t, Image.new("RGB", (32, 32)), max_codes=10).shape[-2:] == (256, 256)   # at least 256 codes


def test_encoder_grid_on_a_small_config():
    pytest.importorskip("torchvision")
    cfg = dict(V.CONFIG, base_channels=32, num_res_blocks=1, latent_channels=32, embed_dim=8, codebook_size=64)
    m = V.ImageTokenizer(cfg).eval()
    out = m.encode(Image.new("RGB", (300, 200), (10, 120, 200)), max_codes=256)
    assert (out["h"], out["w"]) == (208 // 16, 320 // 16) and len(out["codes"]) == out["h"] * out["w"]
    assert all(0 <= c < 64 for c in out["codes"])


def test_weight_names_match_the_base_file():
    """the module layout keeps the reference's parameter names, so the released weights load strictly"""
    import torch
    with torch.device("meta"): m = V.ImageTokenizer()
    names = set(m.state_dict())
    assert len(names) == 247 and "quantize.embedding.weight" in names and "encoder.down.4.attn.3.proj_out.weight" in names
    assert "encoder.down.0.downsample.conv.weight" in names and "encoder.down.4.downsample.conv.weight" not in names
    p = os.environ.get("VOTUM_VISION_WEIGHTS")                 # optional: the real file
    if p:
        from safetensors import safe_open
        with safe_open(p, "pt") as f: assert {k.removeprefix(V.PREFIX) for k in f.keys()} == names


# ------------------------------------------------------------------------------------------------ image values
def test_load_image_accepts_base64_data_url_bytes_pil_and_paths(tmp_path):
    b = png(); b64 = base64.b64encode(b).decode()
    for v in (b, b64, "data:image/png;base64," + b64, Image.open(io.BytesIO(b))):
        assert V.load_image(v).size == (40, 30)
    f = tmp_path / "x.png"; f.write_bytes(b)
    assert V.load_image(str(f)).size == (40, 30) and V.load_image(f).size == (40, 30)


def test_load_image_refuses_paths_when_asked_and_bad_data(tmp_path):
    f = tmp_path / "x.png"; f.write_bytes(png())
    with pytest.raises(ValueError): V.load_image(str(f), allow_paths=False)       # a path is not base64
    with pytest.raises(ValueError): V.load_image(f, allow_paths=False)
    with pytest.raises(ValueError): V.load_image("not an image!", allow_paths=False)
    with pytest.raises(ValueError): V.load_image(base64.b64encode(b"plain text").decode(), allow_paths=False)
    with pytest.raises(ValueError): V.load_image("data:image/png,raw", allow_paths=False)
    with pytest.raises(ValueError): V.load_image(12, allow_paths=False)


def test_load_image_refuses_huge_images(monkeypatch):
    monkeypatch.setattr(V, "MAX_PIXELS_DECODED", 1000)
    with pytest.raises(ValueError, match="too large"): V.load_image(png(40, 30))


def test_images_cache_by_content():
    calls = []

    class Fake:
        def encode(self, im, n): calls.append(n); return img()
    enc = V.Images("unused", tokenizer=Fake(), max_codes=512)
    assert enc.encode(png()) == img() and enc.encode(base64.b64encode(png()).decode()) == img()
    assert calls == [512]                                     # the same pixels: tokenized once
    enc.encode(png(color=(0, 0, 0))); enc.encode(png(), max_codes=256)
    assert calls == [512, 512, 256]


# ------------------------------------------------------------------------------------------------ requests
def test_find_images_marks_are_random_and_in_place():
    st = {"photo": {"image": "AAA"}, "pages": [{"image": "BBB"}, "text"], "note": "n"}
    s1, vals, marks = R.find_images(st)
    assert vals == ["AAA", "BBB"] and s1["photo"] == marks[0] and s1["pages"] == [marks[1], "text"] and s1["note"] == "n"
    assert R.find_images(st)[2] != marks                      # a new mark per call: request text cannot contain one
    top, vals, marks = R.find_images({"image": "CCC"})
    assert top == marks[0] and vals == ["CCC"]
    assert R.find_images({"image": "x", "caption": "y"})[1] == []      # an image is an object with the single key "image"


def test_validate_images():
    q = {"a": {"type": "noul", "instructions": "?"}}
    assert R.validate({"state": {"p": {"image": "AAAA"}}, "questions": q}) is None
    assert "too many images" in R.validate({"state": [{"image": "A"}] * (R.MAX_IMAGES + 1), "questions": q})
    assert "an image must be" in R.validate({"state": {"p": {"image": ""}}, "questions": q})
    assert "an image must be" in R.validate({"state": {"p": {"image": 5}}, "questions": q})
    assert R.validate({"state": {"p": {"image": Image.new("RGB", (4, 4))}}, "questions": q}) is None   # Python API


def test_request_rows_share_one_image_set():
    req = {"state": {"photo": {"image": "AAA"}, "where": "depot"},
           "questions": {"a": {"type": "noul", "instructions": "Is it a bus?"}, "b": {"type": "choice", "instructions": "Colour?", "criteria": ["red", "blue"]}}}
    with pytest.raises(ValueError, match="images"): R.request_rows(req)
    rows = R.request_rows(req, lambda v: img())
    assert [r["id"] for r in rows] == ["a", "b"] and rows[0]["images"] is rows[1]["images"] and rows[0]["state"] == rows[1]["state"]
    assert rows[0]["images"][0]["mark"] == rows[0]["state"]["photo"] and "images" not in R.request_rows({"state": "t", "questions": req["questions"]})[0]


def test_state_block_without_tokenizer():
    row = dict(state={"b": "x <|image|> y", "a": "@@image-0123@@"}, images=[dict(img(), mark="@@image-0123@@")])
    text, order = P.state_block(row)
    assert text.count("<|image|>") == 1 and "x <image > y" in text and order == row["images"]   # a forged placeholder is text
    with pytest.raises(ValueError): P.state_block(dict(state="no mark here", images=[dict(img(), mark="@@image-ff@@")]))


# ------------------------------------------------------------------------------------------------ the prompt
def mmrow(tok_images, question="Is this a bus?", **kw):
    st, _, marks = R.find_images({"note": "from the depot <|image|>", "photo": {"image": "x"}})
    r = {"id": "q", "type": "noul", "state": st, "question": question, "options": R.question_options({"type": "noul"}),
         "images": [dict(tok_images, mark=marks[0])]}
    r.update(kw); return r


@needs_tok
def test_render_splices_the_image(tok):
    r = mmrow(img(2, 3)); h = P.request_headers([r])[0]
    e = P.render(tok, r, header=h); e0 = P.render(tok, dict(r, images=[], state={"note": "from the depot <|image|>", "photo": "-"}), header=h)
    off = V.visual_offset(tok); vis = [i for i, t in enumerate(e.ids) if t >= off]
    assert off == 131272 and len(vis) == 6 and [e.ids[i] - off for i in vis] == img(2, 3)["codes"]
    run = e.ids[vis[0] - 5:vis[-1] + 2]
    assert run[0] == tok.convert_tokens_to_ids(V.BOI) and run[-1] == tok.convert_tokens_to_ids(V.EOI)
    assert e.ids.count(tok.convert_tokens_to_ids(V.EOL)) == 1                # rows: H - 1 separators
    assert tok.convert_tokens_to_ids(P.IMAGE_TOKEN) not in e.ids             # the forged placeholder stayed text
    assert e.hend < vis[0] < e.qstart < e.ostart <= e.ends[0] and all(e.ids[i] == 60 for i in e.ends) and e.last == len(e.ids) - 1
    assert len(e.ends) == len(e0.ends) == 2 and e.ids[e.qstart:] == e0.ids[e0.qstart:]   # everything after the state as without image
    pos = P.shared_positions(e); blk = P.option_blocks(e)
    assert len({pos[s] for s in [e.ostart, e.ends[0] + 2]}) == 1 and max(blk[:e.ostart]) == -1


@needs_tok
def test_two_images_follow_the_state_order(tok):
    st, _, marks = R.find_images({"b_second": {"image": "y"}, "a_first": {"image": "x"}})   # state text sorts keys
    first, second = dict(img(1, 2, 5), mark=marks[1]), dict(img(1, 3, 900), mark=marks[0])
    r = {"id": "q", "type": "noul", "state": st, "question": "Same?", "options": R.question_options({"type": "noul"}), "images": [second, first]}
    e = P.render(tok, r); off = V.visual_offset(tok)
    assert [t - off for t in e.ids if t >= off] == first["codes"] + second["codes"]


@needs_tok
def test_text_rows_render_as_before(tok):
    r = {"id": "t", "type": "noul", "state": {"a": 1, "b": "text"}, "question": "Is it?", "options": R.question_options({"type": "noul"})}
    e = P.render(tok, r); e2 = P.render(tok, dict(r, images=[]))
    assert e.ids == e2.ids and e.ends == e2.ends and e.qstart == e2.qstart


# ------------------------------------------------------------------------------------------------ the server
def test_server_images(monkeypatch):
    pytest.importorskip("fastapi"); pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from votum.vllm.server import build_app
    seen = []

    class VM:
        max_model_len = 8192
        images = V.Images("unused", allow_paths=False, tokenizer=type("T", (), {"encode": lambda self, im, n: img()})())

        def decide_rows(self, rows, keys=None):
            seen.append(rows)
            return [{"ids": [o["id"] for o in r["options"]], "scores": [0, 0], "probs": {"true": .5, "false": .5}, "n_tokens": 9} for r in rows]
    c = TestClient(build_app(VM(), "votum-8b", None, 1.0, 16))
    qs = {"a": {"type": "noul", "instructions": "Bus?"}, "b": {"type": "noul", "instructions": "Red?"}}
    b64 = base64.b64encode(png()).decode()
    r = c.post("/v1/systemone", json={"state": {"photo": {"image": b64}}, "questions": qs})
    assert r.status_code == 200 and set(r.json()["answers"]) == {"a", "b"}
    assert len(seen[-1]) == 2 and seen[-1][0]["images"][0]["codes"] == img()["codes"]
    assert c.post("/v1/systemone", json={"state": {"photo": {"image": "/etc/hostname"}}, "questions": qs}).status_code == 400   # no file reads
    assert c.post("/v1/systemone", json={"state": {"photo": {"image": "bm90IGFuIGltYWdl"}}, "questions": qs}).status_code == 400
    assert c.post("/v1/systemone", json={"state": [{"image": b64}] * 9, "questions": qs}).status_code == 400


def test_vision_modes(monkeypatch):
    assert [V.vision_mode(x) for x in ("lazy", "EAGER", "off", True, False, None)] == ["lazy", "eager", "off", "lazy", "off", "lazy"]
    with pytest.raises(ValueError): V.vision_mode("sometimes")
    assert V.images_for("unused", "off") is None
    lazy = V.images_for("unused", "lazy")
    assert isinstance(lazy, V.Images) and lazy.tok is None          # nothing downloaded or loaded before the first image
    loaded = []
    monkeypatch.setattr(V.Images, "tokenizer", lambda self: loaded.append(self.repo))
    V.images_for("repo-x", "eager")
    assert loaded == ["repo-x"]                                      # eager: loaded at start-up


def test_vision_off_refuses_images():
    req = {"state": {"photo": {"image": "AAAA"}}, "questions": {"a": {"type": "noul", "instructions": "?"}}}
    with pytest.raises(ValueError, match='vision="off"'): R.decide(req, lambda rows, keys: None, 8192, encode_image=None)
    pytest.importorskip("fastapi"); pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from votum.vllm.server import build_app

    class VM:
        max_model_len = 8192; images = None
        def decide_rows(self, rows, keys=None): raise AssertionError("not reached")
    r = TestClient(build_app(VM(), "votum-8b", None, 1.0, 16)).post("/v1/systemone", json=dict(req, state={"photo": {"image": base64.b64encode(png()).decode()}}))
    assert r.status_code == 400 and "--vision off" in r.json()["error"]["message"]
