"""FP8: the conversion (votum quantize), the PyTorch layer that reads it, and the server's memory default (CPU)."""
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")


def tiny_checkpoint(d: Path) -> dict:
    from safetensors.torch import save_file
    g = torch.Generator().manual_seed(0)
    t = {f"model.layers.{i}.{p}.weight": torch.randn(6, 8, generator=g).to(torch.bfloat16)
         for i in range(2) for p in ("self_attn.q_proj", "self_attn.o_proj", "mlp.up_proj")}
    t["model.embed_tokens.weight"] = torch.randn(10, 8, generator=g).to(torch.bfloat16)
    t["model.layers.0.attention_layernorm.weight"] = torch.ones(8, dtype=torch.bfloat16)
    t["pointer.o.weight"] = torch.randn(4, 8, generator=g); t["pointer.q0"] = torch.randn(4, generator=g)
    d.mkdir(); save_file(t, str(d / "model-00001-of-00001.safetensors"))
    (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "model-00001-of-00001.safetensors" for k in t}}))
    (d / "config.json").write_text(json.dumps({"architectures": ["VotumDecisionModel"], "decision_opt_end_id": 60}))
    (d / "votum_config.json").write_text("{}")
    return t


def test_quantize_keeps_everything_but_the_projections(tmp_path):
    from safetensors.torch import load_file
    from votum import fp8
    src = tiny_checkpoint(tmp_path / "Tiny"); out = fp8.quantize(str(tmp_path / "Tiny"), tmp_path / "Tiny-FP8")
    got = load_file(str(out / "model-00001-of-00001.safetensors"))
    for k, w in src.items():
        if ".layers." in k and "proj" in k:
            assert got[k].dtype == torch.float8_e4m3fn and got[k[:-6] + "weight_scale"].shape == (6, 1)
        else:
            assert got[k].dtype == w.dtype and torch.equal(got[k], w)
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["quantization_config"]["votum_fp8"]["source"] == "Tiny" and cfg["decision_opt_end_id"] == 60
    assert fp8.is_quantized(out) and not fp8.is_quantized(tmp_path / "Tiny")
    assert json.loads((out / "model.safetensors.index.json").read_text())["weight_map"].keys() == got.keys()
    with pytest.raises(SystemExit): fp8.quantize(str(out), tmp_path / "again")        # already quantized


def test_quantize_can_keep_layers_and_projections(tmp_path):
    from safetensors.torch import load_file
    from votum import fp8
    tiny_checkpoint(tmp_path / "Tiny"); out = fp8.quantize(str(tmp_path / "Tiny"), tmp_path / "q", ignore_layers={0}, ignore_proj={"up_proj"})
    got = load_file(str(out / "model-00001-of-00001.safetensors"))
    fp8_keys = sorted(k for k, v in got.items() if v.dtype == torch.float8_e4m3fn)
    assert fp8_keys == ["model.layers.1.self_attn.o_proj.weight", "model.layers.1.self_attn.q_proj.weight"]


def test_fp8_linear_matches_the_full_precision_layer(tmp_path):
    from safetensors.torch import load_file
    from votum import fp8
    src = tiny_checkpoint(tmp_path / "Tiny"); out = fp8.quantize(str(tmp_path / "Tiny"), tmp_path / "q")
    got = load_file(str(out / "model-00001-of-00001.safetensors")); k = "model.layers.1.mlp.up_proj.weight"
    lin = fp8.FP8Linear(got[k], got[k[:-6] + "weight_scale"]); x = torch.randn(3, 8)
    ref = x @ src[k].float().T; y = lin(x)
    assert y.dtype == x.dtype and y.shape == ref.shape
    assert (y - ref).abs().max() < 0.07 * ref.abs().max()            # e4m3 keeps 3 mantissa bits: a few percent per weight
    assert sum(b.numel() * b.element_size() for b in lin.buffers()) == 6 * 8 + 6 * 4      # one byte per weight, one fp32 scale per row


def test_cli_lists_quantize(capsys):
    from votum import cli
    cli.main(["--help"]); assert "votum quantize" in capsys.readouterr().out
    with pytest.raises(SystemExit): cli.main(["quantize"])          # --model and --out are required


def test_memory_share_follows_the_weights(tmp_path, monkeypatch):
    from votum.vllm import client
    G = 2**30; d = tmp_path / "m"; d.mkdir(); (d / "config.json").write_text("{}")
    with open(d / "model-00001-of-00001.safetensors", "wb") as f: f.truncate(9 * G)       # sparse: 9 GiB of weights
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (22 * G, 24 * G))
    share, gib = client.memory_share(d); assert gib == 12 and share == pytest.approx(0.5)
    share, gib = client.memory_share(d, gpu_memory_gb=15); assert gib == 15 and share == pytest.approx(15 / 24)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (10 * G, 16 * G))
    with pytest.raises(SystemExit, match="needs about 12 GiB"): client.memory_share(d)
    with pytest.raises(SystemExit, match="less than the weights"): client.memory_share(d, gpu_memory_gb=8)


def test_server_names_the_model_after_its_checkpoint():
    from votum.vllm.server import model_name
    assert model_name("org/Votum-8B-FP8") == "votum-8b-fp8" and model_name("/data/Votum-8B/") == "votum-8b"
