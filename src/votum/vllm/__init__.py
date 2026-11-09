"""Votum on vLLM 0.33: pooling model with the Votum attention layout (model.py, isolate.py), offline client (client.py) and
HTTP server (server.py). The model registers itself through the vLLM plugin entry point `vllm.general_plugins`
(pyproject.toml)."""


def register() -> None:
    from .client import register as _register
    _register()


def __getattr__(name):
    if name == "VotumVLLM":
        from .client import VotumVLLM
        return VotumVLLM
    raise AttributeError(name)
