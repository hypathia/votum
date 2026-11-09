"""votum_config.json: everything besides the weights that a Votum checkpoint needs (layout, mask, readout, temperatures)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

CONFIG_NAME = "votum_config.json"
FORMAT = "votum/1"


@dataclass
class VotumConfig:
    base_model: str = "swiss-ai/Apertus-v1.5-8B"
    base_revision: str = "ef4bbc54d6fe658bc782280ff9f3b53a10a9758f"
    opt_end_token: str = "<|fim_end|>"
    opt_end_id: int = 60
    bidirectional_layers: list = field(default_factory=lambda: list(range(3, 32, 4)))
    pointer_rank: int = 256
    shared_positions: bool = True        # every option block starts at the same position id
    close_last: bool = True              # the prompt's final line break belongs to the last option block
    # header: up to max_q questions of a request listed before the state; choice options listed (all if max_opts is None),
    # sorted by text; chunk: split requests of more than max_q questions into separate headers
    header: dict = field(default_factory=lambda: {"max_q": 8, "max_opts": None, "sort": True, "chunk": False})
    temperatures: dict = field(default_factory=lambda: {"choice": 1.5164094828380006, "noul": 1.3096884485875626,
                                                         "score": 1.1819906765646513})
    max_tokens: int = 8192
    format: str = FORMAT

    @classmethod
    def load(cls, path: str | Path) -> "VotumConfig":
        p = Path(path); p = p / CONFIG_NAME if p.is_dir() else p
        d = json.loads(p.read_text())
        if d.get("format") != FORMAT: raise ValueError(f"{p}: unsupported format {d.get('format')!r} (expected {FORMAT})")
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})

    def save(self, path: str | Path) -> None:
        p = Path(path); p = p / CONFIG_NAME if p.is_dir() else p
        p.write_text(json.dumps(asdict(self), indent=2) + "\n")
