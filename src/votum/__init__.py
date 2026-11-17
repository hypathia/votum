"""Votum: single-pass decision model on Apertus-1.5-8B. One forward pass reads a state, a question and its options and
returns a probability for every option; no text is generated.

    from votum import Votum
    vm = Votum.from_pretrained("<org>/Votum-8B")
    vm.decide({"state": "...", "questions": {"q1": {"type": "choice", "instructions": "...", "criteria": {"a": "...", "b": "..."}}}})
"""
from .config import VotumConfig
from .modeling import Votum

__all__ = ["Votum", "VotumConfig"]
__version__ = "0.1.1"
