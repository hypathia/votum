"""Votum as a vLLM pooling model (vLLM 0.33).

Architecture = vLLM's own Apertus text stack (QK-norm, xIELU MLP, llama3 RoPE; vllm/model_executor/models/apertus.py) with
  * interleaved bidirectional attention: the layers in `config.decision_bidirectional_layers` (3, 7, ..., 31) are built
    non-causal; all other layers stay causal;
  * the option-isolation / header mask of votum.modeling, as one compiled FlexAttention call per layer (isolate.py), over
    prompts packed by the client as [header][state][question block 1]...[question block J] (each question block sees the
    header, its state and itself, with the positions of its own single-question rendering: exactly the single-pass result);
  * the pointer readout inside the pooler: score_j = < q0, W_o LN(h_end_j) > / sqrt(r) at every option-end token, in fp32.
    The pooler returns the option scores, so one request costs a few floats of output.
A checkpoint built with `votum build --exact` keeps the LoRA adapter unmerged and serves it through vLLM's LoRA support; the
default checkpoint has it merged into the weights. Constraints: chunked prefill and prefix caching
must be off (a bidirectional layer needs the whole prompt in one step); client.py sets both, and keeps its own exact
prefix cache for header and state blocks (isolate.PrefixCache).
"""
from __future__ import annotations

from collections.abc import Iterable, Set
from copy import copy

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.model_executor.layers.pooler import Pooler, PoolingParamsUpdate
from vllm.model_executor.layers.pooler.tokwise.methods import AllPool
from vllm.model_executor.models.apertus import ApertusDecoderLayer, ApertusModel
from vllm.model_executor.models.interfaces import SupportsLoRA
from vllm.model_executor.models.interfaces_base import default_pooling_type
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper, extract_layer_index, maybe_prefix
from vllm.v1.pool.metadata import PoolingMetadata

from . import isolate as _iso


class DecisionLayer(ApertusDecoderLayer):
    """An Apertus decoder layer whose attention is bidirectional when its index is listed in the config."""

    def __init__(self, config, cache_config=None, quant_config=None, prefix: str = "") -> None:
        idx = extract_layer_index(prefix)
        cfg = copy(config)
        cfg.is_causal = idx not in set(config.decision_bidirectional_layers)
        super().__init__(config=cfg, cache_config=cache_config, quant_config=quant_config, prefix=prefix)


class PointerPooler(Pooler):
    """constant-query pointer readout over the option-end states of each prompt -> option scores (fp32, prompt order)."""

    def __init__(self, hidden_size: int, rank: int, opt_end_id: int) -> None:
        super().__init__()
        self.o = nn.Linear(hidden_size, rank, bias=False, dtype=torch.float32)   # fp32 as trained
        self.q0 = nn.Parameter(torch.zeros(rank, dtype=torch.float32), requires_grad=False)
        self.rank, self.opt_end_id = rank, opt_end_id
        self.pooling = AllPool()

    def get_supported_tasks(self) -> Set[str]:
        return {"token_classify"}

    def get_pooling_updates(self, task) -> PoolingParamsUpdate:
        return PoolingParamsUpdate(requires_token_ids=True)

    def forward(self, hidden_states: torch.Tensor | list[torch.Tensor], pooling_metadata: PoolingMetadata):
        per_prompt = self.pooling(hidden_states, pooling_metadata)
        token_ids = pooling_metadata.get_prompt_token_ids_cpu()   # vLLM 0.33 fills the CPU copy only
        return [_iso.pointer_scores(h, ids.to(h.device), self.opt_end_id, self.q0, self.o.weight, self.rank)
                for h, ids in zip(per_prompt, token_ids)]


@default_pooling_type(tok_pooling_type="ALL")
class VotumDecisionModel(nn.Module, SupportsLoRA):
    is_pooling_model = True
    packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
    embedding_modules: dict[str, str] = {}
    hf_to_vllm_mapper = WeightsMapper(orig_to_new_stacked={".q_proj": (".qkv_proj", "q"), ".k_proj": (".qkv_proj", "k"),
                                                         ".v_proj": (".qkv_proj", "v")})

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        cfg = vllm_config.model_config.hf_config; self.config = cfg
        self.model = ApertusModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"), layer_type=DecisionLayer)
        self.opt_end_id = int(cfg.decision_opt_end_id)
        self.pooler = PointerPooler(cfg.hidden_size, int(cfg.decision_pointer_rank), self.opt_end_id)
        self.shared_positions = bool(cfg.decision_shared_positions)
        self.header = [int(t) for t in cfg.decision_options_header_ids]
        self.q_header = [int(t) for t in cfg.decision_question_header_ids]
        self.close_last = bool(cfg.decision_close_last)
        self._step = None
        _iso.GEOM.update(layers=cfg.num_hidden_layers, kv_heads=cfg.num_key_value_heads, head_dim=cfg.hidden_size // cfg.num_attention_heads,
                         dtype=vllm_config.model_config.dtype,
                         device=torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu"))
        bidir = set(cfg.decision_bidirectional_layers)
        for i, layer in enumerate(self.model.layers):
            if not hasattr(layer, "self_attn"): continue
            impl = layer.self_attn.attn.impl
            impl.forward = _iso.flex_forward(impl.forward, impl, self, i in bidir, i)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, intermediate_tensors=None,
                inputs_embeds: torch.Tensor | None = None) -> torch.Tensor:
        positions = self._step_layout(input_ids, positions)
        return self.model(input_ids=input_ids, positions=positions, intermediate_tensors=intermediate_tensors, inputs_embeds=inputs_embeds)

    def _step_layout(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """positions and BlockMasks of this step (isolate.py)."""
        from vllm.forward_context import get_forward_context
        self._step = None
        md_all = get_forward_context().attn_metadata
        md = (next(iter(md_all.values())) if isinstance(md_all, dict) else md_all) if md_all else None
        if md is None: return positions                              # profiling / dummy run
        N = int(md.num_actual_tokens); qsl = md.query_start_loc.tolist()
        st = _iso.batch_layout(input_ids[:N].tolist(), qsl, self.opt_end_id, self.shared_positions, self.header, self.q_header, self.close_last)
        self._step = _iso.StepMasks(st, input_ids.device)
        # in place: on steps of <= 512 tokens vLLM replays piecewise CUDA graphs that read its persistent positions buffer
        # (cudagraph_copy_inputs off), so positions written into a copy would be ignored there
        positions[:N].copy_(torch.from_numpy(st["pos"]).to(device=positions.device, dtype=positions.dtype))
        return positions

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        backbone, loaded = [], set()
        for name, w in weights:
            if name == "pointer.q0":
                self.pooler.q0.data.copy_(w.float()); loaded.add("pooler.q0")
            elif name == "pointer.o.weight":
                self.pooler.o.weight.data.copy_(w.float()); loaded.add("pooler.o.weight")
            elif name.startswith("model."):
                backbone.append((name, w))
            # anything else (lm_head, towers) is not needed by a pooling model
        loader = AutoWeightsLoader(self)
        return loaded | loader.load_weights(backbone, mapper=self.hf_to_vllm_mapper)
