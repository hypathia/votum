"""Backbone loading (Apertus-v1.5-8B text stack) and LoRA.

Apertus 1.5 ships one multimodal checkpoint (text + vision tokenizer + audio tokenizer in the same shards). The text stack
is loaded alone through Apertus1p5TextForCausalLM with the key mapping model.language_model.* -> model.*; the tower
weights are ignored. Attention runs on SDPA (all 32 layers are full attention); bidigate.py / flexmask.py replace the masks."""
import time
import torch
import common as C


def text_model(model):
    m = model
    for attr in ("base_model", "model"):   # peft wrapper -> LoraModel -> CausalLM
        while hasattr(m, attr) and not hasattr(m, "layers") and not hasattr(m, "embed_tokens"):
            nxt = getattr(m, attr)
            if nxt is m: break                 # PreTrainedModel.base_model returns self when there is no sub-module
            m = nxt
    if hasattr(m, "model") and hasattr(m.model, "layers"): m = m.model
    return m


def d_model(model):
    return text_model(model).config.hidden_size


def load_backbone(path=None, quant="bf16", device="cuda:0"):
    """Text-only causal LM from the checkpoint at `path` (Apertus 1.5 multimodal or text-only, or Apertus 1.x text)."""
    from transformers import AutoConfig, BitsAndBytesConfig
    path = path or C.backbone_path()[1]
    try: cfg = AutoConfig.from_pretrained(path)
    except ValueError as e:
        if "apertus1p5" in str(e): raise SystemExit("this checkpoint is the Apertus 1.5 multimodal release, which needs the swiss-ai transformers fork (training/README.md, Environment); or fine-tune from the Votum checkpoint instead (finetune.py --mode merged)")
        raise
    mt = cfg.model_type
    kw = dict(dtype=torch.bfloat16, device_map={"": device})
    if mt == "apertus" and getattr(cfg, "decision_pointer_rank", None) is not None:
        # a merged Votum checkpoint (the released adapter already folded in, no lm_head, pointer.* tensors): the text
        # stack alone, through the mainline ApertusModel; a fresh LoRA goes on top (finetune.py --from <Votum repo>)
        from transformers import ApertusModel as Cls; tcfg = cfg
    elif mt == "apertus1p5":
        try: from transformers.models.apertus1p5.modeling_apertus1p5 import Apertus1p5TextForCausalLM as Cls
        except ImportError: raise SystemExit("this checkpoint is the Apertus 1.5 multimodal release, which needs the swiss-ai transformers fork (training/README.md, Environment); "
                                             "or fine-tune from the Votum checkpoint instead (finetune.py --mode merged), which runs on mainline transformers")
        tcfg = cfg.text_config; kw["key_mapping"] = {r"^model\.language_model\.": "model."}
    elif mt == "apertus1p5_text":
        from transformers.models.apertus1p5.modeling_apertus1p5 import Apertus1p5TextForCausalLM as Cls; tcfg = cfg
    elif mt == "apertus":
        from transformers.models.apertus.modeling_apertus import ApertusForCausalLM as Cls; tcfg = cfg
    else:
        raise ValueError(f"unsupported model_type {mt}")
    tcfg._attn_implementation = "sdpa"; kw["config"] = tcfg
    if quant == "nf4":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    t0 = time.time(); m = Cls.from_pretrained(str(path), **kw); m.config.use_cache = False
    if not hasattr(m, "layers") and hasattr(m, "model") and not hasattr(m.model, "layers"): raise ValueError(f"unexpected model structure for {path}")
    for p in m.parameters(): p.requires_grad_(False)
    print(f"loaded {path} ({mt}, {quant}) in {time.time()-t0:.0f}s; GPU {torch.cuda.memory_allocated()/2**30:.1f} GiB", flush=True)
    return m


def add_lora(model, seed, r=None, alpha=None, dropout=None, targets=None):
    from peft import LoraConfig, get_peft_model
    R = C.RECIPE; torch.manual_seed(seed)
    # a bare text stack (ApertusModel, the merged Votum checkpoint) has no generation hooks: plain PEFT wrapper, as votum.modeling does
    task = "CAUSAL_LM" if hasattr(model, "prepare_inputs_for_generation") else None
    cfg = LoraConfig(r=r or R["lora_r"], lora_alpha=alpha or R["lora_alpha"], lora_dropout=R["lora_dropout"] if dropout is None else dropout,
                     target_modules=targets or R["lora_targets"], bias="none", task_type=task)
    return get_peft_model(model, cfg)


def pad_batch(ids_list, device):
    B = len(ids_list); T = max(len(x) for x in ids_list)
    ids = torch.zeros(B, T, dtype=torch.long); am = torch.zeros(B, T, dtype=torch.long)
    for b, x in enumerate(ids_list): ids[b, :len(x)] = torch.tensor(x); am[b, :len(x)] = 1
    return ids.to(device), am.to(device)


def hidden(model, ids_list, device="cuda", positions=None):
    """right-padded forward -> final-norm hidden states [B, T, H] (bf16). positions: optional per-item
    position-id lists (shared option positions); padding positions continue the sequence."""
    ids, am = pad_batch(ids_list, device)
    import bidigate; bidigate.set_padding(am)          # read only when a BidiGate is installed
    kw = {}
    if positions is not None:
        T = ids.shape[1]; pid = torch.zeros_like(ids)
        for b, p in enumerate(positions):
            pid[b, :len(p)] = torch.tensor(p, device=device)
            if len(p) < T: pid[b, len(p):] = torch.arange(len(p), T, device=device) + (max(p) + 1 - len(p) if p else 0)
        kw["position_ids"] = pid
    out = text_model(model)(input_ids=ids, attention_mask=am, use_cache=False, **kw)
    return out.last_hidden_state


def attn_modules(model):
    return [l.self_attn for l in text_model(model).layers]
