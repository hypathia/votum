"""Shared paths, environment and recipe constants.

Paths come from environment variables with local defaults:
  VOTUM_DATA      training data                 (default: data/ next to this file)
  VOTUM_RUNS      checkpoints, logs, caches     (default: runs/ next to this file)
  VOTUM_BACKBONE  backbone repo id              (default: swiss-ai/Apertus-v1.5-8B, from the Hugging Face cache in HF_HOME)
disk_guard() refuses a large write that would leave less than VOTUM_DISK_FLOOR_GB (default 30) free."""
import json, os, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("VOTUM_DATA", ROOT / "data"))
RUNS = Path(os.environ.get("VOTUM_RUNS", ROOT / "runs"))
os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
HF_HOME = Path(os.environ["HF_HOME"])
sys.path.insert(0, str(ROOT))

BACKBONE = "swiss-ai/Apertus-v1.5-8B"   # gated on the Hub; override with VOTUM_BACKBONE

# Training recipe: LoRA rank 16 on the attention and MLP projections, AdamW without weight decay, gradient clip 1. This
# dict is also written into every checkpoint's train_meta.json; the command line (train_meta "args") is authoritative.
# (The released checkpoint's train_meta.json records an older, longer version of this dict.)
RECIPE = dict(
    lora_r=16, lora_alpha=32, lora_dropout=0.0,
    lora_targets=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
    weight_decay=0.0, betas=(0.9, 0.999), grad_clip=1.0,
    warmup=25, total_steps=400, final_decay=100,
    tokens_per_step=65536, micro_tokens=8192, save_every=25,
    seed=1,
)


def snapshot(repo, kind="models"):
    hits = sorted((HF_HOME / "hub" / f"{kind}--{repo.replace('/', '--')}" / "snapshots").glob("*"))
    for h in reversed(hits):
        if kind == "datasets" or (h / "config.json").exists():
            return h
    return None


def backbone_path(repo=None, weights=True):
    """-> (repo, local snapshot path); raises if absent (no implicit download). weights=False accepts a tokenizer-only snapshot."""
    repo = repo or os.environ.get("VOTUM_BACKBONE", BACKBONE)
    p = snapshot(repo)
    if p is not None and (not weights or any(p.glob("*.safetensors"))):
        return repo, p
    raise FileNotFoundError(f"{repo} is not in {HF_HOME}/hub ")


def free_gb(path):
    return shutil.disk_usage(path).free / 2**30


def disk_guard(path, need_gb=0.0, floor_gb=None):
    """Raise before a write that would take the filesystem holding `path` below the floor (default 30 GB)."""
    floor = float(os.environ.get("VOTUM_DISK_FLOOR_GB", 30)) if floor_gb is None else floor_gb
    p = Path(path)
    while not p.exists(): p = p.parent
    f = free_gb(p)
    if f - need_gb < floor:
        raise RuntimeError(f"disk guard: {p} has {f:.1f} GiB free, need {need_gb:.1f} GiB with a {floor:.0f} GiB floor")
    return f


def jdump(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f: json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)


def iter_jsonl(path):
    import gzip
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as f:
        for line in f:
            if line.strip(): yield json.loads(line)


def train_subset_size(ntoks, steps=None, tokens_per_step=None, slack=1.15):
    """number of leading rows of the fixed train order that one run of `steps` steps consumes (+15% slack)."""
    import numpy as np
    steps = steps or RECIPE["total_steps"]; tps = tokens_per_step or RECIPE["tokens_per_step"]
    q_per_step = tps / (float(np.mean(ntoks)) + 16) * 0.85      # 0.85 = packing efficiency of train.Plan
    return min(len(ntoks), int(steps * q_per_step * slack))


def load_tokenizer(path):
    """AutoTokenizer.from_pretrained with a pickle cache under RUNS/cache: the transformers fork converts Apertus's 266k-token
    tokenizer on every load (2-4 min); the pickled object loads in ~40 s and tokenises identically."""
    import hashlib, pickle
    from transformers import AutoTokenizer
    key = hashlib.sha1(str(Path(path).resolve()).encode()).hexdigest()[:10]; pk = RUNS / "cache" / f"tok_{key}.pkl"
    if pk.exists():
        try:
            with open(pk, "rb") as f: return pickle.load(f)
        except Exception: pk.unlink(missing_ok=True)
    tok = AutoTokenizer.from_pretrained(str(path))
    try:
        pk.parent.mkdir(parents=True, exist_ok=True)
        with open(str(pk) + ".tmp", "wb") as f: pickle.dump(tok, f)
        os.replace(str(pk) + ".tmp", pk)
    except Exception: pass
    return tok
