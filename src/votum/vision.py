"""Images in Votum requests: the Apertus 1.5 image tokenizer and the image tokens of a prompt.

Apertus 1.5 reads an image as discrete tokens. Each 16 x 16 pixel patch becomes one code of a 131,072-entry codebook, and
code c is the vocabulary token "<|visual token c|>" (id 131,272 + c) of the same embedding table that holds the text
tokens. An image is written as
    <|img_start|> "H*W" <|img_token_start|> row 1 <|img_end_of_row|> ... row H <|img_end|>      (row = W codes)
exactly as the swiss-ai processor writes it, so Votum needs no model change to read images: only the codes.

The tokenizer is the one of Apertus 1.5 (Apertus1p5VisionTokenizerModel in the Swiss AI Initiative's transformers,
Apache-2.0); this is a standalone copy of its encoder and of Apertus1p5ImageProcessor (RGB, aspect-preserving resize to
multiples of 16, bicubic with antialiasing, x / 127.5 - 1).
It runs in float32: the codes are argmaxes over 131,072 entries, and half precision changes about 8 % of them.

    tok = ImageTokenizer.from_pretrained("votum-ai/Votum-8B")          # 218 M parameters, 0.9 GB
    img = tok.encode(PIL.Image.open("photo.jpg"), max_codes=1024)      # {"h": rows, "w": columns, "codes": [...]}

Needs Pillow and torchvision (pip install "votum[vision]").
"""
from __future__ import annotations

import base64, binascii, io
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

WEIGHTS = "model-vision_tokenizer-model.safetensors"     # the file in the swiss-ai/Apertus-v1.5-8B repository
VISION_DIR = "vision"                                     # optional copy in a Votum repository: vision/<WEIGHTS>
PREFIX = "model.vision_tokenizer."
FACTOR = 16                                               # pixels per code side
MIN_CODES, DEFAULT_CODES = 256, 1024
MAX_PIXELS_DECODED = 64_000_000                           # larger images are refused before decoding (decompression bombs)
IMAGE_TOKEN, BOI, WRAP, EOL, EOI = "<|image|>", "<|img_start|>", "<|img_token_start|>", "<|img_end_of_row|>", "<|img_end|>"
CONFIG = dict(base_channels=256, channel_multiplier=(1, 1, 2, 2, 4), num_res_blocks=4, attn_resolutions=(16,), resolution=256,
              in_channels=3, latent_channels=256, embed_dim=256, codebook_size=131072)


# ------------------------------------------------------------------------------------------------ encoder (swiss-ai port)
class ResnetBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.cin, self.cout = cin, cout
        self.norm1 = nn.GroupNorm(32, cin, eps=1e-6)
        self.conv1 = nn.Conv2d(cin, cout, 3, 1, 1)
        self.norm2 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1)
        if cin != cout:
            self.nin_shortcut = nn.Conv2d(cin, cout, 1)

    def forward(self, x):
        h = self.norm1(x)
        h = self.conv1(h * torch.sigmoid(h))
        h = self.norm2(h)
        h = self.conv2(h * torch.sigmoid(h))
        return (self.nin_shortcut(x) if self.cin != self.cout else x) + h


class AttnBlock(nn.Module):
    """single-head attention over the spatial positions; the plain bmm keeps the reference's numerics"""

    def __init__(self, c: int):
        super().__init__()
        self.norm = nn.GroupNorm(32, c, eps=1e-6)
        self.q, self.k, self.v, self.proj_out = (nn.Conv2d(c, c, 1) for _ in range(4))

    def forward(self, x):
        h = self.norm(x)
        q, k, v = self.q(h), self.k(h), self.v(h)
        b, c, hh, ww = q.shape
        w = F.softmax(torch.bmm(q.reshape(b, c, hh * ww).permute(0, 2, 1), k.reshape(b, c, hh * ww)) * c ** -0.5, dim=2)
        h = torch.bmm(v.reshape(b, c, hh * ww), w.permute(0, 2, 1)).reshape(b, c, hh, ww)
        return x + self.proj_out(h)


class Downsample(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 2, 0)

    def forward(self, x):
        return self.conv(F.pad(x, (0, 1, 0, 1)))       # right and bottom padding, as in the original


class Level(nn.Module):
    def __init__(self, cin: int, cout: int, n_blocks: int, attn: bool, down: bool):
        super().__init__()
        self.block = nn.ModuleList()
        self.attn = nn.ModuleList()
        for _ in range(n_blocks):
            self.block.append(ResnetBlock(cin, cout))
            cin = cout
            if attn:
                self.attn.append(AttnBlock(cout))
        if down:
            self.downsample = Downsample(cout)

    def forward(self, h):
        for k, blk in enumerate(self.block):
            h = blk(h)
            if len(self.attn):
                h = self.attn[k](h)
        return self.downsample(h) if hasattr(self, "downsample") else h


class Mid(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.block_1, self.attn_1, self.block_2 = ResnetBlock(c, c), AttnBlock(c), ResnetBlock(c, c)

    def forward(self, h):
        return self.block_2(self.attn_1(self.block_1(h)))


class Encoder(nn.Module):
    def __init__(self, cfg: dict = CONFIG):
        super().__init__()
        mult = (1,) + tuple(cfg["channel_multiplier"])
        n = len(cfg["channel_multiplier"])
        res = cfg["resolution"]
        self.conv_in = nn.Conv2d(cfg["in_channels"], cfg["base_channels"], 3, 1, 1)
        self.down = nn.ModuleList()
        for i in range(n):            # attention placement follows the reference resolution, as in the original
            cin, cout = cfg["base_channels"] * mult[i], cfg["base_channels"] * cfg["channel_multiplier"][i]
            self.down.append(Level(cin, cout, cfg["num_res_blocks"], res in cfg["attn_resolutions"], i != n - 1))
            res //= 2
        c = cfg["base_channels"] * cfg["channel_multiplier"][-1]
        self.mid = Mid(c)
        self.norm_out = nn.GroupNorm(32, c, eps=1e-6)
        self.conv_out = nn.Conv2d(c, cfg["latent_channels"], 3, 1, 1)

    def forward(self, x):
        h = self.conv_in(x)
        for level in self.down:
            h = level(h)
        h = self.norm_out(self.mid(h))
        return self.conv_out(h * torch.sigmoid(h))


class Codebook(nn.Module):
    def __init__(self, n: int, d: int):
        super().__init__()
        self.embedding = nn.Embedding(n, d)


# ------------------------------------------------------------------------------------------------ tokenizer
def smart_resize(h: int, w: int, min_pixels: int, max_pixels: int, factor: int = FACTOR) -> tuple[int, int]:
    """the output size of Apertus1p5ImageProcessor: the area clamped to [min_pixels, max_pixels], aspect ratio kept, both
    sides rounded half-up to multiples of factor (with the reference's int() truncations)."""
    area = max(min(max_pixels, h * w), min_pixels)
    ar = w / h
    nh = int((area / ar) ** 0.5)
    nw = int(nh * ar)
    nh = (nh + factor // 2) // factor * factor
    nw = (nw + factor // 2) // factor * factor
    return max(nh, factor), max(nw, factor)


class ImageTokenizer(nn.Module):
    def __init__(self, cfg: dict = CONFIG):
        super().__init__()
        self.encoder = Encoder(cfg)
        self.quant_conv = nn.Conv2d(cfg["latent_channels"], cfg["embed_dim"], 1)
        self.quantize = Codebook(cfg["codebook_size"], cfg["embed_dim"])

    @classmethod
    def from_pretrained(cls, repo: str, *, base: str | None = None, revision: str | None = None, device=None,
                        token=None) -> "ImageTokenizer":
        """the tokenizer weights from repo's vision/ folder (a Votum repo or checkpoint directory), else from the Apertus base
        repository (base, default the one pinned in the repo's votum_config.json)."""
        from safetensors.torch import load_file
        from .build import resolve
        from .config import CONFIG_NAME, VotumConfig
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        path = None
        try:
            d = resolve(repo, revision, token, [f"{VISION_DIR}/*", CONFIG_NAME])
            if (d / VISION_DIR / WEIGHTS).exists(): path = d / VISION_DIR / WEIGHTS
        except Exception:
            d = None
        if path is None:
            cfg = VotumConfig.load(d) if d is not None and (d / CONFIG_NAME).exists() else VotumConfig()
            bd = resolve(base or cfg.base_model, None if base else cfg.base_revision, token, [WEIGHTS])
            path = bd / WEIGHTS
            if not path.exists(): raise FileNotFoundError(f"no {WEIGHTS} in {bd}")
        sd = {k.removeprefix(PREFIX): v for k, v in load_file(str(path)).items()}
        m = cls()
        m.load_state_dict(sd, strict=True)
        return m.float().to(device).eval().requires_grad_(False)

    @property
    def device(self):
        return self.quant_conv.weight.device

    def pixels(self, image, max_codes: int = DEFAULT_CODES) -> torch.Tensor:
        """PIL image -> [1, 3, H, W] float32 in [-1, 1], H x W at most max_codes x 256 pixels (at least 256 codes)."""
        import numpy as np
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.v2 import functional as tvF
        x = torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1).contiguous()
        cap = max(MIN_CODES, int(max_codes)) * FACTOR * FACTOR
        nh, nw = smart_resize(x.shape[1], x.shape[2], MIN_CODES * FACTOR * FACTOR, cap)
        x = tvF.resize(x, [nh, nw], interpolation=InterpolationMode.BICUBIC, antialias=True)
        return ((x.to(torch.float32) - 127.5) / 127.5)[None]

    @torch.inference_mode()
    def encode(self, image, max_codes: int = DEFAULT_CODES) -> dict:
        """PIL image -> {"h", "w", "codes"} (row-major codebook indices)."""
        x = self.pixels(image, max_codes).to(self.device)
        with torch.backends.cudnn.flags(enabled=True, allow_tf32=False):      # full fp32 convolutions
            z = self.quant_conv(self.encoder(x))
        h, w = z.shape[-2:]
        flat = z[0].reshape(z.shape[1], h * w)
        codes = torch.cat([torch.einsum("dp,nd->np", flat[:, s:s + 256], self.quantize.embedding.weight).argmax(0)   # 128 MB per chunk
                           for s in range(0, h * w, 256)])
        return {"h": int(h), "w": int(w), "codes": codes.tolist()}


# ------------------------------------------------------------------------------------------------ image values
def load_image(value, allow_paths: bool = True):
    """an image value of a request -> PIL image (RGB). Accepted: a PIL image, bytes, a base64 string or data URL
    ("data:image/png;base64,..."), and, when allow_paths, a file path (str or Path). The HTTP server passes
    allow_paths=False: it never opens files or URLs named by a request."""
    from PIL import Image
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, (bytes, bytearray)):
        data = bytes(value)
    elif isinstance(value, Path) or (isinstance(value, str) and allow_paths and _looks_like_path(value)):
        if not allow_paths: raise ValueError("image paths are not accepted here; send the image as base64")
        data = Path(value).read_bytes()
    elif isinstance(value, str):
        s = value.strip()
        if s.startswith("data:"):
            head, _, s = s.partition(",")
            if ";base64" not in head: raise ValueError("an image data URL must be base64-encoded")
        try:
            data = base64.b64decode(s, validate=True)
        except (binascii.Error, ValueError):
            raise ValueError("an image must be base64 (or a data URL)" + (", a file path" if allow_paths else "")) from None
    else:
        raise ValueError(f"an image must be a base64 string{', a file path, bytes' if allow_paths else ''} or a PIL image, "
                         f"not {type(value).__name__}")
    try:
        im = Image.open(io.BytesIO(data))
        w, h = im.size
        if w * h > MAX_PIXELS_DECODED: raise ValueError(f"image too large ({w} x {h} pixels; at most {MAX_PIXELS_DECODED:,})")
        return im.convert("RGB")
    except ValueError:
        raise
    except Exception:
        raise ValueError("the image could not be decoded (PNG, JPEG, WebP, GIF or BMP)") from None


def _looks_like_path(s: str) -> bool:
    return len(s) < 4096 and not s.startswith("data:") and ("/" in s or "\\" in s or "." in s[-6:]) and Path(s).exists()


def image_ids(tok, img: dict) -> list[int]:
    """the token ids of one image: start, "H*W", wrapper, the rows separated by end-of-row tokens, end."""
    h, w, codes = img["h"], img["w"], img["codes"]
    if len(codes) != h * w: raise ValueError(f"an image of {h} x {w} codes has {len(codes)} codes")
    offset = visual_offset(tok)
    ids = [tok.convert_tokens_to_ids(BOI)] + tok(f"{h}*{w}", add_special_tokens=False).input_ids + [tok.convert_tokens_to_ids(WRAP)]
    eol = tok.convert_tokens_to_ids(EOL)
    for r in range(h):
        if r: ids.append(eol)
        ids += [offset + c for c in codes[r * w:(r + 1) * w]]
    return ids + [tok.convert_tokens_to_ids(EOI)]


def visual_offset(tok) -> int:
    """the id of "<|visual token 0|>" (131,272 in the Apertus 1.5 vocabulary)"""
    off = getattr(tok, "_votum_visual_offset", None)
    if off is None:
        off = tok.convert_tokens_to_ids("<|visual token 0|>")
        if off is None or off == tok.unk_token_id or tok.convert_tokens_to_ids("<|visual token 131071|>") != off + 131071:
            raise ValueError("this tokenizer has no Apertus 1.5 visual tokens; images need the Apertus 1.5 vocabulary")
        try: tok._votum_visual_offset = off
        except AttributeError: pass
    return off


VISION_MODES = ("lazy", "eager", "off")


def vision_mode(v) -> str:
    """the vision option of Votum.from_pretrained / VotumVLLM / votum serve: "lazy" (default: the image tokenizer is
    downloaded and loaded on the first image), "eager" (at start-up) or "off" (images are refused; True / False / None are
    accepted for lazy / off / lazy)"""
    m = {True: "lazy", None: "lazy", False: "off"}.get(v, v) if not isinstance(v, str) else v.lower()
    if m not in VISION_MODES: raise ValueError(f"vision must be one of {VISION_MODES}, not {v!r}")
    return m


def images_for(repo: str, vision="lazy", **kw) -> "Images | None":
    """the Images helper of a model in the given vision mode (None when off)"""
    m = vision_mode(vision)
    if m == "off": return None
    im = Images(repo, **kw)
    if m == "eager": im.tokenizer()
    return im


class Images:
    """the image side of a Votum model: the tokenizer, loaded on first use, and a small cache of codes by image content.

    encode(value) -> {"h", "w", "codes"} for an image value of a request (see load_image). Thread-safe (one image at a
    time). max_codes: the image budget (at most about max_codes x 256 pixels); allow_paths: whether string values may name
    files (the Python API: yes; the HTTP server: no)."""

    def __init__(self, repo: str, *, device=None, max_codes: int = DEFAULT_CODES, allow_paths: bool = True, cache: int = 256,
                 token=None, tokenizer: ImageTokenizer | None = None):
        import threading
        from collections import OrderedDict
        self.repo, self.device, self.max_codes, self.allow_paths, self.token = repo, device, int(max_codes), allow_paths, token
        self.tok = tokenizer; self.lock = threading.Lock(); self.cache: OrderedDict = OrderedDict(); self.size = cache

    def tokenizer(self) -> ImageTokenizer:
        if self.tok is None:
            try:
                import PIL, torchvision  # noqa: F401
            except ImportError:
                raise ImportError('images need Pillow and torchvision: pip install "votum[vision]"') from None
            self.tok = ImageTokenizer.from_pretrained(self.repo, device=self.device, token=self.token)
        return self.tok

    def encode(self, value, max_codes: int | None = None) -> dict:
        import hashlib
        im = load_image(value, self.allow_paths)
        n = int(max_codes or self.max_codes)
        key = hashlib.blake2b(im.tobytes() + repr((im.size, n)).encode(), digest_size=16).digest()
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key); return self.cache[key]
            out = self.tokenizer().encode(im, n)
            self.cache[key] = out
            if len(self.cache) > self.size: self.cache.popitem(last=False)
        return out
