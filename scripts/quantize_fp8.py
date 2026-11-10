"""Quantize a Votum checkpoint to FP8. Same as `votum quantize` (see src/votum/fp8.py for what it does):

    python scripts/quantize_fp8.py --model votum-ai/Votum-8B --out Votum-8B-FP8
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from votum.fp8 import main  # noqa: E402

if __name__ == "__main__":
    main()
