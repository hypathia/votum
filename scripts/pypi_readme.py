"""Rewrite README.md for PyPI: relative links and images become absolute GitHub URLs at the given ref.

    python scripts/pypi_readme.py <git ref, e.g. v0.1.0>

PyPI shows the README without the repository around it, so `assets/x.png` or `docs/serving.md` would not resolve.
Run by the release workflow just before the build; the repository's README stays as it is."""
import re
import sys
from pathlib import Path

REPO = "https://github.com/hypathia/votum"
ref = sys.argv[1]
p = Path(__file__).resolve().parent.parent / "README.md"
s = p.read_text(encoding="utf-8")
raw, blob = f"https://raw.githubusercontent.com/hypathia/votum/{ref}/", f"{REPO}/blob/{ref}/"
absolute = re.compile(r"^(https?:|mailto:|#|/)")


def fix(url: str, image: bool) -> str:
    return url if absolute.match(url) else (raw if image else blob) + url


s = re.sub(r'(src|srcset)="([^"]+)"', lambda m: f'{m[1]}="{fix(m[2], True)}"', s)
s = re.sub(r"(!?)\[([^\]]*)\]\(([^)\s]+)\)", lambda m: f"{m[1]}[{m[2]}]({fix(m[3], m[1] == '!')})", s)
p.write_text(s, encoding="utf-8")
left = [u for u in re.findall(r'(?:src|srcset)="([^"]+)"|\]\(([^)\s]+)\)', s) for u in u if u and not absolute.match(u)]
sys.exit(f"relative links left: {left}" if left else 0)
