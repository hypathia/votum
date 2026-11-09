"""votum command line.

    votum decide --repo <org>/Votum-8B [--exact] [--base <dir>] [--device cuda|cpu] < request.json   PyTorch reference, one request from stdin
    votum serve  --model <org>/Votum-8B [--host 127.0.0.1] [--port 8011]  vLLM HTTP server (/v1/systemone; more options: votum serve --help)
    votum build  --adapter <org>/Votum-8B --out <dir> [--exact] [--link] [--base <dir>]   rebuild a checkpoint from the adapter form
"""
from __future__ import annotations

import json, sys


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__); return
    cmd, rest = argv[0], argv[1:]
    if cmd == "build":
        from .build import main as m; m(rest)
    elif cmd == "serve":
        from .vllm.server import main as m; m(rest)
    elif cmd == "decide":
        import argparse
        ap = argparse.ArgumentParser(prog="votum decide"); ap.add_argument("--repo", required=True); ap.add_argument("--base")
        ap.add_argument("--exact", action="store_true"); ap.add_argument("--device"); a = ap.parse_args(rest)
        from .modeling import Votum
        vm = Votum.from_pretrained(a.repo, exact=a.exact, base=a.base, device=a.device)
        print(json.dumps(vm.decide(json.load(sys.stdin)), indent=1))
    else:
        sys.exit(f"unknown command {cmd!r}\n{__doc__}")


if __name__ == "__main__":
    main()
