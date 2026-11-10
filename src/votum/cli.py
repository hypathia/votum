"""votum command line.

    votum decide --repo <org>/Votum-8B [--request req.json | < req.json] [--exact] [--device cuda|cpu]   PyTorch reference, one request
    votum serve  --model <org>/Votum-8B [--host 127.0.0.1] [--port 8011]  vLLM HTTP server (/v1/systemone; more options: votum serve --help)
    votum quantize --model <org>/Votum-8B --out <dir>                     FP8 copy of a checkpoint, about half the size (votum quantize --help)
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
    elif cmd == "quantize":
        from .fp8 import main as m; m(rest)
    elif cmd == "serve":
        from .vllm.server import main as m; m(rest)
    elif cmd == "decide":
        import argparse
        ap = argparse.ArgumentParser(prog="votum decide", description="Answer one request with the PyTorch reference. The request JSON "
                                     "({\"state\": ..., \"questions\": {...}}) comes from --request or stdin.")
        ap.add_argument("--repo", required=True, help="Votum repo id or local checkpoint directory")
        ap.add_argument("--request", help="JSON file with the request (default: read stdin)")
        ap.add_argument("--exact", action="store_true", help="use the unmerged adapter in the repo's adapter/ folder on the pinned base")
        ap.add_argument("--base", help="with --exact: base checkpoint (repo id or directory) instead of the pinned one")
        ap.add_argument("--device", help="cuda (default when available) or cpu"); a = ap.parse_args(rest)
        if a.request: req = json.load(open(a.request))
        else:
            if sys.stdin.isatty(): print("votum decide: reading the request JSON from stdin (pass --request FILE, or pipe it in)", file=sys.stderr)
            raw = sys.stdin.read()
            if not raw.strip(): sys.exit("votum decide: no request on stdin; pass --request FILE or pipe a JSON request")
            try: req = json.loads(raw)
            except json.JSONDecodeError as e: sys.exit(f"votum decide: stdin is not JSON ({e})")
        from .modeling import Votum
        vm = Votum.from_pretrained(a.repo, exact=a.exact, base=a.base, device=a.device)
        print(json.dumps(vm.decide(req), indent=1))
    else:
        sys.exit(f"unknown command {cmd!r}\n{__doc__}")


if __name__ == "__main__":
    main()
