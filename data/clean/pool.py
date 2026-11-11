#!/usr/bin/env python3
"""The pool pipeline: from converted rows to the published dataset.

    python -I pool.py run    --pool POOL --work WORK --bench FILE [FILE ...]     every stage, one after the other
    python -I pool.py clean  --pool POOL --out WORK/clean --bench FILE [FILE ...]
    python -I pool.py mask   --pool WORK/clean --out WORK/clean_pii
    python -I pool.py export --pool WORK/clean_pii --out WORK/dataset
    python -I pool.py audit  sample|regex|spacy|gliner|report --audit-dir WORK/audit [--pool WORK/clean_pii]

A pool is a directory with train.jsonl.gz and val.jsonl.gz (one JSON row per line; see README.md). What happens to
each source, and every setting of the stages, is in the config file (pool.json next to this script, or --config).
`<command> --help` lists a command's options.

  clean    drops whole sources, filtered rows, and rows that overlap an evaluation benchmark
  mask     drops the sources not cleared for redistribution, masks personal data in the self-collected sources, and
           masks the e-mail addresses left anywhere
  export   writes the parquet files and SOURCES.csv of the dataset repository
  audit    checks a sample of the masked rows independently

Every stage reads its input once, works on all CPU cores and writes rows in input order. The NER model of the mask
stage is the only slow part; its output is cached per text (WORK/ner_cache), so only texts it has not seen go
through it again.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))     # python -I leaves the script's directory off sys.path

from poolkit import engine  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def _part(s):
    k, n = (int(x) for x in s.split("/"))
    if not 0 <= k < n:
        raise argparse.ArgumentTypeError("use k/N with 0 <= k < N")
    return k, n


def _flag(s):
    name, _, path = s.partition("=")
    if not name or not path:
        raise argparse.ArgumentTypeError("use NAME=FILE")
    return name, path


def parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=os.path.join(HERE, "pool.json"), help="the config file (default: pool.json next to this script)")
    common.add_argument("--workers", type=int, default=engine.default_workers(), help="worker processes (default: all cores but two)")

    bench = argparse.ArgumentParser(add_help=False)
    bench.add_argument("--bench", nargs="+", help="benchmark jsonl[.gz] files to decontaminate against (default: clean.benchmarks of the config)")
    bench.add_argument("--info-labels", nargs="*", help="benchmark labels (<file stem>[/<set>]) that are only measured, never used to drop rows")
    bench.add_argument("--no-stoplist", action="store_true", help="disable the stop-list (the plain gram rule)")

    ner = argparse.ArgumentParser(add_help=False)
    ner.add_argument("--device", default="cuda", help="where the NER model runs: cuda, cuda:1 or cpu")
    ner.add_argument("--precision", default="fp32", choices=["fp32", "fp16"],
                     help="fp32 is what the published dataset was made with; fp16 is about twice as fast and changes a few borderline spans")
    ner.add_argument("--flash", action="store_true", help="FlashDeBERTa attention (pip install flashdeberta): about twice as fast again")
    ner.add_argument("--batch", type=int, default=24, help="windows per model batch (lower it on a small or shared GPU)")
    ner.add_argument("--no-ner", action="store_true", help="regular expressions only, whatever mask.ner in the config says")

    flags = argparse.ArgumentParser(add_help=False)
    flags.add_argument("--flag", type=_flag, action="append", default=[], metavar="NAME=FILE",
                       help="add a boolean column NAME, true for the rows whose id is in FILE (one id per line, plain or gzip); repeatable")
    flags.add_argument("--shard-mb", type=int, default=200, help="approximate parquet shard size")

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", parents=[common, bench, ner, flags], help="every stage, one after the other")
    p.add_argument("--pool", required=True, help="directory with the converted train.jsonl.gz and val.jsonl.gz")
    p.add_argument("--work", required=True, help="everything is written here: clean/, clean_pii/, dataset/, audit/, ner_cache/")
    p.add_argument("--from", dest="start", default="clean", choices=["clean", "mask", "export", "audit"], help="start at this stage (the earlier ones have run)")

    p = sub.add_parser("clean", parents=[common, bench], help="drop sources, filtered rows and benchmark overlap")
    p.add_argument("--pool", required=True, help="directory with train.jsonl.gz and val.jsonl.gz")
    p.add_argument("--out", required=True, help="writes train.jsonl.gz, val.jsonl.gz, REPORT.json, REPORT.md, stoplist.tsv, dropped.tsv.gz")
    p.add_argument("--report-only", action="store_true", help="write the reports but not the rows (sensitivity runs)")

    p = sub.add_parser("mask", parents=[common, ner], help="mask personal data, drop the sources not cleared for redistribution")
    p.add_argument("--pool", required=True, help="directory with train.jsonl.gz and val.jsonl.gz (the output of clean)")
    p.add_argument("--out", required=True, help="writes train.jsonl.gz, val.jsonl.gz, pii_report.json, pii_dropped.tsv")
    p.add_argument("--cache", help="span cache directory (default: ner_cache next to --out)")
    p.add_argument("--part", type=_part, metavar="k/N",
                   help="only run the model, on the k-th of N shares of the missing texts (one process per GPU; then run once more without --part)")

    p = sub.add_parser("export", parents=[common, flags], help="write the dataset repository files")
    p.add_argument("--pool", required=True, help="directory with train.jsonl.gz and val.jsonl.gz (the output of mask)")
    p.add_argument("--out", required=True, help="the dataset repository directory")

    p = sub.add_parser("audit", parents=[common], help="check a sample of the masked rows independently")
    p.add_argument("step", choices=["sample", "regex", "spacy", "gliner", "report"])
    p.add_argument("--audit-dir", required=True, help="scratch directory of the audit")
    p.add_argument("--pool", help="sample: the masked pool directory")
    p.add_argument("--n", type=int, default=500, help="sample: rows to draw")
    p.add_argument("--seed", type=int, default=20261107, help="sample: random seed")
    p.add_argument("--device", default="cuda", help="gliner: cuda or cpu")
    p.add_argument("--threshold", type=float, default=0.3, help="gliner: span score threshold")
    return ap


def clean(cfg, a, pool, out):
    from poolkit import decontam
    c = cfg.get("clean", {})
    bench = a.bench or [os.path.join(os.path.dirname(os.path.abspath(a.config)), p) for p in c.get("benchmarks", [])]
    if not bench:
        raise engine.PoolError("give the benchmark files with --bench, or list them as clean.benchmarks in the config")
    info = a.info_labels if a.info_labels is not None else c.get("info_labels", [])
    decontam.run(cfg, pool, out, bench, info, a.workers, report_only=getattr(a, "report_only", False), no_stoplist=a.no_stoplist)


def mask(cfg, a, pool, out, cache=None):
    from poolkit import pii
    cache = cache or os.path.join(os.path.dirname(os.path.abspath(out)), "ner_cache")
    pii.run(cfg, pool, out, cache, a.workers, a.device, a.precision, a.flash, a.batch, a.no_ner, getattr(a, "part", None))


def export(cfg, a, pool, out):
    from poolkit import export as ex
    ex.run(cfg, pool, out, a.flag, a.shard_mb, a.workers)


def audit(cfg, a):
    from poolkit import audit as au
    if a.step == "sample":
        if not a.pool:
            raise engine.PoolError("audit sample needs --pool")
        au.sample(cfg, a.pool, a.audit_dir, a.n, a.seed, a.workers)
    elif a.step == "regex":
        au.regex(a.audit_dir)
    elif a.step == "spacy":
        au.spacy(a.audit_dir)
    elif a.step == "gliner":
        au.gliner(cfg, a.audit_dir, a.device, a.threshold)
    else:
        au.report(cfg, a.audit_dir)


def run(cfg, a):
    from poolkit import audit as au
    w = lambda *p: os.path.join(a.work, *p)
    stages = ["clean", "mask", "export", "audit"]
    for stage in stages[stages.index(a.start):]:
        print(f"== {stage}", flush=True)
        if stage == "clean":
            clean(cfg, a, a.pool, w("clean"))
        elif stage == "mask":
            mask(cfg, a, w("clean"), w("clean_pii"), w("ner_cache"))
        elif stage == "export":
            export(cfg, a, w("clean_pii"), w("dataset"))
        else:                                       # the regex check; `audit spacy` and `audit gliner` are run by hand
            au.sample(cfg, w("clean_pii"), w("audit"), workers=a.workers)
            au.regex(w("audit"))
            au.report(cfg, w("audit"))
    print(f"done: dataset in {w('dataset')} (the dataset card README.md is written by hand)")


def main(argv=None):
    a = parser().parse_args(argv)
    t0 = time.time()
    try:
        cfg = engine.load_config(a.config)
        if a.cmd == "run":
            run(cfg, a)
        elif a.cmd == "audit":
            audit(cfg, a)
        else:
            {"clean": clean, "mask": mask, "export": export}[a.cmd](cfg, a, a.pool, a.out, *([a.cache] if a.cmd == "mask" else []))
    except engine.PoolError as e:
        sys.exit(f"pool.py {a.cmd}: {e}")
    if a.cmd == "run":
        print(f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
