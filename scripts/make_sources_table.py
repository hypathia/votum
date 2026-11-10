"""Write the table of all sources into the dataset card, between the two `sources` markers, from the dataset's SOURCES.csv.

    python scripts/make_sources_table.py <Votum-Pool directory>
"""
import csv, sys
from pathlib import Path

D = Path(sys.argv[1]); START, END = "<!-- sources:start -->", "<!-- sources:end -->"
rows = list(csv.DictReader(open(D / "SOURCES.csv", encoding="utf-8", newline="")))
cell = lambda s: s.replace("|", "·").replace("\n", " ")
out = []
for cfg, title in (("default", "Config `default`"), ("share_alike", "Config `share_alike`")):
    rs = sorted((r for r in rows if r["config"] == cfg), key=lambda r: r["source"].lower())
    n = sum(int(r["train_rows"]) + int(r["validation_rows"]) for r in rs)
    out += [f"**{title}: {len(rs)} sources, {n:,} rows**", "", "| Source | What it is | Licence | Rows |", "|---|---|---|---:|"]
    for r in rs:
        name = f'[{cell(r["source"])}]({r["upstream_release"]})' if r["upstream_release"].startswith("http") else cell(r["source"])
        mark = " †" if r["origin"] == "self-collected" else ""
        out.append(f'| {name}{mark} | {cell(r["description"])} | {cell(r["licence_tags"])} | {int(r["train_rows"]) + int(r["validation_rows"]):,} |')
    out.append("")
p = D / "README.md"; s = p.read_text(encoding="utf-8"); a, b = s.index(START) + len(START), s.index(END)
p.write_text(s[:a] + "\n\n" + "\n".join(out) + "\n" + s[b:], encoding="utf-8")
print(len(rows), "sources written to", p)
