"""v0.4 stage 1 (error analysis): accuracy by a field of the data rows, for several models, with the mismatched-picture
control next to it (how much each slice gains from looking at the picture).

    uv run python scripts/data/n2/slices.py data/n2/dev_sqa.jsonl skill runs/v03/eval/dev_sqa runs/r17-exist/eval/dev_sqa \
        [--min 20] [--where subject=natural\\ science]

Each model argument is the details prefix: <prefix>.details.jsonl (and <prefix>.mismatched.details.jsonl if present).
`field` is a key of the data rows (e.g. ScienceQA's skill, a bench's kind, a count's answer_key); slices with fewer
than --min questions are folded into "(other)". Prints one table, slices sorted by size.
"""
import argparse
import json
import os
from collections import defaultdict
from typing import Dict


def correct(d: dict) -> float:
    g = d["gold"]
    best = max(g.values())
    return float(g.get(d["prediction"], 0.0) == best and best > 0)


def load(prefix: str, suffix: str) -> Dict[str, float]:
    path = "%s%s.details.jsonl" % (prefix, suffix)
    if not os.path.exists(path):
        return {}
    return {d["sample_id"]: correct(d) for d in map(json.loads, open(path))}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data")
    p.add_argument("field")
    p.add_argument("models", nargs="+")
    p.add_argument("--min", type=int, default=20)
    p.add_argument("--where", action="append", default=[], help="key=value filter on the data rows")
    a = p.parse_args(argv)
    rows = [json.loads(l) for l in open(a.data) if l.strip()]
    for w in a.where:
        k, v = w.split("=", 1)
        rows = [r for r in rows if str(r.get(k)) == v]
    key = {r["id"]: str(r.get(a.field)) for r in rows}
    size = defaultdict(int)
    for v in key.values():
        size[v] += 1
    fold = {v: (v if n >= a.min else "(other)") for v, n in size.items()}
    res = [(m, load(m, ""), load(m, ".mismatched")) for m in a.models]
    groups = defaultdict(list)
    for sid, v in key.items():
        groups[fold[v]].append(sid)
    names = [os.path.basename(os.path.dirname(os.path.dirname(m))) or m for m in a.models]
    head = "%-48s %5s" % (a.field, "n") + "".join(" %14s" % n[:14] for n in names)
    print(head + "   (accuracy / mismatched)")
    for g in sorted(groups, key=lambda g: -len(groups[g])) + ["ALL"]:
        ids = list(key) if g == "ALL" else groups[g]
        cells = []
        for _, acc, mis in res:
            s = [acc[i] for i in ids if i in acc]
            t = [mis[i] for i in ids if i in mis]
            cells.append(" %6s / %5s" % ("%.3f" % (sum(s) / len(s)) if s else "-", "%.2f" % (sum(t) / len(t)) if t else "-"))
        print("%-48s %5d" % (g[:48], len(ids)) + "".join(cells))


if __name__ == "__main__":
    main()
