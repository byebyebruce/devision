"""exp14 B - A at the last monitor step (docs/experiments/2026-10-12-exp14-pixmo.md).

    uv run python scripts/exp14/report.py [--step 2000]
"""
import argparse
import json
import os
import sys

from devision.train.compare import load_records, paired_metric

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "r13b"))
RULES = json.load(open(os.path.join(os.path.dirname(__file__), "..", "r13b", "rules.json")))
CHECKS = [("px_wide", "accuracy"), ("px_narrow", "accuracy"), ("px_exist", "balanced_accuracy"),
          ("count_narrow", "accuracy"), ("count_wide", "accuracy"),
          ("exist", "balanced_accuracy"), ("exist", "recall"), ("science", "accuracy"), ("bind_gqa", "accuracy"),
          ("ext", "macro_accuracy"),
          ("px_wide", "nll"), ("count_narrow", "nll"), ("science", "nll")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--step", type=int, default=2000)
    p.add_argument("--arm", default="exp14-pixmo", help="run compared with exp14-base (e.g. exp15-bind)")
    a = p.parse_args()
    d = lambda arm, step: "runs/%s/stage2/monitor/step%d" % (arm, step)
    out = []
    for s, m in CHECKS:
        r = paired_metric(load_records(os.path.join(d("exp14-base", a.step), s + ".details.jsonl")),
                          load_records(os.path.join(d(a.arm, a.step), s + ".details.jsonl")), m,
                          sources=RULES["ext_sources"] if m == "macro_accuracy" else None)
        v12 = load_records(os.path.join(d("exp14-base", 0), s + ".details.jsonl"))
        r["v12"] = paired_metric(v12, v12, m, sources=RULES["ext_sources"] if m == "macro_accuracy" else None)["a"]
        out.append(dict(r, set=s))
        print("%-13s %-18s v12 %.4f  A %.4f  B %.4f  B-A %+.4f [%+.4f, %+.4f]" % (
            s, m, r["v12"], r["a"], r["b"], r["difference"], r["ci95"][0], r["ci95"][1]))
    json.dump(out, open("runs/%s/compare_base.json" % a.arm, "w"), indent=1)


if __name__ == "__main__":
    main()
