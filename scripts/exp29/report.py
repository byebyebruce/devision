"""exp29 readout (docs/experiments/2026-10-17-exp29-perm-avg-plan.md, rules written before running): v0.3's
devision-eval records of every cyclic order (runs/exp29-perm-avg/eval/<panel>.details.jsonl). Per original
question: "single" = order 0 (the original order), "average" = the mean over the k orders of each option's
probability (renormalised), argmax. Both are compared on the same questions, pictures resampled.

Decision: the equal-weight mean over the 6 panels of the accuracy change (average - single) has a lower bound
> 0, and no panel changes by <= -3 points with its upper bound < 0 -> "averaging helps"; else "does not help".

    uv run python scripts/exp29/report.py [run dir]
"""
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

from devision.train.compare import load_records

RUN = "runs/exp29-perm-avg"
PANELS = ("science", "ext", "count_wide", "count_narrow", "flip", "sqa_diagram")


def combine(records):
    """original question -> {"single": record-like, "average": record-like, "image_id", "orders"}."""
    by = defaultdict(list)
    for r in records:
        base, order = r["sample_id"].rsplit(":o", 1)
        by[(base, r["qid"])].append((int(order), r))
    out = {}
    for key, rs in by.items():
        rs.sort(key=lambda x: x[0])
        if [o for o, _ in rs] != list(range(len(rs))):
            raise ValueError("%s: orders %s" % (key, [o for o, _ in rs]))
        gold = rs[0][1]["gold"]
        mean = {k: sum(r["probabilities"][k] for _, r in rs) / len(rs) for k in gold}
        total = sum(mean.values())
        mean = {k: v / total for k, v in mean.items()}
        g = max(gold, key=gold.get)
        single = rs[0][1]
        out[key] = {"image_id": single["image_id"].removesuffix(":flip"), "orders": len(rs),
                    "single": {"correct": single["correct"], "nll": single["nll"]},
                    "average": {"correct": max(mean, key=lambda k: mean[k]) == g,
                                "nll": -sum(gv * math.log(max(mean[k], 1e-12)) for k, gv in gold.items() if gv > 0)},
                    "first_choice": [r["prediction"] == list(r["options"])[0] for _, r in rs]}
    return out


def diff(rows, field="correct", resamples=2000, seed=0):
    pics = defaultdict(lambda: [0.0, 0])
    for r in rows:
        c = pics[r["image_id"]]
        c[0] += float(r["average"][field]) - float(r["single"][field])
        c[1] += 1
    v = np.array(list(pics.values()))
    rng = np.random.default_rng(seed)
    w = rng.multinomial(len(v), [1 / len(v)] * len(v), size=resamples)
    boot = (w @ v[:, 0]) / (w @ v[:, 1])
    return {"value": float(v[:, 0].sum() / v[:, 1].sum()), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "questions": int(v[:, 1].sum()), "pictures": len(v)}


def build(run: str) -> dict:
    per = {p: list(combine(load_records(os.path.join(run, "eval", p + ".details.jsonl"))).values()) for p in PANELS}
    out: dict = {"panels": {}}
    for p, rows in per.items():
        out["panels"][p] = {"single": float(np.mean([r["single"]["correct"] for r in rows])),
                            "average": float(np.mean([r["average"]["correct"] for r in rows])),
                            "accuracy_change": diff(rows), "nll_change": diff(rows, "nll"),
                            "first_option_share": float(np.mean([x for r in rows for x in r["first_choice"]]))}
    # macro: pictures resampled jointly across panels (a picture in two panels moves with both)
    cells = defaultdict(lambda: np.zeros((len(PANELS), 2)))
    for j, p in enumerate(PANELS):
        for r in per[p]:
            cells[r["image_id"]][j] += (float(r["average"]["correct"]) - float(r["single"]["correct"]), 1)
    m = np.stack(list(cells.values()))
    value = lambda w: float(np.mean((w[:, None] * m[:, :, 0]).sum(0) / (w[:, None] * m[:, :, 1]).sum(0)))  # noqa: E731
    rng = np.random.default_rng(0)
    boot = [value(x) for x in rng.multinomial(len(m), [1 / len(m)] * len(m), size=2000)]
    out["macro_change"] = {"value": value(np.ones(len(m))), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}
    out["costs"] = [p for p, d in out["panels"].items()
                    if d["accuracy_change"]["value"] * 100 <= -3 and d["accuracy_change"]["ci95"][1] < 0]
    out["helps"] = out["macro_change"]["ci95"][0] > 0 and not out["costs"]
    return out


def main(argv=None):
    run = (argv if argv is not None else sys.argv[1:] or [RUN])[0]
    out = build(run)
    json.dump(out, open(os.path.join(run, "report.json"), "w"), indent=1)
    for p, d in out["panels"].items():
        a, n = d["accuracy_change"], d["nll_change"]
        print("%-12s single %.3f average %.3f change %+.2f [%+.2f, %+.2f] (%d q) | nll %+.3f | first-option share %.3f" % (
            p, d["single"], d["average"], 100 * a["value"], 100 * a["ci95"][0], 100 * a["ci95"][1], a["questions"],
            n["value"], d["first_option_share"]))
    mc = out["macro_change"]
    print("macro change %+.2f [%+.2f, %+.2f]; costs %s; helps: %s" % (
        100 * mc["value"], 100 * mc["ci95"][0], 100 * mc["ci95"][1], out["costs"] or "none", out["helps"]))


if __name__ == "__main__":
    main()
