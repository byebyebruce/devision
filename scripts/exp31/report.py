"""exp31 readout (critic EXP28-RESULT-02 item 3; rules: report only, it locates the failing level): per model
(v03, exp26B) and level, accuracy with a 95% interval resampling number families; consistency -- a family right
in both pair orders (a, b), right in both wordings (strong / weak) within a pair order (c, d), right in all four
(c, d); the share choosing the "equal / same" option; the share of predictions in each display slot.

    uv run python scripts/exp31/report.py [run dir]
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

from devision.train.compare import load_records

RUN = "runs/exp31-text-force"
MODELS = ("v03", "exp26B")
LEVELS = ("a-smaller-number", "b-closer-pair", "c-rule-distance", "d-rule-size")


def boot(per_family, resamples=2000, seed=0):
    keys = sorted(per_family)
    v = np.array([per_family[k] for k in keys], dtype=float)   # [families, 2] = (right, n)
    w = np.random.default_rng(seed).multinomial(len(v), [1 / len(v)] * len(v), size=resamples)
    b = (w @ v[:, 0]) / (w @ v[:, 1])
    return {"value": float(v[:, 0].sum() / v[:, 1].sum()), "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
            "families": len(v), "n": int(v[:, 1].sum())}


def is_same(option: str) -> bool:
    return "equal" in option or "same" in option


def model_report(rs):
    data = {json.loads(x)["id"]: json.loads(x) for x in open("data/exp31/text_force.jsonl")}
    out = {}
    for lv in LEVELS:
        sel = [r for r in rs if data[r["sample_id"]]["kind"] == lv]
        fam = defaultdict(lambda: [0, 0])
        groups = defaultdict(list)
        for r in sel:
            f = data[r["sample_id"]]["family"]
            fam[f][0] += int(r["correct"])
            fam[f][1] += 1
            groups[f].append(r)
        def all_right(pred):
            return float(np.mean([all(x["correct"] for x in g if pred(x)) for g in groups.values()]))
        res = {"accuracy": boot(dict(fam)), "all_in_family": all_right(lambda x: True),
               "same_share": float(np.mean([is_same(r["prediction"]) for r in sel])),
               "slot_share": [float(np.mean([r["options"].index(r["prediction"]) == s for r in sel])) for s in range(3)]}
        if lv.startswith(("c", "d")):
            pairs = defaultdict(list)
            for r in sel:
                pairs[r["sample_id"].rsplit(":", 1)[0]].append(r["correct"])
            res["both_wordings"] = float(np.mean([all(v) for v in pairs.values()]))
        out[lv] = res
    return out


def build(run):
    return {m: model_report(load_records(os.path.join(run, "eval", m, "text_force.details.jsonl"))) for m in MODELS}


def main(argv=None):
    run = (argv if argv is not None else sys.argv[1:] or [RUN])[0]
    out = build(run)
    json.dump(out, open(os.path.join(run, "report.json"), "w"), indent=1)
    for m in MODELS:
        for lv, r in out[m].items():
            a = r["accuracy"]
            print("%-7s %-17s acc %.3f [%.3f, %.3f] (%d q, %d families) | all in family %.3f | both wordings %s | same %.3f | slots %s" % (
                m, lv, a["value"], a["ci95"][0], a["ci95"][1], a["n"], a["families"], r["all_in_family"],
                "%.3f" % r["both_wordings"] if "both_wordings" in r else "  -  ", r["same_share"],
                "/".join("%.2f" % x for x in r["slot_share"])))


if __name__ == "__main__":
    main()
