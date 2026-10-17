"""exp31 readout (critic EXP28-RESULT-02 item 3, EXP31-R1-01..03; rules: report only, it locates the failing
level): per model (v03, exp26B) and level, accuracy with a 95% interval resampling canonical families (one number
case each, all its pair orders and wordings together); for c, d also the strong and weak wordings separately;
accuracy on unequal and on equal cases; consistency -- a family right in both pair orders (a, b), right in both
wordings within a pair order (c, d), right in all its questions; the gold and predicted shares of the object named
("1", "2", "same"); the share of predictions in each display slot.

    uv run python scripts/exp31/report.py [run dir]
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

from devision.train.compare import load_records

RUN = "runs/exp31-text-force"
DATA = "data/exp31/text_force.jsonl"
MODELS = ("v03", "exp26B")
LEVELS = ("a-compare-numbers", "b-closer-pair", "c-rule-distance", "d-rule-size")


def boot(per_family, resamples=2000, seed=0):
    keys = sorted(per_family)
    v = np.array([per_family[k] for k in keys], dtype=float)   # [families, 2] = (right, n)
    w = np.random.default_rng(seed).multinomial(len(v), [1 / len(v)] * len(v), size=resamples)
    b = (w @ v[:, 0]) / (w @ v[:, 1])
    return {"value": float(v[:, 0].sum() / v[:, 1].sum()), "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
            "families": len(v), "n": int(v[:, 1].sum())}


def accuracy(sel, data):
    fam = defaultdict(lambda: [0, 0])
    for r in sel:
        f = data[r["sample_id"]]["family"]
        fam[f][0] += int(r["correct"])
        fam[f][1] += 1
    return boot(dict(fam)) if fam else None


def model_report(rs, data_path=DATA):
    data = {json.loads(x)["id"]: json.loads(x) for x in open(data_path)}
    out = {}
    for lv in LEVELS:
        sel = [r for r in rs if data[r["sample_id"]]["kind"] == lv]
        groups = defaultdict(list)
        for r in sel:
            groups[data[r["sample_id"]]["family"]].append(r)
        def cat(r, field):
            d = data[r["sample_id"]]
            return d["gold_cat"] if field == "gold" else d["option_cats"][r["prediction"]]
        res = {"accuracy": accuracy(sel, data),
               "unequal": accuracy([r for r in sel if data[r["sample_id"]]["gold_cat"] != "same"], data),
               "equal": accuracy([r for r in sel if data[r["sample_id"]]["gold_cat"] == "same"], data),
               "all_in_family": float(np.mean([all(x["correct"] for x in g) for g in groups.values()])),
               "gold_share": {c: float(np.mean([cat(r, "gold") == c for r in sel])) for c in ("1", "2", "same")},
               "pred_share": {c: float(np.mean([cat(r, "pred") == c for r in sel])) for c in ("1", "2", "same")},
               "slot_share": [float(np.mean([r["options"].index(r["prediction"]) == s for r in sel])) for s in range(3)]}
        if lv.startswith(("c", "d")):
            for w, words in (("strong", ("greater", "stronger")), ("weak", ("smaller", "weaker"))):
                res[w] = accuracy([r for r in sel if r["sample_id"].rsplit(":", 1)[1] in words], data)
            pairs = defaultdict(list)
            for r in sel:
                pairs[r["sample_id"].rsplit(":", 1)[0]].append(r["correct"])
            res["both_wordings"] = float(np.mean([all(v) for v in pairs.values()]))
        out[lv] = res
    return out


def build(run, data_path=DATA):
    return {m: model_report(load_records(os.path.join(run, "eval", m, "text_force.details.jsonl")), data_path) for m in MODELS}


def main(argv=None):
    run = (argv if argv is not None else sys.argv[1:] or [RUN])[0]
    out = build(run)
    json.dump(out, open(os.path.join(run, "report.json"), "w"), indent=1)
    def f(a):
        return "%.3f [%.3f, %.3f] (%d q, %d fam)" % (a["value"], a["ci95"][0], a["ci95"][1], a["n"], a["families"]) if a else "-"
    for m in MODELS:
        for lv, r in out[m].items():
            print("%-7s %-17s acc %s | unequal %s | equal %s" % (m, lv, f(r["accuracy"]), f(r["unequal"]), f(r["equal"])))
            if "strong" in r:
                print("        strong %s | weak %s | both wordings %.3f" % (f(r["strong"]), f(r["weak"]), r["both_wordings"]))
            print("        all in family %.3f | gold 1/2/same %s | pred 1/2/same %s | slots %s" % (
                r["all_in_family"], "/".join("%.2f" % v for v in r["gold_share"].values()),
                "/".join("%.2f" % v for v in r["pred_share"].values()), "/".join("%.2f" % x for x in r["slot_share"])))


if __name__ == "__main__":
    main()
