"""exp18 readout (docs/experiments/2026-10-13-exp18-distill-plan.md, rules written before training), from the
temperature-1 monitor records at step 2000: D (teacher term) against B (exp16-r14-s1, reused) and against v12
(step 0).

    uv run python scripts/exp18/report.py      # -> runs/exp18-distill/report.json, text on stdout

Before anything, D's step 0 must match B's step 0 (both are v12): otherwise B is not D's baseline and no verdict
is given.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

from devision.train.compare import load_records, paired_metric

D = "runs/exp18-distill/stage2/monitor"
B = "runs/exp16-r14-s1/stage2/monitor"
OLD = ("exist", "science", "bind_gqa", "ext", "count_narrow", "flip", "vsr")
NEW = "count_wide"
REPORT = ("px_wide", "px_narrow", "px_exist")
EXT_SOURCES = json.load(open(os.path.join(os.path.dirname(__file__), "..", "r14b", "rules.json")))["ext_sources"]
STEP0_TOLERANCE = 1e-5   # MPS float differences between runs are ~1e-7


def recs(root, step, panel):
    return load_records(os.path.join(root, "step%d" % step, panel + ".details.jsonl"))


def picture(r):
    """Resampling unit: a picture, its mirror image counted with it (flip panel)."""
    return r["image_id"].removesuffix(":flip")


def step0_mismatch():
    worst = 0.0
    for panel in OLD + (NEW,) + REPORT:
        a = {(r["sample_id"], r["qid"]): r for r in recs(B, 0, panel)}
        for r in recs(D, 0, panel):
            x = a[(r["sample_id"], r["qid"])]
            worst = max(worst, max(abs(r["probabilities"][k] - x["probabilities"][k]) for k in r["probabilities"]))
    return worst


def macro_nll(a_dir, b_dir, resamples=2000, seed=0):
    """Equal-weight mean over the old panels of (b - a) mean NLL; pictures resampled jointly across the panels."""
    cells = defaultdict(lambda: np.zeros((len(OLD), 2)))   # picture -> per panel [sum of NLL differences, count]
    for j, panel in enumerate(OLD):
        a = {(r["sample_id"], r["qid"]): r for r in recs(*a_dir, panel)}
        b = recs(*b_dir, panel)
        if len(a) != len(b):
            raise ValueError("%s: the two runs hold different questions" % panel)
        for r in b:
            cells[picture(r)][j] += (r["nll"] - a[(r["sample_id"], r["qid"])]["nll"], 1)
    m = np.stack(list(cells.values()))             # pictures x panels x 2

    def value(w):
        s = (w[:, None] * m[:, :, 0]).sum(0)
        n = (w[:, None] * m[:, :, 1]).sum(0)
        return float(np.mean(s / n))

    w = np.random.default_rng(seed).multinomial(len(m), [1 / len(m)] * len(m), size=resamples)
    boot = sorted(value(x) for x in w)
    return {"difference": value(np.ones(len(m))), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "pictures": len(m), "panels": list(OLD)}


def pm(a_dir, b_dir, panel, metric):
    return paired_metric(recs(*a_dir, panel), recs(*b_dir, panel), metric,
                         sources=EXT_SOURCES if metric == "macro_accuracy" else None)


def main():
    out = {"step0_max_abs_difference": step0_mismatch()}
    print("step 0, D vs B, largest probability difference: %.2e" % out["step0_max_abs_difference"])
    if out["step0_max_abs_difference"] > STEP0_TOLERANCE:
        print("D and B do not start from the same model: no verdict")
        sys.exit(2)
    d, b, v12 = (D, 2000), (B, 2000), (D, 0)

    p1 = macro_nll(b, d)
    gain = pm(v12, d, NEW, "accuracy")
    keep = pm(b, d, NEW, "accuracy")
    out["P1_old_nll_D_minus_B"], out["P2_count_wide_D_minus_v12"], out["P2_count_wide_D_minus_B"] = p1, gain, keep
    print("P1 old-panel NLL, D - B (equal-weight mean of %d panels): %+.4f [%+.4f, %+.4f]"
          % (len(OLD), p1["difference"], *p1["ci95"]))
    print("P2 count_wide accuracy, D - v12: %+.2f [%+.2f, %+.2f] points; D - B: %+.2f [%+.2f, %+.2f]"
          % tuple(100 * x for x in (gain["difference"], *gain["ci95"], keep["difference"], *keep["ci95"])))

    ok1 = p1["ci95"][1] < 0
    ok2 = gain["ci95"][0] > 0 and keep["ci95"][0] > -0.05
    if ok1 and ok2:
        verdict = "effective"
    elif p1["difference"] >= 0 or keep["ci95"][1] < -0.05:
        verdict = "not effective"
    else:
        verdict = "inconclusive"
    out["verdict"] = verdict
    print("verdict:", verdict)

    out["report_only"] = {}
    print("\nreport only (accuracy in points, NLL raw; temperature 1):")
    for panel in OLD + (NEW,) + REPORT:
        row = {}
        for name, base in (("D-B", b), ("D-v12", v12)):
            row[name] = {m: pm(base, d, panel, m) for m in ("accuracy", "nll")}
        if panel == "ext":
            row["D-B"]["macro_accuracy"] = pm(b, d, panel, "macro_accuracy")
        if panel == "exist":
            row["D-B"]["recall"] = pm(b, d, panel, "recall")
        out["report_only"][panel] = row
        print("  %-13s " % panel + "  ".join(
            "%s acc %+.2f [%+.2f, %+.2f] nll %+.4f [%+.4f, %+.4f]" % (
                name, *(100 * x for x in (r["accuracy"]["difference"], *r["accuracy"]["ci95"])),
                r["nll"]["difference"], *r["nll"]["ci95"]) for name, r in row.items()))
    os.makedirs("runs/exp18-distill", exist_ok=True)
    json.dump(out, open("runs/exp18-distill/report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
