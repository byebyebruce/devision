"""exp19 readout (docs/experiments/2026-10-14-exp19-vision-plan.md, rules written before training), from the
temperature-1 monitor records at step 2000: D (top SigLIP layers trained) against B (exp16-r14b-s1, reused) and
against v12 (step 0).

    uv run python scripts/exp19/report.py      # -> runs/exp19-vision/report.json, text on stdout

D's step 0 must match B's (the training already checked this before its first update); otherwise no verdict and
exit status 2. Any other failure also exits non-zero.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

from devision.train.compare import load_records, paired_metric, same_start

D = "runs/exp19-vision/stage2/monitor"
B = "runs/exp16-r14b-s1/stage2/monitor"
TARGET = ("science", "flip", "vsr")                               # P1: what a trained tower should help
GUARD = ("exist", "bind_gqa", "ext", "count_narrow", "count_wide")   # P2: no clear early loss
REPORT = ("px_wide", "px_narrow", "px_exist")
EXT_SOURCES = json.load(open(os.path.join(os.path.dirname(__file__), "..", "r14b", "rules.json")))["ext_sources"]
TOLERANCE = 1e-5   # the same as configs/exp19-vision.yaml's start_tolerance


def recs(root, step, panel, by_original=False):
    rs = load_records(os.path.join(root, "step%d" % step, panel + ".details.jsonl"))
    return [dict(r, image_id=picture(r)) for r in rs] if by_original else rs


def picture(r):
    """Resampling unit: a picture, its mirror image counted with it (flip panel)."""
    return r["image_id"].removesuffix(":flip")


def macro(a_dir, b_dir, panels, field, resamples=2000, seed=0):
    """Equal-weight mean over `panels` of the (b - a) mean of `field` ("correct" or "nll"); pictures resampled
    jointly across the panels, a mirror image with its original."""
    cells = defaultdict(lambda: np.zeros((len(panels), 2)))
    for j, panel in enumerate(panels):
        a = {(r["sample_id"], r["qid"]): r for r in recs(*a_dir, panel)}
        b = recs(*b_dir, panel)
        if len(a) != len(b):
            raise ValueError("%s: the two runs hold different questions" % panel)
        for r in b:
            cells[picture(r)][j] += (float(r[field]) - float(a[(r["sample_id"], r["qid"])][field]), 1)
    m = np.stack(list(cells.values()))

    def value(w):
        return float(np.mean((w[:, None] * m[:, :, 0]).sum(0) / (w[:, None] * m[:, :, 1]).sum(0)))

    w = np.random.default_rng(seed).multinomial(len(m), [1 / len(m)] * len(m), size=resamples)
    boot = [value(x) for x in w]
    return {"difference": value(np.ones(len(m))), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "pictures": len(m), "panels": list(panels)}


def pm(a_dir, b_dir, panel, metric, by_original=False):
    return paired_metric(recs(*a_dir, panel, by_original), recs(*b_dir, panel, by_original), metric,
                         sources=EXT_SOURCES if metric == "macro_accuracy" else None)


def pts(r):
    return tuple(100 * x for x in (r["difference"], *r["ci95"]))


def main():
    try:
        out = {"step0": same_start(os.path.join(D, "step0"), os.path.join(B, "step0"), TOLERANCE)}
    except ValueError as e:
        print("D and B do not start from the same model: no verdict (%s)" % e)
        sys.exit(2)
    print("step 0, D vs B, largest probability difference: %.2e" % out["step0"]["largest"])
    d, b, v12 = (D, 2000), (B, 2000), (D, 0)

    p1 = macro(b, d, TARGET, "correct")
    guard = {panel: pm(b, d, panel, "accuracy") for panel in GUARD}
    hurt = [panel for panel, r in guard.items() if r["difference"] <= -0.03 and r["ci95"][1] < 0]
    out.update(P1_target_accuracy_D_minus_B=p1, P2_guard_D_minus_B=guard, P2_clear_losses=hurt)
    print("P1 target accuracy, D - B (equal-weight mean of %s): %+.2f [%+.2f, %+.2f] points" % (", ".join(TARGET), *pts(p1)))
    for panel, r in guard.items():
        print("P2 %-13s accuracy D - B: %+.2f [%+.2f, %+.2f]" % (panel, *pts(r)))
    print("P2 clear early losses (<= -3 points with upper bound < 0):", hurt or "none")

    if p1["ci95"][0] > 0 and not hurt:
        verdict = "effective"
    elif p1["difference"] <= 0:
        verdict = "not effective"
    else:
        verdict = "inconclusive"
    out["verdict"] = verdict
    print("verdict:", verdict)
    print("(P1 is the equal-weight mean accuracy of the three target panels against the frozen run, at step 2000;"
          " not each panel separately, and not the end of a full run)")

    out["report_only"] = {"nll_all_panels_D_minus_B": macro(b, d, TARGET + GUARD, "nll")}
    r = out["report_only"]["nll_all_panels_D_minus_B"]
    print("\nreport only: equal-weight mean NLL of all 8 panels, D - B: %+.4f [%+.4f, %+.4f]" % (r["difference"], *r["ci95"]))
    print("per panel (accuracy in points, NLL raw; temperature 1):")
    for panel in TARGET + GUARD + REPORT:
        row = {name: {m: pm(base, d, panel, m) for m in ("accuracy", "nll")} for name, base in (("D-B", b), ("D-v12", v12))}
        if panel == "flip":
            for name, base in (("D-B", b), ("D-v12", v12)):
                row[name]["accuracy_by_original"] = pm(base, d, panel, "accuracy", by_original=True)
        out["report_only"][panel] = row
        print("  %-13s " % panel + "  ".join("%s acc %+.2f [%+.2f, %+.2f] nll %+.4f [%+.4f, %+.4f]" % (
            name, *pts(x["accuracy"]), x["nll"]["difference"], *x["nll"]["ci95"]) for name, x in row.items()))
    os.makedirs("runs/exp19-vision", exist_ok=True)
    json.dump(out, open("runs/exp19-vision/report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
