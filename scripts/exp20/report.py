"""exp20 readout (docs/experiments/2026-10-15-exp20-tokens256-plan.md, rules written before training), from the
temperature-1 monitor records at step 2000: D (256 visual tokens, folded projector) against B = post-r15-order-a
(64 tokens, the same data, order, seed and settings; reused), plus D's step 0 against r15's (what the fold costs
before training) and the CPU latency of both token counts.

    uv run python scripts/exp20/report.py      # -> runs/exp20-tokens256/report.json, text on stdout

The two runs must hold the same questions on every panel; any failure exits non-zero.
"""
import json
import os
from collections import defaultdict

import numpy as np

from devision.train.compare import load_records, paired_metric

D = "runs/exp20-tokens256/stage2/monitor"
B = "runs/post-r15-order-a/stage2/monitor"
LATENCY = "runs/exp20-tokens256/latency"
TARGET = ("science", "count_wide", "count_narrow", "flip")       # P1: details a 4x finer token grid should help
GUARD = ("exist", "ext", "count_wide_reverse", "count_narrow_reverse")   # P2: no clear loss
REPORT = ()
GUARD_METRIC = {"ext": "macro_accuracy"}   # 13 sources weighted equally (critic EXP20-N02, fixed before running)
EXT_SOURCES = json.load(open(os.path.join(os.path.dirname(__file__), "..", "r14b", "rules.json")))["ext_sources"]


def recs(root, step, panel):
    """A panel's records with the resampling unit as image_id: a mirror image counts as its original (critic
    EXP20-I01), for every statistic of this report."""
    rs = load_records(os.path.join(root, "step%d" % step, panel + ".details.jsonl"))
    return [dict(r, image_id=picture(r)) for r in rs]


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


def pm(a_dir, b_dir, panel, metric):
    return paired_metric(recs(*a_dir, panel), recs(*b_dir, panel), metric,
                         sources=EXT_SOURCES if metric == "macro_accuracy" else None)


def pts(r):
    return tuple(100 * x for x in (r["difference"], *r["ci95"]))


def main():
    out = {}
    lat = {}
    for name in ("r15", "tokens256"):
        r = json.load(open(os.path.join(LATENCY, name + ".json")))
        lat[name] = {k: r["request_latency_ms"][k] for k in ("p50", "p95")}
    out["cpu_latency_ms"] = lat
    print("CPU latency, one question per request, 4 threads: r15 P50 %.1f / P95 %.1f ms; 256 tokens P50 %.1f / P95 %.1f ms"
          % (lat["r15"]["p50"], lat["r15"]["p95"], lat["tokens256"]["p50"], lat["tokens256"]["p95"]))
    fold = macro((B, 0), (D, 0), TARGET + GUARD, "correct")
    out["fold_step0_D_minus_r15"] = fold
    print("before training, 256 tokens - r15 (equal-weight accuracy of all 8 panels): %+.2f [%+.2f, %+.2f] points" % pts(fold))
    d, b, r15 = (D, 2000), (B, 2000), (B, 0)

    p1 = macro(b, d, TARGET, "correct")
    guard = {panel: pm(b, d, panel, GUARD_METRIC.get(panel, "accuracy")) for panel in GUARD}
    hurt = [panel for panel, r in guard.items() if r["difference"] <= -0.03 and r["ci95"][1] < 0]
    out.update(P1_target_accuracy_D_minus_B=p1, P2_guard_D_minus_B=guard, P2_clear_losses=hurt)
    print("P1 target accuracy, D - B (equal-weight mean of %s): %+.2f [%+.2f, %+.2f] points" % (", ".join(TARGET), *pts(p1)))
    for panel, r in guard.items():
        print("P2 %-20s %s D - B: %+.2f [%+.2f, %+.2f]" % (panel, GUARD_METRIC.get(panel, "accuracy"), *pts(r)))
    print("P2 clear early losses (<= -3 points with upper bound < 0):", hurt or "none")

    if p1["ci95"][0] > 0 and not hurt:
        verdict = "effective"
    elif p1["difference"] <= 0:
        verdict = "not effective"
    else:
        verdict = "inconclusive"
    out["verdict"] = verdict
    print("verdict:", verdict)
    print("(P1 is the equal-weight mean accuracy of the four target panels against the 64-token run on the same data,"
          " at step 2000; not each panel separately, and not the end of a full run)")

    out["report_only"] = {"nll_all_panels_D_minus_B": macro(b, d, TARGET + GUARD, "nll")}
    r = out["report_only"]["nll_all_panels_D_minus_B"]
    print("\nreport only: equal-weight mean NLL of all 8 panels, D - B: %+.4f [%+.4f, %+.4f]" % (r["difference"], *r["ci95"]))
    print("per panel (accuracy in points, NLL raw; temperature 1):")
    for panel in TARGET + GUARD + REPORT:
        row = {name: {m: pm(base, d, panel, m) for m in ("accuracy", "nll")} for name, base in (("D-B", b), ("D-r15", r15))}
        if panel == "ext":
            row["D-B"]["macro_accuracy"] = pm(b, d, panel, "macro_accuracy")
        out["report_only"][panel] = row
        print("  %-20s (%d pictures) " % (panel, row["D-B"]["accuracy"]["pictures"]) + "  ".join(
            "%s acc %+.2f [%+.2f, %+.2f] nll %+.4f [%+.4f, %+.4f]" % (
                name, *pts(x["accuracy"]), x["nll"]["difference"], *x["nll"]["ci95"]) for name, x in row.items()))
    os.makedirs("runs/exp20-tokens256", exist_ok=True)
    json.dump(out, open("runs/exp20-tokens256/report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
