"""exp21 readout (docs/experiments/2026-10-15-exp21-reading-plan.md section 4, rules written before running),
from the temperature-1 monitor records of runs/exp21-reading/stage2/monitor/step{0,2000} (step 0 = r15).

    uv run python scripts/exp21/report.py      # -> runs/exp21-reading/report.json, text on stdout

Reading (section 4.2): per height, read / compare / both readings / compare given both / follows the value change,
classified pass / fail / near threshold (readout.classify). Transfer (4.3): 68:114 temp / force accuracy change
and picture-gain change, step 2000 - step 0. Protection (4.4): r15's panels, a cost when a panel is <= -3 points
with its upper bound < 0. Mechanism labels are screening labels, not causal proofs (critic EXP21-R1-03).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from readout import FAIL, HEIGHTS, PASS, accuracy, classify, height_gap, syn_index, syn_stat, transfer  # noqa: E402

from devision.train.compare import load_records, paired_metric  # noqa: E402

RUN = "runs/exp21-reading"
GUARD = ("exist", "science", "ext", "count_wide", "count_narrow", "flip", "vsr")
EXT_SOURCES = json.load(open(os.path.join(os.path.dirname(__file__), "..", "r14b", "rules.json")))["ext_sources"]
WHAT = ("read", "readA", "readB", "cmp", "both_read", "cmp_given_both", "cf_read", "cf_cmp")


def recs(mon, step, panel):
    rs = load_records(os.path.join(mon, "step%d" % step, panel + ".details.jsonl"))
    if panel == "flip":   # a mirror image counts as its original
        rs = [dict(r, image_id=r["image_id"].removesuffix(":flip")) for r in rs]
    return rs


def reading_label(syn: dict, gaps: dict) -> str:
    c = {h: classify(syn[str(h)]["read"]) for h in HEIGHTS}
    cf = all(syn[str(h)]["cf_read"]["value"] >= 0.60 for h in (6, 8))
    if c[6] == PASS and c[8] == PASS and cf:
        return "small numbers readable (6 and 8 px pass, readings follow the value change)"
    if c[16] == PASS and c[6] == FAIL and c[8] == FAIL and gaps["16-8"]["ci95"][0] > 0:
        return "scale-sensitive (16 px passes, 6 and 8 px fail)"
    if c[16] == FAIL:
        return "not learned (16 px fails)"
    return "uncertain"


def compare_label(syn: dict, text_cmp: dict) -> dict:
    out = {}
    for h in HEIGHTS:
        s = syn[str(h)]
        if classify(s["read"]) != PASS:
            out[str(h)] = "reading does not pass"
        elif classify(s["cmp"]) == PASS:
            out[str(h)] = "compares"
        elif s["cmp"]["value"] < 0.60 and s["both_read"]["value"] >= 0.70 and text_cmp["value"] >= 0.80:
            out[str(h)] = "reads both but compares poorly on the picture (combination / readout)"
        else:
            out[str(h)] = "uncertain"
    return out


def build(mon: str, step: int = 2000) -> dict:
    syn = syn_index(recs(mon, step, "syn"))
    out = {"step": step,
           "syn": {str(h): {w: syn_stat(syn, h, w) for w in WHAT} for h in HEIGHTS},
           "syn_step0": {str(h): {w: syn_stat(syn_index(recs(mon, 0, "syn")), h, w) for w in ("read", "cmp")}
                         for h in HEIGHTS},
           "erased": {w: syn_stat(syn_index(recs(mon, step, "syn_erased")), None, w) for w in ("read", "cmp")},
           "text": {w: syn_stat(syn_index(recs(mon, step, "syn_text")), None, w) for w in ("read", "cmp")},
           "sqa_text": accuracy(recs(mon, step, "sqa_text")), "sqa_blank": accuracy(recs(mon, step, "sqa_blank"))}
    out["height_gap"] = {"16-8": height_gap(syn, 16, 8), "16-6": height_gap(syn, 16, 6)}
    out["valid"] = out["erased"]["read"]["value"] < 0.50 and out["erased"]["cmp"]["value"] < 0.55
    out["reading"] = reading_label(out["syn"], out["height_gap"]) if out["valid"] else "invalid: erased control"
    out["compare"] = compare_label(out["syn"], out["text"]["cmp"])
    real = lambda s: {k: recs(mon, s, "sqa_" + k) for k in ("temp", "force", "temp_mism", "force_mism")}  # noqa: E731
    out["transfer"] = transfer(real(0), real(step))
    t = out["transfer"]
    out["transfer_observed"] = t["accuracy_change"]["ci95"][0] > 0 and t["gain_change"]["ci95"][0] > 0
    out["guard"] = {}
    for p in GUARD + ("val",):
        a, b = recs(mon, 0, p), recs(mon, step, p)
        g = {"accuracy": paired_metric(a, b, "accuracy"), "nll": paired_metric(a, b, "nll")}
        if p == "ext":
            g["macro_accuracy"] = paired_metric(a, b, "macro_accuracy", sources=EXT_SOURCES)
        out["guard"][p] = g
    def cost(p):
        m = out["guard"][p]["macro_accuracy" if p == "ext" else "accuracy"]
        return m["difference"] * 100 <= -3 and m["ci95"][1] < 0
    out["costs"] = [p for p in GUARD if cost(p)]
    if out["reading"].startswith("small") and out["transfer_observed"]:
        out["next"] = "write the r16 plan (diagnostic result, not an adoptable model); costs: %s" % (out["costs"] or "none")
    else:
        out["next"] = "owner decides; candidate hypotheses only (512 / OCR alignment are not started automatically)"
    return out


def main():
    mon = os.path.join(RUN, "stage2", "monitor")
    out = build(mon)
    json.dump(out, open(os.path.join(RUN, "report.json"), "w"), indent=1)
    f = lambda s: "%.3f [%.3f, %.3f]" % (s["value"], s["ci95"][0], s["ci95"][1])  # noqa: E731
    for h in HEIGHTS:
        s = out["syn"][str(h)]
        print("h%-2d read %s (step 0 %s) | cmp %s | both read %s | cmp|both %s n=%d | cf read %s | cf cmp %s" % (
            h, f(s["read"]), f(out["syn_step0"][str(h)]["read"]), f(s["cmp"]), f(s["both_read"]),
            f(s["cmp_given_both"]), s["cmp_given_both"]["n"], f(s["cf_read"]), f(s["cf_cmp"])))
    print("read 16-8 px %s, 16-6 px %s" % (f(out["height_gap"]["16-8"]), f(out["height_gap"]["16-6"])))
    print("erased read %s cmp %s | text read %s cmp %s" % (f(out["erased"]["read"]), f(out["erased"]["cmp"]),
                                                          f(out["text"]["read"]), f(out["text"]["cmp"])))
    print("reading:", out["reading"])
    print("compare:", out["compare"])
    t = out["transfer"]
    print("transfer (68:114 temp/force): accuracy %s, picture gain %s; temp %s (%d q), force %s (%d q)" % (
        f(t["accuracy_change"]), f(t["gain_change"]), f(t["temp"]["accuracy_change"]), t["temp"]["questions"],
        f(t["force"]["accuracy_change"]), t["force"]["questions"]))
    print("ScienceQA 24 transcribed: with values %s, blank %s" % (f(out["sqa_text"]), f(out["sqa_blank"])))
    for p, g in out["guard"].items():
        m = g.get("macro_accuracy", g["accuracy"])
        print("guard %-12s %+.2f [%+.2f, %+.2f]  nll %+.3f" % (p, 100 * m["difference"], 100 * m["ci95"][0],
                                                             100 * m["ci95"][1], g["nll"]["difference"]))
    print("costs:", out["costs"] or "none")
    print("next:", out["next"])


if __name__ == "__main__":
    main()
