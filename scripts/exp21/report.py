"""exp21 readout (docs/experiments/2026-10-15-exp21-reading-plan.md section 4, rules written before running),
from the temperature-1 monitor records of runs/exp21-reading/stage2/monitor/step{0,2000} (step 0 = r15).

    uv run python scripts/exp21/report.py [run dir]   # -> <run>/report.json, text on stdout

Reading (section 4.2): per height, read / compare / both readings / compare given both / follows the value change,
classified pass / fail / near threshold (readout.classify). Transfer (4.3): 68:114 temp / force accuracy change
and picture-gain change, step 2000 - step 0. Protection (4.4): r15's panels, a cost when a panel is <= -3 points
with its upper bound < 0. Mechanism labels are screening labels, not causal proofs (critic EXP21-R1-03).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from readout import (FAIL, HEIGHTS, PASS, accuracy, classify, fmt_stat, height_gap, paired_by_occurrence,  # noqa: E402
                     syn_index, syn_stat, transfer)

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
    # an invalid erased control voids every synthetic mechanism label, compare included (critic EXP21-I03); the
    # numbers themselves stay in the JSON
    out["compare"] = (compare_label(out["syn"], out["text"]["cmp"]) if out["valid"]
                      else {str(h): "invalid: erased control" for h in HEIGHTS})
    real = lambda s: {k: recs(mon, s, "sqa_" + k) for k in ("temp", "force", "temp_mism", "force_mism")}  # noqa: E731
    out["transfer"] = transfer(real(0), real(step))
    t = out["transfer"]
    out["transfer_observed"] = t["accuracy_change"]["ci95"][0] > 0 and t["gain_change"]["ci95"][0] > 0
    out["guard"] = {}
    for p in GUARD:
        a, b = recs(mon, 0, p), recs(mon, step, p)
        g = {"accuracy": paired_metric(a, b, "accuracy"), "nll": paired_metric(a, b, "nll")}
        if p == "ext":
            g["macro_accuracy"] = paired_metric(a, b, "macro_accuracy", sources=EXT_SOURCES)
        out["guard"][p] = g
    # val (report only) holds 242 ids twice: copies matched by occurrence, all 3,892 kept (critic EXP21-I01)
    a, b = recs(mon, 0, "val"), recs(mon, step, "val")
    out["guard"]["val"] = {"accuracy": paired_by_occurrence(a, b, "correct"), "nll": paired_by_occurrence(a, b, "nll")}
    def cost(p):
        m = out["guard"][p]["macro_accuracy" if p == "ext" else "accuracy"]
        return m["difference"] * 100 <= -3 and m["ci95"][1] < 0
    out["costs"] = [p for p in GUARD if cost(p)]
    if out["reading"].startswith("small") and out["transfer_observed"]:
        out["next"] = "write the r16 plan (diagnostic result, not an adoptable model); costs: %s" % (out["costs"] or "none")
    else:
        out["next"] = "owner decides; candidate hypotheses only (512 / OCR alignment are not started automatically)"
    return out


def render(out: dict) -> str:
    """The text report; a conditional statistic without cases prints as not estimable (critic EXP21-I02)."""
    f = lambda s: fmt_stat(s)  # noqa: E731
    g = lambda s: fmt_stat({k: v for k, v in s.items() if k != "n"})  # noqa: E731
    lines = []
    for h in HEIGHTS:
        s = out["syn"][str(h)]
        lines.append("h%-2d read %s (step 0 %s) | cmp %s | both read %s | cmp|both %s | cf read %s | cf cmp %s" % (
            h, f(s["read"]), g(out["syn_step0"][str(h)]["read"]), f(s["cmp"]), f(s["both_read"]),
            f(s["cmp_given_both"]), f(s["cf_read"]), f(s["cf_cmp"])))
    lines.append("read 16-8 px %s, 16-6 px %s" % (g(out["height_gap"]["16-8"]), g(out["height_gap"]["16-6"])))
    lines.append("erased read %s cmp %s | text read %s cmp %s" % (f(out["erased"]["read"]), f(out["erased"]["cmp"]),
                                                                 f(out["text"]["read"]), f(out["text"]["cmp"])))
    lines.append("reading: " + out["reading"])
    lines.append("compare: " + json.dumps(out["compare"]))
    t = out["transfer"]
    lines.append("transfer (68:114 temp/force): accuracy %s, picture gain %s; temp %s (%d q), force %s (%d q)" % (
        g(t["accuracy_change"]), g(t["gain_change"]), g(t["temp"]["accuracy_change"]), t["temp"]["questions"],
        g(t["force"]["accuracy_change"]), t["force"]["questions"]))
    lines.append("ScienceQA 24 transcribed: with values %s, blank %s" % (f(out["sqa_text"]), f(out["sqa_blank"])))
    for p, gd in out["guard"].items():
        m = gd.get("macro_accuracy", gd["accuracy"])
        lines.append("guard %-12s %+.2f [%+.2f, %+.2f]  nll %+.3f%s" % (
            p, 100 * m["difference"], 100 * m["ci95"][0], 100 * m["ci95"][1], gd["nll"]["difference"],
            "  (report only, %d questions)" % m["questions"] if p == "val" else ""))
    lines.append("costs: %s" % (out["costs"] or "none"))
    lines.append("next: " + out["next"])
    return "\n".join(lines)


def main(argv=None):
    run = (argv if argv is not None else sys.argv[1:] or [RUN])[0]
    out = build(os.path.join(run, "stage2", "monitor"))
    json.dump(out, open(os.path.join(run, "report.json"), "w"), indent=1)
    print(render(out))


if __name__ == "__main__":
    main()
