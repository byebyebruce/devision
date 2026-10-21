"""exp27 readout (docs/experiments/2026-10-17-exp27-transfer-plan.md, rules written before running): exp26's
step-5000 weights against v0.3 on the same questions (devision-eval records in runs/exp27-transfer/eval/<model>/).

  - transfer (the decision): ScienceQA temperature + force accuracy change, weighted 68 : 114, and the change of
    the picture gain (true - fixed mismatched picture), as exp21's readout.transfer, v0.3 in place of step 0;
  - report only: each of the two forms, concentration and attract / repel accuracy, the 24 transcribed questions,
    and r15's 7 protection panels (exp26 - v0.3; a cost when <= -3 points with the upper bound < 0).

    uv run python scripts/exp27/report.py [run dir]     # -> <run>/report.json, text on stdout
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "exp21"))
from readout import accuracy, fmt_stat, transfer  # type: ignore  # noqa: E402

from devision.train.compare import load_records, paired_metric  # noqa: E402

RUN = "runs/exp27-transfer"
GUARD = ("exist", "science", "ext", "count_wide", "count_narrow", "flip", "vsr")
EXT_SOURCES = json.load(open(os.path.join(os.path.dirname(__file__), "..", "r14b", "rules.json")))["ext_sources"]
REPORT_ONLY = ("sqa_conc", "sqa_repel", "sqa_text", "sqa_blank")


def recs(run, model, name):
    rs = load_records(os.path.join(run, "eval", model, name + ".details.jsonl"))
    if name == "flip":   # a mirror image counts as its original
        rs = [dict(r, image_id=r["image_id"].removesuffix(":flip")) for r in rs]
    return rs


def build(run: str) -> dict:
    real = lambda m: {k: recs(run, m, "sqa_" + k) for k in ("temp", "force", "temp_mism", "force_mism")}  # noqa: E731
    out: dict = {"transfer": transfer(real("v03"), real("exp26"))}
    t = out["transfer"]
    out["transfer_observed"] = t["accuracy_change"]["ci95"][0] > 0 and t["gain_change"]["ci95"][0] > 0
    out["accuracy"] = {m: {n: accuracy(recs(run, m, n)) for n in ("sqa_temp", "sqa_force", "sqa_temp_mism",
                                                                  "sqa_force_mism") + REPORT_ONLY}
                       for m in ("v03", "exp26")}
    out["report_only"] = {n: paired_metric(recs(run, "v03", n), recs(run, "exp26", n), "accuracy") for n in REPORT_ONLY}
    out["guard"] = {}
    for p in GUARD:
        a, b = recs(run, "v03", p), recs(run, "exp26", p)
        g = {"accuracy": paired_metric(a, b, "accuracy")}
        if p == "ext":
            g["macro_accuracy"] = paired_metric(a, b, "macro_accuracy", sources=EXT_SOURCES)
        out["guard"][p] = g
    def cost(p):
        m = out["guard"][p]["macro_accuracy" if p == "ext" else "accuracy"]
        return m["difference"] * 100 <= -3 and m["ci95"][1] < 0
    out["costs"] = [p for p in GUARD if cost(p)]
    if out["transfer_observed"]:
        out["outcome"] = "transfer observed: temperature / force accuracy and picture gain both up against v0.3"
    elif t["accuracy_change"]["ci95"][0] > 0:
        out["outcome"] = "accuracy up against v0.3, but a larger picture gain is not shown"
    else:
        out["outcome"] = "no transfer shown on temperature / force"
    return out


def render(out: dict) -> str:
    t = out["transfer"]
    pct = lambda d: "%+.1f [%+.1f, %+.1f]" % (100 * d["value"], 100 * d["ci95"][0], 100 * d["ci95"][1])  # noqa: E731
    lines = ["transfer (68:114 temp/force, exp26 - v0.3): accuracy %s, picture gain %s" % (
        pct(t["accuracy_change"]), pct(t["gain_change"]))]
    for k in ("temp", "force"):
        lines.append("  %s (%d q, %d pictures): accuracy %s, picture gain %s | v0.3 %s / mism %s, exp26 %s / mism %s" % (
            k, t[k]["questions"], t[k]["pictures"], pct(t[k]["accuracy_change"]), pct(t[k]["gain_change"]),
            fmt_stat(out["accuracy"]["v03"]["sqa_" + k]), fmt_stat(out["accuracy"]["v03"]["sqa_%s_mism" % k]),
            fmt_stat(out["accuracy"]["exp26"]["sqa_" + k]), fmt_stat(out["accuracy"]["exp26"]["sqa_%s_mism" % k])))
    for n, d in out["report_only"].items():
        lines.append("report only %-10s v0.3 %s, exp26 %s, change %+.1f [%+.1f, %+.1f]" % (
            n, fmt_stat(out["accuracy"]["v03"][n]), fmt_stat(out["accuracy"]["exp26"][n]),
            100 * d["difference"], 100 * d["ci95"][0], 100 * d["ci95"][1]))
    for p, g in out["guard"].items():
        m = g.get("macro_accuracy", g["accuracy"])
        lines.append("guard %-12s %+.2f [%+.2f, %+.2f]" % (p, 100 * m["difference"], 100 * m["ci95"][0], 100 * m["ci95"][1]))
    lines.append("costs: %s" % (out["costs"] or "none"))
    lines.append("transfer observed: %s" % out["transfer_observed"])
    lines.append("outcome: " + out["outcome"])
    return "\n".join(lines)


def main(argv=None):
    run = (argv if argv is not None else sys.argv[1:] or [RUN])[0]
    out = build(run)
    json.dump(out, open(os.path.join(run, "report.json"), "w"), indent=1)
    print(render(out))


if __name__ == "__main__":
    main()
