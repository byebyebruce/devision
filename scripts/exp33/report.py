"""exp33 readout (docs/experiments/2026-10-17-exp33-decision-protect-plan.md, rules written before running): the
alignment-with-decisions checkpoint against v0.3 (exp27's records) and against exp30 (the same alignment without
decision supervision), on the same questions.

  - reading kept: exp24's gate (real - shuffled >= 0.30 on the held-out text pictures);
  - protection: the 7 panels against v0.3 (a cost when <= -3 points with the upper bound < 0; ext 13-source macro,
    flip mirrors with their originals), and each panel against exp30;
  - outcome: gate fails -> "decision supervision pushed out the reading gate"; no pre-written cost against v0.3 ->
    "passes this protection screen"; science, count_wide and flip all above exp30 (lower bounds > 0) but costs
    remain -> "improves on exp30, pre-written costs remain"; otherwise -> "direct decision supervision in
    alignment did not stop the costs". Screening labels: no cost found is not equivalence.

    uv run python scripts/exp33/report.py [run dir]
"""
import importlib.util
import json
import os
import sys

_spec = importlib.util.spec_from_file_location("exp30_report", os.path.join(os.path.dirname(__file__), "..", "exp30", "report.py"))
assert _spec is not None and _spec.loader is not None
_exp30 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_exp30)
GUARD, metric, recs = _exp30.GUARD, _exp30.metric, _exp30.recs

RUN = "runs/exp33-align-decide"
BASE = {"v03": "runs/exp27-transfer/eval/v03", "exp30": "runs/exp30-align-replay/eval"}
FOCUS = ("science", "count_wide", "flip")


def build(run: str) -> dict:
    gate = json.load(open(os.path.join(run, "stage1", "align_report.json")))["val"][-1]
    out: dict = {"gate_gap": gate["val/mlm_acc_gap"], "reading_kept": round(gate["val/mlm_acc_gap"], 9) >= 0.30,
                 "vs_v03": {}, "vs_exp30": {}}
    ev = os.path.join(run, "eval")
    for p in GUARD:
        for base in ("v03", "exp30"):
            m = metric(recs(BASE[base], p), recs(ev, p), p)
            out["vs_" + base][p] = {"difference": m["difference"], "ci95": m["ci95"]}
    out["costs"] = [p for p, d in out["vs_v03"].items() if d["difference"] * 100 <= -3 and d["ci95"][1] < 0]
    better = all(out["vs_exp30"][p]["ci95"][0] > 0 for p in FOCUS)
    if not out["reading_kept"]:
        out["outcome"] = "decision supervision pushed out the reading gate"
    elif not out["costs"]:
        out["outcome"] = "passes this protection screen (no pre-written cost against v0.3)"
    elif better:
        out["outcome"] = "improves on exp30, pre-written costs remain"
    else:
        out["outcome"] = "direct decision supervision in alignment did not stop the costs"
    out["sqa"] = {k: sum(r["correct"] for r in recs(ev, k)) / len(recs(ev, k)) for k in ("sqa_temp", "sqa_force", "sqa_conc")}
    return out


def main(argv=None):
    run = (argv if argv is not None else sys.argv[1:] or [RUN])[0]
    out = build(run)
    json.dump(out, open(os.path.join(run, "report.json"), "w"), indent=1)
    print("reading gate gap %.3f -> %s" % (out["gate_gap"], "passed" if out["reading_kept"] else "not passed"))
    for p in GUARD:
        a, b = out["vs_v03"][p], out["vs_exp30"][p]
        print("%-12s vs v0.3 %+.2f [%+.2f, %+.2f] | vs exp30 %+.2f [%+.2f, %+.2f]" % (
            p, 100 * a["difference"], 100 * a["ci95"][0], 100 * a["ci95"][1], 100 * b["difference"],
            100 * b["ci95"][0], 100 * b["ci95"][1]))
    print("costs vs v0.3: %s" % (out["costs"] or "none"))
    print("ScienceQA dev (report only): %s" % {k: round(v, 3) for k, v in out["sqa"].items()})
    print("outcome: " + out["outcome"])


if __name__ == "__main__":
    main()
