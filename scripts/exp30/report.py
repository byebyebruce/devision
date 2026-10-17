"""exp30 readout (docs/experiments/2026-10-17-exp30-align-replay-plan.md, rules written before running): the
alignment-with-replay checkpoint against v0.3 (exp27's records) and against exp24 phase A (exp28's records, the
same alignment without replay), on the same questions.

  - reading kept: exp24's gate on the held-out text pictures (mixed caption completion, real - shuffled >= 0.30);
  - protection: the 7 panels against v0.3 (a cost when <= -3 points with the upper bound < 0; ext 13-source macro,
    flip mirrors with their originals) and exist / science against exp24A;
  - outcome: "replay protects" = reading kept and no cost against v0.3; "replay helps, not enough" = reading kept,
    exist and science both above exp24A (lower bounds > 0), but costs remain; "replay does not help" otherwise
    (or "reading lost" when the gate fails).

    uv run python scripts/exp30/report.py [run dir]
"""
import json
import os
import sys

from devision.train.compare import load_records, paired_metric

RUN = "runs/exp30-align-replay"
GUARD = ("exist", "science", "ext", "count_wide", "count_narrow", "flip", "vsr")
EXT_SOURCES = json.load(open(os.path.join(os.path.dirname(__file__), "..", "r14b", "rules.json")))["ext_sources"]
BASE = {"v03": "runs/exp27-transfer/eval/v03", "exp24A": "runs/exp28-diagnose/eval/exp24A"}


def recs(d, name):
    rs = load_records(os.path.join(d, name + ".details.jsonl"))
    return [dict(r, image_id=r["image_id"].removesuffix(":flip")) for r in rs] if name == "flip" else rs


def metric(a, b, p):
    return paired_metric(a, b, "macro_accuracy", sources=EXT_SOURCES) if p == "ext" else paired_metric(a, b, "accuracy")


def build(run: str) -> dict:
    gate = json.load(open(os.path.join(run, "stage1", "align_report.json")))["val"][-1]
    out: dict = {"gate_gap": gate["val/mlm_acc_gap"], "reading_kept": round(gate["val/mlm_acc_gap"], 9) >= 0.30,
                 "vs_v03": {}, "vs_exp24A": {}}
    ev = os.path.join(run, "eval")
    for p in GUARD:
        m = metric(recs(BASE["v03"], p), recs(ev, p), p)
        out["vs_v03"][p] = {"difference": m["difference"], "ci95": m["ci95"]}
        m = metric(recs(BASE["exp24A"], p), recs(ev, p), p)
        out["vs_exp24A"][p] = {"difference": m["difference"], "ci95": m["ci95"]}
    out["costs"] = [p for p, d in out["vs_v03"].items() if d["difference"] * 100 <= -3 and d["ci95"][1] < 0]
    better = all(out["vs_exp24A"][p]["ci95"][0] > 0 for p in ("exist", "science"))
    if not out["reading_kept"]:
        out["outcome"] = "reading lost: the gate fails"
    elif not out["costs"]:
        out["outcome"] = "replay protects: reading kept and no cost against v0.3"
    elif better:
        out["outcome"] = "replay helps, not enough: exist and science above exp24A, costs remain"
    else:
        out["outcome"] = "replay does not help"
    out["sqa"] = {k: sum(r["correct"] for r in recs(ev, k)) / len(recs(ev, k)) for k in ("sqa_temp", "sqa_force", "sqa_conc")}
    return out


def main(argv=None):
    run = (argv if argv is not None else sys.argv[1:] or [RUN])[0]
    out = build(run)
    json.dump(out, open(os.path.join(run, "report.json"), "w"), indent=1)
    print("reading gate gap %.3f -> %s" % (out["gate_gap"], "kept" if out["reading_kept"] else "lost"))
    for p in GUARD:
        a, b = out["vs_v03"][p], out["vs_exp24A"][p]
        print("%-12s vs v0.3 %+.2f [%+.2f, %+.2f] | vs exp24A %+.2f [%+.2f, %+.2f]" % (
            p, 100 * a["difference"], 100 * a["ci95"][0], 100 * a["ci95"][1], 100 * b["difference"],
            100 * b["ci95"][0], 100 * b["ci95"][1]))
    print("costs vs v0.3: %s" % (out["costs"] or "none"))
    print("ScienceQA dev (report only): %s" % {k: round(v, 3) for k, v in out["sqa"].items()})
    print("outcome: " + out["outcome"])


if __name__ == "__main__":
    main()
