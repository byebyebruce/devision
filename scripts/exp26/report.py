"""exp26 phase B readout (docs/experiments/2026-10-16-exp26-bind-align-plan.md, rules written before running):
exp25's readout at step 5000 (outcome rules and thresholds unchanged) on runs/exp26-read-compare, plus, against
exp25 (same data, order, seed and settings; only the init differs):

  - the paired 16 px reading change on the same fit and dev16 questions (families resampled);
  - on pictures whose A and B values differ: the share answering "A?" and "B?" with the same option (secondary
    threshold 0.50, a measure of whether the answers separate, not of right binding; invalid when the erased
    control is), and among wrong readings the share that picked the other panel's value (exp25 on dev16: 0.976
    and 0.907).

    uv run python scripts/exp26/report.py [exp26 run] [exp25 run]   # -> <exp26 run>/report.json, text on stdout
"""
import importlib.util
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "exp21"))
from readout import fmt_stat  # type: ignore  # noqa: E402

from devision.train.compare import load_records  # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, "..", rel))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


exp25 = _load("exp25_report_for26", "exp25/report.py")
exp24 = _load("exp24_report_for26", "exp24/report.py")
LAST = exp25.LAST
BOUND_SAME = 0.50


def binding(path: str) -> dict:
    """On pictures with A != B: same answer to readA and readB; wrong readings that chose the other panel's value."""
    by = defaultdict(dict)
    for r in load_records(path):
        by[r["sample_id"]][r["qid"]] = r
    pics = [q for q in by.values() if "readA" in q and "readB" in q and q["readA"]["gold_answer"] != q["readB"]["gold_answer"]]
    same = sum(q["readA"]["prediction"] == q["readB"]["prediction"] for q in pics)
    wrong = other = 0
    for q in pics:
        for qid, alt in (("readA", "readB"), ("readB", "readA")):
            if not q[qid]["correct"]:
                wrong += 1
                other += q[qid]["prediction"] == q[alt]["gold_answer"]
    return {"pictures": len(pics), "same_answer": same / len(pics) if pics else None,
            "wrong_readings": wrong, "other_panel": other / wrong if wrong else None}


def build(run: str, control: str) -> dict:
    out = exp25.build(run)
    step = lambda r, n: os.path.join(r, "stage2", "monitor", "step%d" % LAST, n + ".details.jsonl")  # noqa: E731
    out["vs_exp25"] = {n: exp24.paired_read(step(control, n), step(run, n)) for n in ("val", "dev16")}
    out["binding"] = {n: {"exp26": binding(step(run, n)), "exp25": binding(step(control, n))} for n in ("val", "dev16")}
    # secondary readout, threshold written before running: on dev16 pictures with A != B, at most half answer "A?"
    # and "B?" with the same option (exp25: 0.976). It measures whether the two answers separate, not whether they
    # are bound right (swapped answers also separate), and it inherits an invalid erased control (critic EXP26-I01).
    same = out["binding"]["dev16"]["exp26"]["same_answer"]
    if not out["valid"] or same is None:
        out["same_answer_threshold_met"], out["same_answer_label"] = None, "invalid: erased control" if not out["valid"] else "n/a"
    else:
        met = same <= BOUND_SAME
        out["same_answer_threshold_met"] = met
        out["same_answer_label"] = ("same-answer share at or below %.2f: the answers to A? and B? separate more often "
                                    "(not by itself evidence of right binding)" % BOUND_SAME if met else
                                    "same-answer share above %.2f: the pre-written threshold is not met" % BOUND_SAME)
    return out


def main(argv=None):
    args = argv if argv is not None else sys.argv[1:]
    run = args[0] if args else "runs/exp26-read-compare"
    control = args[1] if len(args) > 1 else "runs/exp25-read-compare"
    out = build(run, control)
    json.dump(out, open(os.path.join(run, "report.json"), "w"), indent=1)
    print(exp25.exp22.render(out))
    print("text compare %s: %s | picture compare (dev16) %s: %s" % (
        fmt_stat(out["text"]["cmp"]), out["text_cmp_class"], fmt_stat(out["dev16"]["cmp"]), out["picture_cmp_class"]))
    print("outcome:", out["outcome"])
    print("same answer to A?/B? (secondary): " + out["same_answer_label"])
    for n, label in (("val", "fit"), ("dev16", "dev16")):
        b = out["binding"][n]
        f = lambda x: "n/a" if x is None else "%.3f" % x  # noqa: E731
        print("vs exp25, 16 px read %-5s %s | same answer to A?/B? (A != B, %d pictures) %s vs %s | wrong readings "
              "on the other panel's value %s vs %s" % (label, fmt_stat(out["vs_exp25"][n]), b["exp26"]["pictures"],
                                                      f(b["exp26"]["same_answer"]), f(b["exp25"]["same_answer"]),
                                                      f(b["exp26"]["other_panel"]), f(b["exp25"]["other_panel"])))


if __name__ == "__main__":
    main()
