"""Frozen exp80 decision from T=1 decide outputs; paired bootstrap by picture."""
import json
from pathlib import Path

from devision.train.compare import load_records, paired_metric
from structure_report import occurrences


def records(arm, name, control="real"):
    return occurrences(load_records(f"runs/exp80/{arm}/raw/{name}.{control}.details.jsonl"))


def summarize_arm(arm):
    base = records("baseline", "chart")
    new = records(arm, "chart")
    p1 = paired_metric(base, new, seed=80)
    # Direction is real minus mismatched, not the reverse.
    p2 = paired_metric(records(arm, "chart", "mismatched"), new, seed=80)
    old, protect = records("baseline", "protect"), records(arm, "protect")
    guard = paired_metric(old, protect, "nll", seed=80)
    groups = {}
    for source in sorted({r["source"] for r in old}):
        groups[source] = paired_metric([r for r in old if r["source"] == source],
                                      [r for r in protect if r["source"] == source], seed=80)
    harmful = [s for s, d in groups.items() if d["difference"] <= -.03 and d["ci95"][1] < 0]
    reversed_rows = {(r["sample_id"], r["qid"]): r for r in records(arm, "chart", "reversed")}
    choice = [r for r in new if r["type"] == "choice"]
    if {(r["sample_id"], r["qid"]) for r in choice} != set(reversed_rows):
        raise ValueError("Reversed choice panel is incomplete or mismatched")
    flips = sum(r["prediction"] != reversed_rows[(r["sample_id"], r["qid"])]["prediction"] for r in choice) / len(choice)
    fit = paired_metric(records("baseline", "fit"), records(arm, "fit"), seed=80)
    pass1, pass2, pass3 = p1["difference"] >= .03 and p1["ci95"][0] > 0, p2["ci95"][0] > 0, not harmful and guard["difference"] <= .05
    return {"P1_accuracy": p1, "P2_real_minus_mismatched": p2, "P3_protect_nll": guard,
            "protect_sources": groups, "harmful_sources": harmful, "reversed_prediction_flip_rate": flips,
            "fit_accuracy": fit, "pass": {"P1": pass1, "P2": pass2, "P3": pass3}, "expand": pass1 and pass2 and pass3}


def main():
    arms = {a: summarize_arm(a) for a in ("r", "c")}
    diff = paired_metric(records("r", "chart"), records("c", "chart"), seed=80)
    out = {"protocol": "exp80-v1", "temperature": 1, "arms": arms, "C_minus_R": diff,
           "decision": "do_not_expand"}
    valid = [a for a in arms if arms[a]["expand"]]
    if len(valid) == 1:
        out["decision"] = "expand_" + valid[0]
    elif len(valid) == 2:
        if diff["ci95"][0] > 0:
            out["decision"] = "expand_c"
        elif diff["ci95"][1] < 0:
            out["decision"] = "expand_r"
        else:
            out["decision"] = "expand_c_no_detected_objective_difference"
    p = Path("runs/exp80/report.json")
    p.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
