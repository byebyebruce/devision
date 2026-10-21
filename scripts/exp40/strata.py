"""Real-layout reading questions split by whether the two objects' values are the same (critic EXP40-M-RESULT-02):
per details file, over the mass / speed / distance reading questions of data/real_layout/questions_tune.jsonl,
right answers on same-value pairs and on different-value pairs; on different-value pairs: both objects right, both
predicted the same, and wrong answers equal to the other object's gold. Descriptive counts only (no gate).

    uv run python scripts/exp40/strata.py runs/exp32-fit/eval_v03/rl_tune.details.jsonl runs/exp40-m/eval/rl_tune.details.jsonl
"""
import json
import os
import sys
from collections import defaultdict
from typing import Dict

sys.path.insert(0, os.path.dirname(__file__))
from build import field_of  # type: ignore  # noqa: E402

TUNE = "data/real_layout/questions_tune.jsonl"
FIELDS = ("mass", "speed", "distance")


def strata(details: str, data_path: str = TUNE) -> Dict:
    field = {}
    for s in map(json.loads, open(data_path)):
        for qid, q in s["questions"].items():
            f = field_of(q["instructions"])
            if f != "cmp":
                field[(s["id"], qid)] = f
    pairs = defaultdict(list)
    for r in map(json.loads, open(details)):
        f = field.get((r["sample_id"], r["qid"]))
        if f:
            pairs[(r["sample_id"], f)].append(r)
    if any(len(v) != 2 for v in pairs.values()) or len(pairs) * 2 != len(field):
        raise SystemExit("%s: not exactly two reading questions per picture and field" % details)
    out: Dict = {"same": {"questions": 0, "right": 0}, "different": {"questions": 0, "right": 0, "pairs": 0, "both_right": 0,
                 "same_prediction": 0, "wrong": 0, "wrong_is_other_gold": 0}, "by_field": {}}
    for (sid, f), (a, b) in sorted(pairs.items()):
        bf = out["by_field"].setdefault(f, {"pairs": 0, "both_right": 0, "wrong": 0, "wrong_is_other_gold": 0})
        if a["gold_answer"] == b["gold_answer"]:
            out["same"]["questions"] += 2
            out["same"]["right"] += a["correct"] + b["correct"]
            continue
        d = out["different"]
        d["questions"] += 2
        d["right"] += a["correct"] + b["correct"]
        d["pairs"] += 1
        bf["pairs"] += 1
        both = bool(a["correct"] and b["correct"])
        d["both_right"] += both
        bf["both_right"] += both
        d["same_prediction"] += a["prediction"] == b["prediction"]
        for x, y in ((a, b), (b, a)):
            if not x["correct"]:
                hit = x["prediction"] == y["gold_answer"]
                d["wrong"] += 1
                d["wrong_is_other_gold"] += hit
                bf["wrong"] += 1
                bf["wrong_is_other_gold"] += hit
    out["by_field"] = {f: out["by_field"][f] for f in FIELDS if f in out["by_field"]}
    out["both_right_macro"] = sum(v["both_right"] / v["pairs"] for v in out["by_field"].values()) / len(out["by_field"])
    return out


if __name__ == "__main__":
    for path in sys.argv[1:]:
        print(path)
        print(json.dumps(strata(path), indent=1))
