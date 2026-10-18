"""exp44 number-validity erratum (critic EXP44-FINAL-02), CPU only, from the frozen outputs as they are: the first
readout counted "1,", "390," and other malformed strings as valid numbers (rule "loose-v1"). This recounts validity
with the strict rule (scripts/exp44/aux_readout.py NUMBER, "strict-v2") on J's four number readouts and M's two,
checks that no "correct" changes (a malformed string never equalled a gold number), and records the sha256 of every
source it read -- J's FROZEN.json, each readout's summary and details, the original report.json / report.txt -- which
it does not modify. Writes runs/exp44-erratum.json.

    uv run python scripts/exp44/erratum.py
"""
import hashlib
import importlib.util
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("exp44_readout", os.path.join(HERE, "aux_readout.py"))
assert _spec is not None and _spec.loader is not None
readout = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(readout)

READOUTS = {"J": ("runs/exp44-j", ("aux_fit_O", "aux_fit_L", "aux_rl_tune", "aux_rl_tune_L")),
            "M": ("runs/exp44-m-aux", ("aux_fit_O", "aux_rl_tune"))}
REPORT = ("runs/exp44-report.json", "runs/exp44-report.txt")


def sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def recount(details: str) -> dict:
    recs = [json.loads(x) for x in open(details) if x.strip()]
    out: dict = {"questions": len(recs), "valid_recorded": sum(r["valid"] for r in recs), "correct": sum(r["correct"] for r in recs)}
    strict = [r["prediction"] is not None and bool(readout.NUMBER.match(r["prediction"])) for r in recs]
    out["valid_strict"] = sum(strict)
    out["recorded_valid_but_malformed"] = sum(r["valid"] and not s for r, s in zip(recs, strict))
    if any(r["correct"] != (s and r["prediction"] == r["gold_number"]) for r, s in zip(recs, strict)):
        raise SystemExit("%s: a correct answer would change under the strict rule" % details)
    bad = Counter(r["prediction"] for r, s in zip(recs, strict) if r["valid"] and not s)
    out["malformed_most_common"] = bad.most_common(5)
    return out


def main() -> int:
    res: dict = {"rule": readout.NUMBER_RULE, "pattern": readout.NUMBER.pattern, "sources": {}, "readouts": {}}
    res["sources"]["runs/exp44-j/FROZEN.json"] = sha("runs/exp44-j/FROZEN.json")
    for p in REPORT:
        res["sources"][p] = sha(p)
    for who, (run, names) in READOUTS.items():
        for n in names:
            for ext in (".json", ".details.jsonl"):
                p = os.path.join(run, "eval", n + ext)
                res["sources"][p] = sha(p)
            res["readouts"]["%s/%s" % (who, n)] = recount(os.path.join(run, "eval", n + ".details.jsonl"))
    fr = json.load(open("runs/exp44-j/FROZEN.json"))
    for n in READOUTS["J"][1]:                       # the J files are still the ones recorded under its frozen identity
        for f, h in fr["evals"][n].items():
            if res["sources"][os.path.join("runs/exp44-j/eval", f)] != h:
                raise SystemExit("%s changed after it was recorded" % f)
    speed: Counter = Counter()
    tune = {json.loads(x)["id"]: json.loads(x) for x in open("data/real_layout/questions_tune.jsonl")}
    for r in (json.loads(x) for x in open("runs/exp44-j/eval/aux_rl_tune.details.jsonl")):
        if "speed" in tune[r["sample_id"]]["questions"][r["qid"]]["instructions"]:
            speed[r["prediction"]] += 1
    res["rl_tune_speed_predictions_most_common"] = speed.most_common(5)
    json.dump(res, open("runs/exp44-erratum.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
