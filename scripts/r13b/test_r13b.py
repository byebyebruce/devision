"""The v12 retrain helpers: the 0..10 rewrite of counting questions and the frozen-rule judge."""
import json
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
import build  # noqa: E402
import judge  # noqa: E402

from devision.train.evaluate import score_records  # noqa: E402


def sample(instr, options, gold):
    return {"id": "x", "image_id": "i", "image": "i.jpg", "source": "tallyqa", "answer_position": 1,
            "questions": {"q": {"type": "choice", "instructions": instr, "criteria": {o: None for o in options}}},
            "gold": {"q": {"probabilities": gold}}}


def test_counting_questions_get_every_candidate_0_to_10():
    s = build.wide(sample("How many dogs are there?", ["2", "3", "five"], {"2": 0.0, "3": 0.25, "five": 0.75}),
                   random.Random(0))
    q, g = s["questions"]["q"], s["gold"]["q"]["probabilities"]
    assert sorted(q["criteria"], key=int) == [str(n) for n in range(11)]
    assert g["3"] == 0.25 and g["5"] == 0.75 and sum(g.values()) == 1.0   # soft gold kept, words mapped
    assert "answer_position" not in s


@pytest.mark.parametrize("instr, options, gold", [
    ("What color is it?", ["2", "3"], {"2": 1.0, "3": 0.0}),                  # not a counting question
    ("How many people?", ["11", "12"], {"11": 1.0, "12": 0.0}),               # answer above 10: kept as it is
    ("How many years passed?", ["35 years", "20 years"], {"35 years": 1.0, "20 years": 0.0}),
])
def test_other_questions_are_left_alone(instr, options, gold):
    s = sample(instr, options, gold)
    assert build.wide(s, random.Random(0))["questions"] == s["questions"]


def test_option_only_rules():
    ss = [sample("How many?", ["1", "2", "3"], {"1": 0.0, "2": 1.0, "3": 0.0}),
          sample("How many?", ["0", "4"], {"0": 1.0, "4": 0.0})]
    r = build.option_rules(ss)
    assert r["smallest"] == 0.5 and r["lower_median"] == 1.0


def rec(sid, image, gold_answer, pred, p_pred=0.9):
    probs = {"true": p_pred if pred == "true" else 1 - p_pred, "false": p_pred if pred == "false" else 1 - p_pred}
    return score_records([{"sample_id": sid, "qid": "q", "image_id": image, "source": "s", "type": "noul",
                           "options": ["false", "true"], "gold": {"true": float(gold_answer == "true"),
                                                                 "false": float(gold_answer == "false")},
                           "gold_answer": gold_answer, "probabilities": probs, "prediction": pred}])[0]


def write(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_holds_is_strict_at_the_thresholds():
    r = {"difference": -0.03, "ci95": [-0.05, -0.01]}
    assert judge.holds({"difference_le": -0.03, "hi_lt": 0}, r)
    assert not judge.holds({"difference_le": -0.031, "hi_lt": 0}, r)
    assert not judge.holds({"lo_gt": -0.05}, r)          # lo must be strictly above
    assert judge.holds({"hi_lt": 0}, r) and not judge.holds({"hi_lt": -0.01}, r)


def test_monitor_needs_two_consecutive_nll_hits(tmp_path):
    rules = {"ext_sources": [], "monitor": [
        {"id": "nll", "set": "exist", "metric": "nll", "stop_if": {"difference_ge": 0.10, "lo_gt": 0}, "consecutive": 2}]}
    good = [rec("p%d" % i, "i%d" % i, "true", "true", 0.9) for i in range(20)]
    worse = [rec("p%d" % i, "i%d" % i, "true", "true", 0.6) for i in range(20)]   # nll 0.105 -> 0.51
    mon = tmp_path / "monitor"
    write(str(mon / "step0" / "exist.details.jsonl"), good)
    write(str(mon / "step2000" / "exist.details.jsonl"), worse)
    write(str(mon / "step4000" / "exist.details.jsonl"), good)
    assert judge.monitor(rules, str(tmp_path)) == 0          # one hit, then back
    write(str(mon / "step6000" / "exist.details.jsonl"), worse)
    write(str(mon / "step8000" / "exist.details.jsonl"), worse)
    assert judge.monitor(rules, str(tmp_path)) == 3
    assert json.load(open(mon / "judge.json"))["stop"] == [[8000, "nll"]]


def test_the_frozen_rules_parse():
    rules = json.load(open(judge.RULES))
    assert len(rules["ext_sources"]) == 13
    for r in rules["monitor"] + rules["final"]:
        assert ("stop_if" in r) or ("require" in r) or ("fail_if" in r)
