"""Round 14 acceptance (critic R14-R1-01 / 02) on hand-made details, no model."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
import accept  # noqa: E402

from devision.train.evaluate import score_records  # noqa: E402


def stats(lo=0.0, hi=0.0, nll=0.0, nll_ci=(0.0, 0.0)):
    return {"accuracy": {"ci95": [lo, hi]}, "nll": {"difference": nll, "ci95": list(nll_ci)}}


def all_sets(**override):
    sets = {s: stats() for s in accept.REQUIRED}
    sets["bench_countbench"] = stats(0.10, 0.19)    # primary met
    sets.update(override)
    return sets


@pytest.mark.parametrize("ci, held, worse", [
    ((-0.015, -0.005), True, True),     # a small proven drop inside the floor: passes, reported as proven worse
    ((-0.025, -0.005), False, True),
    ((-0.020, 0.005), False, False),    # the floor is strict: exactly -2 points is not held
    ((-0.015, 0.005), True, False),
])
def test_held_and_proven_worse_are_separate(ci, held, worse):
    assert accept.protection(*ci) == {"held": held, "proven_worse": worse}
    v = accept.decide(all_sets(test_relation_flip=stats(*ci)))
    assert v["protection"] is held and v["candidate"] is held
    assert ("test_relation_flip" in v["proven_worse"]) is worse


def test_unrounded_lower_bound_decides():
    v = accept.decide(all_sets(bench_lv_aokvqa=stats(-0.02021, 0.00527)))   # shows as -2.0 when rounded
    assert v["not_held"] == ["bench_lv_aokvqa"] and not v["candidate"]


def test_nll_warning_blocks_the_candidate():
    v = accept.decide(all_sets(bench_whatsup=stats(-0.02, 0.01, 0.051, (0.038, 0.065))))
    assert v["warnings"] == ["bench_whatsup"] and v["protection"] and not v["candidate"]


@pytest.mark.parametrize("gone", ["test_relation_flip", "bench_whatsup", "bench_countbench"])
def test_decide_refuses_without_a_required_set(gone):
    sets = all_sets()
    del sets[gone]
    with pytest.raises(ValueError, match=gone):
        accept.decide(sets)


def write_details(d, name, correct):
    os.makedirs(d, exist_ok=True)
    pred = "true" if correct else "false"
    r = score_records([{"sample_id": name, "qid": "q", "image_id": "i-" + name, "source": "s", "type": "noul",
                        "options": ["false", "true"], "gold": {"false": 0.0, "true": 1.0}, "gold_answer": "true",
                        "probabilities": {pred: 0.9, ("false" if correct else "true"): 0.1}, "prediction": pred}])[0]
    with open(os.path.join(d, name + ".details.jsonl"), "w") as f:
        f.write(json.dumps(r) + "\n")


def fixture(tmp_path):
    base, cand = str(tmp_path / "v12"), str(tmp_path / "cand")
    for s in accept.REQUIRED:
        write_details(base, s, s != "bench_countbench")
        write_details(cand, s, True)                 # counting up, the rest identical
    return base, cand


def test_complete_files_give_a_verdict(tmp_path, capsys):
    base, cand = fixture(tmp_path)
    accept.main(["--cand", cand, "--base", base])
    out = json.load(open(os.path.join(cand, "compare", "accept_vs_v12.json")))
    assert out["primary"] and out["protection"] and out["candidate"]
    assert "candidate for v0.3" in capsys.readouterr().out


@pytest.mark.parametrize("gone, side", [("test_relation_flip", "cand"), ("bench_whatsup", "base"),
                                        ("bench_countbench", "cand")])
def test_a_missing_file_stops_without_a_verdict(tmp_path, gone, side):
    base, cand = fixture(tmp_path)
    os.remove(os.path.join(cand if side == "cand" else base, gone + ".details.jsonl"))
    with pytest.raises(SystemExit, match=gone):
        accept.main(["--cand", cand, "--base", base])
    assert not os.path.exists(os.path.join(cand, "compare"))
