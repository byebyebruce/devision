"""Decision samples are checked before training (broken gold is an error, never a uniform target), and scene
families stay adjacent when the training order is shuffled."""
import json
import random

import pytest

from devision.train.rlcd import grouped_order
from devision.train.samples import read_decision_jsonl, sample_errors


def sample(**over):
    s = {"id": "s1", "image": "a.jpg",
         "questions": {"q": {"type": "choice", "instructions": "Colour?", "criteria": {"red": None, "green": None}}},
         "gold": {"q": {"probabilities": {"red": 0.0, "green": 1.0}}}}
    s.update(over)
    return s


def test_a_valid_soft_or_hard_label_passes():
    assert sample_errors(sample()) == []
    assert sample_errors(sample(gold={"q": {"probabilities": {"red": 0.5, "green": 0.5}}})) == []
    noul = {"q": {"type": "noul", "instructions": "Dog?"}}
    assert sample_errors(sample(questions=noul, gold={"q": {"probabilities": {"true": 0.8, "false": 0.2}}})) == []


@pytest.mark.parametrize("gold", [
    {"q": {"probabilities": {"red": 0.0, "green": 0.0}}},      # all zero
    {"q": {"probabilities": {"blue": 1.0}}},                    # names no option
    {"q": {"probabilities": {"red": -1.0, "green": 2.0}}},      # negative
    {"q": {"probabilities": {"red": float("nan"), "green": 1}}},
    {},                                                         # missing
])
def test_broken_gold_is_reported(gold):
    assert sample_errors(sample(gold=gold))


def test_the_reader_names_file_line_and_id(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text(json.dumps(sample()) + "\n" + json.dumps(sample(id="bad", gold={"q": {"probabilities": {"x": 1}}})) + "\n")
    with pytest.raises(ValueError, match=r"d.jsonl:2 \(bad\)"):
        read_decision_jsonl(str(p))


def test_grouped_order_keeps_each_scene_family_together_and_shuffles_families():
    items = [{"group": "a"}, {"group": "b"}, {"group": "a"}, {"group": "c"}, {"group": "b"}, {}]
    order = grouped_order(items, random.Random(0))
    assert sorted(order) == list(range(6))
    seen = [items[i].get("group", i) for i in order]
    for g in ("a", "b"):
        first = seen.index(g)
        assert seen[first:first + 2] == [g, g]
    assert {tuple(grouped_order(items, random.Random(s))) for s in range(10)} != {tuple(order)}
