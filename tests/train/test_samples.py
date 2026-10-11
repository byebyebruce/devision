"""Decision samples are checked before training (broken gold is an error, never a uniform target), and scene
families' counterfactual units never straddle a micro batch."""
import json
import random

import pytest

from devision.train.rlcd import plan_batches
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


def units_of(items):
    u = {}
    for i, it in enumerate(items):
        if "group" in it:
            u.setdefault((it["group"], it["qid"]), set()).add(i)
    return u


def check(items, mb, seed):
    batches = plan_batches(items, mb, random.Random(seed))
    flat = [i for b in batches for i in b]
    assert sorted(flat) == list(range(len(items)))          # every item once
    assert all(1 <= len(b) <= mb for b in batches)
    where = {i: n for n, b in enumerate(batches) for i in b}
    for key, idx in units_of(items).items():
        if len(idx) <= mb:
            assert len({where[i] for i in idx}) == 1, key     # a unit never crosses a batch
    return batches


def test_counterfactual_units_share_a_micro_batch():
    rng = random.Random(0)
    items = []
    for fam in range(40):                     # families of 4 pictures x 9-11 questions: 36-44 items, > a batch
        for qid in range(9 + fam % 3):
            for _ in range(4):
                items.append({"group": "f%d" % fam, "qid": "q%d" % qid})
    items += [{"qid": "q"} for _ in range(37)]   # replay / GQA without groups, odd count for a ragged tail
    rng.shuffle(items)
    for seed in range(5):
        batches = check(items, 8, seed)
        assert len(batches) == -(-len(items) // 8)   # packs tightly: no extra steps
    assert check(items, 8, 0) != check(items, 8, 1)


def test_a_unit_larger_than_a_batch_is_split_into_batch_sized_pieces():
    items = [{"group": "g", "qid": "q"} for _ in range(11)] + [{"qid": "x"}]
    batches = plan_batches(items, 4, random.Random(0))
    assert sorted(len(b) for b in batches) == [4, 4, 4]          # pieces 4 + 4 + 3, the singleton fills the gap
    assert sum(1 for b in batches if any(i < 11 for i in b)) == 3
    assert sorted(i for b in batches for i in b) == list(range(12))
