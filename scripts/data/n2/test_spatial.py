import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spatial import caption, relations, to_sample, valid  # noqa: E402


def box(x, y, w, h):
    return {"x": x, "y": y, "w": w, "h": h}


def test_captions_use_the_right_article():
    assert caption("apple", "on", "table") == "An apple on a table"
    assert caption("cup", "to the left of", "owl") == "A cup to the left of an owl"


def test_on_needs_the_subject_over_the_object_and_left_needs_a_gap():
    table, cup = box(100, 200, 300, 100), box(200, 150, 40, 40)
    assert valid("on", "vertical", cup, table, 1000, 1000)
    assert not valid("under", "vertical", cup, table, 1000, 1000)
    assert not valid("to the left of", "vertical", cup, table, 1000, 1000)   # overlaps horizontally
    left = box(0, 190, 40, 40)
    assert valid("to the left of", "vertical", left, table, 1000, 1000)
    assert not valid("to the left of", "depth", left, table, 1000, 1000)     # bottoms differ: depth unknown
    assert valid("to the left of", "depth", box(0, 240, 40, 60), table, 1000, 1000)


def test_contradicted_relations_are_dropped():
    scene = {"objects": {"1": {"name": "cup", "relations": [{"name": "on", "object": "2"}]},
                         "2": {"name": "table", "relations": [{"name": "on", "object": "1"}]},
                         "3": {"name": "dog", "relations": [{"name": "behind", "object": "2"}]}}}
    objs = {"cup": scene["objects"]["1"], "table": scene["objects"]["2"], "dog": scene["objects"]["3"]}
    assert relations(scene, objs) == [("dog", "behind", "table")]


def test_one_option_is_gold_and_yes_no_is_consistent():
    rng = random.Random(0)
    for _ in range(50):
        r = to_sample("1", "vg:1", "gqa/images/1.jpg", "depth", "dog", "behind", "car", rng)
        probs = r["gold"]["q"]["probabilities"]
        assert sum(probs.values()) == 1.0 and max(probs.values()) == 1.0
        if r["questions"]["q"]["type"] == "choice":
            assert len(probs) == 4 and probs["A dog behind a car"] == 1.0
