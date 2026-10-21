import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from count_val import OPTIONS, countable, pick, plural, row  # noqa: E402


def ann(c, area, crowd=0):
    return {"category_id": c, "area": area, "iscrowd": crowd}


def test_crowd_or_tiny_instances_make_the_count_ambiguous():
    assert countable([ann(1, 500), ann(1, 600)], 1, 10000, 0.01) == 2
    assert countable([ann(1, 500), ann(1, 600, crowd=1)], 1, 10000, 0.01) is None
    assert countable([ann(1, 500), ann(1, 50)], 1, 10000, 0.01) is None
    assert countable([ann(2, 500)], 1, 10000, 0.01) == 0


def test_one_question_per_picture_and_counts_are_balanced():
    names = {1: "dog", 2: "cat"}
    images = [{"id": i, "width": 100, "height": 100} for i in range(40)]
    anns = {i: [ann(1, 500)] * (i % 4) for i in range(40)}  # 0..3 dogs, never a cat
    out = pick(images, anns, names, 3, 0.01, random.Random(0))
    ids = [im["id"] for im, _, _ in out]
    assert len(ids) == len(set(ids))
    per = {}
    for _, _, n in out:
        per[n] = per.get(n, 0) + 1
    assert per == {0: 3, 1: 3, 2: 3, 3: 3}


def test_options_never_depend_on_the_gold():
    im = {"id": 7, "width": 10, "height": 10}
    r = row(im, "dog", 4, "val2014", random.Random(0))
    assert sorted(r["questions"]["q"]["criteria"], key=int) == OPTIONS
    assert r["gold"]["q"]["probabilities"]["4"] == 1.0 and sum(r["gold"]["q"]["probabilities"].values()) == 1.0
    assert r["questions"]["q"]["instructions"] == "How many dogs are there in the image?"


def test_names_without_a_clean_plural_are_not_asked():
    assert plural("person") == "people" and plural("skis") is None
