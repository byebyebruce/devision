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


def test_mix_gives_counting_questions_the_full_options():
    from mix import full_count_options, is_count
    r = {"id": "t", "source": "tallyqa", "kind": "count", "image_id": "x", "image": "x.jpg",
         "questions": {"q": {"type": "choice", "instructions": "How many chairs?",
                             "criteria": {"2": None, "0": None, "4": None, "5": None}}},
         "gold": {"q": {"probabilities": {"2": 1.0, "0": 0.0, "4": 0.0, "5": 0.0}}}, "answer_key": "2"}
    assert is_count(r)
    r2 = full_count_options(r, random.Random(0))
    assert sorted(r2["questions"]["q"]["criteria"], key=int) == OPTIONS
    assert r2["gold"]["q"]["probabilities"]["2"] == 1.0 and sum(r2["gold"]["q"]["probabilities"].values()) == 1.0
    big = json_copy(r)
    big["gold"]["q"]["probabilities"] = {"2": 0.0, "0": 0.0, "4": 0.0, "12": 1.0}
    assert full_count_options(big, random.Random(0)) is None


def json_copy(r):
    import json
    return json.loads(json.dumps(r))
