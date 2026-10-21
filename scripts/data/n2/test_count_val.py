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


def test_soft_count_golds_are_kept():
    from mix import COUNT_OPTIONS, full_count_options
    soft = {"id": "s", "source": "tallyqa", "kind": "count", "image_id": "x", "image": "x.jpg",
            "questions": {"q": {"type": "choice", "instructions": "How many cats?",
                                "criteria": {"1": None, "2": None, "3": None}}},
            "gold": {"q": {"probabilities": {"1": 0.0, "2": 0.7, "3": 0.3}}}, "answer_key": "2"}
    r = full_count_options(soft, random.Random(0))
    assert r["gold"]["q"]["probabilities"]["2"] == 0.7 and r["gold"]["q"]["probabilities"]["3"] == 0.3
    full = json_copy(r)
    assert full_count_options(full, random.Random(0)) is full
    assert sorted(r["questions"]["q"]["criteria"], key=int) == COUNT_OPTIONS


def test_soften_keeps_the_answer_and_sums_to_one():
    from soft_mix import soften
    r = {"gold": {"q": {"probabilities": {"true": 1.0, "false": 0.0}}}}
    g = soften(r, 0.1)["gold"]["q"]["probabilities"]
    assert g == {"true": 0.9, "false": 0.1}
    r3 = {"gold": {"q": {"probabilities": {"a": 0.0, "b": 1.0, "c": 0.0}}}}
    g3 = soften(r3, 0.1)["gold"]["q"]["probabilities"]
    assert abs(sum(g3.values()) - 1.0) < 1e-9 and g3["b"] == 0.9
