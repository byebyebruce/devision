"""COCO-box question generator: questions must follow from the boxes."""
import random
from collections import Counter

from cocoqa import image_questions

NAMES = {1: "dog", 2: "cup", 3: "laptop", 4: "fork", 5: "apple"}
IMAGE = {"id": 7, "width": 100, "height": 100}


def box(cat, x, y, w, h):
    return {"category_id": cat, "bbox": [x, y, w, h], "area": w * h}


def questions(anns, cooccur=None, seed=0):
    cooccur = cooccur or {c: Counter() for c in NAMES}
    out = []
    for s in range(20):  # the generator samples; collect over seeds
        out += image_questions(IMAGE, anns, NAMES, cooccur, list(NAMES), "train2014", random.Random(seed + s))
    return out


def answer(s):
    probs = s["gold"]["q"]["probabilities"]
    return max(probs, key=probs.get)


def test_an_object_on_the_left_is_said_to_be_on_the_left():
    qs = [s for s in questions([box(1, 5, 40, 20, 20)]) if s["kind"] == "absolute"]
    assert qs and all(answer(s) == "left" for s in qs if "left" in s["questions"]["q"]["criteria"])


def test_an_object_in_the_middle_gets_no_left_right_question():
    qs = [s for s in questions([box(1, 40, 5, 20, 20)]) if s["kind"] == "absolute"]
    assert qs and all("left" not in s["questions"]["q"]["criteria"] for s in qs)


def test_relative_position_follows_the_boxes():
    anns = [box(2, 5, 40, 15, 15), box(3, 60, 40, 30, 30)]  # cup well left of laptop, same height
    qs = [s for s in questions(anns) if s["kind"] == "relative"]
    assert qs
    for s in qs:
        q = s["questions"]["q"]
        truth = "to the left of" if q["instructions"].startswith("Is the cup") else "to the right of"
        if q["type"] == "choice":
            assert answer(s) == truth
        else:
            assert answer(s) == ("true" if truth in q["instructions"] else "false")


def test_an_object_with_two_instances_is_never_asked_about_by_name():
    anns = [box(1, 5, 40, 20, 20), box(1, 70, 40, 20, 20), box(3, 40, 5, 20, 20)]
    qs = [s for s in questions(anns) if s["kind"] in ("absolute", "relative", "size")]
    assert all("dog" not in s["questions"]["q"]["instructions"] for s in qs)


def test_a_negative_existence_question_names_something_not_in_the_image():
    cooccur = {c: Counter() for c in NAMES}
    cooccur[3] = Counter({4: 10, 5: 1})
    qs = [s for s in questions([box(3, 10, 10, 50, 50)], cooccur) if s["kind"] == "exist" and answer(s) == "false"]
    assert qs and all("laptop" not in s["questions"]["q"]["instructions"] for s in qs)
    assert any("fork" in s["questions"]["q"]["instructions"] for s in qs)  # the usual companion is asked


def test_size_question_picks_the_larger_box():
    anns = [box(1, 5, 5, 60, 60), box(2, 75, 75, 15, 15)]
    qs = [s for s in questions(anns) if s["kind"] == "size"]
    assert qs and all(answer(s) == "dog" for s in qs)
