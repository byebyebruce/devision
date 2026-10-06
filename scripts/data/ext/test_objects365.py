"""Extension pack, Objects365: exist / count / left-right questions from exhaustive boxes (no network)."""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import problems, question_only_baseline  # noqa: E402
from objects365 import (balance_positions, boxes_by_class, confusable, cooccurrence, count_of,  # noqa: E402
                        left_right_pairs, plausible_absent, questions_for, thing, visible)

W, H = 1000, 1000


def box(cat, x0, y0, x1, y1, crowd=0, fake=0, refl=0):
    return {"category": cat, "bbox": [x0, y0, x1, y1], "iscrowd": crowd, "isfake": fake, "isreflected": refl}


def test_counts_come_from_boxes_and_unknowable_counts_are_refused():
    b = boxes_by_class([box("Cup", 0, 0, 200, 200), box("Cup", 300, 0, 500, 200),
                        box("Bottle", 0, 0, 200, 200), box("Bottle", 0, 0, 50, 50),          # 0.25%: tiny
                        box("Chair", 0, 0, 300, 300), box("Chair", 0, 0, 300, 300, crowd=1),
                        box("Plate", 0, 0, 300, 300, refl=1),
                        box("Glasses", 0, 0, 300, 300)], W, H)
    assert count_of(b, "Cup") == 2
    assert count_of(b, "Bottle") is None        # a tiny box: a person would count it too
    assert count_of(b, "Chair") is None         # a crowd box: count unknowable
    assert count_of(b, "Plate") is None         # a reflection
    assert count_of(b, "Glasses") is None       # "pair of glasses" is not counted
    assert count_of(b, "Dog") == 0              # absent from an exhaustive annotation
    many = boxes_by_class([box("Apple", i * 10, 0, i * 10 + 150, 150) for i in range(11)], W, H)
    assert count_of(many, "Apple") is None      # above 10


def test_visible_needs_a_real_box_of_at_least_one_percent():
    b = boxes_by_class([box("Cup", 0, 0, 50, 50), box("Dog", 0, 0, 400, 400, refl=1), box("Cat", 0, 0, 100, 100),
                        box("Other Shoes", 0, 0, 500, 500)], W, H)
    assert visible(b) == ["Cat"]                # tiny cup, reflected dog, never-asked "Other Shoes"


def test_negatives_are_never_in_the_picture_nor_confusable_with_it():
    pictures = [["Car", "Person", "Traffic Sign", "Dog"], ["Car", "Traffic Light"], ["Car", "SUV"],
                ["Person", "Dog", "Stop Sign"], ["Car", "Bicycle", "Street Lights"]] * 5
    cooc = cooccurrence(pictures)
    b = boxes_by_class([box("Car", 0, 0, 300, 300), box("Person", 500, 0, 600, 300),
                        box("Traffic Sign", 700, 0, 800, 100)], W, H)
    seen = {c for s in range(200) for c in plausible_absent(b, cooc, random.Random(s), k=2)}
    assert seen
    assert not seen & {"Car", "Person", "Traffic Sign"}
    assert not seen & {"SUV", "Stop Sign", "Traffic Light"}     # SUV ~ car, stop sign / traffic light ~ traffic sign
    assert seen <= {"Dog", "Bicycle", "Street Lights"}
    assert confusable("Cabbage", "Red Cabbage") and confusable("Coffee Table", "Dinning Table")
    assert not confusable("Dog", "Cup")


def test_left_right_from_single_instance_box_centres():
    b = boxes_by_class([box("Dog", 0, 0, 200, 200), box("Cat", 700, 0, 900, 200),       # centres 0.10 / 0.80
                        box("Cup", 150, 500, 250, 600),                                    # centre 0.20: too close to dog
                        box("Chair", 400, 0, 600, 300), box("Chair", 0, 0, 300, 300)],   # two chairs: ambiguous
                       W, H)
    pairs = left_right_pairs(b)
    assert ("Cat", "Dog", False) in pairs       # the cat is not left of the dog
    assert ("Cat", "Cup", False) in pairs
    assert not any("Chair" in p[:2] for p in pairs)
    assert not any(set(p[:2]) == {"Cup", "Dog"} for p in pairs)
    assert not left_right_pairs(boxes_by_class([box("Car", 0, 0, 200, 200), box("SUV", 700, 0, 900, 200)], W, H))


def test_questions_are_well_formed_and_balanced_positions_say_nothing_by_wording():
    rng = random.Random(0)
    cooc = cooccurrence([["Dog", "Cat", "Cup"], ["Dog", "Bowl/Basin"], ["Cat", "Bowl/Basin"]])
    rows = []
    for i in range(400):
        left, right = ("Dog", "Cat") if i % 3 else ("Cat", "Dog")       # dogs mostly on the left
        b = boxes_by_class([box(left, 0, 0, 200, 200), box(right, 700, 0, 900, 200)], W, H)
        rows += questions_for(str(i), "x.jpg", b, cooc, rng)
    assert all(problems(r) == [] for r in rows)
    kinds = {r["kind"] for r in rows}
    assert kinds == {"exist", "count", "position"}
    assert len({r["id"] for r in rows}) == len(rows)
    no = [r for r in rows if r["kind"] == "exist" and r["answer_key"] == "false"]
    assert no and all(any(w in r["questions"]["q"]["instructions"] for w in ("bowl", "cup")) for r in no)
    pos = balance_positions([r for r in rows if r["kind"] == "position"])
    assert len(pos) > 100
    assert question_only_baseline(pos, "wording")["accuracy"] < 0.62
    assert {r["answer_key"] for r in pos} == {"left", "right"}


def test_articles():
    assert thing("Apple") == "an apple" and thing("SUV") == "an SUV" and thing("Rice") == "rice"
    assert thing("Bakset") == "a basket"


def test_cap_keeps_one_of_each_kind_first():
    from objects365 import cap_mixed
    rows = [{"image_id": "o:1", "kind": k, "n": i} for i, k in enumerate(["exist"] * 4 + ["count", "position"])]
    kept = cap_mixed(rows, 3, random.Random(0))
    assert sorted(r["kind"] for r in kept) == ["count", "exist", "position"]
