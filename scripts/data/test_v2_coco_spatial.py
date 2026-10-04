"""v2 COCO size / position / relation questions: answers follow the boxes, names carry no prior."""
import random
from collections import Counter, defaultdict

from v2_coco_spatial import build

NAMES = {1: "dog", 2: "cup", 3: "laptop"}


def box(cat, x, y, w, h, crowd=0):
    return {"category_id": cat, "bbox": [x, y, w, h], "area": w * h, "iscrowd": crowd}


def run(scenes, caps=None):
    images = [{"id": i, "width": 100, "height": 100} for i in range(len(scenes))]
    anns = dict(enumerate(scenes))
    return build(images, anns, NAMES, "train2014", set(), caps or {"position": 50, "relation": 50, "size": 50},
                 random.Random(0))


def answer(s):
    g = s["gold"]["q"]["probabilities"]
    return max(g, key=g.get)


def test_position_follows_the_box():
    rows = run([[box(1, 2, 40, 20, 20)], [box(1, 78, 40, 20, 20)]])["position"]
    for s in rows:
        left_box = s["image_id"] == "coco:0"
        assert answer(s) == ("left" if left_box else "right")


def test_a_category_seen_on_one_side_only_is_not_asked():
    rows = run([[box(1, 2, 40, 20, 20)]] * 6).get("position", [])
    assert not rows  # every dog is on the left: the name would give the answer away


def test_each_answer_is_equally_common_per_category():
    scenes = [[box(1, 2, 40, 20, 20)]] * 5 + [[box(1, 78, 40, 20, 20)]] * 2
    rows = run(scenes)["position"]
    assert Counter(answer(s) for s in rows) == Counter({"left": 2, "right": 2})


def test_size_names_the_larger_box_and_drops_one_sided_pairs():
    big_dog = [box(1, 0, 0, 60, 60), box(2, 80, 80, 20, 20)]
    big_cup = [box(1, 0, 0, 20, 20), box(2, 40, 40, 60, 60)]
    rows = run([big_dog, big_cup])["size"]
    assert {answer(s) for s in rows} == {"dog", "cup"}
    for s in rows:
        assert answer(s) == ("dog" if s["image_id"] == "coco:0" else "cup")
    assert not run([big_dog, big_dog]).get("size")


def test_relations_read_correctly_whichever_object_is_the_subject():
    cup_left = [box(2, 2, 40, 15, 15), box(3, 60, 40, 30, 30)]
    cup_right = [box(2, 80, 40, 15, 15), box(3, 5, 40, 30, 30)]
    rows = run([cup_left, cup_right] * 4)["relation"]
    assert rows
    for s in rows:
        cup_is_left = s["image_id"] in ("coco:0", "coco:2", "coco:4", "coco:6")
        q = s["questions"]["q"]
        subject_cup = q["instructions"].startswith("Is the cup")
        truth = "to the left of" if cup_is_left == subject_cup else "to the right of"
        if q["type"] == "choice":
            assert answer(s) == truth
        else:
            assert answer(s) == ("true" if truth in q["instructions"] else "false")


def test_objects_with_two_instances_or_too_small_are_never_named():
    rows = run([[box(1, 2, 40, 20, 20), box(1, 70, 40, 20, 20)], [box(1, 2, 40, 5, 5)]] * 3)
    assert not any(rows.values())


def test_a_big_object_spanning_the_middle_gets_no_position_question():
    # centre at y=0.35 (upper part) but 40% of the box lies in the lower half
    rows = run([[box(1, 30, 0, 40, 70)], [box(1, 30, 0, 40, 70)]]).get("position", [])
    assert not [s for s in rows if s["axis"] == "tb"]


def test_size_is_not_asked_about_objects_too_small_to_compare():
    tiny_cup = [box(1, 0, 0, 60, 60), box(2, 80, 80, 12, 12)]   # cup 1.4% of the image
    tiny_dog = [box(1, 0, 0, 12, 12), box(2, 40, 40, 60, 60)]
    assert not run([tiny_cup, tiny_dog]).get("size")
