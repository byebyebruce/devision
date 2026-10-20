import random

from v2_coco_exist import build


def scene():
    images = [{"id": i, "width": 100, "height": 100} for i in (1, 2)]
    anns = {1: [{"category_id": 7, "area": 20.0, "iscrowd": 0}, {"category_id": 8, "area": 2000.0, "iscrowd": 0}],
            2: [{"category_id": 8, "area": 2000.0, "iscrowd": 0}]}
    return images, anns, {7: "spoon", 8: "dining table"}, {1: ["a table"], 2: ["a table"]}


def asked_yes(rows):
    return {(r["image_id"], r["category"]) for r in rows if r["gold"]["q"]["probabilities"]["true"] == 1.0}


def test_a_tiny_thing_is_a_yes_question_only_without_the_minimum_area():
    images, anns, names, caps = scene()
    default = build(images, anns, names, caps, "train2014", 5, set(), random.Random(0))
    no_min = build(images, anns, names, caps, "train2014", 5, set(), random.Random(0), min_pos_area=0.0)
    assert ("coco:1", "spoon") not in asked_yes(default)
    assert ("coco:1", "spoon") in asked_yes(no_min)
    spoon = [r for r in no_min if r["category"] == "spoon"]
    assert sum(r["answer_key"] == "true" for r in spoon) == sum(r["answer_key"] == "false" for r in spoon)
