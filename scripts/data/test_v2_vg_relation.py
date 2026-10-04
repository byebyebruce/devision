"""Above / below questions from scene graphs: answers follow the boxes, and the names carry no prior."""
import random
from collections import Counter, defaultdict

from v2_vg_relation import build


def obj(name, y, h=10, x=10, w=10):
    return {"name": name, "x": x, "y": y, "w": w, "h": h, "relations": [], "attributes": []}


def scenes(pairs):
    """{vg id: scene} with `cup` at y_cup and `lamp` at y_lamp in a 100x100 picture."""
    return {str(i): {"width": 100, "height": 100,
                     "objects": {"a": obj("cup", yc), "b": obj("lamp", yl, x=50)}} for i, (yc, yl) in enumerate(pairs)}


def where(vg_id):
    return "vg:%s" % vg_id, "gqa/images/%s.jpg" % vg_id


def run(sc, held_out=(), where=where):
    return build(sc, where, set(held_out), cap=50, rng=random.Random(0), min_name_pictures=1)


def truth(s):
    """Whether the subject of the question is above its object, from the gold answer."""
    q, g = s["questions"]["q"], s["gold"]["q"]["probabilities"]
    ans = max(g, key=g.get)
    if q["type"] == "choice":
        return ans == "above"
    asked_above = " above " in q["instructions"]
    return asked_above == (ans == "true")


def test_answers_follow_the_boxes():
    rows = run(scenes([(5, 60)] * 4 + [(60, 5)] * 4))   # cup above lamp in 0-3, below in 4-7
    assert rows
    for s in rows:
        cup_above = int(s["image_id"][3:]) < 4
        subject_is_cup = s["questions"]["q"]["instructions"].startswith("Is the cup")
        assert truth(s) == (cup_above == subject_is_cup)


def test_each_pair_is_asked_both_ways_equally_and_one_way_pairs_are_dropped():
    rows = run(scenes([(5, 60)] * 6 + [(60, 5)] * 2))
    assert Counter(s["answer_key"] for s in rows) == {"above": 2, "below": 2}
    assert run(scenes([(5, 60)] * 5)) == []


def test_unclear_repeated_tiny_and_background_objects_are_not_asked():
    touching = scenes([(5, 17), (17, 5)])           # gap of 2% < 5%
    assert run(touching) == []
    twice = {"0": {"width": 100, "height": 100, "objects": {"a": obj("cup", 5), "b": obj("cup", 60),
                                                             "c": obj("lamp", 60, x=50)}},
             "1": {"width": 100, "height": 100, "objects": {"a": obj("cup", 60), "c": obj("lamp", 5, x=50)}}}
    assert all("cup" not in s["category"] for s in run(twice))
    sky = {str(i): {"width": 100, "height": 100, "objects": {"a": obj("sky", y), "b": obj("lamp", 65 - y)}}
           for i, y in enumerate([5, 60])}
    assert run(sky) == []
    tiny = {str(i): {"width": 100, "height": 100, "objects": {"a": obj("cup", y, h=3, w=3), "b": obj("lamp", 65 - y)}}
            for i, y in enumerate([5, 60])}
    assert run(tiny) == []


def test_held_out_and_missing_pictures_are_skipped_under_either_id():
    sc = scenes([(5, 60), (60, 5), (5, 60), (60, 5)])
    coco = lambda k: ("coco:%s" % k, "coco/train2014/x.jpg") if k != "3" else None
    rows = run(sc, held_out={"vg:0", "coco:1"}, where=coco)
    assert {s["image_id"] for s in rows} <= {"vg:2"} and len(rows) == 0  # only vg:2 left: one-way, dropped
    rows = run(sc, held_out=set(), where=coco)
    assert "vg:3" not in {s["image_id"] for s in rows}


def test_ids_are_unique_and_samples_have_the_training_format():
    rows = run(scenes([(5, 60)] * 3 + [(60, 5)] * 3))
    assert len({s["id"] for s in rows}) == len(rows)
    for s in rows:
        assert set(s) >= {"id", "source", "image_id", "image", "questions", "gold"}
        q = s["questions"]["q"]
        assert q["type"] in ("noul", "choice")
        assert abs(sum(s["gold"]["q"]["probabilities"].values()) - 1) < 1e-9


def test_a_name_coco_boxed_more_than_once_is_not_asked_on_that_picture():
    sc = scenes([(5, 60), (60, 5)] * 2)
    counts = lambda k: {"cup": 3} if k in ("0", "1") else {"cup": 1}
    rows = build(sc, where, set(), cap=50, rng=random.Random(0), min_name_pictures=1, coco_counts=counts)
    assert rows and {s["image_id"] for s in rows} <= {"vg:2", "vg:3"}


def test_plural_and_clothing_names_are_not_asked():
    sc = {str(i): {"width": 100, "height": 100, "objects": {"a": obj(n, y), "b": obj("lamp", 65 - y, x=50)}}
          for i, (n, y) in enumerate([("cups", 5), ("cups", 60), ("helmet", 5), ("helmet", 60)])}
    assert run(sc) == []

