"""ArxivQA label parsing, depth-question geometry (clear margins only, balanced slots) and DriveLM distractors."""
import numpy as np

from v05_board_arxivqa import parse, strictly_longest
from v05_board_depth import backproject, camera_question, distance_question, point_question
from v05_board_drive import describe_question, element_answer


def test_arxivqa_labels_in_every_format():
    opts = ["A. one", "B. two", "C. four", "D. five"]
    assert parse({"options": opts, "label": "C"})[0] == (["one", "two", "four", "five"], 2)
    assert parse({"options": ["A) One", "B) Two", "C) Four"], "label": "C) Four"})[0] == (["One", "Two", "Four"], 2)
    assert parse({"options": ["red", "blue"], "label": "blue"})[0] == (["red", "blue"], 1)
    assert parse({"options": opts, "label": "[Choose based on the figure]"})[1] == "bad_label"
    assert parse({"options": ["A. x", "B. X"], "label": "A"})[1] == "bad_options"      # duplicate ignoring case
    assert strictly_longest(["a", "bbb", "cc"], 1) and not strictly_longest(["aa", "bb"], 0)


def obj(cat, depth, cx, cy):
    return {"cat": cat, "depth": depth, "cx": cx, "cy": cy, "bbox": [cx - 5, cy - 5, 10, 10]}


def test_distance_question_needs_a_clear_margin_and_rotates_the_gold():
    objs = [obj("chair", 2.0, 300, 240), obj("cup", 2.1, 320, 250), obj("bed", 5.0, 100, 240)]
    for slot in (0, 1):
        z, blue, green, gold = distance_question(objs, 640, 480, "k", slot) or (None, None, None, None)
        assert z is not None and blue is not None and green is not None
        assert ({z["cat"], blue["cat"], green["cat"]} == {"chair", "cup", "bed"})
        closer = blue if slot == 0 else green
        assert closer["cat"] == gold
    tie = [obj("chair", 2.0, 300, 240), obj("cup", 2.0, 340, 240), obj("bed", 2.0, 260, 240)]
    assert distance_question(tie, 640, 480, "k", 0) is None
    assert backproject(320, 240, 3.0, 640, 480) == (0.0, 0.0, 3.0)


def test_camera_and_point_questions():
    red, blue, gold = camera_question([obj("tv", 4.0, 0, 0), obj("cat", 1.0, 0, 0)], "k", 0) or (None, None, None)
    assert red is not None and red["cat"] == gold == "cat"
    assert camera_question([obj("tv", 1.1, 0, 0), obj("cat", 1.0, 0, 0)], "k", 0) is None
    dm = np.ones((60, 80), dtype=np.float32)
    dm[:, 40:] = 3.0                                                  # right half far
    got = point_question(dm, "k", 1)
    assert got is not None
    a, b, gold = got
    assert gold == "B is closer" and dm[b[1], b[0]] < dm[a[1], a[0]]
    assert point_question(np.ones((60, 80), dtype=np.float32), "k", 0) is None   # flat: no clear pair


def test_drive_describe_distractors_are_same_category_and_differ():
    pool = {"Vehicle": ["Black sedan.", "black sedan", "White truck.", "Red bus.", "Blue SUV."]}
    got = describe_question({"Category": "Vehicle", "Visual_description": "Black sedan."}, pool, "k", 2)
    assert got is not None
    opts, gold = got
    assert gold == "Black sedan." and opts.index(gold) == 2 and len(opts) == 4
    assert sum(o.lower().startswith("black sedan") for o in opts) == 1
    assert describe_question({"Category": "Vehicle", "Visual_description": "Black sedan."},
                             {"Vehicle": ["Red bus."]}, "k", 0) is None
    assert element_answer("Yes, there are traffic elements") == "Yes" and element_answer("No, none.") == "No"
