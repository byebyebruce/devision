"""v2 GQA filter: only questions whose objects are big enough to see; left/right marked."""
from v2_gqa import tag, visible

GRAPH = {"width": 500, "height": 400, "objects": {
    "1": {"w": 200, "h": 150},   # 15% of the image
    "2": {"w": 30, "h": 20},     # 0.3%
}}


def rec(*args):
    return {"semantic": [{"operation": "select", "argument": a, "dependencies": []} for a in args]}


def test_a_question_about_a_large_object_is_kept():
    assert visible(rec("dog (1)"), GRAPH)


def test_a_question_touching_a_tiny_object_is_dropped():
    assert not visible(rec("dog (1)", "_,to the left of,s (2)"), GRAPH)


def test_unknown_objects_and_missing_graphs_are_dropped():
    assert not visible(rec("cat (99)"), GRAPH)
    assert not visible(rec("dog (1)"), None)


def test_questions_about_absent_things_have_no_object_to_check():
    assert visible(rec("cat (-)"), GRAPH)


def test_left_right_questions_are_marked_spatial():
    left = {"questions": {"q": {"type": "choice", "instructions": "Is the cup to the left or right of the plate?",
                                "criteria": {"to the left of": None, "to the right of": None}}},
            "gold": {"q": {"probabilities": {"to the left of": 1.0, "to the right of": 0.0}}}}
    colour = {"questions": {"q": {"type": "choice", "instructions": "Is the shirt red or blue?",
                                  "criteria": {"red": None, "blue": None}}},
              "gold": {"q": {"probabilities": {"red": 0.0, "blue": 1.0}}}}
    assert tag(left)["kind"] == "spatial" and tag(colour)["kind"] == "gqa"
    assert tag(colour)["answer_key"] == "blue" and tag(colour)["group"] == "is the shirt red or blue?"
