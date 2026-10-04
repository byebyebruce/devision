"""Mirrored left/right questions: same words, opposite answer."""
from v6_flip import flipped


def sample(q, probs, answer_key):
    return {"id": "v2-position:x", "image": "coco/train2014/a.jpg", "image_id": "coco:1", "axis": "lr",
            "answer_key": answer_key, "questions": {"q": q}, "gold": {"q": {"probabilities": probs}}}


def test_a_choice_answer_swaps_and_the_options_keep_their_order():
    q = {"type": "choice", "instructions": "Is the cup on the left or the right?", "criteria": {"left": None, "right": None}}
    f = flipped(sample(q, {"left": 1.0, "right": 0.0}, "left"), "v6/images/flip/a.jpg")
    assert f["gold"]["q"]["probabilities"] == {"left": 0.0, "right": 1.0}
    assert list(f["questions"]["q"]["criteria"]) == ["left", "right"] and f["answer_key"] == "right"
    assert f["image"] == "v6/images/flip/a.jpg" and f["id"] == "flip:v2-position:x"


def test_a_relation_statement_flips_between_true_and_false():
    q = {"type": "noul", "instructions": "Is the tie to the left of the phone?"}
    f = flipped(sample(q, {"false": 0.0, "true": 1.0}, "to the left of"), "x.jpg")
    assert f["gold"]["q"]["probabilities"] == {"false": 1.0, "true": 0.0}
    assert f["answer_key"] == "to the right of"


def test_the_mirrored_picture_is_a_different_picture_id():
    q = {"type": "choice", "instructions": "?", "criteria": {"to the left of": None, "to the right of": None}}
    f = flipped(sample(q, {"to the left of": 0.0, "to the right of": 1.0}, "to the right of"), "x.jpg")
    assert f["image_id"] == "coco:1:flip"   # per-picture caps count it apart from the original
