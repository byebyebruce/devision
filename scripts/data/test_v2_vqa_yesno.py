"""v2 VQAv2 yes/no selection: no reading questions, recurring wording balanced, eval images out."""
import random
from collections import Counter

from v2_vqa_yesno import needs_reading, select


def q(i, text, p_yes, image=None):
    return {"id": "vqav2:%d" % i, "source": "vqav2", "image_id": image or "coco:%d" % i, "image": "x.jpg",
            "questions": {"q": {"type": "noul", "instructions": text}},
            "gold": {"q": {"probabilities": {"false": 1 - p_yes, "true": p_yes}}}}


def test_questions_about_written_text_are_recognised():
    assert needs_reading("Does the sign say stop?")
    assert needs_reading("Is this a 7?")
    assert needs_reading("Is the brand Nike?")
    assert not needs_reading("Is the dog running?")
    assert not needs_reading("Is this a kitchen?")


def test_recurring_wording_gets_as_many_yes_as_no():
    rows = [q(i, "Is it sunny?", 0.9) for i in range(20)] + [q(100 + i, "Is it sunny?", 0.1) for i in range(5)]
    out = select(rows, 1000, set(), random.Random(0))
    assert Counter(s["gold"]["q"]["probabilities"]["true"] > 0.5 for s in out) == Counter({True: 5, False: 5})


def test_reading_questions_ties_and_eval_images_are_dropped():
    rows = [q(1, "Does the sign say stop?", 1.0), q(2, "Is the dog running?", 0.5),
            q(3, "Is the cat asleep?", 1.0, image="coco:999"), q(4, "Is the cat awake?", 0.0)]
    out = select(rows, 1000, {"coco:999"}, random.Random(0))
    assert out == []  # the only clean question is a "no" with no "yes" to pair it with


def test_an_image_gets_at_most_three_questions():
    rows = [q(i, "Question %d about it?" % i, i % 2, image="coco:1") for i in range(10)]
    assert len(select(rows, 1000, set(), random.Random(0))) == 3
