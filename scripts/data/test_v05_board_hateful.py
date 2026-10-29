"""Hateful memes: yes / no keeps the board's option order, labels are balanced, and dev / test ids never pass."""
from v05_board_hateful import balance, sample


def test_balance_equalises_labels_deterministically():
    rows = [{"id": i, "label": int(i % 3 == 0)} for i in range(90)]
    a, b = balance(rows), balance(list(reversed(rows)))
    assert a == b and sum(r["label"] for r in a) == len(a) // 2 == 30


def test_sample_is_yes_no_in_board_order():
    s = sample({"id": 42953, "label": 1}, "x.jpg")
    assert s["questions"]["q"]["criteria"] == {"A": "Yes", "B": "No"} and s["image_id"] == "hm:42953"
    assert s["gold"]["q"]["probabilities"] == {"A": 1.0, "B": 0.0}
    assert sample({"id": 7, "label": 0}, "x.jpg")["gold"]["q"]["probabilities"]["B"] == 1.0
