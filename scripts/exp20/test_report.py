"""exp20's readout resamples a mirror image together with its original in every statistic (critic EXP20-I01),
on hand-made records, no model."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import report  # noqa: E402


def write(root, step, panel, rows):
    d = root / ("step%d" % step)
    d.mkdir(parents=True, exist_ok=True)
    (d / (panel + ".details.jsonl")).write_text("".join(json.dumps(r) + "\n" for r in rows))


def rec(i, mirror, correct):
    p = 0.9 if correct else 0.1
    return {"sample_id": "s%d%s" % (i, "m" if mirror else ""), "qid": "q", "image_id": "coco:%d%s" % (i, ":flip" if mirror else ""),
            "type": "noul", "options": ["false", "true"], "gold": {"false": 0.0, "true": 1.0}, "gold_answer": "true",
            "probabilities": {"false": 1 - p, "true": p}, "prediction": "true" if correct else "false",
            "correct": correct, "nll": -__import__("math").log(p)}


def test_flip_is_resampled_by_original_picture(tmp_path):
    a = [rec(i, m, i % 3 == 0) for i in range(30) for m in (False, True)]
    b = [rec(i, m, i % 2 == 0) for i in range(30) for m in (False, True)]
    write(tmp_path / "a", 2000, "flip", a)
    write(tmp_path / "b", 2000, "flip", b)
    r = report.pm((str(tmp_path / "a"), 2000), (str(tmp_path / "b"), 2000), "flip", "accuracy")
    assert r["pictures"] == 30                       # 60 records: 30 originals, each with its mirror
    assert abs(r["difference"] - (sum(x["correct"] for x in b) - sum(x["correct"] for x in a)) / 60) < 1e-12
    n = report.pm((str(tmp_path / "a"), 2000), (str(tmp_path / "b"), 2000), "flip", "nll")
    assert n["pictures"] == 30
