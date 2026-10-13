"""Paired statistics beyond accuracy (v12 retrain plan, section 10.1), on hand-checkable records."""
import pytest

from devision.train.compare import paired_metric
from devision.train.evaluate import score_records


def rec(sid, image, gold_answer, pred, source="s", qtype="noul", options=("false", "true"), gold=None, probs=None):
    gold = gold or {o: float(o == gold_answer) for o in options}
    probs = probs or {o: (0.9 if o == pred else 0.1 / (len(options) - 1)) for o in options}
    return score_records([{"sample_id": sid, "qid": "q", "image_id": image, "source": source, "type": qtype,
                           "options": list(options), "gold": gold, "gold_answer": gold_answer,
                           "probabilities": probs, "prediction": pred}])[0]


def flip(r, pred):
    return rec(r["sample_id"], r["image_id"], r["gold_answer"], pred, r["source"], r["type"], r["options"])


def panel():
    # 3 positives, 1 negative; pictures: i1 holds two questions
    return [rec("a", "i1", "true", "true"), rec("b", "i1", "true", "false"),
            rec("c", "i2", "true", "true"), rec("d", "i3", "false", "true")]


@pytest.mark.parametrize("metric", ["accuracy", "recall", "fpr", "balanced_accuracy", "nll"])
def test_identical_predictions_differ_by_zero(metric):
    r = paired_metric(panel(), panel(), metric, resamples=200)
    assert r["difference"] == 0 and r["ci95"] == [0, 0]


def test_balanced_accuracy_with_unequal_classes():
    a = panel()
    b = [a[0], flip(a[1], "true"), a[2], flip(a[3], "false")]
    ra = paired_metric(a, b, "balanced_accuracy", resamples=200)
    # A: recall 2/3, negatives right 0/1 -> 1/3; B: 1 and 1 -> 1
    assert ra["a"] == pytest.approx(1 / 3) and ra["b"] == pytest.approx(1.0)
    assert paired_metric(a, b, "recall", resamples=10)["difference"] == pytest.approx(1 / 3)
    assert paired_metric(a, b, "fpr", resamples=10)["difference"] == pytest.approx(-1.0)
    assert ra["pictures"] == 3          # two questions on i1, one picture


def test_soft_gold_nll():
    gold = {"x": 0.7, "y": 0.3}
    r = rec("s", "i", "x", "x", qtype="choice", options=("x", "y"), gold=gold, probs={"x": 0.6, "y": 0.4})
    import math
    assert r["nll"] == pytest.approx(-(0.7 * math.log(0.6) + 0.3 * math.log(0.4)))
    assert paired_metric([r], [r], "nll", resamples=10)["a"] == pytest.approx(r["nll"])


def test_macro_weights_sources_equally():
    # source big: 3 questions, A all right, B all wrong; source small: 1 question, A wrong, B right
    a = [rec("b%d" % i, "p%d" % i, "true", "true", source="big") for i in range(3)] + \
        [rec("s0", "q0", "true", "false", source="small")]
    b = [flip(r, "false") for r in a[:3]] + [flip(a[3], "true")]
    macro = paired_metric(a, b, "macro_accuracy", sources=["big", "small"], resamples=200)
    per_q = paired_metric(a, b, "accuracy", resamples=10)
    assert macro["difference"] == pytest.approx(0.0)          # (-1 + 1) / 2
    assert per_q["difference"] == pytest.approx(-0.5)         # (-3 + 1) / 4
    assert macro["ci95"] == [pytest.approx(0.0), pytest.approx(0.0)]   # every resample keeps both sources


def test_a_picture_shared_by_two_sources_moves_whole():
    # the shared picture carries one question of each source; the other pictures are identical in A and B
    a = [rec("x1", "shared", "true", "false", source="s1"), rec("x2", "shared", "true", "true", source="s2"),
         rec("y1", "only1", "true", "true", source="s1"), rec("y2", "only2", "true", "true", source="s2")]
    b = [flip(a[0], "true"), flip(a[1], "false"), a[2], a[3]]
    r = paired_metric(a, b, "macro_accuracy", sources=["s1", "s2"], resamples=300)
    assert r["pictures"] == 3
    # each stratum ({s1,s2}, {s1}, {s2}) holds one picture, so every resample is the full panel
    assert r["ci95"] == [pytest.approx(r["difference"]), pytest.approx(r["difference"])]


def test_panels_must_match_exactly():
    a = panel()
    with pytest.raises(ValueError, match="differ in questions"):
        paired_metric(a, a[:3], "accuracy")
    other = [rec(r["sample_id"], r["image_id"], "false" if r["gold_answer"] == "true" else "true", r["prediction"])
             for r in a]
    with pytest.raises(ValueError, match="gold"):
        paired_metric(a, other, "accuracy")
    with pytest.raises(ValueError, match="without questions"):
        paired_metric(a, a, "macro_accuracy", sources=["s", "other"])
    mixed = a + [rec("e", "i4", "true", "true", source="t")]
    with pytest.raises(ValueError, match="fixed list"):
        paired_metric(mixed, mixed, "macro_accuracy", sources=["s"])
