"""Evaluation records and controls: every number is recomputable, controls change only what they say."""
import json
import random

from conftest import tiny_decider
from PIL import Image

from devision.train.compare import compare
from devision.train.evaluate import evaluate, order_sensitivity, picture_pairs, summarize
from devision.train.rlcd import holdout_by_image


def samples(root, n=8):
    (root / "img").mkdir(parents=True)
    out = []
    for i in range(n):
        Image.new("RGB", (40, 30), (40 * i % 255, 90, 160)).save(root / "img" / ("%d.jpg" % i))
        out.append({"id": "s%d" % i, "source": "pope-random" if i % 2 else "colors", "kind": "color",
                    "image_id": "syn:%d" % (i // 2), "image": "img/%d.jpg" % i,
                    "questions": {"q": {"type": "choice", "instructions": "Which colour?",
                                        "criteria": {"red": None, "blue": "dark blue", "green": None}},
                                  "p": {"type": "noul", "instructions": "Is it blue?"}},
                    "gold": {"q": {"probabilities": {"red": 0.0, "blue": 1.0, "green": 0.0}},
                             "p": {"probabilities": {"false": 0.3, "true": 0.7}}}})
    return out


def test_the_summary_is_recomputable_from_the_records(tmp_path):
    rows = samples(tmp_path)
    records = []
    result = evaluate(tiny_decider(), rows, tmp_path, records_out=records)
    again = summarize(json.loads(json.dumps(records)))  # as if read back from the details file
    for key in ("n", "images", "accuracy_all", "accuracy", "nll", "brier", "ece", "by_source", "by_kind",
                "by_options", "pope", "noul_confusion", "reliability", "thresholds"):
        assert again[key] == result[key], key
    assert len(records) == 2 * len(rows)
    soft = next(r for r in records if r["qid"] == "p")
    assert soft["gold"] == {"false": 0.3, "true": 0.7} and soft["nll"] > 0  # NLL is against the soft label
    assert set(result["pope"]["pope-random"]) >= {"precision", "recall", "f1", "yes_ratio"}
    assert sum(b["n"] for b in result["reliability"]) == len(records)
    assert result["thresholds"]["0.6"]["coverage"] >= result["thresholds"]["0.9"]["coverage"]


def test_mismatched_pictures_are_never_their_own():
    rows = [{"image_id": "p%d" % (i // 3), "image": "%d.jpg" % (i // 3)} for i in range(30)]
    pairs = picture_pairs(rows, seed=1)
    assert all(pairs[pid] != "%s.jpg" % pid[1:] for pid in pairs)
    assert pairs == picture_pairs(rows, seed=1) and pairs != picture_pairs(rows, seed=2)


def test_reversing_options_keeps_keys_and_gold_and_reports_flips(tmp_path):
    rows = samples(tmp_path)
    plain, rev = [], []
    evaluate(tiny_decider(), rows, tmp_path, records_out=plain)
    evaluate(tiny_decider(), rows, tmp_path, control="reversed", records_out=rev)
    r = next(x for x in rev if x["type"] == "choice")
    assert r["options"] == ["green", "blue", "red"] and r["gold_answer"] == "blue"
    sens = order_sensitivity(plain, rev)
    assert sens["n"] == len(rows) and 0.0 <= sens["answer_flip_rate"] <= 1.0


def test_comparison_resamples_whole_pictures(tmp_path):
    def recs(correct):
        return [{"sample_id": "s%d" % i, "qid": "q", "image_id": "pic%d" % (i // 5), "source": "x",
                 "correct": c} for i, c in enumerate(correct)]
    a = recs([False] * 50)
    b = recs([True] * 25 + [False] * 25)
    out = compare(a, b, resamples=500)
    assert out["all"]["pictures"] == 10 and abs(out["all"]["difference"] - 0.5) < 1e-9
    assert not out["all"]["covers_zero"]
    assert compare(a, a, resamples=200)["all"]["covers_zero"]


def test_fallback_holdout_never_splits_an_image():
    items = [{"image": "im%d" % (i % 7)} for i in range(70)]
    random.seed(0)
    held, rest = holdout_by_image(items, 15)
    assert held and rest
    assert not {it["image"] for it in held} & {it["image"] for it in rest}
