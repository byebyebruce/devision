"""Evaluation records and controls: every number is recomputable, controls change only what they say."""
import json
import math
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
    p_true = soft["probabilities"]["true"]
    # by hand: NLL against the soft label, Brier over both options, accuracy against the majority answer
    assert abs(soft["nll"] - -(0.3 * math.log(1 - p_true) + 0.7 * math.log(p_true))) < 1e-9
    assert abs(soft["brier"] - ((1 - p_true - 0.3) ** 2 + (p_true - 0.7) ** 2)) < 1e-9
    assert soft["gold_answer"] == "true" and soft["correct"] == (p_true >= 0.5)
    assert set(result["by_options"]["3"]) >= {"accuracy", "nll", "ece"}
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
    assert {x["type"] for x in rev} == {"choice"}  # noul questions have no order to reverse: not asked
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
    sub = compare(a, b, resamples=200, only={"s%d" % i for i in range(25, 50)})
    assert sub["all"]["questions"] == 25 and sub["all"]["difference"] == 0.0


def test_fallback_holdout_never_splits_an_image():
    items = [{"image": "im%d" % (i % 7)} for i in range(70)]
    random.seed(0)
    held, rest = holdout_by_image(items, 15)
    assert held and rest
    assert not {it["image"] for it in held} & {it["image"] for it in rest}


def per_picture_samples(root, pictures=3, per_picture=10):
    """Several one-question samples per picture file (as in POPE), of different lengths and option
    counts, two of them per picture with a text state, so grouped requests mix padding lengths."""
    (root / "img").mkdir(parents=True)
    colours = ["red", "blue", "green", "black", "white"]
    out = []
    for p in range(pictures):
        Image.new("RGB", (40 + 9 * p, 30), (80 * p % 255, 40, 200)).save(root / "img" / ("%d.jpg" % p))
        for k in range(per_picture):
            if k % 2:
                opts = colours[: 2 + k % 4]
                q = {"type": "choice", "instructions": "What color is the " + "dog " * (k % 3) + "?",
                     "criteria": {o: ("dark " + o if k % 3 == 0 else None) for o in opts}}
                gold = {o: float(o == opts[k % len(opts)]) for o in opts}
            else:
                q = {"type": "noul", "instructions": "Is there a " + ("cat" if k % 4 else "person sitting") + "?"}
                gold = {"false": 0.4, "true": 0.6} if k % 4 else {"false": 1.0, "true": 0.0}
            s = {"id": "p%d-%d" % (p, k), "source": "pope-adversarial" if k < 4 else "colors", "kind": "color",
                 "image_id": "pic:%d" % p, "image": "img/%d.jpg" % p,
                 "questions": {"q": q}, "gold": {"q": {"probabilities": gold}}}
            if k % 5 == 4:  # k = 4, 9: a noul and a choice question with a text state
                s["state_text"] = "the photo is on the left"
            out.append(s)
    random.Random(0).shuffle(out)
    return out


def close(a, b, tol=1e-5):
    """Equal, floats within `tol` (NLL and ECE are means of per-question floats)."""
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(close(a[k], b[k], tol) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) < tol
    return a == b


def test_grouping_questions_by_picture_changes_no_answer(tmp_path):
    rows = per_picture_samples(tmp_path)
    decider = tiny_decider()

    def run(control, max_questions):
        records = []
        summary = evaluate(decider, rows, tmp_path, control=control, records_out=records,
                           max_questions=max_questions)
        return sorted(records, key=lambda r: (r["sample_id"], r["qid"])), summary

    plain = {}
    for control in ("none", "mismatched", "reversed"):
        alone, s_alone = run(control, None)
        grouped, s_grouped = run(control, 32)
        capped, _ = run(control, 3)  # a picture's questions split over several requests
        assert {r["questions_in_request"] for r in alone} == {1}
        assert max(r["questions_in_request"] for r in grouped) > 3
        assert max(r["questions_in_request"] for r in capped) == 3
        for other in (grouped, capped):
            assert len(other) == len(alone)
            for a, b in zip(alone, other):
                assert a.keys() == b.keys()
                for key in a:
                    if key in ("latency_ms", "questions_in_request"):
                        continue
                    if key == "probabilities":
                        assert a[key].keys() == b[key].keys()
                        assert all(abs(a[key][o] - b[key][o]) < 1e-5 for o in a[key]), (control, a, b)
                    elif key in ("nll", "brier", "p_prediction"):
                        assert abs(a[key] - b[key]) < 1e-5, (control, key)
                    else:
                        assert a[key] == b[key], (control, key)
        for key in ("n", "images", "accuracy_all", "accuracy", "nll", "brier", "ece", "reliability", "by_source",
                    "by_kind", "by_options", "pope", "noul_confusion", "thresholds", "control"):
            assert close(s_grouped.get(key), s_alone.get(key)), (control, key)
        # timing: each request counted once; single-question latency only from one-question requests
        assert s_alone["request_latency_ms"]["requests"] == len(alone) == s_alone["latency_ms"]["requests"]
        assert s_grouped["request_latency_ms"]["requests"] == 3 * 2  # per picture: without / with text state
        assert s_grouped["questions_per_request"]["max"] > 3
        singles = [r for r in grouped if r["questions_in_request"] == 1]
        assert s_grouped.get("latency_ms", {}).get("requests", 0) == len(singles)
        plain[control] = (alone, grouped)
    a = order_sensitivity(plain["none"][0], plain["reversed"][0])
    b = order_sensitivity(plain["none"][1], plain["reversed"][1])
    assert a["n"] == b["n"] > 0 and a["answer_flip_rate"] == b["answer_flip_rate"]
    assert abs(a["mean_abs_probability_change"] - b["mean_abs_probability_change"]) < 1e-5
