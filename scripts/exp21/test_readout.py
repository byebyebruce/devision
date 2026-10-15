"""exp21 readout on fixture records, no model: denominators, the family pairing, the fixed 68:114 transfer weights,
the precheck exits and an end-to-end report on a fabricated monitor directory."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
import precheck  # noqa: E402
import report  # noqa: E402
from readout import FAIL, NEAR, PASS, classify, syn_index, syn_stat, transfer  # noqa: E402


def syn_records(right, families=40, heights=(6, 8, 11, 16), suffix=""):
    """right(family index, view, height, qid) -> correct."""
    out = []
    for f in range(families):
        for v in (0, 1):
            for h in heights:
                for q in ("readA", "readB", "cmp"):
                    out.append({"sample_id": "exp21:d%04d:v%d:h%d%s" % (f, v, h, suffix), "qid": q,
                                "image_id": "exp21:d%04d:v%d:h%d" % (f, v, h), "source": "exp21-syn",
                                "correct": bool(right(f, v, h, q)), "nll": 0.5})
    return out


def test_compare_given_both_counts_only_pictures_read_right():
    # readA right on even families only, readB always, cmp always
    idx = syn_index(syn_records(lambda f, v, h, q: q != "readA" or f % 2 == 0, families=10))
    s = syn_stat(idx, 6, "cmp_given_both", resamples=50)
    assert s["n"] == 5 * 2 and s["value"] == 1.0
    assert syn_stat(idx, 6, "both_read", resamples=50)["value"] == 0.5
    assert syn_stat(idx, 6, "read", resamples=50)["n"] == 10 * 2 * 2


def test_following_the_value_change_needs_both_views():
    idx = syn_index(syn_records(lambda f, v, h, q: v == 0, families=10))    # right only in view 0
    assert syn_stat(idx, 8, "cf_cmp", resamples=50)["value"] == 0.0
    assert syn_stat(idx, 8, "cmp", resamples=50)["value"] == 0.5


def test_classify():
    assert classify({"value": 0.85, "ci95": [0.80, 0.90]}) == PASS
    assert classify({"value": 0.82, "ci95": [0.70, 0.90]}) == NEAR
    assert classify({"value": 0.50, "ci95": [0.45, 0.55]}) == FAIL
    assert classify({"value": 0.58, "ci95": [0.50, 0.66]}) == NEAR


def real(n, right):
    return [{"sample_id": "s%d" % i, "qid": "q", "image_id": "p%d" % i, "correct": right(i)} for i in range(n)]


def test_transfer_uses_the_fixed_weights():
    zero = {k: real(10, lambda i: False) for k in ("temp", "force", "temp_mism", "force_mism")}
    after = dict(zero, temp=real(10, lambda i: True))
    t = transfer(zero, after, resamples=100)
    assert t["accuracy_change"]["value"] == pytest.approx(68 / 182)
    assert t["gain_change"]["value"] == pytest.approx(68 / 182)
    assert t["temp"]["questions"] == 10 and t["force"]["accuracy_change"]["value"] == 0.0
    # a gain from the mismatched picture also rising is not a picture gain
    both = dict(after, temp_mism=real(10, lambda i: True))
    assert transfer(zero, both, resamples=100)["gain_change"]["value"] == pytest.approx(0.0)


def test_transfer_refuses_different_questions():
    zero = {k: real(10, lambda i: False) for k in ("temp", "force", "temp_mism", "force_mism")}
    with pytest.raises(ValueError):
        transfer(zero, dict(zero, force=real(9, lambda i: False)), resamples=10)


def write(d, name, rows):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, name + ".details.jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def control(right, kind):
    rows = syn_records(lambda f, v, h, q: right(q), families=40, heights=(0,), suffix=":" + kind)
    return rows


def precheck_dir(tmp_path, syn_right, erased_right, text_right):
    d = str(tmp_path / "pre")
    write(d, "dev_syn", syn_records(syn_right))
    write(d, "dev_syn_erased", control(erased_right, "erased"))
    write(d, "dev_syn_text", control(text_right, "text"))
    write(d, "sqa_text", real(24, lambda i: i % 2 == 0))
    write(d, "sqa_blank", real(24, lambda i: i % 3 == 0))
    return d


@pytest.mark.parametrize("syn,erased,text,train,word", [
    (lambda f, v, h, q: f % 2 == 0, lambda q: False, lambda q: True, True, "train"),
    (lambda f, v, h, q: True, lambda q: False, lambda q: True, False, "already"),
    (lambda f, v, h, q: f % 2 == 0, lambda q: q != "cmp", lambda q: True, False, "leak"),
    (lambda f, v, h, q: f % 2 == 0, lambda q: False, lambda q: q != "cmp", False, "text"),
])
def test_precheck_exits(tmp_path, syn, erased, text, train, word):
    out = precheck.decide(precheck_dir(tmp_path, syn, erased, text))
    assert out["train"] is train and out["decision"].startswith(word)


def panel(n, right, source="x"):
    return [{"sample_id": "%s%d" % (source, i), "qid": "q", "image_id": "%s%d" % (source, i // 2), "source": source,
             "correct": bool(right(i)), "nll": 0.3, "type": "choice", "gold": {"a": 1.0, "b": 0.0}} for i in range(n)]


def test_report_end_to_end(tmp_path):
    mon = str(tmp_path / "mon")
    for step, good in ((0, False), (2000, True)):
        d = os.path.join(mon, "step%d" % step)
        # step 2000: reads everything at 11/16 px, nothing at 6/8 px
        write(d, "syn", syn_records(lambda f, v, h, q: good and h >= 11 if q != "cmp" else good and h >= 11))
        write(d, "syn_erased", control(lambda q: False, "erased"))
        write(d, "syn_text", control(lambda q: True, "text"))
        for k in ("temp", "force"):
            write(d, "sqa_" + k, real(20, lambda i: good or i % 3 == 0))
            write(d, "sqa_%s_mism" % k, real(20, lambda i: i % 3 == 0))
        write(d, "sqa_text", real(24, lambda i: True))
        write(d, "sqa_blank", real(24, lambda i: False))
        for p in report.GUARD + ("val",):
            if p == "ext":
                rows = [r for s in report.EXT_SOURCES for r in panel(10, lambda i: i % 2 == 0, s)]
            else:
                rows = panel(40, lambda i: i % 2 == 0 if p != "flip" else True)
            if p == "flip":
                rows = [dict(r, image_id=r["image_id"] + (":flip" if int(r["sample_id"][1:]) % 2 else "")) for r in rows]
            write(d, p, rows)
    out = report.build(mon)
    assert out["valid"] and out["reading"].startswith("scale-sensitive")
    assert out["compare"]["16"] == "compares" and out["compare"]["6"] == "reading does not pass"
    assert out["transfer_observed"] and out["costs"] == []
    assert out["guard"]["flip"]["accuracy"]["difference"] == 0.0
