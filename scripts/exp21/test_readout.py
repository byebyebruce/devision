"""exp21 readout on fixture records, no model: denominators, the family pairing, the fixed 68:114 transfer weights,
the precheck exits and an end-to-end report on a fabricated monitor directory."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
import precheck  # noqa: E402
import report  # noqa: E402
from readout import FAIL, NEAR, PASS, classify, paired_by_occurrence, syn_index, syn_stat, transfer  # noqa: E402


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
    """right(q) or right(f, v, q)."""
    two = right.__code__.co_argcount == 3
    return syn_records(lambda f, v, h, q: right(f, v, q) if two else right(q), families=40, heights=(0,),
                       suffix=":" + kind)


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


def val_rows():
    """data/r13b/val.jsonl's identity structure: some ids twice, the copies with different options and gold."""
    rows = []
    for i in range(30):
        rows.append({"sample_id": "v%d" % i, "qid": "q", "image_id": "vp%d" % (i // 2), "source": "v",
                     "options": ["a", "b"], "gold": {"a": 1.0, "b": 0.0}, "correct": i % 2 == 0, "nll": 0.4})
        if i % 3 == 0:   # the rewritten copy: same id, other option order
            rows.append({"sample_id": "v%d" % i, "qid": "q", "image_id": "vp%d" % (i // 2), "source": "v",
                         "options": ["b", "a"], "gold": {"b": 0.0, "a": 1.0}, "correct": True, "nll": 0.2})
    return rows


def monitor(tmp_path, syn_right=None, erased_right=lambda q: False):
    """A fabricated run directory: step 2000 reads 11/16 px and not 6/8 px by default (so compare-given-both has
    no cases at 6/8 px); step 0 reads nothing."""
    run = str(tmp_path / "run")
    mon = os.path.join(run, "stage2", "monitor")
    syn_right = syn_right or (lambda f, v, h, q: h >= 11)
    for step, good in ((0, False), (2000, True)):
        d = os.path.join(mon, "step%d" % step)
        write(d, "syn", syn_records(lambda f, v, h, q: good and syn_right(f, v, h, q)))
        write(d, "syn_erased", control(erased_right, "erased"))
        write(d, "syn_text", control(lambda q: True, "text"))
        for k in ("temp", "force"):
            write(d, "sqa_" + k, real(20, lambda i: good or i % 3 == 0))
            write(d, "sqa_%s_mism" % k, real(20, lambda i: i % 3 == 0))
        write(d, "sqa_text", real(24, lambda i: True))
        write(d, "sqa_blank", real(24, lambda i: False))
        for p in report.GUARD:
            if p == "ext":
                rows = [r for s in report.EXT_SOURCES for r in panel(10, lambda i: i % 2 == 0, s)]
            else:
                rows = panel(40, lambda i: i % 2 == 0 if p != "flip" else True)
            if p == "flip":
                rows = [dict(r, image_id=r["image_id"] + (":flip" if int(r["sample_id"][1:]) % 2 else "")) for r in rows]
            write(d, p, rows)
        write(d, "val", val_rows())
    return run


def test_report_end_to_end(tmp_path, capsys):
    run = monitor(tmp_path)
    report.main([run])
    text = capsys.readouterr().out
    out = json.load(open(os.path.join(run, "report.json")))
    assert out["valid"] and out["reading"].startswith("scale-sensitive")
    assert out["compare"]["16"] == "compares" and out["compare"]["6"] == "reading does not pass"
    assert out["transfer_observed"] and out["costs"] == []
    assert out["guard"]["flip"]["accuracy"]["difference"] == 0.0
    # 6 / 8 px: nothing read, so compare-given-both has no cases: null in JSON, "not estimable" in the text
    assert out["syn"]["6"]["cmp_given_both"]["value"] is None and out["syn"]["6"]["cmp_given_both"]["n"] == 0
    assert "not estimable (n=0)" in text and "next:" in text
    # val keeps every copy (30 + 10 repeated ids) and identical runs differ by 0
    assert out["guard"]["val"]["accuracy"]["questions"] == 40 and out["guard"]["val"]["accuracy"]["difference"] == 0.0
    assert "report only, 40 questions" in text


@pytest.mark.parametrize("erased,valid", [
    (lambda f, v, q: q == "readA", False),                      # read exactly 0.50
    (lambda f, v, q: q == "cmp" and f * 2 + v < 44, False),     # compare exactly 0.55 (44 / 80)
    (lambda f, v, q: q != "readB", False),                      # both above
    (lambda f, v, q: q == "cmp" and f * 2 + v < 43, True),      # compare 0.5375, read 0: normal
])
def test_an_invalid_erased_control_voids_every_mechanism_label(tmp_path, capsys, erased, valid):
    run = monitor(tmp_path, erased_right=erased)
    report.main([run])
    out = json.load(open(os.path.join(run, "report.json")))
    assert out["valid"] is valid
    if valid:
        assert out["compare"]["16"] == "compares"
    else:
        assert out["reading"].startswith("invalid")
        assert set(out["compare"].values()) == {"invalid: erased control"}
        assert out["syn"]["16"]["cmp"]["value"] == 1.0          # the numbers are kept


def test_val_copies_must_agree():
    a = val_rows()
    assert paired_by_occurrence(a, a, resamples=50)["questions"] == 40
    for change in ({"options": ["a", "c"]}, {"gold": {"a": 0.0, "b": 1.0}}, {"image_id": "other"}):
        b = [dict(r, **change) if r["sample_id"] == "v3" and r["options"] == ["b", "a"] else r for r in a]
        with pytest.raises(ValueError):
            paired_by_occurrence(a, b, resamples=10)
    with pytest.raises(ValueError):
        paired_by_occurrence(a, a[:-1], resamples=10)
    # the duplicate check of the judged panels is untouched
    from devision.train.compare import paired_metric
    with pytest.raises(ValueError, match="duplicate"):
        paired_metric(a, a)


def test_precheck_command_exit_status(tmp_path, capsys):
    d = precheck_dir(tmp_path, lambda f, v, h, q: f % 2 == 0, lambda q: False, lambda q: True)
    with pytest.raises(SystemExit) as e:
        precheck.main([d])
    assert e.value.code == 0 and "decision: train" in capsys.readouterr().out
    d2 = precheck_dir(tmp_path / "b", lambda f, v, h, q: True, lambda q: False, lambda q: True)
    with pytest.raises(SystemExit) as e:
        precheck.main([d2])
    assert e.value.code == 10
