"""scripts/release/delivery.py and calibration.py on fabricated records: dev16 labels follow the original rule
only with a valid erased control; the transfer is recomputed from the given records and compared with exp32's;
the calibration comparison keeps every copy of a repeated question id and refuses different copies."""
import importlib.util
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts", "exp21"))
from test_readout import syn_records  # type: ignore  # noqa: E402


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", "release", name + ".py"))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rows(n, right, source):
    return [{"sample_id": "%s%d" % (source, i), "qid": "q", "image_id": "%s-p%d" % (source, i), "source": source,
             "gold": {"a": 1.0, "b": 0.0}, "correct": bool(right(i)), "nll": 0.5} for i in range(n)]


IDENT = {"checkpoint": "runs/c/stage2", "files": {"model.safetensors": "ab", "config.json": "cd"}, "frozen": "t"}


def records(d, new):
    out = {}
    for k, n in (("temp", 41), ("force", 120)):
        out["sqa_" + k] = rows(n, (lambda i: True) if new else (lambda i: i % 3 == 0), k)
        out["sqa_%s_mism" % k] = rows(n, lambda i: i % 3 == 0, k)
    for p in d.GUARD:
        out[p] = [r for s in d.EXT_SOURCES for r in rows(20, lambda i: True, s)] if p == "ext" else rows(60, lambda i: True, p)
    return out


@pytest.mark.parametrize("erased_right,expect", [(lambda f, v, h, q: f % 3 == 0, "pass"),
                                                 (lambda f, v, h, q: True, "void: erased control invalid")])
def test_delivery(erased_right, expect):
    d = load("delivery")
    cand, v03 = records(d, True), records(d, False)
    cand["dev16"] = syn_records(lambda f, v, h, q: True, heights=(16,))
    cand["syn_erased"] = syn_records(erased_right, heights=(0,), suffix=":erased")
    report = {"outcome": "x", "transfer": {"accuracy_change": {"value": 0.0}, "gain_change": {"value": 0.0}}}
    out = d.build(cand, v03, IDENT, d.screening(IDENT, "expN", report, "ab"))
    assert out["dev16"]["read_class"] == expect and out["dev16"]["cmp_class"] == expect
    assert out["primary"]["target_met"] and not out["primary"]["screening_report_matches"] and out["protection"]["costs"] == []
    assert out["primary"]["screening"]["outcome"] == "x"
    assert out["identity"]["model.safetensors"] == "ab"


def test_calibration_keeps_repeated_ids():
    c = load("calibration")
    def recs(p, opts_of_copy2=("a", "b", "c")):
        rs = [{"sample_id": "s%d" % i, "qid": "q", "type": "choice", "options": ["a", "b"], "correct": i % 2 == 0,
               "nll": 0.7, "p_prediction": p} for i in range(10)]
        return rs + [dict(rs[0], options=list(opts_of_copy2))]          # s0 held twice, with different options
    out = c.compare(recs(0.9), recs(0.6))
    assert out["all"]["raw"]["n"] == 11 and out["all"]["raw"]["ece"] > out["all"]["fitted"]["ece"] and "choice:3" in out
    with pytest.raises(SystemExit):
        c.compare(recs(0.9), recs(0.6)[:-1])
    with pytest.raises(SystemExit):
        c.compare(recs(0.9), recs(0.6, ("a", "c", "b")))


def test_screening_labels_are_bound_to_their_weights():
    """Two candidates cannot borrow each other's screening label: the label is carried only when the report's
    weights are the frozen candidate's."""
    d = load("delivery")
    first = {"files": {"model.safetensors": "aaa"}}
    second = {"files": {"model.safetensors": "bbb"}}
    report = {"outcome": "delivery target met", "transfer": None}
    assert d.screening(first, "expA", report, "aaa")["outcome"] == "delivery target met"
    borrowed = d.screening(second, "expA", report, "aaa")
    assert borrowed["outcome"] is None and borrowed["status"].startswith("unverifiable")
    assert d.screening(second, "expB", report, "")["status"].startswith("unverifiable")
    assert d.screening(second, "expB", None, "bbb")["status"] == "no screening report"
