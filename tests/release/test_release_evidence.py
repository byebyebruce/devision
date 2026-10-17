"""scripts/release/delivery.py and calibration.py on fabricated records: dev16 labels follow the original rule
only with a valid erased control; the calibration comparison pairs raw and fitted runs of the same questions."""
import importlib.util
import json
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


REPORT = {"transfer": {"accuracy_change": {"value": 0.12, "ci95": [0.03, 0.2]}, "gain_change": {"value": 0.1, "ci95": [0.01, 0.2]}},
          "target_met": True, "outcome": "delivery target met with no pre-written protection cost", "guard": {}, "costs": []}
IDENT = {"checkpoint": "runs/c/stage2", "files": {"model.safetensors": "ab", "config.json": "cd"}, "frozen": "t"}


@pytest.mark.parametrize("erased_right,expect", [(lambda f, v, h, q: f % 3 == 0, "pass"),
                                                 (lambda f, v, h, q: True, "void: erased control invalid")])
def test_dev16_needs_a_valid_erased_control(erased_right, expect):
    d = load("delivery")
    dev16 = syn_records(lambda f, v, h, q: True, heights=(16,))
    erased = syn_records(erased_right, heights=(0,), suffix=":erased")
    out = d.build(REPORT, dev16, erased, IDENT)
    assert out["dev16"]["read_class"] == expect and out["dev16"]["cmp_class"] == expect
    assert out["primary"]["exp32_outcome"] == REPORT["outcome"] and out["identity"]["model.safetensors"] == "ab"


def test_calibration_pairs_same_questions():
    c = load("calibration")
    def recs(p):
        return [{"sample_id": "s%d" % i, "qid": "q", "type": "choice", "options": ["a", "b"], "correct": i % 2 == 0,
                 "nll": 0.7, "p_prediction": p} for i in range(10)]
    out = c.compare(recs(0.9), recs(0.6))
    assert out["all"]["raw"]["ece"] > out["all"]["fitted"]["ece"] and "choice:2" in out
    with pytest.raises(SystemExit):
        c.compare(recs(0.9), recs(0.6)[:5])
