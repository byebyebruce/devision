"""exp22 fit-test data and readout, no model: the training families are exp21's, every one at 16 px in both
views; the fit set is part of the training set; the readout labels each exit and prints n=0 statistics."""
import json
import os
import sys

import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "exp21"))
sys.path.insert(0, os.path.join(HERE, "..", "data"))
import importlib.util  # noqa: E402


def _load(name):   # exp21 has its own report.py on the path: load these two by file
    spec = importlib.util.spec_from_file_location("exp22_" + name, os.path.join(HERE, name + ".py"))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


exp22_build, report = _load("build"), _load("report")
from exp21_reading import build as exp21_build  # type: ignore  # noqa: E402
from test_readout import control, syn_records, write  # type: ignore  # noqa: E402


def test_training_families_are_exp21s_at_16_px_in_both_views(tmp_path):
    replay = tmp_path / "replay.jsonl"
    replay.write_text(json.dumps({"id": "r", "source": "r14b", "image_id": "x", "image": "x.png",
                                  "questions": {"q": {"type": "noul", "instructions": "?"}},
                                  "gold": {"q": {"probabilities": {"false": 0.0, "true": 1.0}}}}) + "\n")
    exp21_build(str(tmp_path), str(replay), n_train=30, n_dev=6, n_controls=2, total=10)
    m = exp22_build.build(str(tmp_path), n_train=30, n_fit=5)
    rows = lambda n: [json.loads(x) for x in open(tmp_path / n)]  # noqa: E731
    train, fit, dev16 = rows("exp22/train.jsonl"), rows("exp22/fit.jsonl"), rows("exp22/dev16.jsonl")
    old = {s["family"]: s["questions"] for s in rows("exp21/train_syn.jsonl")}
    assert {s["family"] for s in train} == set(old)
    assert all(s["questions"] == old[s["family"]] for s in train)          # same candidates and option order
    assert all(s["height"] == 16 for s in train + fit + dev16)
    assert sorted((s["family"], s["view"]) for s in train) == sorted((f, v) for f in old for v in (0, 1))
    assert {s["id"] for s in fit} <= {s["id"] for s in train} and len({s["family"] for s in fit}) == 5
    assert not {s["family"] for s in dev16} & set(old)
    assert m["files"]["train"]["questions"] == 30 * 2 * 3


def monitor(tmp_path, fit_right, dev_right, erased=lambda q: False):
    run = str(tmp_path / "run")
    for step in report.STEPS:
        d = os.path.join(run, "stage2", "monitor", "step%d" % step)
        last = step == 2000
        write(d, "val", syn_records(lambda f, v, h, q: last and fit_right(f, q), heights=(16,)))
        write(d, "dev16", syn_records(lambda f, v, h, q: last and dev_right(f, q), heights=(16,)))
        write(d, "dev", syn_records(lambda f, v, h, q: last and dev_right(f, q) and h == 16))
        write(d, "erased", control(erased, "erased"))
        write(d, "text", control(lambda q: True, "text"))
    return run


@pytest.mark.parametrize("fit,dev,erased,start", [
    (lambda f, q: True, lambda f, q: True, lambda q: False, "fitted and generalizes"),
    (lambda f, q: True, lambda f, q: False, lambda q: False, "fitted but does not generalize"),
    (lambda f, q: False, lambda f, q: False, lambda q: False, "not fitted"),
    (lambda f, q: f % 10 < 7, lambda f, q: False, lambda q: False, "uncertain"),
    (lambda f, q: True, lambda f, q: True, lambda q: q == "readA", "invalid"),
])
def test_labels(tmp_path, capsys, fit, dev, erased, start):
    run = monitor(tmp_path, fit, dev, erased)
    report.main([run])
    text = capsys.readouterr().out
    out = json.load(open(os.path.join(run, "report.json")))
    assert out["label"].startswith(start)
    assert set(out["trajectory"]) == {str(s) for s in report.STEPS}
    if start == "not fitted":     # nothing read anywhere: compare-given-both has no cases
        assert "not estimable (n=0)" in text
