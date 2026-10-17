"""exp33 without a model: the decision draw keeps r14b's block proportions; the report's four outcomes."""
import importlib.util
import json
import os

import pytest

HERE = os.path.dirname(__file__)


def load(name):
    spec = importlib.util.spec_from_file_location("exp33_" + name, os.path.join(HERE, name + ".py"))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_draw_keeps_block_proportions(tmp_path):
    (tmp_path / "r14b").mkdir()
    with open(tmp_path / "r14b" / "train.jsonl", "w") as f:
        for i in range(1000):
            f.write(json.dumps({"id": "x%d" % i, "block": "a" if i < 700 else "b" if i < 950 else "c",
                                "questions": {"q": {}}}) + "\n")
    m = load("build").build(str(tmp_path), n=200)
    assert m["by_block"] == {"a": 140, "b": 50, "c": 10} and m["rows"] == 200


@pytest.mark.parametrize("gap,v03,vs30,start", [
    (0.2, -0.05, 0.05, "decision supervision pushed out"),
    (0.4, -0.01, 0.00, "passes this protection screen"),
    (0.4, -0.05, 0.05, "improves on exp30"),
    (0.4, -0.05, 0.00, "direct decision supervision"),
])
def test_outcomes(tmp_path, monkeypatch, gap, v03, vs30, start):
    r = load("report")
    (tmp_path / "stage1").mkdir()
    json.dump({"val": [{"val/mlm_acc_gap": gap}]}, open(tmp_path / "stage1" / "align_report.json", "w"))
    monkeypatch.setattr(r, "recs", lambda d, name: [{"d": d, "correct": True}])
    def metric(a, b, p):
        base = a[0]["d"]
        d = v03 if base == r.BASE["v03"] else vs30
        return {"difference": d, "ci95": [d - 0.02, d - 0.001] if d < 0 else [d - 0.01 if d else -0.02, d + 0.02]}
    monkeypatch.setattr(r, "metric", metric)
    assert r.build(str(tmp_path))["outcome"].startswith(start)
