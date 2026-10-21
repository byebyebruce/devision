"""exp37 without a model: the report's main comparison and conditions on fabricated records (L right where S is
not), and the renderer's shared layout (an S picture and its L picture differ only inside the number boxes, every
digit inside its box, L digit ink 15-17 px at 256)."""
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "exp36"))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_report(tmp_path, monkeypatch):
    r = load("exp37_report", os.path.join(HERE, "report.py"))
    t36 = load("exp36_test", os.path.join(HERE, "..", "exp36", "test_exp36.py"))
    runs = {}
    for m, right in (("S", lambda f, x: False), ("L", lambda f, x: True)):
        for inp in r.INPUTS:
            d = tmp_path / m / ("numbers_" + inp)
            d.mkdir(parents=True)
            with open(d / "numbers.jsonl", "w") as fh:
                for x in t36.records(right, lambda f, x: False):
                    fh.write(json.dumps(x) + "\n")
        runs[m] = str(tmp_path / m)
    monkeypatch.setattr(r.r36._exp30, "recs", lambda d, name: [])
    monkeypatch.setattr(r.r36._exp30, "metric", lambda a, b, p: {"difference": 0.0, "ci95": [-0.01, 0.01]})
    out = r.build(runs)
    assert out["L_minus_S"]["whole"]["value"] == 1.0
    assert out["conditions"]["whole_macro"] and out["conditions"]["picture_dependence"]


def test_shared_layout():
    import numpy as np
    R = load("exp37_render", os.path.join(HERE, "render.py"))
    seen = {"temp": 0, "force": 0}
    for i, split, form, s, twin in R.rl.families(6000, 300, 300):
        if seen[form] >= 3:
            if min(seen.values()) >= 3:
                break
            continue
        seen[form] += 1
        a, nums = R.render(form, s, "S")
        b, _ = R.render(form, s, "L")
        blank, _ = R.render(form, s, "L", digits=False)
        boxes = [n["box"] for n in nums]
        assert R.outside_boxes_equal(a, b, boxes) and R.outside_boxes_equal(b, blank, boxes)
        bare, _ = R.render(form, s, "L", strip=True)
        assert all(15 <= R.ink_rows(bare, blank, n["box"], bare) <= 17 for n in nums)
        assert len(nums) == (4 if form == "temp" else 2)
