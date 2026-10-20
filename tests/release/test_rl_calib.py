"""scripts/release/rl_calib.py (critic V04-K-PREP-02-R1-01), no model: the dense magnet pool is the original units, ranges
and precision on integer ticks and contains the old values; the partition is fixed and both views of every drawn family
sit on the calibration side; the base magnet scenes split evenly; a small real drawing renders dense values (which the
old FORCE_VALUES.index could not) with the drawn label equal to the annotation, the gap growing with the distance under
one mapping per unit, and real_layout's own pool restored afterwards."""
import importlib.util
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def rl(monkeypatch):
    monkeypatch.delitem(sys.modules, "build", raising=False)
    spec = importlib.util.spec_from_file_location("rl_calib_t", os.path.join(ROOT, "scripts", "release", "rl_calib.py"))
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_dense_pool(rl):
    old = rl.rl.FORCE_VALUES
    assert {u: (v[0], v[-1]) for u, v in rl.DENSE.items()} == {"in": (0.25, 5.0), "cm": (0.1, 8.0), "mm": (5.5, 70.2)}
    assert all(set(old[u]) <= set(rl.DENSE[u]) for u in old)               # the old values are in it (same floats)
    assert len(rl.DENSE["in"]) == 96 and len(rl.DENSE["cm"]) == 159 and len(rl.DENSE["mm"]) == 648
    assert all(round(x * 100) == x * 100 or abs(round(x * 100) - x * 100) < 1e-9 for x in rl.DENSE["cm"])


def test_families_isolated_partitioned_and_stratified(rl):
    stats: dict = {}
    fams = list(rl.families(set(), {"temp": 30, "force": 30}, stats=stats))
    assert stats["have"] == {"temp": 30, "force": 30} and stats["magnet_base"] == {"distance": 10, "size": 10, "equal": 10}
    for _, form, s, twin in fams:
        keys = {rl.rl.case_key(form, s), rl.rl.case_key(form, twin)}
        assert all(rl.partition(k) == "calib" for k in keys)
    used = {rl.rl.case_key(f, s) for _, f, s, _ in fams[:5]}
    again = list(rl.families(used, {"temp": 30, "force": 30}))
    assert not any({rl.rl.case_key(f, s), rl.rl.case_key(f, t)} & used for _, f, s, t in again)


def test_drawing_dense_values(rl, tmp_path, monkeypatch):
    from PIL import ImageDraw
    drawn = []
    real_text = ImageDraw.ImageDraw.text

    def record(self, xy, text, *a, **k):
        drawn.append(text)
        return real_text(self, xy, text, *a, **k)
    monkeypatch.setattr(ImageDraw.ImageDraw, "text", record)
    monkeypatch.setattr(rl, "used_cases", lambda root: set())
    old = rl.rl.FORCE_VALUES
    man = rl.build(str(tmp_path), want={"temp": 1, "force": 6})
    assert rl.rl.FORCE_VALUES is old and man["pictures"] == 14                # real_layout's pool restored
    rows = [json.loads(x) for x in open(tmp_path / "rl_calib" / "questions.jsonl")]
    dense_only = 0
    for r in rows:
        assert os.path.exists(tmp_path / r["image"])
        if r["kind"] != "force":
            continue
        case = eval(r["case"])
        for d, _ in case[2]:
            label = "%s %s" % (rl.rl.fmt(d), case[1])
            assert label in drawn, label                                  # the annotation is what the picture shows
            dense_only += d not in old[case[1]]
    assert dense_only > 0                                                  # values the old pool did not have were drawn
    s = rl.force_scene_dense(__import__("random").Random(3), "distance")
    gaps = []
    with rl.dense_values():
        for d in sorted(rl.DENSE[s["unit"]])[::40]:
            gaps.append(rl.rl.force_geometry(dict(s, d=[d, d]))["pairs"][0]["gap"])
    assert gaps == sorted(gaps) and gaps[0] < gaps[-1]                     # one mapping per unit, growing with the distance
