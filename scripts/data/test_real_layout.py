"""Real-layout bridge data: labels follow the drawn values (temperature M * V^2; force closer / larger), the reading
questions share one candidate set per field, one magnet scale and distance mapping per force picture, twins change
only the values (pictures and requests), splits share no canonical case."""
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from real_layout import (TEMP_SAME, build, force_geometry, force_items, force_render, force_scene, force_twin,  # noqa: E402
                         temp_items, temp_render, temp_scene, temp_twin)


def test_temperature_labels():
    rng = random.Random(1)
    for k in range(300):
        s = temp_scene(rng)
        cap, items = temp_items(s, k, [s, temp_twin(s)])
        ka, kb = (s[x]["mass"] * s[x]["speed"] ** 2 for x in "AB")
        assert items[0]["gold"] == ("sample A" if ka > kb else "sample B" if kb > ka else TEMP_SAME)
        assert all(it["gold"] in it["options"] and len(set(it["options"])) == 3 for it in items)
        assert items[1]["options"] == items[2]["options"] and items[3]["options"] == items[4]["options"]
        assert "mass of each particle %d u" % s["A"]["mass"] in cap and cap.index("Sample A") < cap.index("Sample B")


def test_force_labels():
    rng = random.Random(2)
    for k in range(300):
        s = force_scene(rng)
        cap, items = force_items(s, k, [s, force_twin(s)])
        g = items[0]["gold"]
        if s["case"] == "equal":
            assert "same" in g
        else:
            stronger = (1 if s["d"][0] < s["d"][1] else 2) if s["case"] == "distance" else (1 if all(s["big"][0]) else 2)
            weak = "weaker" in g or "smaller" in g
            assert ("Pair %d" % (3 - stronger if weak else stronger)) in g
        assert items[1]["options"] == items[2]["options"] and all(it["gold"] in it["options"] for it in items)


def test_force_geometry():
    """Equal distances -> equal gaps; a longer distance -> a wider gap; every large magnet the same size, every
    small one the same size, in both pairs; magnets inside their box."""
    rng = random.Random(4)
    for _ in range(500):
        s = force_scene(rng)
        for v in (s, force_twin(s)):
            g = force_geometry(v)
            a, b = g["pairs"]
            if v["d"][0] == v["d"][1]:
                assert a["gap"] == b["gap"]
            else:
                assert (a["gap"] < b["gap"]) == (v["d"][0] < v["d"][1])
            sizes = {(v["big"][p][j], g["pairs"][p]["sizes"][j]) for p in range(2) for j in range(2)}
            assert len({z for big, z in sizes if big}) == 1 and len({z for big, z in sizes if not big}) <= 1
            for q in g["pairs"]:
                assert q["box"] <= q["start"] and q["start"] + sum(q["widths"]) + q["gap"] <= q["box"] + g["pw"]


def test_twins_change_only_values():
    """Within a family: the questions (wording, options, their order) are the same; the pictures differ only where
    the values are drawn (the text rows, particle sizes / magnets and gaps), not in the layout."""
    rng = random.Random(5)
    for k in range(60):
        s = temp_scene(rng)
        t = temp_twin(s)
        qa, qb = temp_items(s, k, [s, t])[1], temp_items(t, k, [s, t])[1]
        assert [(x["q"], x["options"]) for x in qa] == [(x["q"], x["options"]) for x in qb]
        assert qa[0]["gold"] != qb[0]["gold"] or s["case"] == "equal"
        ia, ib = np.asarray(temp_render(s)), np.asarray(temp_render(t))
        assert ia.shape == ib.shape
        if s["case"] != "equal" and s["A"]["mass"] == s["B"]["mass"]:   # same particles, only the speeds swap
            assert (ia[:305] == ib[:305]).all()
        f = force_scene(rng)
        g = force_twin(f)
        fa, fb = force_items(f, k, [f, g])[1], force_items(g, k, [f, g])[1]
        assert [(x["q"], x["options"]) for x in fa] == [(x["q"], x["options"]) for x in fb]
        assert {x: f[x] for x in f if x not in ("d", "big", "case")} == {x: g[x] for x in g if x not in ("d", "big", "case")}
        assert np.asarray(force_render(f))[:22].tolist() == np.asarray(force_render(g))[:22].tolist()   # the titles


def test_pictures_and_build(tmp_path):
    rng = random.Random(3)
    for _ in range(20):
        assert temp_render(temp_scene(rng)).size[1] >= 405
        assert 580 <= force_render(force_scene(rng)).size[0] <= 700
    m = build(str(tmp_path), n=40, n_tune=10, n_confirm=10)
    assert m["families"] == {"train_temp": 20, "train_force": 20, "tune_temp": 5, "tune_force": 5, "confirm_temp": 5, "confirm_force": 5}
    assert m["questions_train"]["rows"] == 80 and m["captions_tune"]["rows"] == 20
    rows = {sp: [json.loads(x) for x in open(tmp_path / "real_layout" / ("questions_%s.jsonl" % sp))] for sp in ("train", "tune", "confirm")}
    cases = [{r["case"] for r in rows[sp]} for sp in rows]
    fams = [{r["family"] for r in rows[sp]} for sp in rows]
    assert not (cases[0] & cases[1] or cases[0] & cases[2] or cases[1] & cases[2])
    assert not (fams[0] & fams[1] or fams[0] & fams[2] or fams[1] & fams[2])
    assert set(m["case_overlap"].values()) == {0}
