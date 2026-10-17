"""Real-layout bridge data: labels follow the drawn values (temperature M * V^2; force closer / larger), the reading
questions share one candidate set per field, every gold is an option, pictures are drawn inside their frame."""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
from real_layout import TEMP_SAME, build, force_items, force_render, force_scene, temp_items, temp_render, temp_scene  # noqa: E402


def test_temperature_labels():
    rng = random.Random(1)
    for k in range(300):
        s = temp_scene(rng)
        cap, items = temp_items(s, k)
        ka, kb = (s[x]["mass"] * s[x]["speed"] ** 2 for x in "AB")
        assert items[0]["gold"] == ("sample A" if ka > kb else "sample B" if kb > ka else TEMP_SAME)
        assert all(it["gold"] in it["options"] and len(set(it["options"])) == 3 for it in items)
        assert set(items[1]["options"]) == set(items[2]["options"]) and set(items[3]["options"]) == set(items[4]["options"])
        assert "mass of each particle %d u" % s["A"]["mass"] in cap


def test_force_labels():
    rng = random.Random(2)
    for k in range(300):
        s = force_scene(rng)
        cap, items = force_items(s, k)
        g = items[0]["gold"]
        if s["case"] == "equal":
            assert "same" in g
        else:
            stronger = (1 if s["d"][0] < s["d"][1] else 2) if s["case"] == "distance" else (1 if all(s["big"][0]) else 2)
            weak = "weaker" in g or "smaller" in g
            assert ("Pair %d" % (3 - stronger if weak else stronger)) in g
        assert set(items[1]["options"]) == set(items[2]["options"])


def test_pictures_and_build(tmp_path):
    rng = random.Random(3)
    for _ in range(20):
        assert temp_render(temp_scene(rng)).size[1] >= 405
        im = force_render(force_scene(rng))
        assert 580 <= im.size[0] <= 700
    m = build(str(tmp_path), n=8, n_tune=2, n_confirm=2)
    assert m["questions_train"]["rows"] == 16 and m["captions_tune"]["rows"] == 4 and m["questions_confirm"]["rows"] == 4
    import json
    rows = {sp: [json.loads(x) for x in open(tmp_path / "real_layout" / ("questions_%s.jsonl" % sp))] for sp in ("train", "tune", "confirm")}
    fams = [{r["family"] for r in rows[sp]} for sp in rows]
    assert not (fams[0] & fams[1] or fams[0] & fams[2] or fams[1] & fams[2])


def test_twins_change_only_values():
    from real_layout import force_twin, temp_twin
    rng = random.Random(5)
    for k in range(200):
        s = temp_scene(rng)
        t = temp_twin(s)
        assert {x: s[x] for x in s if x not in ("A", "B", "case")} == {x: t[x] for x in t if x not in ("A", "B", "case")}
        assert temp_items(s, k)[1][0]["gold"] != temp_items(t, k)[1][0]["gold"] or s["case"] == "equal"
        f = force_scene(rng)
        g = force_twin(f)
        assert {x: f[x] for x in f if x not in ("d", "big", "case")} == {x: g[x] for x in g if x not in ("d", "big", "case")}
