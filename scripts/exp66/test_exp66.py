# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportOptionalSubscript=false
"""exp66 without a model: the moved-column renderer is the generator at shift 0 (pixels, rows, bands); every variant's
layout is its family's (shared by both views, recomputed from the hash), inside its physical limits, labelled by the
drawn rows; a cut glyph or an ink-removing crop is refused; train_C / tune_C / earlier families are isolated; every
train_C question has its parent; the audit's ink measures on a drawn picture."""
import importlib.util
import json
import os
import random
from collections import Counter, defaultdict

import numpy as np
import pytest

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.dirname(HERE))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, rel))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


b = _load("exp66_build", "build.py")
au = _load("exp66_audit", "audit.py")
rl, b45 = b.rl, b.b45
has = os.path.exists(os.path.join(ROOT, "data", "exp66", "MANIFEST.json"))
data = pytest.mark.skipif(not has, reason="data/exp66 not built")


@pytest.fixture(autouse=True)
def in_root(monkeypatch):
    monkeypatch.chdir(ROOT)


def scenes(n=12):
    rng = random.Random(5)
    return [rl.temp_scene(rng) for _ in range(n)]


def test_shift_zero_is_the_generator():
    for s in scenes():
        assert np.array_equal(np.asarray(b.temp_render_at(s, 0)), np.asarray(rl.temp_render(s)))
        assert b.temp_rows_at(s, 0) == b45.temp_rows(s)
        for o in "AB":
            for f in ("mass", "speed"):
                band, text, _ = b45.original_box("temp", s, o, f)
                assert b.temp_band(s, 0, o, f) == pytest.approx(band) and b.temp_texts(s, 0)[o][f] == pytest.approx(text)


def test_shift_moves_the_column_and_its_labels():
    s = scenes(1)[0]
    p = {"top": 0, "shift": 10, "bottom_margin": 6}
    img, boxes, _ = b.render("temp", s, p)
    _, base, _ = b.render("temp", s, dict(p, shift=0))
    left = "A" if s.get("a_left", True) else "B"
    assert boxes[("mass", left)][1][0] == pytest.approx(base[("mass", left)][1][0] - 10)       # the text moved out by 10
    a = np.asarray(img)
    t = [int(round(v)) for v in boxes[("mass", left)][1]]
    assert b.text_mask(a[t[1]:t[3], t[0]:t[2]]).any()                                          # and is drawn there


def test_cut_text_and_ink_crop_are_refused():
    s = scenes(1)[0]
    sm = b.max_shift([s])
    with pytest.raises(SystemExit):
        b.render("temp", s, {"top": 0, "shift": sm + 40, "bottom_margin": 0})
    first, _ = b.ink_rows(rl.temp_render(s))
    with pytest.raises(SystemExit, match="crop"):
        b.render("temp", s, {"top": first + 1, "shift": 0, "bottom_margin": 0})
    img, boxes, _ = b.render("temp", s, {"top": 0, "shift": sm, "bottom_margin": 0})                # the limit itself is fine
    assert all(t[0] >= 0 and t[2] <= img.size[0] for _, t in boxes.values())


@data
def test_variants():
    qs = [json.loads(x) for x in open("data/exp66/questions.jsonl")]
    m = json.load(open("data/exp66/MANIFEST.json"))
    assert m["questions"] == {"train_C": 1200, "tune_C": 1800, "train_C_parent": 1200}
    by = defaultdict(list)
    for q in qs:
        by[(q["set"], q["family"])].append(q)
    sc = b.b61.scenes({f for _, f in by})
    for (st, fam), rows in by.items():
        if st == "train_C_parent":
            continue
        form = sc[(fam, "v0")][0]
        p = b.params(fam, form, [sc[(fam, "v0")][1], sc[(fam, "v1")][1]])
        assert all(q["layout_params"] == p for q in rows)                            # the family's, shared by its views
        assert len({(q["width"], q["height"]) for q in rows}) == 1
        if form == "temp":
            assert -p["shift_in_max"] <= p["shift"] <= p["shift_max"]
        for q in rows:
            t = q["text_box"]
            assert 0 <= t[0] and t[2] <= q["width"] and 0 <= t[1] and t[3] <= q["height"]
            assert q["target"] == b.target_of(form, sc[(fam, q["view"])][1], q["field"], q["object"])
    fams = {st: {f for s, f in by if s == st} for st in ("train_C", "tune_C")}
    used = {json.loads(x)["family"] for x in open("data/exp49/questions.jsonl")}
    used61 = {json.loads(x)["family"] for x in open("data/exp61/questions.jsonl") if json.loads(x)["set"] == "train_V"}
    assert not (fams["train_C"] & fams["tune_C"]) and not (fams["train_C"] & (used | used61))
    cases = {st: {q["case"] for q in qs if q["set"] == st} for st in ("train_C", "tune_C")}
    assert not cases["train_C"] & cases["tune_C"]
    keys = {q["key"] for q in qs if q["set"] == "train_C"}
    pairs = Counter(q["pair"] for q in qs if q["set"] == "train_C_parent")
    assert set(pairs) == keys and set(pairs.values()) == {1}
    assert Counter(sc[(f, "v0")][0] for f in fams["train_C"]) == {"temp": 100, "force": 100}


def test_audit_measures_drawn_ink(tmp_path):
    s = scenes(1)[0]
    img, boxes, _ = b.render("temp", s, {"top": 0, "shift": 0, "bottom_margin": 3})
    p = str(tmp_path / "x.png")
    img.save(p)
    left = "A" if s.get("a_left", True) else "B"
    box, text = boxes[("speed" if not s.get("speed_first") else "mass", left)]
    m = au.measure(p, img.size[0], img.size[1], box, au.xyxy(box))
    assert m["bottom_margin"] == 3 and m["holds"] and abs(m["col_x"] - 0.25) < 0.01
    assert m["side_margin"] == pytest.approx(m["ink"][0]) and 0.3 < m["rel_cy"] < 0.7
