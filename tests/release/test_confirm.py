"""scripts/release/confirm.py (critic V04-K-CONFIRM-PREP-01), no model, no confirmation set: a preview (explicit seed,
"preview:" ids, at most 4 + 4 pictures) is drawn by the independent renderer only -- real_layout's renderers are never
called -- with the drawn values equal to the questions' facts and golds recomputed from the rules; selection keeps the
classes, the A / B balance, the confirm side and the isolation; the wrong-picture pairing is a derangement inside each
domain; the formal entry refuses (before drawing anything) without a complete frozen identity of this checkpoint with
the K rule and the LIVE-K lineage, after the checkpoint changed, or into an existing directory, and accepts the freeze of
a tiny LIVE-K-like candidate (its set is not generated here); the pre-written outcome."""
import importlib.util
import json
import os
import sys

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture
def cf(monkeypatch):
    monkeypatch.delitem(sys.modules, "build", raising=False)
    return load("confirm_t", "scripts/release/confirm.py")


def gold_of(r, q):
    p = r["gold"][q]["probabilities"]
    return max(p, key=p.get)


def test_preview_is_drawn_by_the_independent_renderer(cf, tmp_path, monkeypatch):
    from PIL import ImageDraw
    drawn = []
    real_text = ImageDraw.ImageDraw.text
    monkeypatch.setattr(ImageDraw.ImageDraw, "text", lambda self, xy, text, *a, **k: (drawn.append(text), real_text(self, xy, text, *a, **k))[1])
    for name in ("temp_render", "force_render"):
        monkeypatch.setattr(cf.rl, name, lambda *a, **k: pytest.fail("real_layout's renderer was used"))
    monkeypatch.setattr(cf, "excluded", lambda root: set())
    man = cf.preview(str(tmp_path / "p"), seed=7, n_temp=4, n_force=3)
    rows = [json.loads(x) for x in open(tmp_path / "p" / "preview.jsonl")]
    assert man["preview"] is True and man["pictures"] == 7 and all(r["id"].startswith("preview:") for r in rows)
    for r in rows:
        case = eval(r["case"])
        assert os.path.exists(tmp_path / "p" / r["image"])
        if r["kind"] == "temp":
            (ma, va), (mb, vb) = case[2]
            texts = {"Mass of each particle: %d u" % ma, "Mass of each particle: %d u" % mb,
                     "Average particle speed: %s m/s" % cf.rl.fmt(va), "Average particle speed: %s m/s" % cf.rl.fmt(vb)}
            assert texts <= set(drawn)
            assert len(r["questions"]) == 5 and gold_of(r, "q1").endswith(" u")
        else:
            for d, _ in case[2]:
                assert "%s %s" % (cf.rl.fmt(d), case[1]) in drawn
            assert len(r["questions"]) == 3
    with pytest.raises(SystemExit, match="at most"):
        cf.preview(str(tmp_path / "big"), seed=7, n_temp=5)


def test_golds_follow_the_rules(cf):
    import random
    rng = random.Random(3)
    for case in ("same mass", "same speed", "both differ"):
        for hotter in "AB":
            s = cf.temp_scene(rng, case, hotter)
            items = cf.items_of("temp", s, 1)
            ka, kb = s["A"]["mass"] * s["A"]["speed"] ** 2, s["B"]["mass"] * s["B"]["speed"] ** 2
            assert items[0]["gold"] == "sample %s" % hotter and (ka > kb) == (hotter == "A")
    s = cf.temp_scene(rng, "equal", None)
    assert cf.items_of("temp", s, 1)[0]["gold"] == cf.rl.TEMP_SAME
    for k in range(8):                                             # the four wordings, gold flipped as defined
        s = cf.force_scene(rng, "distance")
        g = cf.items_of("force", s, k)[0]["gold"]
        closer = 1 if s["d"][0] < s["d"][1] else 2
        strong = ("greater" in g or "stronger" in g)
        assert ("Pair %d" % closer in g) == strong


def test_selection_classes_balance_isolation_and_side(cf):
    used = set()
    first = cf.select(11, used, {"same mass": 4, "same speed": 4, "both differ": 4, "equal": 2}, {"distance": 3, "size": 3, "equal": 3})
    keys = [cf.rl.case_key(f, s) for f, s in first]
    assert len(set(keys)) == len(keys) and all(cf.rlc.partition(k) == "confirm" for k in keys)
    from collections import Counter
    temp = [(s["case"], "A" if s["A"]["mass"] * s["A"]["speed"] ** 2 > s["B"]["mass"] * s["B"]["speed"] ** 2 else "B") for f, s in first if f == "temp" and s["case"] != "equal"]
    c = Counter(temp)
    assert all(abs(c[(k, "A")] - c[(k, "B")]) <= 1 for k in ("same mass", "same speed", "both differ"))
    again = cf.select(11, set(keys[:5]), {"same mass": 4, "same speed": 4, "both differ": 4, "equal": 2}, {"distance": 3, "size": 3, "equal": 3})
    assert not {cf.rl.case_key(f, s) for f, s in again} & set(keys[:5])


def test_wrong_pictures_are_a_derangement_inside_each_domain(cf):
    rows = [{"id": "t%d" % i, "kind": "temp", "image": "t%d.png" % i, "image_id": "t%d" % i} for i in range(6)] + \
           [{"id": "f%d" % i, "kind": "force", "image": "f%d.png" % i, "image_id": "f%d" % i} for i in range(5)]
    mism = cf.mismatched(rows, 5)
    assert len(mism) == 11
    for m_, r in zip(mism, [r for f in ("temp", "force") for r in rows if r["kind"] == f]):
        assert m_["image"] != r["image"] and m_["image"][0] == r["image"][0]


def frozen_tiny(tmp_path, monkeypatch):
    ks = load("ks_confirm", "tests/release/test_k_source.py")
    pins = ks.livek_run(tmp_path)
    ks.source(tmp_path, pins)
    cand = ks.candidate(tmp_path, pins)
    ks.rb.bind(str(cand), pins)
    ce = ks.ce
    setattr(ce, "SOURCE_PINS", pins)
    (tmp_path / "set.jsonl").write_text("")
    conf = tmp_path / "full.yaml"
    conf.write_text(yaml.safe_dump({"name": "full", "stages": [], "evaluate": {"checkpoint": str(cand), "sets": {"a": {"path": str(tmp_path / "set.jsonl"), "role": "heldout"}}}}))
    assert ce.main(["freeze", str(conf)]) == 0
    return pins, cand, tmp_path / "round" / "eval" / "IDENTITY.json"


def test_the_formal_entry_needs_the_freeze(cf, tmp_path, monkeypatch):
    pins, cand, ident = frozen_tiny(tmp_path, monkeypatch)
    assert cf.freeze_problems(str(ident), str(cand), pins) == []          # a valid freeze is accepted (no set generated here)
    monkeypatch.setattr(cf, "select", lambda *a, **k: pytest.fail("selected before the freeze checks"))
    out = tmp_path / "confirm"
    with pytest.raises(SystemExit, match="no frozen identity"):
        cf.generate(str(out), str(tmp_path / "nope.json"), str(cand), pins=pins)
    partial = tmp_path / "partial.json"
    j = json.loads(ident.read_text())
    j.pop("code")
    partial.write_text(json.dumps(j))
    with pytest.raises(SystemExit, match="not a complete frozen identity"):
        cf.generate(str(out), str(partial), str(cand), pins=pins)
    with pytest.raises(SystemExit, match="is frozen for"):
        cf.generate(str(out), str(ident), str(tmp_path / "end"), pins=pins)
    with pytest.raises(SystemExit, match="not the pinned source|not the required source"):
        cf.generate(str(out), str(ident), str(cand), pins=None)              # the release pins (real LIVE-K), not this tiny source
    with open(cand / "reading" / "head.pt", "ab") as fh:
        fh.write(b"x")
    with pytest.raises(SystemExit, match="differs from the frozen identity"):
        cf.generate(str(out), str(ident), str(cand), pins=pins)
    assert not out.exists()


def test_the_formal_entry_writes_only_new_output(cf, tmp_path, monkeypatch):
    pins, cand, ident = frozen_tiny(tmp_path, monkeypatch)
    out = tmp_path / "exists"
    out.mkdir()
    with pytest.raises(SystemExit, match="fresh only"):
        cf.generate(str(out), str(ident), str(cand), pins=pins)


def test_outcome(cf):
    ids = ["v04-confirm:p%03d" % i for i in range(120)]
    v03 = {i: n % 2 == 0 for n, i in enumerate(ids)}                        # 0.5
    good = {i: n % 10 != 0 for n, i in enumerate(ids)}                      # 0.9
    wrong = {i: n % 3 == 0 for n, i in enumerate(ids)}
    assert cf.outcome(good, wrong, v03, ids)["branch"] == "synthetic confirmation passed"
    assert cf.outcome(v03, wrong, v03, ids)["branch"].startswith("synthetic confirmation not passed")
    assert cf.outcome(good, good, v03, ids)["branch"].startswith("synthetic confirmation not passed")    # no own picture gain
    assert cf.outcome(good, wrong, dict(list(v03.items())[:-1]), ids)["branch"].startswith("no verdict")


@pytest.mark.parametrize("case", ["common_missing", "one_picture", "empty", "foreign_ids", "not_bool", "not_bool_int"])
def test_outcome_needs_the_whole_formal_set(cf, case):
    """critic V04-K-CONFIRM-R1-02: the formal 120 ids, each once in every set, explicit bools -- else no verdict."""
    ids = ["v04-confirm:p%03d" % i for i in range(120)]
    t: dict = {i: True for i in ids}
    w = {i: False for i in ids}
    v = {i: False for i in ids}
    if case == "common_missing":
        for d in (t, w, v):
            d.pop(ids[5])
    elif case == "one_picture":
        t, w, v = ({ids[0]: x} for x in (True, False, False))
    elif case == "empty":
        t, w, v = {}, {}, {}
    elif case == "foreign_ids":
        t, w, v = ({"other:%03d" % i: x for i in range(120)} for x in (True, False, False))
    elif case == "not_bool":
        t[ids[3]] = "yes"
    else:
        t[ids[3]] = 1
    for expected in (ids, ids[:1], []):
        res = cf.outcome(t, w, v, expected)
        assert res["branch"].startswith("no verdict"), (case, len(expected), res["branch"])
    assert cf.outcome({i: True for i in ids}, {i: False for i in ids}, {i: False for i in ids}, ids)["branch"] == "synthetic confirmation passed"
    import inspect
    assert list(inspect.signature(cf.outcome).parameters) == ["cand_true", "cand_wrong", "v03_true", "expected"] and cf.PICTURES == 120
    one = ids[:1]                                                        # a one-picture "set" can never pass (no n override)
    assert cf.outcome({one[0]: True}, {one[0]: False}, {one[0]: False}, one)["branch"].startswith("no verdict")


@pytest.mark.parametrize("case,why", [("forged", "not a runtime code record"), ("missing_module", "required but not loaded"),
                                      ("missing_devision", "module devision: required but not loaded"),
                                      ("missing_cli", "module devision.train.cli: required but not loaded"),
                                      ("changed_module", "not this implementation"), ("moved_same_content", None)])
def test_the_frozen_code_must_be_this_implementation(cf, tmp_path, monkeypatch, case, why):
    """critic V04-K-CONFIRM-R1-01: checked before anything is selected or drawn."""
    pins, cand, ident = frozen_tiny(tmp_path, monkeypatch)
    j = json.loads(ident.read_text())
    if case == "forged":
        j["code"] = {"anything": 1}
    elif case == "missing_module":
        j["code"]["modules"].pop("devision.image")
    elif case == "missing_devision":
        j["code"]["modules"].pop("devision")
    elif case == "missing_cli":
        j["code"]["modules"].pop("devision.train.cli")
    elif case == "changed_module":
        j["code"]["modules"]["devision.reading"]["sha256"] = "0" * 64      # the frozen reading.py is not today's
    else:
        j["code"]["modules"]["devision.reading"]["path"] = "/elsewhere/devision/reading.py"
    edited = tmp_path / "IDENTITY_edited.json"
    edited.write_text(json.dumps(j))
    monkeypatch.setattr(cf, "select", lambda *a, **k: pytest.fail("selected before the freeze checks"))
    out = tmp_path / "confirm"
    if why is None:
        assert cf.freeze_problems(str(edited), str(cand), pins) == []
        return
    with pytest.raises(SystemExit, match=why):
        cf.generate(str(out), str(edited), str(cand), pins=pins)
    assert not out.exists()
