# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false
"""exp61 without a model: the layout variants remove blank rows only, keep every field box on its own drawn value (the
EXP45 semantics, the magnets' wide label band) inside the new canvas; family / split isolation; the training loop starts
from the given selector state and changes only selector weights; EXP53-style statistics, family pairing and the
pre-written gates / branch; changed variant data and changed / incomplete read files are refused."""
import importlib.util
import json
import os

import numpy as np
import pytest
import torch
from PIL import Image

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.dirname(HERE))
spec = importlib.util.spec_from_file_location("exp61", os.path.join(HERE, "exp61.py"))
assert spec is not None and spec.loader is not None
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
bspec = importlib.util.spec_from_file_location("exp61_build", os.path.join(HERE, "build.py"))
b = importlib.util.module_from_spec(bspec)
bspec.loader.exec_module(b)

has_data = os.path.exists(os.path.join(ROOT, m.V_MANIFEST))


@pytest.fixture(autouse=True)
def in_root(monkeypatch):
    monkeypatch.chdir(ROOT)


@pytest.mark.skipif(not has_data, reason="data/exp61 not built")
def test_variants_keep_ink_and_boxes():
    qs = m.jsonl(m.V_QUESTIONS)
    sc = b.scenes({q["family"] for q in qs[:60]})
    seen = set()
    for q in qs[:60]:
        img = Image.open(os.path.join("data", q["picture"]))
        assert img.size == (q["width"], q["height"])
        x, y, w, h = q["box"]
        assert x >= 0 and y >= 0 and x + w <= q["width"] + 1e-6 and y + h <= q["height"] + 1e-6
        t = q["text_box"]
        assert x - 1 <= t[0] and y - 1 <= t[1] and x + w + 1 >= t[2] and y + h + 1 >= t[3]          # its value inside
        if q["field"] == "distance":
            assert w / q["width"] > 0.3                                                         # the wide label band
        if q["picture"] in seen:
            continue
        seen.add(q["picture"])
        form, scene = sc[(q["family"], q["view"])]
        orig = b.rl.temp_render(scene) if form == "temp" else b.rl.force_render(scene)
        a = np.asarray(orig.convert("RGB"))
        ink = np.flatnonzero((a < 250).any(-1).any(-1))
        tag = q["picture"].split("/")[-1].rsplit(".", 1)[0]
        top, end = b.crop_params(tag, orig)
        assert top <= ink[0] and end > ink[-1] and img.size[1] == end - top                       # blank rows only
        assert np.array_equal(np.asarray(img.convert("RGB")), a[top:end])                          # nothing else changed


@pytest.mark.skipif(not has_data, reason="data/exp61 not built")
def test_family_split_isolation():
    qs = m.jsonl(m.V_QUESTIONS)
    by = {}
    for q in qs:
        by.setdefault(q["family"], set()).add(q["set"])
    assert all(len(v) == 1 for v in by.values())
    used = {q["family"] for q in m.jsonl(m.QUESTIONS)}
    train = {f for f, v in by.items() if "train_V" in v}
    assert not (train & used) and len(train) == 200
    assert {q["view"] for q in qs if q["family"] == next(iter(train))} == {"v0", "v1"}


def tiny_inputs(n_q=16):
    torch.manual_seed(1)
    return m.sel.Inputs(torch.randn(4, 256, 768).half(), torch.rand(4, 256, 4), [i % 4 for i in range(n_q)],
                        torch.randn(n_q, 5, 1024).half(), torch.ones(n_q, 5, dtype=torch.bool))


def test_training_starts_from_the_given_selector_and_only_moves_it():
    inp = tiny_inputs()
    torch.manual_seed(7)
    init = m.sel.Selector().state_dict()
    targets = torch.rand(16, 4) * 0.5 + 0.25
    same, _ = m.train_from(init, inp, list(range(16)), targets, steps=0)
    assert all(torch.equal(same.state_dict()[k], init[k]) for k in init)
    a, stats = m.train_from(init, inp, list(range(16)), targets, steps=3)
    b2, _ = m.train_from(init, inp, list(range(16)), targets, steps=3)
    assert stats["steps"] == 3 and any(not torch.equal(a.state_dict()[k], init[k]) for k in init)
    assert all(torch.equal(a.state_dict()[k], b2.state_dict()[k]) for k in init)                # seeded
    assert set(a.state_dict()) == set(init)                                                     # the selector only


def rec(fam, case, view, field, obj, target, read, twin_target):
    other = "v1" if view == "v0" else "v0"
    return {"key": "s|%s:%s|%s:%s" % (fam, view, field, obj), "family": fam, "case": case, "view": view, "field": field, "object": obj,
            "picture": "%s_%s" % (fam, view), "target": target, "read": read, "right": read == target,
            "twin": "s|%s:%s|%s:%s" % (fam, other, field, obj), "twin_changed": target != twin_target}


def test_both_counts_only_unequal_pairs():
    """Equal-value pairs read perfectly, unequal pairs poorly: both must follow the unequal pairs only."""
    rs = []
    for k in range(10):
        fam, unequal = "f%d" % k, k < 3
        for v in ("v0", "v1"):
            ta, tb = ("10", "20") if unequal else ("10", "10")
            rs += [rec(fam, "c%d" % k, v, "mass", "A", ta, ta, ta), rec(fam, "c%d" % k, v, "mass", "B", tb, ta if unequal else tb, tb)]
    s = m.stats(rs)
    assert s["both:mass"] == {"value": 0.0, "n": 6} and s["both"] == 0.0               # not 14 / 20
    assert s["twin"] is None and s["real_minus_donor"] is None and s["real_minus_donor_ci"] is None   # no legal sample: missing


def test_macro_over_fields_with_different_denominators():
    rs = []
    for k in range(4):                                      # mass: 4 families with changed values, all right
        rs += [rec("m%d" % k, "cm%d" % k, v, "mass", "A", t, t, o) for v, t, o in (("v0", "1", "2"), ("v1", "2", "1"))]
    rs += [rec("s0", "cs0", v, "speed", "A", t, "9", o) for v, t, o in (("v0", "3", "4"), ("v1", "4", "3"))]   # speed: 1, wrong
    s = m.stats(rs)
    assert s["real_minus_donor:mass"]["n"] == 8 and s["real_minus_donor:speed"]["n"] == 2
    assert s["real_minus_donor"] == pytest.approx((1.0 + 0.0) / 2)                 # macro, not (8 + 0) / 10
    assert s["exact"] == pytest.approx((1.0 + 0.0) / 2)


def test_clusters_link_families_sharing_a_case():
    rs = [rec("f%d" % k, "shared" if k < 5 else "c%d" % k, v, "mass", "A", t, t, o)
          for k in range(8) for v, t, o in (("v0", "1", "2"), ("v1", "2", "1"))]
    assert m.stats(rs)["clusters"] == 4                                              # 5 linked families + 3 alone
    d = m.paired(rs, [dict(r, read="0", right=False) for r in rs])
    assert d["value"] == 1.0 and d["clusters"] == 4


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "runs/exp53/report.json")), reason="runs/exp53 not here")
def test_reproduces_exp53_on_the_registered_old_reads():
    old = m.stats(m.old_rl_tune(m.question_sets()["rl_tune"]))
    ref = json.load(open("runs/exp53/report.json"))["paths"]["auto"]["readout"]["tune_O"]
    for k in ("exact", "both", "twin", "same", "real_minus_donor", "donor_follows", "real_minus_donor_ci", "clusters"):
        assert old[k] == ref[k], k
    assert all(old[k] == v for k, v in ref.items() if isinstance(v, dict))
    assert old["both"] == pytest.approx(0.7071163422497765) and old["clusters"] == 165


def stat(**over):
    x = {"exact": 0.9, "both": 0.8, "twin": 0.8, "real_minus_donor": 0.7, "real_minus_donor_ci": [0.6, 0.8],
         **{"exact:" + f: {"value": 0.9, "n": 10} for f in m.FIELDS}}
    x.update(over)
    return x


def good(**over):
    res = {"tune_V": {"new": stat(), "oracle": stat(), "new_minus_old": {"all": {"ci95": [0.1, 0.3]}}},
           "rl_tune": {"new": stat(), "old": stat()}, "dev24": {"speed_new": 20, "speed_new_minus_old": {"ci95": [0.5, 0.9]}}}
    for k, v in over.items():
        path = k.split("__")
        d = res
        for p_ in path[:-1]:
            d = d[p_]
        d[path[-1]] = v
    return res


@pytest.mark.parametrize("over,text", [
    ({}, "selector repaired by this recipe"),
    ({"tune_V__oracle__exact:speed": {"value": 0.5, "n": 10}}, "region reading unresolved"),
    ({"rl_tune__new__exact:distance": {"value": 0.85, "n": 10}}, "old-layout regression"),
    ({"tune_V__new_minus_old__all": {"ci95": [0.0, 0.2]}}, "did not repair"),
    ({"tune_V__new__exact": 0.79}, "did not repair"),
    ({"tune_V__new__both": 0.49}, "did not repair"),
    ({"dev24__speed_new": 17}, "coverage gap"),
    ({"tune_V__new__twin": None}, "incomplete"),
])
def test_gates_and_branch(over, text):
    assert text in m.branch(m.gates(good(**over)))
    assert m.branch(m.gates({})).startswith("incomplete")


@pytest.mark.skipif(not has_data, reason="data/exp61 not built")
def test_changed_variant_data_is_refused(tmp_path, monkeypatch):
    q = tmp_path / "q.jsonl"
    lines = open(m.V_QUESTIONS).read().splitlines()
    lines[0] = lines[0].replace('"target": "', '"target": "9')
    q.write_text("\n".join(lines) + "\n")
    monkeypatch.setattr(m, "V_QUESTIONS", str(q))
    with pytest.raises(SystemExit, match="differs from its MANIFEST"):
        m.check_variants()


def test_read_files_are_checked(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "START.json").write_text("{}")
    qs = [{"_pic": "p1", "_template": "t1", "target": "5", "key": "k1"}, {"_pic": "p2", "_template": "t1", "target": "6", "key": "k2"}]
    ident = {"identity": {"start_sha256": m.sha(str(out / "START.json")), "set": "x"}}
    rows = [{"picture": "p1", "reads": [{"question": "t1", "decoded": "5", "valid": True}]}]

    def write(rs):
        (out / "x.jsonl").write_text("".join(json.dumps(r) + "\n" for r in [ident] + rs))
        m.register(str(out), "reads_x", {"x.jsonl": m.sha(str(out / "x.jsonl"))})
    write(rows)
    with pytest.raises(SystemExit, match="missing"):
        m.slice_records(str(out), "x", qs, lambda x: x["picture"])
    write(rows + rows)
    with pytest.raises(SystemExit, match="twice"):
        m.slice_records(str(out), "x", qs, lambda x: x["picture"])
    write(rows + [{"picture": "p2", "reads": [{"question": "t1", "decoded": "7", "valid": True}]}])
    got = m.slice_records(str(out), "x", qs, lambda x: x["picture"])
    assert [g["right"] for g in got] == [True, False]
    (out / "x.jsonl").write_text((out / "x.jsonl").read_text() + "\n")
    with pytest.raises(SystemExit, match="registered"):
        m.slice_records(str(out), "x", qs, lambda x: x["picture"])


@pytest.fixture
def started(tmp_path, monkeypatch):
    """A real model-free START in tmp, with copies of the variant list / MANIFEST the test may change."""
    if not has_data:
        pytest.skip("data/exp61 not built")
    out = str(tmp_path / "out")
    q, man = tmp_path / "q.jsonl", tmp_path / "MANIFEST.json"
    q.write_text(open(m.V_QUESTIONS).read())
    man.write_text(open(m.V_MANIFEST).read())
    monkeypatch.setattr(m, "V_QUESTIONS", str(q))
    monkeypatch.setattr(m, "V_MANIFEST", str(man))
    m.stage_start(0, out=out)
    return out, q, man


@pytest.mark.parametrize("field", ["box", "target"])
def test_variant_list_changed_after_start_is_refused_without_pictures(started, field):
    out, q, man = started
    m.check_start(out, data_root="")
    lines = q.read_text().splitlines()
    r = json.loads(lines[0])
    r[field] = [v + 1 for v in r["box"]] if field == "box" else r["target"] + "1"
    lines[0] = json.dumps(r)
    q.write_text("\n".join(lines) + "\n")
    mm = json.loads(man.read_text())
    mm["questions_sha256"] = m.sha(str(q))                     # the current MANIFEST updated too
    man.write_text(json.dumps(mm))
    with pytest.raises(SystemExit, match="data/exp61 differs from START"):
        m.check_start(out, data_root="")                        # train / report path (no picture hashing)
    rep = m.report(out)
    assert rep["branch"].startswith("invalid")


def fake_train(out):
    import shutil
    os.makedirs(os.path.join(out, "bundle"), exist_ok=True)
    open(os.path.join(out, "selector.pt"), "wb").write(b"new selector")
    shutil.copy(os.path.join(out, "selector.pt"), os.path.join(out, "bundle", "selector.pt"))
    json.dump({"sha256": {"selector.pt": m.sha(os.path.join(out, "selector.pt"))}}, open(os.path.join(out, "bundle", "reading.json"), "w"))
    json.dump({"selector_sha256": m.sha(os.path.join(out, "selector.pt")), "bundle_reading_json": m.sha(os.path.join(out, "bundle", "reading.json"))},
              open(os.path.join(out, "train.json"), "w"))
    open(os.path.join(out, "boxes.jsonl"), "w").write("")
    m.register(out, "selector", {f: m.sha(os.path.join(out, f)) for f in ("selector.pt", "boxes.jsonl", "train.json")})
    return m.sha(os.path.join(out, "selector.pt"))


def test_replaced_selector_with_old_products_is_refused(started):
    out, _, _ = started
    st = json.load(open(os.path.join(out, "START.json")))
    h = fake_train(out)
    assert m.new_selector_sha(out) == h
    name = "tune_V_new"
    ident = m.read_identity(out, name, h, st)
    qs = m.question_sets()["tune_V"][:3]
    by = {}
    for q in qs:
        by.setdefault(q["picture"], []).append({"question": q["text"], "decoded": q["target"], "valid": True})
    rows = [{"picture": k, "reads": v} for k, v in by.items()]
    open(os.path.join(out, name + ".jsonl"), "w").write("".join(json.dumps(x) + "\n" for x in [{"identity": ident}] + rows))
    m.register(out, "reads_" + name, {name + ".jsonl": m.sha(os.path.join(out, name + ".jsonl"))})
    assert len(m.slice_records(out, name, qs, lambda x: x["picture"], ident)) == 3
    open(os.path.join(out, "selector.pt"), "wb").write(b"another selector")          # swapped and re-registered
    m.register(out, "selector", {f: m.sha(os.path.join(out, f)) for f in ("selector.pt", "boxes.jsonl", "train.json")})
    with pytest.raises(SystemExit, match="do not name one selector"):
        m.new_selector_sha(out)
    with pytest.raises(SystemExit, match="identity does not bind"):                   # the old reads under the new end
        m.slice_records(out, name, qs, lambda x: x["picture"], m.read_identity(out, name, m.sha(os.path.join(out, "selector.pt")), st))
    rep = m.report(out)
    assert rep["states"]["train"].startswith("missing / invalid") and rep["states"][name].startswith("missing / invalid")
    assert rep["branch"].startswith("incomplete") and rep["candidate_eligible"] is False
