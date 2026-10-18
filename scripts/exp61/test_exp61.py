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


def recs(right_fn, n_fam=10):
    out = []
    for f in range(n_fam):
        for v in ("v0", "v1"):
            for o in ("A", "B"):
                k = "s|f%d:%s|speed:%s" % (f, v, o)
                tgt = str(100 + f + (v == "v1") * 10 + (o == "B"))
                ok = right_fn(f, v, o)
                out.append({"key": k, "family": "f%d" % f, "field": "speed", "object": o, "picture": "p%d%s" % (f, v), "target": tgt,
                            "twin": "s|f%d:%s|speed:%s" % (f, "v1" if v == "v0" else "v0", o), "twin_changed": True,
                            "read": tgt if ok else "0", "right": ok})
    return out


def test_statistics_and_pairing():
    s = m.stats(recs(lambda f, v, o: True))
    assert s["all"] == 1 and s["both"] == 1 and s["twin"] == 1 and s["true_minus_donor"]["value"] == 1
    half = recs(lambda f, v, o: f % 2 == 0)
    d = m.paired(recs(lambda f, v, o: True), half)
    assert d["value"] == 0.5 and d["ci95"][0] > 0


def good(**over):
    st = {"all": 0.9, "all:mass": 0.9, "all:speed": 0.9, "all:distance": 0.9, "both": 0.8, "twin": 0.8,
          "true_minus_donor": {"value": 0.7, "ci95": [0.6, 0.8]}}
    res = {"tune_V": {"new": dict(st), "oracle": dict(st), "new_minus_old": {"all": {"ci95": [0.1, 0.3]}}},
           "rl_tune": {"new": dict(st), "old": dict(st)}, "dev24": {"speed_new": 20, "speed_new_minus_old": {"ci95": [0.5, 0.9]}}}
    for k, v in over.items():
        path = k.split("__")
        d = res
        for p in path[:-1]:
            d = d[p]
        d[path[-1]] = v
    return res


@pytest.mark.parametrize("over,text", [
    ({}, "selector repaired by this recipe"),
    ({"tune_V__oracle__all:speed": 0.5}, "region reading unresolved"),
    ({"rl_tune__new__all:distance": 0.85}, "old-layout regression"),
    ({"tune_V__new_minus_old__all": {"ci95": [0.0, 0.2]}}, "did not repair"),
    ({"tune_V__new__all": 0.79}, "did not repair"),
    ({"dev24__speed_new": 17}, "coverage gap"),
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
