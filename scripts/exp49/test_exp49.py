"""exp49 without a real model: targets and parsing (EOS needed, strict number, no option matching); the loss is
normalised per sample over its valid slots; the head learns a real signal from region tokens alone and stays at chance
with random labels or shuffled picture / label pairs (it cannot read an id); the region cells and their relative
positions follow the box; the extraction keeps only region tokens from the tiny model's own forward; the readout counts
the donor test, both objects and twins on fabricated predictions; features not bound to the question file are refused."""
import importlib.util
import json
import os
import sys

import numpy as np
import pytest
import torch
from PIL import Image

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tests"))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pb = load("exp49_probe", os.path.join(HERE, "probe.py"))
ex = load("exp49_extract", os.path.join(HERE, "extract.py"))
bd = load("exp49_build", os.path.join(HERE, "build.py"))


def test_targets_and_parsing():
    assert bd.target_of("1,360 m/s") == "1360" and bd.target_of("0.75 in") == "0.75"
    with pytest.raises(SystemExit):
        bd.target_of("1, m/s")
    assert pb.labels("12.5") == [1, 2, 10, 5, pb.EOS]
    assert pb.decode([1, 3, 6, 0, pb.EOS]) == {"text": "1360", "valid": True}
    assert not pb.decode([1, 3, 6, 0, 0])["valid"]                                  # no EOS
    assert not pb.decode([10, 5, pb.EOS, 0, 0])["valid"] and not pb.decode([pb.EOS, 1, 1, 1, 1])["valid"]   # ".5", empty
    assert not pb.decode([1, 10, pb.EOS, 0, 0])["valid"]                             # "1."
    with pytest.raises(ValueError):
        pb.labels("12345")                                                         # 5 characters + EOS > 5 slots


def test_loss_is_normalised_per_sample():
    logits = torch.zeros(2, pb.SLOTS, pb.CLASSES, requires_grad=True)
    y = torch.tensor([pb.labels("1"), pb.labels("1234")])
    loss = pb.sample_loss(logits, y)
    assert float(loss) == pytest.approx(float(torch.log(torch.tensor(float(pb.CLASSES)))))   # uniform logits
    loss.backward()
    assert logits.grad is not None and (logits.grad[0, 2:] == 0).all()             # slots after EOS ignored


def synthetic(n, shuffle=False, seed=0):
    """Region tokens that carry the digits of a 2-3 digit target in their first channels."""
    g = torch.Generator().manual_seed(seed)
    targets = [str(int(torch.randint(10, 999, (1,), generator=g))) for _ in range(n)]
    x = torch.randn(n, 6, 32, generator=g) * 0.1
    for i, t in enumerate(targets):
        for k, c in enumerate(t):
            x[i, k, int(c)] += 3.0
            x[i, k, 10 + k] += 3.0
    mask = torch.ones(n, 6, dtype=torch.bool)
    rel = torch.stack([torch.zeros(6, dtype=torch.long), torch.arange(6)], -1).expand(n, -1, -1).contiguous()
    labels_t = targets[:]
    if shuffle:
        perm = torch.randperm(n, generator=g).tolist()
        labels_t = [targets[p] for p in perm]
    return x, mask, rel, torch.tensor([pb.labels(t) for t in labels_t]), labels_t


def exact(head, x, m, r, targets):
    pred = pb.predict(head, x, m, r)
    return sum(pb.decode(p)["valid"] and pb.decode(p)["text"] == t for p, t in zip(pred, targets)) / len(targets)


def test_head_learns_signal_but_not_random_or_shuffled_labels():
    x, m, r, y, t = synthetic(300)
    head = pb.train_head(x[:200], m[:200], r[:200], y[:200], steps=300)
    assert exact(head, x[200:], m[200:], r[200:], t[200:]) > 0.9
    xs, ms, rs, ys, ts = synthetic(300, shuffle=True, seed=1)                      # labels not matching the tokens
    head = pb.train_head(xs[:200], ms[:200], rs[:200], ys[:200], steps=300)
    assert exact(head, xs[200:], ms[200:], rs[200:], ts[200:]) < 0.05


def test_region_cells_and_relative_positions():
    idx, rel = ex.region([0, 0, 64, 32], 256, 256, 16)                              # top-left 64 x 32 px: 4 x 2 cells
    assert idx == [0, 1, 2, 3, 16, 17, 18, 19] and rel[0] == (0, 0) and rel[-1] == (1, 3)
    idx8, rel8 = ex.region([128, 128, 32, 32], 256, 256, 8)
    assert idx8 == [36] and rel8 == [(0, 0)]


def tiny_setup(tmp_path):
    from conftest import tiny_decider  # type: ignore
    d = tiny_decider(seed=1)
    d.save(tmp_path / "run" / "stage2")
    (tmp_path / "data").mkdir()
    qs = []
    for i in range(2):
        Image.new("RGB", (60, 40), (90 * i, 40, 160)).save(tmp_path / "data" / ("%d.png" % i))
        for obj, x in (("sample A", 0), ("sample B", 30)):
            qs.append({"key": "fit_O|%d|%s" % (i, obj), "set": "fit_O", "picture": "%d.png" % i, "box": [x, 10, 30, 20],
                       "width": 60, "height": 40, "field": "mass", "object": obj, "family": "f%d" % i, "case": "c%d" % i,
                       "view": "v0", "target": "12", "twin": None, "twin_changed": False})
    (tmp_path / "q.jsonl").write_text("".join(json.dumps(q) + "\n" for q in qs))
    return d, qs


def test_extraction_keeps_only_region_tokens(tmp_path):
    d, qs = tiny_setup(tmp_path)
    ex.main(["--questions", str(tmp_path / "q.jsonl"), "--run", str(tmp_path / "run"), "--out", str(tmp_path / "out"),
             "--data-root", str(tmp_path / "data"), "--device", "cpu", "--expect-sha", "0"])
    z = np.load(tmp_path / "out" / "features.npz")
    info = json.load(open(tmp_path / "out" / "features.json"))
    assert z["siglip"].shape[0] == 4 and z["proj"].shape[0] == 4
    for i, q in enumerate(qs):
        idx, _ = ex.region(q["box"], q["width"], q["height"], 16)
        assert z["siglip_mask"][i].sum() == len(idx) and not z["siglip_mask"][i, len(idx):].any()
    from devision.train.rlcd import _Images
    with torch.no_grad():
        px = torch.stack([_Images(str(tmp_path / "data"), d)("0.png")])
        patches = d.model.vision_patches(px)
    idx, _ = ex.region(qs[0]["box"], 60, 40, 16)
    assert np.allclose(z["siglip"][0, :len(idx)], patches[0, idx].numpy().astype(np.float16), atol=1e-3)
    assert info["questions_sha256"] and info["features_sha256"]
    (tmp_path / "q.jsonl").write_text((tmp_path / "q.jsonl").read_text() + "\n")    # questions changed after extraction
    with pytest.raises(SystemExit):
        pb.main(["--questions", str(tmp_path / "q.jsonl"), "--features", str(tmp_path / "out")])


def test_readout_counts_on_fabricated_predictions():
    def q(key, setname, pic, obj, target, family, view, twin, changed):
        return {"key": key, "set": setname, "picture": pic, "object": obj, "field": "mass", "target": target, "family": family,
                "case": family, "view": view, "twin": twin, "twin_changed": changed}
    qs = []
    for s in pb.SET_NAME:
        qs += [q(s + "a0", s, s + "p0", "A", "10", "f1", "v0", s + "a1", True), q(s + "b0", s, s + "p0", "B", "20", "f1", "v0", s + "b1", True),
               q(s + "a1", s, s + "p1", "A", "20", "f1", "v1", s + "a0", True), q(s + "b1", s, s + "p1", "B", "10", "f1", "v1", s + "b0", True)]
    y = [pb.labels(x["target"]) for x in qs]
    pred = y[:]                                                                    # every real prediction right
    key_i = {x["key"]: i for i, x in enumerate(qs)}
    donor = [y[key_i[x["twin"]]] for x in qs]                                      # donor features read the twin's value
    r = pb.readout(qs, [[c if c != -100 else 0 for c in p] for p in pred], [[c if c != -100 else 0 for c in p] for p in donor], y)
    for name in ("fit_O", "tune_O"):
        assert r[name]["exact"] == 1.0 and r[name]["both"] == 1.0 and r[name]["twin"] == 1.0 and r[name]["same"] == 0.0
        assert r[name]["real_minus_donor"] == 1.0 and r[name]["donor_follows"] == 1.0


def test_ambiguous_regions_are_flagged():
    base = {"set": "fit_O", "picture": "p.png", "width": 256, "height": 256, "family": "f", "case": "c", "view": "v0"}
    qs = [dict(base, key="a", box=[0, 0, 60, 10], target="1"), dict(base, key="b", box=[0, 4, 60, 8], target="2"),
          dict(base, key="c", box=[128, 128, 60, 10], target="3")]
    assert pb.ambiguous(qs, 8) == [True, True, False]                              # rows 0-9 and 4-11 share the same cells
    assert pb.ambiguous(qs, 16) == [True, True, False]                             # both inside the first 16-px row
