"""exp51 without the real model: the region tokens are exactly the patches (full picture + 12 views) overlapping the field
box on view content, with rel / geometry as documented; gathered features are visual_inputs' keys / values at those
indices, the full-picture tokens identical for T and C; the donor swap changes only the features and refuses a
different region; the vision-identity check refuses one changed byte; the conditions / attribution / branch are
computed from per-question predictions and are incomplete without both conditions."""
import importlib.util
import json
import os
import struct
import sys

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tests"))


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, rel))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


reg = load("exp51_regions", "regions.py")
ext = load("exp51_extract", "extract.py")
prb = load("exp51_probe", "probe.py")
vid = load("exp51_vision", "vision_identity.py")


def test_region_tokens_are_the_overlapping_content_patches():
    from devision.image import patch_geometry, view_rects
    w, h, box = 560, 405, [10.0, 300.0, 260.0, 26.0]
    r = reg.region(box, w, h)
    views = [(0, 0, w, h)] + view_rects(w, h, 12, 0.25)
    want = set()
    for k, v in enumerate(views):                       # brute force on a 0.5-px lattice
        g = patch_geometry(v, w, h, 256, 16).numpy()
        for i, (cx, cy, pw, ph) in enumerate(g):
            xs = np.arange(max(cx * w - pw * w / 2, v[0], box[0]) + 0.25, min(cx * w + pw * w / 2, v[2], box[0] + box[2]), 0.5)
            ys = np.arange(max(cy * h - ph * h / 2, v[1], box[1]) + 0.25, min(cy * h + ph * h / 2, v[3], box[1] + box[3]), 0.5)
            if len(xs) and len(ys):
                want.add(k * 256 + i)
    assert set(r["index"]) == want and 0 in r["view"] and len(set(r["view"])) > 1
    assert all(0 <= a < 16 and 0 <= b < 16 for a, b in r["rel"])
    for k in set(r["view"]):                            # rel from the first selected cell of the same view
        cells = [i % 256 for i, v in zip(r["index"], r["view"]) if v == k]
        rel = [rr for rr, v in zip(r["rel"], r["view"]) if v == k]
        assert min(a for a, _ in rel) == 0 and min(b for _, b in rel) == 0 and len(cells) == len(rel)
    gx = [g[0] for g in r["geom"]]                      # geometry relative to the box: centres around [0, 1]
    assert min(gx) < 0.1 and max(gx) > 0.9


def test_region_ignores_the_letterbox_margin():
    """A box touching only a view's grey margin area gets no token from that view."""
    w, h = 650, 140
    r = reg.region([0.0, 0.0, 2.0, 2.0], w, h)          # the top-left corner: only views holding it
    from devision.image import view_rects
    views = [(0, 0, w, h)] + view_rects(w, h, 12, 0.25)
    assert all(views[k][0] <= 2 and views[k][1] <= 2 for k in set(r["view"]))


def tiny_local():
    from conftest import tiny_decider  # type: ignore
    d = tiny_decider(seed=3)
    d.model.add_question_conditioning()
    d.model.enable_local_views(12)
    return d.model.eval()


def test_gathered_tokens_are_visual_inputs_patches_and_full_branch_is_shared():
    from devision.image import local_pixels, to_pixel_values, view_rects
    m = tiny_local()
    img = Image.new("RGB", (560, 405), "white")
    ImageDraw.Draw(img).text((20, 310), "Mass 1,360 u", fill="black")
    r = reg.region([10.0, 300.0, 260.0, 26.0], 560, 405)
    with torch.no_grad():
        kv = {s: m.visual_inputs(local_pixels(img, 256, 12, 0.25, s)[None])[0][0].numpy() for s in "TC"}
        rects = view_rects(560, 405, 12, 0.25)
        k = max(r["view"])
        own = m.vision_patches(to_pixel_values(img.crop(rects[k - 1]), 256)[None])[0].numpy()
    t = ext.gather(kv["T"], r)
    for n, (i, v) in enumerate(zip(r["index"], r["view"])):
        if v == k:
            assert np.allclose(t[n], own[i % 256], atol=1e-5)
    c = ext.gather(kv["C"], r)
    full = [n for n, v in enumerate(r["view"]) if v == 0]
    assert np.array_equal(t[full], c[full])                                       # same full-picture tokens
    assert not np.array_equal(t, c)                                               # views differ in pixels


def test_donor_swap_and_refusal():
    regs = [{"index": [1, 2], "rel": [[0, 0], [0, 1]], "geom": [[0, 0, 1, 1], [1, 0, 1, 1]]},
            {"index": [1, 2], "rel": [[0, 0], [0, 1]], "geom": [[0, 0, 1, 1], [1, 0, 1, 1]]},
            {"index": [5], "rel": [[0, 0]], "geom": [[0, 0, 1, 1]]}]
    flat = np.arange(5 * 3, dtype=np.float16).reshape(5, 3)
    tok = prb.Tokens(flat, np.array([0, 2, 4, 5]), regs)
    x, m, r, g = tok.batch([0], [1])
    assert torch.equal(x[0], torch.from_numpy(flat[2:4].astype(np.float32))) and torch.equal(r[0], torch.tensor([[0, 0], [0, 1]]))
    with pytest.raises(SystemExit):
        tok.batch([0], [2])


def fake_safetensors(path, data: bytes):
    header = json.dumps({"vision.a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
                         "decision.b": {"dtype": "F32", "shape": [1], "data_offsets": [8, 12]}}).encode()
    open(path, "wb").write(struct.pack("<Q", len(header)) + header + data)


def test_vision_identity(tmp_path):
    runs = {}
    for k in "XTC":
        runs[k] = str(tmp_path / (k + ".safetensors"))
        fake_safetensors(runs[k], b"\x00" * 8 + (b"\x01\x02\x03\x04" if k == "T" else b"\x09\x09\x09\x09"))
    assert vid.check(runs)["T"]["tensors"] == 1                                    # decision tensors may differ
    fake_safetensors(runs["C"], b"\x00" * 7 + b"\x01" + b"\x09" * 4)
    with pytest.raises(SystemExit):
        vid.check(runs)


def test_conditions_branch_and_incomplete():
    qs = [json.loads(x) for x in open(os.path.join(ROOT, "data", "exp49", "questions.jsonl"))]
    perfect = [prb.p49.labels(q["target"]) for q in qs]
    key_i = {q["key"]: i for i, q in enumerate(qs)}
    donor = [perfect[key_i[q["twin"]]] if q["twin"] else perfect[i] for i, q in enumerate(qs)]
    wrong = [[prb.p49.EOS] + [-100] * 4 for _ in qs]                              # empty: invalid everywhere
    res = prb.report_from(qs, {"T": (perfect, donor), "C": (wrong, wrong)})
    assert res["conditions"]["T"]["status"]["original"] == "positive" and res["attribution"] is True
    assert res["T_minus_C"]["rl_tune"]["value"] == pytest.approx(1.0)
    assert res["conditions"]["C"]["status"]["original"].startswith("not learned")
    assert res["branch"].startswith("a head was not learned")                       # an unlearned C: not located
    res2 = prb.report_from(qs, {"T": (perfect, donor), "C": (perfect, donor)})
    assert res2["attribution"] is False and res2["branch"].startswith("T and C both readable")
    assert prb.report_from(qs, {"T": (perfect, donor)})["branch"].startswith("incomplete")
    assert res["conditions"]["T"]["cut_slice"] == {"right": 15, "of": 15}
