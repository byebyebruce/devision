"""exp50's fixed local views (docs/experiments/2026-10-18-exp50-local-views-plan.md): the views cover the picture edge
to edge and depend only on its size; a view's pixels are the original's (T) or the reduced picture's (C) over the same
rectangle, the full branch identical in both; each patch's geometry points back at the original pixels it shows; the
packed pixels round-trip; off by default (old checkpoints answer exactly as before); decide, training's forward and the
once-per-picture batching agree; the vision tower stays frozen while projector / conditioning / geometry train; the two
arms start from the same tensors and the same first step; malformed settings are refused."""
import numpy as np
import pytest
import torch
from conftest import tiny_decider
from PIL import Image, ImageDraw
from test_distill import CONFIG

from devision import model as M
from devision.image import (degraded, letterbox, local_pixels, local_views, patch_geometry, to_pixel_values,
                            unpack_local, view_rects)
from devision.infer import Decider, image_tensor
from devision.train import rlcd
from devision.train.rlcd import TrainConfig, train

N, OVERLAP = 4, 0.5


def checker(w=120, h=80, cell=1) -> Image.Image:
    """Fine detail (a checkerboard of `cell`-pixel squares) a full 256 letterbox of a large picture cannot keep."""
    a = (((np.indices((h, w)) // cell).sum(0) % 2) * 255).astype("uint8")
    return Image.fromarray(np.stack([a] * 3, -1))


@pytest.mark.parametrize("w,h,n", [(560, 405, 16), (650, 140, 16), (560, 405, 12), (300, 300, 4)])
def test_views_cover_the_picture_and_depend_only_on_its_size(w, h, n):
    rects = view_rects(w, h, n, 0.5)
    assert len(rects) == n and len(set(rects)) == n
    assert min(r[0] for r in rects) == 0 and min(r[1] for r in rects) == 0
    assert max(r[2] for r in rects) == w and max(r[3] for r in rects) == h
    cover = np.zeros((h, w), bool)
    for x0, y0, x1, y1 in rects:
        cover[y0:y1, x0:x1] = True
    assert cover.all()
    a, b = Image.new("RGB", (w, h), "red"), checker(w, h)
    assert local_views(a, 256, n, 0.5, "T")[1] == local_views(b, 256, n, 0.5, "C")[1] == rects   # no content, box or text input


def test_t_crops_the_original_c_the_reduced_picture_full_branch_identical():
    img = checker(600, 400)
    t, c = local_pixels(img, 256, N, OVERLAP, "T"), local_pixels(img, 256, N, OVERLAP, "C")
    assert torch.equal(t[:3], c[:3]) and torch.equal(t[:3], to_pixel_values(img, 256))         # the full branch
    assert torch.equal(t[-1], c[-1])                                                           # same rectangles
    rects = view_rects(600, 400, N, OVERLAP)
    for k, r in enumerate(rects):
        assert torch.equal(t[3 + 3 * k:6 + 3 * k], to_pixel_values(img.crop(r), 256))
        assert torch.equal(c[3 + 3 * k:6 + 3 * k], to_pixel_values(degraded(img, 256).crop(r), 256))
    # T keeps detail (4-pixel squares, 16 views of a 1200 x 800 picture: x0.57) that C, like the full letterbox
    # (x0.21), loses
    fine = checker(1200, 800, cell=4)
    t, c = local_pixels(fine, 256, 16, OVERLAP, "T"), local_pixels(fine, 256, 16, OVERLAP, "C")
    assert float(t[3:-1].std()) > 3 * float(c[3:-1].std())
    # the reduced picture holds what the full letterbox holds: reduced again, it gives the same letterbox content
    diff = np.abs(np.asarray(letterbox(degraded(img, 256), 256), float) - np.asarray(letterbox(img, 256), float))
    assert diff.mean() < 8


def test_patch_geometry_points_at_the_original_pixels():
    w, h = 560, 405
    for x, y in [(30, 40), (280, 200), (530, 380)]:
        img = Image.new("RGB", (w, h), "black")
        ImageDraw.Draw(img).rectangle((x - 2, y - 2, x + 2, y + 2), fill="white")
        views, rects = local_views(img, 256, 16, 0.5, "T")
        for k, r in enumerate(rects):
            if not (r[0] + 3 <= x < r[2] - 3 and r[1] + 3 <= y < r[3] - 3):
                continue
            white = (views[k].mean(0) > 0.5).float()                         # the dot, not the grey letterbox margin
            energy = white.unfold(0, 16, 16).unfold(1, 16, 16).mean((-1, -2)).flatten()   # per 16 x 16 patch
            g = patch_geometry(r, w, h, 256, 16)[int(energy.argmax())]
            assert abs(float(g[0]) * w - x) <= float(g[2]) * w and abs(float(g[1]) * h - y) <= float(g[3]) * h
    full = patch_geometry((0, 0, w, h), w, h, 256, 16)
    assert float(full[:, 0].min()) > 0 and float(full[:, 0].max()) < 1      # wide picture: x spans it, y has margins
    assert float(full[:, 1].min()) < 0 and float(full[:, 1].max()) > 1


def test_packing_round_trips_and_wrong_shapes_are_refused():
    img = checker(333, 222)
    px = torch.stack([local_pixels(img, 256, N, OVERLAP, "T")] * 2)
    full, views, meta = unpack_local(px, N)
    assert full.shape == (2, 3, 256, 256) and views.shape == (2, N, 3, 256, 256)
    assert meta[0] == (333, 222, view_rects(333, 222, N, OVERLAP))
    with pytest.raises(ValueError):
        unpack_local(px, N + 1)
    with pytest.raises(ValueError):
        local_views(img, 256, N, OVERLAP, "X")


def local_decider(source="T", seed=0):
    d = tiny_decider(seed=seed)
    d.model.add_question_conditioning()
    d.cfg.question_conditioning = True
    d.model.enable_local_views(N)
    d.cfg.local_views, d.cfg.local_overlap, d.cfg.local_source = N, OVERLAP, source
    assert d.model.condition is not None
    for p in d.model.condition.parameters():         # a trained-looking module, so the local patches matter
        torch.nn.init.normal_(p, std=0.05)
    return d


QS = {"q": {"type": "choice", "instructions": "what color is it ?", "criteria": {"red": None, "blue": None, "green": None}},
      "y": {"type": "noul", "instructions": "is there a dog ?"}}


def test_off_by_default_old_checkpoints_answer_as_before(tmp_path):
    d = tiny_decider(seed=3)
    d.model.add_question_conditioning()
    d.cfg.question_conditioning = True
    assert d.model.condition is not None
    torch.nn.init.normal_(d.model.condition.out.weight, std=0.05)
    assert d.cfg.local_views == 0 and d.model.condition.geom is None
    img = checker(120, 80)
    # the path before exp50, written out: image_features -> forward without geometry
    from laya.common import collate_items
    items = [rlcd._items(d, [{"id": "s", "image": "x", "image_id": "x", "questions": {"q": QS["q"]},
                              "gold": {"q": {"probabilities": {"red": 1.0, "blue": 0.0, "green": 0.0}}}}])[0]]
    batch = collate_items([items], d.tok.pad_token_id)
    assert batch is not None
    with torch.no_grad():
        patches, visual = d.model.image_features(to_pixel_values(img, 256)[None])
        old = d.model(*(batch[k] for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")),
                      visual=visual, patches=patches)
        new = rlcd._forward(d.model.eval(), batch, image_tensor(img, d.cfg)[None], "cpu")
    assert torch.equal(old, new)
    d.save(tmp_path / "old")
    cfg = (tmp_path / "old" / "config.json").read_text()
    (tmp_path / "old" / "config.json").write_text(cfg.replace('"local_views": 0,', "").replace('"local_overlap": 0.0,', "")
                                                  .replace('"local_source": "",', ""))
    again = Decider.load(str(tmp_path / "old"), device="cpu")
    assert again.cfg.local_views == 0 and again.decide({}, QS, image=img) == d.decide({}, QS, image=img)


def test_decide_training_forward_and_once_per_picture_batching_agree(tmp_path, monkeypatch):
    from laya.common import collate_items
    d = local_decider("T").model.eval()
    dec = Decider(d, tiny_decider().tok, local_decider("T").cfg)
    dec.model.load_state_dict(d.state_dict())
    imgs = [checker(120, 80), Image.new("RGB", (90, 140), (20, 200, 40))]
    pix = [image_tensor(i, dec.cfg) for i in imgs]
    items = rlcd._items(dec, [{"id": "s%d" % i, "image": "x", "image_id": "x", "questions": {"q": QS["q"]},
                               "gold": {"q": {"probabilities": {"red": 1.0, "blue": 0.0, "green": 0.0}}}} for i in range(4)])
    batch = collate_items([items], dec.tok.pad_token_id)
    assert batch is not None
    order = [0, 1, 1, 0]
    with torch.no_grad():
        per_q = rlcd._forward(dec.model, batch, torch.stack([pix[i] for i in order]), "cpu")
        once = rlcd._forward(dec.model, batch, torch.stack(pix), "cpu", index=torch.tensor(order))
        monkeypatch.setattr(M, "VIEW_CHUNK", 3)                             # views through the tower in small chunks
        chunked = rlcd._forward(dec.model, batch, torch.stack(pix), "cpu", index=torch.tensor(order))
    assert torch.allclose(per_q, once, atol=1e-5) and torch.allclose(once, chunked, atol=1e-5)
    probs = torch.softmax(per_q[1, :3] / dec.cfg.temperature_for(0, 3), -1)
    dec.save(tmp_path / "local")
    back = Decider.load(str(tmp_path / "local"), device="cpu")
    assert (back.cfg.local_views, back.cfg.local_overlap, back.cfg.local_source) == (N, OVERLAP, "T")
    assert back.decide({}, {"q": QS["q"]}, image=imgs[1]) == dec.decide({}, {"q": QS["q"]}, image=imgs[1])
    got = dec.decide({}, {"q": QS["q"]}, image=imgs[1])["answers"]["q"]
    assert sorted(got["probabilities"].values()) == pytest.approx(sorted(probs.tolist()), abs=1e-4)


def test_geometry_is_needed_exactly_with_local_views():
    d = local_decider("T")
    assert d.model.condition is not None
    with pytest.raises(ValueError):
        d.model.condition(torch.zeros(1, 64, 64), torch.zeros(1, 10, 32), torch.zeros(1, 5, 64), torch.ones(1, 5, dtype=torch.bool))
    plain = tiny_decider()
    plain.model.add_question_conditioning()
    assert plain.model.condition is not None
    with pytest.raises(ValueError):
        plain.model.condition(torch.zeros(1, 64, 64), torch.zeros(1, 10, 32), torch.zeros(1, 5, 64),
                              torch.ones(1, 5, dtype=torch.bool), torch.zeros(1, 10, 4))
    with pytest.raises(ValueError):
        tiny_decider().model.enable_local_views(N)                          # no question conditioning


def local_samples(root):
    (root / "images").mkdir(parents=True)
    opts = ["red", "blue", "green"]
    out = []
    for i in range(20):
        a = np.asarray(checker(600 + 50 * (i % 3), 400, cell=2), np.float32) * np.array([i / 20, 0.5, 1 - i / 40])
        Image.fromarray(a.astype("uint8")).save(root / "images" / ("%d.png" % i))      # detail T keeps and C loses
        gold = opts[i % 3]
        out.append({"id": "s%d" % i, "image": "images/%d.png" % i, "image_id": "s%d" % i, "source": "synthetic",
                    "questions": {"q": {"type": "choice", "instructions": "what color is it ?", "criteria": {o: None for o in opts}}},
                    "gold": {"q": {"probabilities": {o: float(o == gold) for o in opts}}}})
    return out


def run(tmp_path, name, samples, **over):
    cfg = {**CONFIG, "epochs": 1, "objective": "ce", "question_conditioning": True, "local_views": N,
           "local_overlap": OVERLAP, "local_source": "T", "save_every": 0, **over}
    d = tiny_decider(seed=1)
    before = {k: v.clone() for k, v in d.model.state_dict().items()}
    report = train(d, samples[:16], data_root=tmp_path / "data", out_dir=tmp_path / name, val_samples=samples[16:],
                   config=TrainConfig(**cfg))
    return d, before, report


def test_frozen_tower_trainable_reader_and_identical_arm_starts(tmp_path, monkeypatch):
    samples = local_samples(tmp_path / "data")
    inits = []
    real_opt = torch.optim.AdamW

    def opt(groups, **kw):                                                     # the arm's tensors before the first update
        inits.append(([len(g["params"]) for g in groups], [p.detach().clone() for g in groups for p in g["params"]]))
        return real_opt(groups, **kw)
    monkeypatch.setattr(torch.optim, "AdamW", opt)
    t, before, rt = run(tmp_path, "t", samples, local_source="T")
    c, _, rc = run(tmp_path, "c", samples, local_source="C")
    after = t.model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before if k.startswith("vision."))             # frozen tower
    assert not torch.equal(before["projector.mlp.1.weight"], after["projector.mlp.1.weight"])
    assert float(after["condition.geom.weight"].abs().sum()) > 0                                        # geometry trains
    (gt, t0), (gc, c0) = inits
    assert gt == gc and len(t0) == len(c0) and all(torch.equal(a, b) for a, b in zip(t0, c0))           # same groups, tensors
    assert rt["losses"][0] == rc["losses"][0] and rt["losses"][1:] != rc["losses"][1:]                  # pixels differ after
    assert (t.cfg.local_source, c.cfg.local_source) == ("T", "C") and rt["local"]["vision_trainable"] == 0
    assert set(rt["local"]["groups"]) == {"projector", "head", "lora", "cond"}


def test_bad_settings_are_refused(tmp_path):
    samples = local_samples(tmp_path / "data")
    for over in ({"question_conditioning": False}, {"vision_unfreeze_top": 1}, {"local_source": "X"},
                 {"local_overlap": 1.0}, {"aux_weight": 1.0}):
        with pytest.raises(ValueError):
            run(tmp_path, "bad", samples, **over)
    d = local_decider("C")
    with pytest.raises(ValueError, match="other local views"):
        train(d, samples[:16], data_root=tmp_path / "data", out_dir=tmp_path / "x", val_samples=samples[16:],
              config=TrainConfig(**{**CONFIG, "question_conditioning": True, "local_views": N, "local_overlap": OVERLAP,
                                    "local_source": "T"}))
