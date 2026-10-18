"""exp45's region supervision of the question-conditioning attention (docs/experiments/2026-10-18-exp45-region-plan.md):
off by default; the zero-initialised module leaves the answers as they were; both arms (weight 0 and 0.5) run the same
attention path, batch order and first step, and only the weighted one trains differently; the KL's gradient reaches
the attention's query / key projections and depends on the box; a non-finite KL stops before any update; boxes are
required and must lie inside the picture; decide never needs them and answers the same after saving / loading."""
import json
from typing import cast

import pytest
import torch
from conftest import image_b64, tiny_decider
from test_distill import CONFIG
from test_number_aux import number_samples

from devision.infer import Decider
from devision.train import rlcd
from devision.train.rlcd import TrainConfig, box_grid, region_kl, region_targets, train

QS = {"q": {"type": "choice", "instructions": "what color is it?", "criteria": {"red": None, "blue": None}}}


@pytest.fixture
def setup(tmp_path):
    samples = number_samples(tmp_path / "data", 24)
    side = tmp_path / "regions.jsonl"
    with open(side, "w") as f:
        for i, s in enumerate(samples):
            w, h = 40 + i, 32
            f.write(json.dumps({"id": s["id"], "box": [0 if i % 2 else w // 2, 0, w // 2, h // 2], "width": w, "height": h}) + "\n")
    return samples, str(side)


def run(tmp_path, name, samples, **over):
    cfg = {**CONFIG, "objective": "ce", "question_conditioning": True, **over}
    return train(tiny_decider(seed=1), samples[:16], data_root=tmp_path / "data", out_dir=tmp_path / name,
                 val_samples=samples[16:], config=TrainConfig(**cfg))


def test_zero_initialised_module_answers_as_before():
    d = tiny_decider(seed=1)
    before = d.decide([{"type": "image", "base64": image_b64()}], QS)["answers"]["q"]["probabilities"]
    d.model.add_question_conditioning()
    d.cfg.question_conditioning = True
    cond = d.model.condition
    assert cond is not None
    cond.keep_weights = True
    after = d.decide([{"type": "image", "base64": image_b64()}], QS)["answers"]["q"]["probabilities"]
    assert all(abs(before[k] - after[k]) < 1e-6 for k in before)
    assert cond.last_weights is not None and abs(float(cond.last_weights.sum()) - 1.0) < 1e-4


def test_both_arms_share_order_and_first_step_and_only_the_weighted_one_differs(tmp_path, setup, monkeypatch):
    samples, side = setup
    orders = {}
    real_save = rlcd._save_state
    monkeypatch.setattr(rlcd, "_save_state", lambda path, state: (orders.setdefault(str(path), state["order"]), real_save(path, state)))
    s = run(tmp_path, "s", samples, save_every=1, region_file=side, region_weight=0.0)
    g = run(tmp_path, "g", samples, save_every=1, region_file=side, region_weight=0.5)
    o_g, o_s = (v for _, v in sorted(orders.items()))
    assert o_g == o_s and s["losses"][0] == g["losses"][0] and s["losses"][1:] != g["losses"][1:]
    assert s["region"]["weight"] == 0.0 and g["region"]["weight"] == 0.5 and g["region"]["smooth"] == 0.1
    loaded = Decider.load(str(tmp_path / "g"), device="cpu")                     # decide needs no box
    assert loaded.cfg.question_conditioning and loaded.model.condition is not None and not loaded.model.condition.keep_weights
    ans = loaded.decide([{"type": "image", "base64": image_b64()}], QS)["answers"]["q"]["probabilities"]
    assert set(ans) == {"red", "blue"}


def test_off_by_default_and_region_needs_the_module(tmp_path, setup):
    samples, side = setup
    a = run(tmp_path, "a", samples, question_conditioning=False)
    assert "region" not in a
    with pytest.raises(ValueError):
        run(tmp_path, "b", samples, question_conditioning=False, region_file=side, region_weight=0.5)


def test_boxes_are_required_and_inside_the_picture(tmp_path, setup):
    samples, side = setup
    with pytest.raises(ValueError):
        region_targets(side, samples + [dict(samples[0], id="extra")], 16, 256)
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"id": samples[0]["id"], "box": [30, 0, 40, 10], "width": 40, "height": 32}) + "\n")
    with pytest.raises(ValueError):
        region_targets(str(bad), samples[:1], 16, 256)
    t = region_targets(side, samples[:1], 16, 256)[samples[0]["id"]]
    assert len(t) == 256 and abs(sum(t) - 1) < 1e-6 and min(t) >= 0.1 / 256 - 1e-9


def test_nonfinite_region_kl_stops_before_any_update(tmp_path, setup, monkeypatch):
    samples, side = setup
    monkeypatch.setattr(rlcd, "region_kl", lambda w, t: torch.tensor(float("nan")))
    with pytest.raises(RuntimeError, match="region KL"):
        run(tmp_path, "nan", samples, region_file=side, region_weight=0.5)
    assert not (tmp_path / "nan" / "model.safetensors").exists()


def test_kl_gradient_reaches_query_and_key_and_follows_the_box():
    d = tiny_decider(seed=1)
    d.model.add_question_conditioning()
    cond = d.model.condition
    assert cond is not None
    cond.keep_weights = True
    torch.manual_seed(0)
    visual, patches = torch.randn(2, 4, 64), torch.randn(2, 16, 32)
    words, qmask = torch.randn(2, 5, 64), torch.ones(2, 5, dtype=torch.bool)

    def grads(box):
        cond.zero_grad()
        cond(visual, patches, words, qmask)
        t = torch.tensor([box_grid([box], 64, 64, 256, grid=4, smooth=0.1)] * 2)
        assert cond.last_weights is not None
        region_kl(cond.last_weights, t).backward()
        return [cast(torch.Tensor, p.grad).clone() for p in (cond.question.weight, cond.patches.weight, cond.attend.in_proj_weight)]
    a, b = grads([0, 0, 16, 16]), grads([48, 48, 16, 16])
    assert all(x.abs().sum() > 0 for x in a)                                      # query (question), key (patches) paths
    assert any(not torch.allclose(x, y) for x, y in zip(a, b))                     # another box, another gradient
