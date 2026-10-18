"""exp48's four-corner interaction loss (docs/experiments/2026-10-18-exp48-interaction-plan.md): J matches the formula
and its corner gradients have the right signs; additive picture / object / constant biases cancel; the right
interaction lowers the loss and the reverse raises it; reordering the options with remapped a / b changes nothing and
padded options get no gradient; malformed quads are refused; a non-finite loss stops before the update; off by default;
both arms share order and first step, and the term's gradient reaches the shared model, not a separate head."""
import json
import math

import pytest
import torch
from conftest import tiny_decider
from test_distill import CONFIG

from devision.train import rlcd
from devision.train.rlcd import TrainConfig, interaction_loss, interaction_table, train

C = [("q", 0, 0, 0, 1), ("q", 0, 1, 0, 1), ("q", 1, 0, 0, 1), ("q", 1, 1, 0, 1)]


def margins(m00, m01, m10, m11, k: int = 4):
    """Logits whose a (index 0) - b (index 1) margins are the given ones."""
    z = torch.zeros(4, k)
    z[:, 0] = torch.tensor([m00, m01, m10, m11])
    return z.requires_grad_(True)


def test_j_matches_the_formula_and_gradient_signs():
    z = margins(2.0, -1.0, 0.5, 1.5)
    loss, j = interaction_loss(z, C)
    assert float(j) == pytest.approx((2.0 - (-1.0) - 0.5 + 1.5) / 4)
    assert float(loss) == pytest.approx(math.log1p(math.exp(1 - float(j))))
    loss.backward()
    assert z.grad is not None
    g = z.grad[:, 0]
    assert g[0] < 0 and g[1] > 0 and g[2] > 0 and g[3] < 0                     # lowering the loss raises J's + corners


def test_additive_biases_cancel_and_the_interaction_direction_counts():
    u, w, k = [0.7, -0.3], [1.1, 2.4], 5.0
    z = margins(*[u[v] + w[o] + k for v in (0, 1) for o in (0, 1)], k=4)
    loss, j = interaction_loss(z, C)
    assert float(j) == pytest.approx(0.0, abs=1e-6) and float(loss) == pytest.approx(math.log1p(math.e))
    right, _ = interaction_loss(margins(1, -1, -1, 1), C)
    wrong, _ = interaction_loss(margins(-1, 1, 1, -1), C)
    assert float(right) < float(loss) < float(wrong)


def test_option_order_does_not_matter_and_padding_gets_no_gradient():
    torch.manual_seed(0)
    z = torch.randn(4, 3)
    perm = [2, 0, 1]                                                          # options reordered: a was 0 -> now 1, b 1 -> 2
    zp = torch.cat([z[:, perm], torch.zeros(4, 2)], 1).requires_grad_(True)   # plus two padded slots
    a, b = perm.index(0), perm.index(1)
    l1, _ = interaction_loss(z, C)
    l2, _ = interaction_loss(zp, [(q, v, o, a, b) for q, v, o, _, _ in C])
    assert float(l1) == pytest.approx(float(l2))
    l2.backward()
    assert zp.grad is not None
    assert (zp.grad[:, 3:] == 0).all() and (zp.grad[:, perm.index(2)] == 0).all()
    # exchanging the views (and so a / b) keeps the meaning
    sw = interaction_loss(z[[2, 3, 0, 1]][:, [1, 0, 2]], [(q, v, o, 0, 1) for q, v, o, _, _ in C])
    assert float(sw[0]) == pytest.approx(float(l1))


def test_a_quad_split_across_the_batch_is_refused():
    with pytest.raises(ValueError):
        interaction_loss(margins(1, 1, 1, 1)[:3], C[:3])


def table_samples():
    qs = lambda o: {"q": {"type": "choice", "instructions": o, "criteria": {"4 u": None, "5 u": None, "6 u": None}}}  # noqa: E731
    return [{"id": "r%d" % i, "questions": qs("x"), "gold": {"q": {"probabilities": {"4 u": 1.0, "5 u": 0.0, "6 u": 0.0}}}} for i in range(4)]


def test_malformed_sidecars_are_refused(tmp_path):
    good = [{"id": "r%d" % i, "qid": "q", "quad": "Q", "view": i // 2, "object": i % 2, "a_index": 0, "b_index": 1} for i in range(4)]

    def check(rows, samples=None):
        p = tmp_path / "side.jsonl"
        p.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return interaction_table(str(p), samples or table_samples())
    assert len(check(good)) == 4
    for bad in (good[:3],                                                      # a corner missing
                good + [good[0]],                                              # a row repeated
                [dict(good[0], object=1)] + good[1:],                          # a corner twice
                [dict(good[0], a_index=1, b_index=1)] + good[1:],              # a == b
                [dict(good[0], a_index=7)] + good[1:]):                        # not an option
        with pytest.raises(ValueError):
            check(bad)


def interaction_samples(root):
    from PIL import Image
    (root / "images").mkdir(parents=True)
    out, side = [], []
    opts = ["red", "blue", "green"]
    for u in range(6):
        for view in (0, 1):
            Image.new("RGB", (40, 30), (40 * u, 100 * view, 90)).save(root / "images" / ("%d_%d.png" % (u, view)))
            for obj in (0, 1):
                gold = opts[(obj + view) % 2]                                  # exchanged between the views
                sid = "u%d:v%d:o%d" % (u, view, obj)
                out.append({"id": sid, "image": "images/%d_%d.png" % (u, view), "image_id": sid, "source": "synthetic",
                            "group": "g%d" % u, "questions": {"q": {"type": "choice", "instructions": "what color is it ? %s" % ("left" if obj == 0 else "right"),
                                                                     "criteria": {o: None for o in opts}}},
                            "gold": {"q": {"probabilities": {o: float(o == gold) for o in opts}}}})
                side.append({"id": sid, "qid": "q", "quad": "Q%d" % u, "view": view, "object": obj, "a_index": 0, "b_index": 1})
    (root / "quads.jsonl").write_text("".join(json.dumps(r) + "\n" for r in side))
    return out


def run(tmp_path, name, samples, monkeypatch=None, **over):
    cfg = {**CONFIG, "objective": "ce", "question_conditioning": True, "micro_batch": 4, **over}
    return train(tiny_decider(seed=1), samples[:16], data_root=tmp_path / "data", out_dir=tmp_path / name,
                 val_samples=samples[16:], config=TrainConfig(**cfg))


@pytest.fixture
def isamples(tmp_path):
    return interaction_samples(tmp_path / "data")


def test_off_by_default_and_both_arms_share_order_and_first_step(tmp_path, isamples, monkeypatch):
    side = str(tmp_path / "data" / "quads.jsonl")
    base = run(tmp_path, "base", isamples)
    orders = {}
    real = rlcd._save_state
    monkeypatch.setattr(rlcd, "_save_state", lambda path, state: (orders.setdefault(str(path), state["order"]), real(path, state)))
    c = run(tmp_path, "c", isamples, save_every=1, interaction_file=side, interaction_weight=0.0)
    x = run(tmp_path, "x", isamples, save_every=1, interaction_file=side, interaction_weight=1.0)
    assert "interaction" not in base and c["losses"] == base["losses"]          # weight 0: the CE run itself
    o_c, o_x = (v for _, v in sorted(orders.items()))
    assert o_c == o_x and c["losses"][0] == x["losses"][0] and c["losses"][1:] != x["losses"][1:]
    assert x["interaction"]["weight"] == 1.0 and x["interaction"]["formula"] == rlcd.INTERACTION_FORMULA


def test_nonfinite_interaction_stops_before_any_update(tmp_path, isamples, monkeypatch):
    side = str(tmp_path / "data" / "quads.jsonl")
    monkeypatch.setattr(rlcd, "interaction_loss", lambda logits, corners: (torch.tensor(float("nan")), torch.tensor(0.0)))
    with pytest.raises(RuntimeError):
        run(tmp_path, "nan", isamples, interaction_file=side, interaction_weight=1.0)
    assert not (tmp_path / "nan" / "model.safetensors").exists()


def test_gradient_reaches_the_shared_model():
    from laya.common import collate_items
    d = tiny_decider(seed=1)
    d.model.add_question_conditioning()
    torch.manual_seed(0)
    items = rlcd._items(d, [{"id": "s%d" % i, "image": "x", "image_id": "x", "questions": {"q": {"type": "choice",
             "instructions": "what color is it ?", "criteria": {"red": None, "blue": None}}},
             "gold": {"q": {"probabilities": {"red": 1.0, "blue": 0.0}}}} for i in range(4)])
    batch = collate_items([items], d.tok.pad_token_id)
    assert batch is not None
    pixels = torch.randn(4, 3, 256, 256)
    logits = rlcd._forward(d.model, batch, pixels, "cpu").float()
    loss, _ = interaction_loss(logits, C)
    loss.backward()
    grads = [p.grad for p in d.model.projector.parameters()] + [p.grad for p in d.model.decision.parameters()]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)
