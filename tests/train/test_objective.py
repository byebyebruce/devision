"""exp42's objective switch (docs/experiments/2026-10-18-exp42-objective-plan.md): "rlcd" (the default) is the
loss every run before it used; "ce" is the standard gold cross-entropy over the real options (padding excluded),
value and gradient, drawing the same RLCD noise so the random stream, the batch order and the first step match."""
import pytest
import torch
from test_distill import run, samples  # noqa: F401  (the samples fixture)

from devision.train.rlcd import TrainConfig, _rlcd_loss


def batch():
    torch.manual_seed(3)
    logits = torch.randn(5, 4, requires_grad=True)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
    gold = torch.tensor([2, 0, 3, 1, 1])
    target = torch.zeros(5, 4).scatter_(1, gold.unsqueeze(1), 1.0)
    return logits, mask, target, gold, torch.zeros(5, dtype=torch.long)


def test_default_is_rlcd_and_unchanged():
    assert TrainConfig().objective == "rlcd"
    logits, mask, target, _, qtype = batch()
    torch.manual_seed(7)
    a = _rlcd_loss(logits, mask, target, qtype, 0.3, 4)
    torch.manual_seed(7)
    b = _rlcd_loss(logits, mask, target, qtype, 0.3, 4, "rlcd")
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert not torch.allclose(a[0], a[1])                     # the policy term is there


def test_ce_is_the_standard_gold_cross_entropy_with_padding():
    logits, mask, target, gold, qtype = batch()
    torch.manual_seed(7)
    loss, ce = _rlcd_loss(logits, mask, target, qtype, 0.3, 4, "ce")
    (g,) = torch.autograd.grad(loss, logits)
    ref_logits = logits.detach().clone().requires_grad_(True)
    ref = torch.stack([torch.nn.functional.cross_entropy(ref_logits[i, :int(m.sum())].unsqueeze(0), gold[i:i + 1])
                       for i, m in enumerate(mask)]).mean()
    (rg,) = torch.autograd.grad(ref, ref_logits)
    assert torch.allclose(loss, ref, atol=1e-6) and torch.equal(loss, ce)
    assert torch.allclose(g, rg, atol=1e-6) and (g[~mask] == 0).all()      # no gradient into padded options


def test_ce_draws_the_same_noise():
    logits, mask, target, _, qtype = batch()
    states = []
    for objective in ("rlcd", "ce"):
        torch.manual_seed(7)
        _rlcd_loss(logits, mask, target, qtype, 0.3, 4, objective)
        states.append(torch.get_rng_state())
    assert torch.equal(states[0], states[1])


def test_unknown_objective_is_refused(tmp_path, samples, monkeypatch):  # noqa: F811
    with pytest.raises(ValueError):
        run(tmp_path, "bad", samples, monkeypatch, objective="sigma0")


def test_ce_run_same_order_and_first_step_different_training(tmp_path, samples, monkeypatch):  # noqa: F811
    base, order = run(tmp_path, "base", samples, monkeypatch)
    explicit, order_e = run(tmp_path, "explicit", samples, monkeypatch, objective="rlcd")
    ce, order_c = run(tmp_path, "ce", samples, monkeypatch, objective="ce")
    assert explicit["losses"] == base["losses"] and order_e == order        # the default is the explicit rlcd run
    assert base["objective"] == "rlcd" and ce["objective"] == "ce"
    assert order_c == order and ce["losses"][0] == base["losses"][0]        # same init, batch, gold before the first update
    assert ce["losses"][1:] != base["losses"][1:]
