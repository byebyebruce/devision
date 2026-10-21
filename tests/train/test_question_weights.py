# pyright: reportOptionalMemberAccess=false, reportArgumentType=false
"""exp65's per-question CE weights (docs/experiments/2026-10-19-exp65-weight-plan.md): off by default; an all-ones file
trains the same as no file; the weighted loss and its gradient are sum(w_i * CE_i) / batch questions; the RLCD noise is
still drawn (same random stream); malformed files and non-CE objectives are refused."""
import json

import pytest
import torch
from test_distill import CONFIG
from test_text_only import exp50_like, samples

from devision.train import rlcd
from devision.train.rlcd import TrainConfig, train


def test_weighted_loss_and_gradient_are_the_explicit_sum():
    torch.manual_seed(0)
    logits = torch.randn(5, 3, requires_grad=True)
    mask = torch.ones(5, 3, dtype=torch.bool)
    target = torch.nn.functional.one_hot(torch.tensor([0, 2, 1, 1, 0]), 3).float()
    w = torch.tensor([1.0, 3.0, 0.5, 2.0, 1.0])
    torch.manual_seed(1)
    loss, ce = rlcd._rlcd_loss(logits, mask, target, torch.zeros(5, dtype=torch.long), 0.1, 4, "ce", w)
    after_weighted = torch.randn(1)
    loss.backward()
    g = logits.grad.clone()
    logits.grad = None
    per = -(target * torch.log_softmax(logits, -1)).sum(-1)
    explicit = (w * per).sum() / 5
    explicit.backward()
    assert torch.allclose(loss, explicit) and torch.allclose(g, logits.grad) and torch.allclose(ce, per.mean())
    torch.manual_seed(1)
    rlcd._rlcd_loss(logits.detach(), mask, target, torch.zeros(5, dtype=torch.long), 0.1, 4, "ce")
    assert torch.equal(torch.randn(1), after_weighted)                     # the same random stream with or without weights


def write_weights(path, ss, w=lambda s, q: 1.0, skip=0, extra=False, dup=False):
    rows = [{"id": s["id"], "qid": q, "weight": w(s, q)} for s in ss for q in s["questions"]][skip:]
    if extra:
        rows.append({"id": "nope", "qid": "q", "weight": 1.0})
    if dup:
        rows.append(dict(rows[0]))
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def run(tmp_path, name, weight_file=""):
    d = exp50_like(seed=1)
    ss = samples(8)
    cfg = {**CONFIG, "epochs": 1, "objective": "ce", "micro_batch": 4, "save_every": 0, "text_only": True, "question_weight_file": weight_file}
    rep = train(d, ss, data_root=tmp_path / "nowhere", out_dir=tmp_path / name, val_samples=ss[:2], config=TrainConfig(**cfg))
    return d, rep


def test_all_ones_trains_as_no_file(tmp_path):
    assert TrainConfig().question_weight_file == ""
    a, ra = run(tmp_path, "a")
    b, rb = run(tmp_path, "b", write_weights(tmp_path / "w.jsonl", samples(8)))
    assert ra["losses"] == pytest.approx(rb["losses"], rel=1e-6, abs=1e-7)
    sa, sb = a.model.state_dict(), b.model.state_dict()
    assert all(torch.allclose(sa[k].float(), sb[k].float(), atol=1e-6) for k in sa)


def test_non_uniform_weights_change_the_update(tmp_path):
    a, _ = run(tmp_path, "a")
    b, _ = run(tmp_path, "b", write_weights(tmp_path / "w.jsonl", samples(8), w=lambda s, q: 5.0 if s["id"] == "t1" else 1.0))
    sa, sb = a.model.state_dict(), b.model.state_dict()
    assert any(not torch.allclose(sa[k].float(), sb[k].float()) for k in sa)


@pytest.mark.parametrize("kind", ["missing", "extra", "dup", "zero"])
def test_malformed_weight_files_are_refused(tmp_path, kind):
    ss = samples(8)
    f = write_weights(tmp_path / "w.jsonl", ss, skip=1 if kind == "missing" else 0, extra=kind == "extra", dup=kind == "dup",
                      w=(lambda s, q: 0.0) if kind == "zero" else (lambda s, q: 1.0))
    with pytest.raises(ValueError, match="question_weight_file"):
        rlcd.question_weights(f, ss)


def test_weights_need_the_ce_objective(tmp_path):
    f = write_weights(tmp_path / "w.jsonl", samples(8))
    with pytest.raises(ValueError, match="objective ce"):
        train(exp50_like(), samples(8), data_root=tmp_path, out_dir=tmp_path / "o",
              config=TrainConfig(**{**CONFIG, "objective": "rlcd", "text_only": True, "question_weight_file": f}))
