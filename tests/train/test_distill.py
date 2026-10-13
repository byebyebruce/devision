"""exp18's teacher term (docs/experiments/2026-10-13-exp18-distill-plan.md): only the named blocks are
constrained, the RLCD / gold training and its randomness are those of a run without it, and it is off by default."""
import pytest
import torch
from conftest import tiny_decider
from test_pipeline import synthetic_samples

from devision.train import rlcd
from devision.train.rlcd import TrainConfig, train

CONFIG = dict(epochs=2, micro_batch=4, lr_new=3e-3, lr_head=3e-3, lr_lora=3e-3, warmup=2, eval_every=0,
              seed=0, device="cpu", save_every=1)


@pytest.fixture
def samples(tmp_path):
    ss = synthetic_samples(tmp_path / "data", 12)
    return [dict(s, block="old" if i % 3 else "new") for i, s in enumerate(ss)]


@pytest.fixture
def teacher(tmp_path):
    path = tmp_path / "teacher"
    tiny_decider(seed=2).save(path)     # not the student's init, so the term is non-zero from the first step
    return str(path)


def run(tmp_path, name, samples, monkeypatch, **over):
    """Train a fresh tiny model; returns the report and the batch order the run used."""
    orders = []
    real_save = rlcd._save_state
    monkeypatch.setattr(rlcd, "_save_state", lambda path, state: (orders.append(state["order"]), real_save(path, state)))
    report = train(tiny_decider(seed=1), samples[:16], data_root=tmp_path / "data", out_dir=tmp_path / name,
                   val_samples=samples[16:], config=TrainConfig(**{**CONFIG, **over}))
    monkeypatch.setattr(rlcd, "_save_state", real_save)
    return report, orders[0]


def test_weight_zero_is_the_run_without_distillation(tmp_path, samples, teacher, monkeypatch):
    base, order = run(tmp_path, "base", samples, monkeypatch)
    off, order_off = run(tmp_path, "off", samples, monkeypatch, distill_teacher=teacher, distill_blocks="old")
    assert off["losses"] == base["losses"]
    assert order_off == order


def test_the_teacher_term_changes_training_but_not_its_order_or_first_step(tmp_path, samples, teacher, monkeypatch):
    base, order = run(tmp_path, "base", samples, monkeypatch)
    dist, order_d = run(tmp_path, "dist", samples, monkeypatch, distill_weight=1.0, distill_teacher=teacher,
                        distill_blocks="old")
    assert order_d == order                         # the teacher is loaded before the seeds
    assert dist["losses"][0] == base["losses"][0]   # same init, batch and gold before the first update
    assert dist["losses"][1:] != base["losses"][1:]


def test_only_the_named_blocks_are_constrained(tmp_path, samples, teacher, monkeypatch):
    rows = []
    real = rlcd._distill_kl
    monkeypatch.setattr(rlcd, "_distill_kl", lambda s, t, m: (rows.append(s.shape[0]), real(s, t, m))[1])
    run(tmp_path, "dist", samples, monkeypatch, distill_weight=1.0, distill_teacher=teacher, distill_blocks="old")
    old = sum(len(s["questions"]) for s in samples[:16] if s["block"] == "old")
    assert sum(rows) == CONFIG["epochs"] * old


def test_a_block_missing_from_the_data_is_refused(tmp_path, samples, teacher, monkeypatch):
    with pytest.raises(ValueError, match="not in the training data"):
        run(tmp_path, "x", samples, monkeypatch, distill_weight=1.0, distill_teacher=teacher, distill_blocks="replay")
    with pytest.raises(ValueError, match="needs distill_teacher"):
        run(tmp_path, "y", samples, monkeypatch, distill_weight=1.0)


def test_kl_over_the_options_only():
    mask = torch.tensor([[True, True, False], [True, True, True]])
    t = torch.tensor([[2.0, 0.0, 0.0], [0.5, 1.0, -1.0]])
    assert rlcd._distill_kl(t, t, mask).abs().max() < 1e-6
    s = torch.tensor([[0.0, 0.0, 9.0], [0.0, 0.0, 0.0]])
    kl = rlcd._distill_kl(s, t, mask)
    s2 = s.clone()
    s2[0, 2] = -5.0                                  # padding is not an option: changing it changes nothing
    assert torch.equal(rlcd._distill_kl(s2, t, mask), kl)
    p = torch.softmax(t[0, :2], -1)
    assert kl[0] == pytest.approx(float((p * (p.log() - torch.log_softmax(s[0, :2], -1))).sum()), abs=1e-6)


def test_the_noise_draws_see_the_same_random_state(tmp_path, samples, teacher, monkeypatch):
    """The RNG state at every RLCD noise draw is the same with and without the teacher (critic EXP18-N01): the
    seeding, LoRA init and batch order before the first draw, and the teacher's forward passes between draws,
    consume nothing. (A first-step gold NLL alone would not show this.)"""
    states = {}
    real = rlcd._rlcd_loss

    def spy(name):
        def f(*a):
            states.setdefault(name, []).append(torch.get_rng_state().clone())
            return real(*a)
        return f

    monkeypatch.setattr(rlcd, "_rlcd_loss", spy("base"))
    run(tmp_path, "base", samples, monkeypatch)
    monkeypatch.setattr(rlcd, "_rlcd_loss", spy("dist"))
    run(tmp_path, "dist", samples, monkeypatch, distill_weight=1.0, distill_teacher=teacher, distill_blocks="old")
    assert len(states["base"]) == len(states["dist"]) > 2
    assert all(torch.equal(a, b) for a, b in zip(states["base"], states["dist"]))
