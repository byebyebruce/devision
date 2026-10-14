"""exp19's option (docs/experiments/2026-10-14-exp19-vision-plan.md): the SigLIP tower's top N layers train,
the rest of it does not, the trained tower is what the saved checkpoint answers with, and nothing else about the
run (batch order, randomness, step 0) changes."""
import pytest
import torch
from conftest import tiny_decider
from test_pipeline import synthetic_samples

from devision.infer import Decider
from devision.train import rlcd
from devision.train.rlcd import TrainConfig, train

CONFIG = dict(epochs=2, micro_batch=4, lr_new=3e-3, lr_head=3e-3, lr_lora=3e-3, warmup=2, eval_every=0,
              seed=0, device="cpu", monitor_details=True)


@pytest.fixture
def samples(tmp_path):
    return synthetic_samples(tmp_path / "data", 12)


def run(tmp_path, name, samples, **over):
    d = tiny_decider(seed=1)
    before = {n: p.detach().clone() for n, p in d.model.vision.named_parameters()}
    report = train(d, samples[:16], data_root=tmp_path / "data", out_dir=tmp_path / name, val_samples=samples[16:],
                   config=TrainConfig(**{**CONFIG, **over}))
    return report, before


def changed(path, before):
    after = dict(Decider.load(path, device="cpu").model.vision.named_parameters())
    return {n for n in before if not torch.equal(after[n], before[n])}


def test_off_by_default_the_tower_stays_as_it_was(tmp_path, samples):
    _, before = run(tmp_path, "frozen", samples)
    assert changed(tmp_path / "frozen", before) == set()


def test_only_the_top_layers_and_final_norm_train(tmp_path, samples):
    _, before = run(tmp_path, "top1", samples, vision_unfreeze_top=1, lr_vision=1e-2)
    moved = changed(tmp_path / "top1", before)    # the tiny tower has one encoder layer: layer 0 is the top one
    assert moved and all(".layers.0." in n or "post_layernorm." in n for n in moved)
    assert not any("embeddings." in n or "head." in n for n in moved)


def test_the_same_order_randomness_and_start_as_the_frozen_run(tmp_path, samples, monkeypatch):
    states = {}
    real = rlcd._rlcd_loss

    def spy(name):
        def f(*a):
            states.setdefault(name, []).append(torch.get_rng_state().clone())
            return real(*a)
        return f

    monkeypatch.setattr(rlcd, "_rlcd_loss", spy("frozen"))
    frozen, _ = run(tmp_path, "frozen", samples)
    monkeypatch.setattr(rlcd, "_rlcd_loss", spy("open"))
    opened, _ = run(tmp_path, "open", samples, vision_unfreeze_top=1, lr_vision=1e-2,
                    start_check=str(tmp_path / "frozen" / "monitor" / "step0"))
    assert len(states["frozen"]) == len(states["open"]) > 2
    assert all(torch.equal(a, b) for a, b in zip(states["frozen"], states["open"]))
    assert opened["losses"][0] == frozen["losses"][0]
    assert opened["losses"][1:] != frozen["losses"][1:]


def test_a_layer_count_out_of_range_is_refused(tmp_path, samples):
    with pytest.raises(ValueError, match="vision_unfreeze_top"):
        run(tmp_path, "x", samples, vision_unfreeze_top=5)


def test_a_stopped_run_resumes_with_the_trained_tower(tmp_path, samples, monkeypatch):
    whole, before = run(tmp_path, "whole", samples, vision_unfreeze_top=1, lr_vision=1e-2, save_every=2)
    real_save = rlcd._save_state

    def save_then_stop(path, state):
        real_save(path, state)
        if len(state["losses"]) == 4:
            raise KeyboardInterrupt

    monkeypatch.setattr(rlcd, "_save_state", save_then_stop)
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path, "parts", samples, vision_unfreeze_top=1, lr_vision=1e-2, save_every=2)
    monkeypatch.setattr(rlcd, "_save_state", real_save)
    parts, _ = run(tmp_path, "parts", samples, vision_unfreeze_top=1, lr_vision=1e-2, save_every=2)
    assert parts["losses"] == pytest.approx(whole["losses"])
    a = dict(Decider.load(tmp_path / "whole", device="cpu").model.vision.named_parameters())
    b = dict(Decider.load(tmp_path / "parts", device="cpu").model.vision.named_parameters())
    assert all(torch.allclose(a[n], b[n]) for n in a)
