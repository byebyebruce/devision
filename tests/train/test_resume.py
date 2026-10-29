"""Stage-2 training is reproducible on CPU, and a run stopped after a resume state continues to the
same model as an uninterrupted one."""
import pytest
from conftest import image_b64, tiny_decider
from test_pipeline import synthetic_samples

from devision.train import rlcd
from devision.train.rlcd import TrainConfig, train

CONFIG = dict(epochs=3, micro_batch=4, lr_new=3e-3, lr_head=3e-3, lr_lora=3e-3, warmup=2, eval_every=0,
              seed=0, device="cpu")


def answers(decider):
    state = [{"type": "image", "base64": image_b64(color=c)} for c in ((220, 20, 20), (20, 20, 220))]
    out = []
    for s in state:
        res = decider.decide([s], {"a": {"type": "noul", "instructions": "Is the image red?"},
                                   "b": {"type": "choice", "instructions": "What color is it, red or blue?",
                                         "criteria": {"red": None, "blue": None}}})
        out.append([res["answers"]["a"]["noul"], *res["answers"]["b"]["probabilities"].values()])
    return out


def run(tmp_path, name, samples, **over):
    d = tiny_decider(seed=1)
    report = train(d, samples[:24], data_root=tmp_path / "data", out_dir=tmp_path / name,
                   val_samples=samples[24:], config=TrainConfig(**{**CONFIG, **over}))
    return d, report


@pytest.fixture
def samples(tmp_path):
    return synthetic_samples(tmp_path / "data", 16)


def test_the_same_seed_trains_the_same_model(tmp_path, samples):
    a, ra = run(tmp_path, "a", samples)
    b, rb = run(tmp_path, "b", samples)
    assert ra["losses"] == rb["losses"]
    assert answers(a) == answers(b)


def test_a_stopped_run_resumes_to_the_same_model(tmp_path, samples, monkeypatch):
    whole, r_whole = run(tmp_path, "whole", samples, save_every=4)
    assert not (tmp_path / "whole" / rlcd.RESUME).exists()   # removed once training finishes

    real_save = rlcd._save_state

    def save_then_stop(path, state):
        real_save(path, state)
        if len(state["losses"]) == 8:      # mid-epoch: 6 steps per epoch
            raise KeyboardInterrupt

    monkeypatch.setattr(rlcd, "_save_state", save_then_stop)
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path, "parts", samples, save_every=4)
    assert (tmp_path / "parts" / rlcd.RESUME).exists()
    monkeypatch.setattr(rlcd, "_save_state", real_save)
    parts, r_parts = run(tmp_path, "parts", samples, save_every=4)

    assert r_parts["losses"] == pytest.approx(r_whole["losses"])
    assert sum(answers(parts), []) == pytest.approx(sum(answers(whole), []))


def test_a_resume_state_from_different_settings_is_refused(tmp_path, samples, monkeypatch):
    real_save = rlcd._save_state

    def stop(path, state):
        real_save(path, state)
        raise KeyboardInterrupt

    monkeypatch.setattr(rlcd, "_save_state", stop)
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path, "x", samples, save_every=4)
    monkeypatch.undo()
    with pytest.raises(ValueError, match="different run"):
        run(tmp_path, "x", samples, save_every=4, lr_head=1e-4)


def test_a_stopped_run_with_counterfactual_groups_resumes_to_the_same_model(tmp_path, samples, monkeypatch):
    grouped = [dict(s, group="fam%d" % (i // 3)) for i, s in enumerate(samples)]
    whole, r_whole = run(tmp_path, "gwhole", grouped, save_every=2)
    real_save = rlcd._save_state

    def save_then_stop(path, state):
        real_save(path, state)
        if len(state["losses"]) == 4:
            raise KeyboardInterrupt

    monkeypatch.setattr(rlcd, "_save_state", save_then_stop)
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path, "gparts", grouped, save_every=2)
    monkeypatch.setattr(rlcd, "_save_state", real_save)
    parts, r_parts = run(tmp_path, "gparts", grouped, save_every=2)
    assert r_parts["losses"] == pytest.approx(r_whole["losses"])
    assert sum(answers(parts), []) == pytest.approx(sum(answers(whole), []))


def test_an_invalid_loss_stops_training_and_keeps_the_last_good_resume_state(tmp_path, samples, monkeypatch):
    """A device that starts returning zeros (the 2026-10-10 MPS failure) stops the run instead of training on."""
    real_loss = rlcd._rlcd_loss
    calls = {"n": 0}

    real_forward = rlcd._forward

    def zero_after_five(*args, **kw):          # what the failing GPU did: every number came back 0
        calls["n"] += 1
        out = real_forward(*args, **kw)
        return out * 0 if calls["n"] > 5 else out

    def loss_of_zeros(logits, *args, **kw):
        loss, nll = real_loss(logits, *args, **kw)
        return (loss * 0, nll * 0) if not logits.abs().sum() else (loss, nll)

    monkeypatch.setattr(rlcd, "_forward", zero_after_five)
    monkeypatch.setattr(rlcd, "_rlcd_loss", loss_of_zeros)
    with pytest.raises(RuntimeError, match="invalid results"):
        run(tmp_path, "bad", samples, save_every=4)
    import torch
    state = torch.load(tmp_path / "bad" / rlcd.RESUME, weights_only=False)
    assert len(state["losses"]) == 4 and all(x > 0 for x in state["losses"])


def weights(path):
    from safetensors.torch import load_file
    return load_file(str(path / "model.safetensors"))


def test_preparing_batches_ahead_trains_the_same_model(tmp_path, samples):
    """--prefetch only moves picture decoding to background threads: the saved checkpoint is bit-identical."""
    import torch
    off, r_off = run(tmp_path, "off", samples, prefetch=0)
    on, r_on = run(tmp_path, "on", samples, prefetch=3)
    assert r_on["losses"] == r_off["losses"]
    w_off, w_on = weights(tmp_path / "off"), weights(tmp_path / "on")
    assert w_on.keys() == w_off.keys() and all(torch.equal(w_on[k], w_off[k]) for k in w_off)
    assert answers(on) == answers(off)
