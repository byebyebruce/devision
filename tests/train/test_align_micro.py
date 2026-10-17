"""exp39: alignment run in parts of micro_batch rows is the whole-batch step (tiny model, CPU). The same draws,
masks and number rewrites (parts with unequal masked-token counts), the same per-step loss, the same scheduler steps,
the same evaluation (its batch rotation kept), parameters within float tolerance; a non-finite part fails before the
step and nothing is saved; micro batches are refused with decision supervision or when they do not divide batch."""
import pytest
import torch
from conftest import tiny_decider
from test_pipeline import caption_samples, synthetic_samples

from devision.infer import Decider
from devision.train.align import AlignConfig, align


@pytest.fixture
def data(tmp_path):
    root = tmp_path / "data"
    samples = synthetic_samples(root, 16)
    caps = caption_samples(samples)
    return root, caps[:24], caps[24:]


def run(tmp_path, root, caps, val, name, micro, **kw):
    report = align(tiny_decider(seed=1), caps, data_root=root, out_dir=tmp_path / name, val_samples=val,
                   config=AlignConfig(epochs=2, batch=4, lr=3e-3, warmup=2, mlm_head="", eval_every=3, lora_r=4,
                                      lr_lora=3e-3, mask_prob=0.5, seed=0, device="cpu", micro_batch=micro, **kw))
    return report, Decider.load(tmp_path / name, device="cpu").model.state_dict()


def test_parts_equal_the_whole_batch(tmp_path, data):
    root, caps, val = data
    full, wf = run(tmp_path, root, caps, val, "full", 0)
    part, wp = run(tmp_path, root, caps, val, "part", 2)
    assert len(full["losses"]) == len(part["losses"]) > 4             # same number of optimizer / scheduler steps
    assert all(abs(a - b) < 1e-5 for a, b in zip(full["losses"], part["losses"]))
    assert [e["step"] for e in full["val"]] == [e["step"] for e in part["val"]]
    for a, b in zip(full["val"], part["val"]):
        assert all(abs(a[k] - b[k]) < 1e-5 for k in a if k != "step")
    assert all(torch.allclose(wf[k], wp[k], atol=1e-5) for k in wf)


def test_a_non_finite_part_stops_before_the_step(tmp_path, data, monkeypatch):
    import devision.train.align as A
    root, caps, val = data
    real = A._mlm_logits
    calls = {"n": 0}

    def second_part_nan(*a, **k):
        calls["n"] += 1
        out = real(*a, **k)
        return out * float("nan") if calls["n"] == 2 else out
    monkeypatch.setattr(A, "_mlm_logits", second_part_nan)
    with pytest.raises(RuntimeError):
        align(tiny_decider(seed=1), caps, data_root=root, out_dir=tmp_path / "nan",
              config=AlignConfig(epochs=1, batch=4, lr=3e-3, warmup=1, mlm_head="", eval_every=0, seed=0, device="cpu",
                                 micro_batch=2))
    assert not (tmp_path / "nan" / "model.safetensors").exists()


def test_refused_combinations(tmp_path, data):
    import json
    root, caps, _ = data
    with open(tmp_path / "x.jsonl", "w") as fh:
        for s in synthetic_samples(tmp_path / "d2", 2):
            fh.write(json.dumps(dict(s, image="../d2/" + s["image"])) + "\n")
    for micro, dec in ((3, ""), (2, str(tmp_path / "x.jsonl"))):
        with pytest.raises(ValueError):
            align(tiny_decider(seed=1), caps, data_root=root, out_dir=tmp_path / "r",
                  config=AlignConfig(epochs=1, batch=4, mlm_head="", eval_every=0, seed=0, device="cpu",
                                     micro_batch=micro, decision_data=dec))
