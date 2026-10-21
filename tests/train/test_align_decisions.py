"""exp33: alignment with old decision questions supervised alongside the captions (tiny model, CPU). Without decision
data (or weight 0) the run is exactly the caption-only one; with them the decision CE reaches the projector and
LoRA, the frozen decision head and vision tower do not change, the caption / mask draws are unchanged, and the
checkpoint saves and loads."""
import json

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
    with open(root / "decisions.jsonl", "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    return root, caption_samples(samples), str(root / "decisions.jsonl")


def run(tmp_path, root, captions, name, **kw):
    d = tiny_decider(seed=1)
    report = align(d, captions, data_root=root, out_dir=tmp_path / name,
                   config=AlignConfig(epochs=2, batch=4, lr=3e-3, warmup=2, mlm_head="", eval_every=0, lora_r=4,
                                      lr_lora=3e-3, seed=0, device="cpu", **kw))
    return d, report


def weights(d):
    return {k: v.detach().clone() for k, v in d.model.state_dict().items()}


def test_no_decisions_or_weight_zero_is_the_caption_only_run(tmp_path, data):
    root, captions, dec = data
    _, base = run(tmp_path, root, captions, "base")
    _, zero = run(tmp_path, root, captions, "zero", decision_data=dec, decision_weight=0.0)
    assert base["losses"] == zero["losses"]
    a, b = Decider.load(tmp_path / "base", device="cpu"), Decider.load(tmp_path / "zero", device="cpu")
    assert all(torch.equal(x, y) for x, y in zip(a.model.state_dict().values(), b.model.state_dict().values()))


def test_decisions_train_projector_and_lora_only(tmp_path, data):
    root, captions, dec = data
    start = weights(tiny_decider(seed=1))
    _, base = run(tmp_path, root, captions, "base")
    _, with_dec = run(tmp_path, root, captions, "dec", decision_data=dec, decision_weight=1.0, decision_micro_batch=4)
    assert base["losses"][0] == with_dec["losses"][0]        # same first caption batch and masks, before any update
    assert base["losses"] != with_dec["losses"]              # the decision term changes training
    after = Decider.load(tmp_path / "dec", device="cpu").model.state_dict()
    plain = Decider.load(tmp_path / "base", device="cpu").model.state_dict()
    changed = {k for k in after if not torch.equal(after[k], plain[k])}
    assert any(k.startswith("projector.") for k in changed)
    assert any(".encoder." in k for k in changed)            # LoRA merged into ModernBERT
    frozen = [k for k in start if k.startswith("vision.") or (k.startswith("decision.") and ".encoder." not in k)]
    assert frozen and all(torch.equal(after[k], start[k]) for k in frozen)
