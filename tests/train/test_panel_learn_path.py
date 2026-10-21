# pyright: reportOptionalMemberAccess=false, reportAttributeAccessIssue=false, reportArgumentType=false
"""PANEL-LEARN-PREP-01 compatibility (docs/experiments/2026-10-19-panel-learn-plan.md): the two existing paths the
diagnostic would use, on a tiny checkpoint shaped like the K candidate (question conditioning + fixed local views):
  - D1: align (caption MLM, no decision supervision, micro_batch 0) runs on such a checkpoint; it changes the projector
    and the LoRA-merged text encoder, never SigLIP or the conditioning module;
  - D2: train with mixed_input (picture rows and text rows scored with image=None) runs on D1's output; it changes only
    the LoRA-merged encoder and the decision head -- projector, conditioning and SigLIP stay;
  - align's decision supervision is not usable here: it is refused with micro_batch, and it always loads a picture, so
    a text row (image=None in decide) is not scored as text -- the plan does not use it and never substitutes a blank
    picture."""
import json

import pytest
import torch
from PIL import Image
from test_mixed_input import exp72_like, rows

from devision.infer import Decider
from devision.train.align import AlignConfig, align
from devision.train.rlcd import TrainConfig, train


def snapshot(d):
    return {k: v.detach().clone() for k, v in d.model.state_dict().items()}


def changed(before, after, prefix):
    keys = [k for k in before if k.startswith(prefix)]
    assert keys, prefix
    return any(not torch.equal(before[k], after[k]) for k in keys)


def captions(root, n=12):
    (root / "cap").mkdir(parents=True, exist_ok=True)
    out = []
    for i in range(n):
        colour = [(200, 30, 30), (30, 30, 200)][i % 2]
        Image.new("RGB", (40, 30), colour).save(root / "cap" / ("%d.png" % i))
        word = ["red", "blue"][i % 2]
        out.append({"id": "c%d" % i, "source": "panel", "image_id": "c%d" % i, "image": "cap/%d.png" % i,
                    "captions": ["A: value %d red. B: value %d %s." % (i, i + 1, word)]})
    return out


@pytest.fixture
def chain(tmp_path):
    d = exp72_like()
    d.save(tmp_path / "start")
    start = Decider.load(tmp_path / "start")
    s0 = snapshot(start)
    caps = captions(tmp_path)
    align(start, caps[:8], data_root=tmp_path, out_dir=tmp_path / "d1", val_samples=caps[8:],
          config=AlignConfig(epochs=2, batch=4, lr=1e-2, warmup=1, mlm_head="", eval_every=0, lora_r=4, lr_lora=1e-2, seed=0, device="cpu"))
    d1 = Decider.load(tmp_path / "d1")
    s1 = snapshot(d1)
    rs = rows(tmp_path)
    for name, part in (("train", rs), ("val", rs[:4] + rs[-4:])):
        (tmp_path / ("%s.jsonl" % name)).write_text("".join(json.dumps(r) + "\n" for r in part))
    train(d1, rs, data_root=tmp_path, out_dir=tmp_path / "d2", val_samples=rs[:4] + rs[-4:],
          config=TrainConfig(epochs=2, micro_batch=4, lr_head=1e-2, lr_lora=1e-2, lora_r=4, warmup=1, mixed_input=True, seed=0,
                             device="cpu"))
    s2 = snapshot(Decider.load(tmp_path / "d2"))
    return s0, s1, s2, d1


def test_d1_align_on_a_local_view_checkpoint_changes_projector_and_encoder_only(chain):
    s0, s1, _, d1 = chain
    assert d1.cfg.local_views and d1.cfg.question_conditioning          # the shape survives the alignment
    assert not changed(s0, s1, "vision.") and not changed(s0, s1, "condition.")
    assert changed(s0, s1, "projector.") and changed(s0, s1, "decision.encoder.")


def test_d2_mixed_input_changes_lora_encoder_and_head_only(chain):
    _, s1, s2, _ = chain
    assert not changed(s1, s2, "vision.") and not changed(s1, s2, "projector.") and not changed(s1, s2, "condition.")
    assert changed(s1, s2, "decision.encoder.")


def test_align_decision_supervision_is_not_this_path(tmp_path):
    d = exp72_like()
    rs = rows(tmp_path)
    (tmp_path / "dec.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rs))
    caps = captions(tmp_path)
    with pytest.raises(ValueError, match="micro_batch is not combined with decision supervision"):
        align(d, caps, data_root=tmp_path, out_dir=tmp_path / "x",
              config=AlignConfig(epochs=1, batch=4, micro_batch=2, mlm_head="", decision_data=str(tmp_path / "dec.jsonl"), device="cpu"))
    with pytest.raises((FileNotFoundError, OSError, ValueError)):          # a text row's named picture is opened, not image=None
        align(d, caps, data_root=tmp_path, out_dir=tmp_path / "y",
              config=AlignConfig(epochs=1, batch=4, mlm_head="", decision_data=str(tmp_path / "dec.jsonl"), decision_micro_batch=8,
                                 warmup=1, eval_every=0, device="cpu"))


def test_align_uses_the_same_visual_tokens_as_decide(tmp_path):
    """The minimal fix: on a local-view checkpoint, caption alignment projects the full picture exactly as
    model.visual_inputs does for decide (the packed views are not fed to SigLIP as channels)."""
    from devision.infer import image_tensor
    from devision.train.align import _full_picture
    d = exp72_like()
    pics = torch.stack([image_tensor(Image.new("RGB", (40 + 7 * i, 30), (60 * i, 90, 30)), d.cfg) for i in range(3)])
    with torch.no_grad():
        want = d.model.visual_inputs(pics)[1]
        got = d.model.projector(d.model.vision_patches(_full_picture(d.model, pics)))
    assert torch.equal(got, want)
    plain = __import__("conftest").tiny_decider(seed=2)
    p2 = torch.stack([image_tensor(Image.new("RGB", (40, 30), "red"), plain.cfg)])
    assert torch.equal(_full_picture(plain.model, p2), p2)                 # no local views: unchanged
