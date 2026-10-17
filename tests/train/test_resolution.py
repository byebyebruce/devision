"""exp38: a checkpoint read at 512 instead of 256 (tiny model, CPU). Converting changes only the config; the 512
model keeps every parameter, gives 4x the visual tokens through the same 2x2 projector, answers through decide,
and trains in both stages with gradients reaching the projector / LoRA (and the unfrozen vision layers in decision
training) while the frozen parts stay; a size other than the tower's own without interpolation is refused."""
import importlib.util
import json
import os

import pytest
import torch
from conftest import image_b64, tiny_decider
from test_pipeline import caption_samples, synthetic_samples

from devision.infer import Decider
from devision.model import ModelConfig, build_model, text_encoder
from devision.train.align import AlignConfig, align
from devision.train.rlcd import TrainConfig, train

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
spec = importlib.util.spec_from_file_location("exp38_convert", os.path.join(ROOT, "scripts", "exp38", "convert.py"))
assert spec is not None and spec.loader is not None
convert = importlib.util.module_from_spec(spec)
spec.loader.exec_module(convert)


@pytest.fixture
def ckpt(tmp_path):
    d = tiny_decider(seed=3)
    d.save(tmp_path / "c256")
    convert.convert(str(tmp_path / "c256"), str(tmp_path / "c512"), 512)
    return tmp_path


def test_conversion_keeps_every_weight(ckpt):
    a = Decider.load(ckpt / "c256", device="cpu")
    b = Decider.load(ckpt / "c512", device="cpu")
    sa, sb = a.model.state_dict(), b.model.state_dict()
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
    assert b.cfg.image_size == 512 and b.cfg.interpolate_pos_encoding and a.cfg.image_size == 256
    px = torch.zeros(1, 3, 512, 512)
    with torch.no_grad():
        assert b.model.encode_image(px).shape[1] == 4 * a.model.encode_image(torch.zeros(1, 3, 256, 256)).shape[1]
    out = b.decide(state=[{"type": "image", "base64": image_b64()}], questions={"q": {"type": "noul", "instructions": "Is it red?"}})
    assert 0.0 <= out["answers"]["q"]["noul"] <= 1.0


def test_other_size_without_interpolation_is_refused():
    d = tiny_decider()
    with pytest.raises(ValueError):
        build_model(text_encoder(d.model.decision).config, d.model.vision.config, ModelConfig(image_size=512))


def test_both_stages_train_at_512(ckpt):
    root = ckpt / "data"
    samples = synthetic_samples(root, 8)
    start = Decider.load(ckpt / "c512", device="cpu").model.state_dict()
    align(Decider.load(ckpt / "c512", device="cpu"), caption_samples(samples), data_root=root, out_dir=ckpt / "a512",
          config=AlignConfig(epochs=1, batch=4, lr=3e-3, warmup=1, mlm_head="", eval_every=0, lora_r=4, lr_lora=3e-3,
                             seed=0, device="cpu"))
    aligned = Decider.load(ckpt / "a512", device="cpu")
    after = {k: v.detach().clone() for k, v in aligned.model.state_dict().items()}
    assert aligned.cfg.image_size == 512 and aligned.cfg.interpolate_pos_encoding
    assert any(not torch.equal(after[k], start[k]) for k in after if k.startswith("projector."))
    assert all(torch.equal(after[k], start[k]) for k in after if k.startswith("vision."))
    train(aligned, samples, data_root=root, out_dir=ckpt / "t512",
          config=TrainConfig(epochs=1, micro_batch=4, lr_new=3e-3, lr_head=3e-3, lr_lora=3e-3, lora_r=4, warmup=1,
                             vision_unfreeze_top=1, lr_vision=3e-3, seed=0, device="cpu"))
    trained = Decider.load(ckpt / "t512", device="cpu")
    final = trained.model.state_dict()
    assert trained.cfg.image_size == 512
    assert any(not torch.equal(final[k], after[k]) for k in final if k.startswith("vision.") and "encoder.layers" in k)
    assert json.load(open(ckpt / "t512" / "config.json"))["interpolate_pos_encoding"] is True
