"""exp20's 64 -> 256 visual-token conversion on the tiny model, no real checkpoint: where every 2x2 block of
patches is uniform, each of the 4 new tokens equals the old token of its block; the converted model still answers."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tests"))
sys.path.insert(0, os.path.dirname(__file__))
from conftest import image_b64, tiny_decider  # noqa: E402
from fold import fold  # noqa: E402


def trained_like(d):
    """Non-trivial LayerNorm affine, as a trained projector has (a fresh one is weight 1, bias 0)."""
    torch.manual_seed(3)
    ln = d.model.projector.mlp[0]
    with torch.no_grad():
        ln.weight.uniform_(0.5, 1.5)
        ln.bias.normal_(0, 0.2)
    return d


def test_uniform_blocks_give_the_old_tokens():
    old = trained_like(tiny_decider())
    new = fold(old)
    assert new.cfg.visual_shuffle == 1
    dim = old.model.vision.config.hidden_size
    g = 16                                   # 256 patches, 8 x 8 blocks of 2 x 2
    torch.manual_seed(0)
    blocks = torch.randn(1, g // 2, g // 2, dim)
    patches = blocks.repeat_interleave(2, 1).repeat_interleave(2, 2).reshape(1, g * g, dim)
    with torch.no_grad():
        a = old.model.projector(patches).reshape(g // 2, g // 2, -1)
        b = new.model.projector(patches).reshape(g, g, -1)
    for i in range(g):
        for j in range(g):
            assert torch.allclose(b[i, j], a[i // 2, j // 2], atol=1e-5)


def test_other_weights_and_settings_are_kept():
    old = trained_like(tiny_decider())
    new = fold(old)
    so, sn = old.model.state_dict(), new.model.state_dict()
    assert {k for k in so if not k.startswith("projector.")} == {k for k in sn if not k.startswith("projector.")}
    assert all(torch.equal(so[k], sn[k]) for k in so if not k.startswith("projector."))
    assert new.cfg.temperature == old.cfg.temperature


def test_the_converted_model_answers(tmp_path):
    new = fold(trained_like(tiny_decider()))
    new.save(tmp_path / "m")
    from devision.infer import Decider
    d = Decider.load(tmp_path / "m", device="cpu")
    res = d.decide([{"type": "image", "base64": image_b64()}], {"a": {"type": "noul", "instructions": "Is the image red?"}})
    p = res["answers"]["a"]["noul"]
    assert 0.0 <= p <= 1.0


def test_only_a_2x2_checkpoint_is_converted():
    with pytest.raises(ValueError, match="2x2"):
        fold(tiny_decider(visual_shuffle=1))
