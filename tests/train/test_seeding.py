"""A fresh model's random projector follows --seed (it is built before training seeds its sampling)."""
import torch
from conftest import tiny_decider

from devision.train.cli import seed_all


def fresh_projector(seed):
    """Build the way `_decider` does for a new model: nothing seeds torch in between."""
    from devision.model.network import Projector

    seed_all(seed)
    return Projector(32, 64, 2).state_dict()


def test_the_same_seed_gives_the_same_projector_and_another_seed_does_not():
    a, b, c = fresh_projector(0), fresh_projector(0), fresh_projector(1)
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert not all(torch.equal(a[k], c[k]) for k in a)


def test_loading_a_checkpoint_does_not_depend_on_the_seed(tmp_path):
    from devision.model import Decider

    tiny_decider(seed=3).save(tmp_path / "ckpt")
    seed_all(0)
    x = Decider.load(tmp_path / "ckpt").model.state_dict()
    seed_all(1)
    y = Decider.load(tmp_path / "ckpt").model.state_dict()
    assert all(torch.equal(x[k], y[k]) for k in x)
