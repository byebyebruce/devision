"""exp27 export on the tiny model: a resume state written at the last step, exported onto the init checkpoint,
gives the same weights and answers as the checkpoint train() saved itself (temperatures aside)."""
import os
import sys

import pytest
import torch

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "tests"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tests", "train"))
from conftest import tiny_decider  # type: ignore  # noqa: E402
from export import export  # noqa: E402
from test_pipeline import synthetic_samples  # type: ignore  # noqa: E402

from devision.infer import Decider  # noqa: E402
from devision.train import rlcd  # noqa: E402
from devision.train.rlcd import TrainConfig, train  # noqa: E402

CONFIG: dict = dict(epochs=3, micro_batch=4, lr_new=3e-3, lr_head=3e-3, lr_lora=3e-3, warmup=2, eval_every=0, seed=0,
              device="cpu", save_every=6, vision_unfreeze_top=1, lr_vision=1e-3)


@pytest.fixture
def trained(tmp_path, monkeypatch):
    samples = synthetic_samples(tmp_path / "data", 16)
    monkeypatch.setattr(rlcd.os, "remove", lambda path: None)   # keep the last resume state (step 18)
    init = tiny_decider(seed=1)
    init.save(tmp_path / "init")
    d = Decider.load(str(tmp_path / "init"), device="cpu")
    train(d, samples[:24], data_root=tmp_path / "data", out_dir=tmp_path / "run", val_samples=samples[24:],
          config=TrainConfig(**CONFIG))
    return tmp_path


def test_export_matches_the_saved_checkpoint(trained):
    state = torch.load(trained / "run" / rlcd.RESUME, map_location="cpu", weights_only=False)
    assert len(state["losses"]) == 18
    out = export(Decider.load(str(trained / "init"), device="cpu"), state)
    saved = Decider.load(str(trained / "run"), device="cpu")
    a, b = out.model.state_dict(), saved.model.state_dict()
    assert a.keys() == b.keys()
    for k in a:
        assert torch.allclose(a[k], b[k], atol=1e-6), k
    vision_changed = any(not torch.equal(a[k], Decider.load(str(trained / "init"), device="cpu").model.state_dict()[k])
                         for k in a if k.startswith("vision."))
    assert vision_changed       # the unfrozen top layer was trained and exported


def test_a_state_from_another_setup_is_refused(trained):
    state = torch.load(trained / "run" / rlcd.RESUME, map_location="cpu", weights_only=False)
    state["config"]["vision_unfreeze_top"] = 0
    with pytest.raises(ValueError, match="does not match"):
        export(Decider.load(str(trained / "init"), device="cpu"), state)
