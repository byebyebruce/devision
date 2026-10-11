"""Per-question monitor files written during training (v12 retrain plan, section 10): the same records
devision-eval writes for the same model and temperatures, so devision-compare can pair them."""
import json

import pytest
from conftest import tiny_decider
from test_pipeline import synthetic_samples

from devision.train.evaluate import evaluate_records
from devision.train.rlcd import TrainConfig, train

CONFIG = dict(epochs=1, micro_batch=4, lr_new=3e-3, lr_head=3e-3, lr_lora=3e-3, warmup=2, eval_every=3,
              seed=0, device="cpu", monitor_details=True)


def rows(path):
    return [json.loads(line) for line in open(path)]


def same(monitor, evaluated):
    assert [(r["sample_id"], r["qid"]) for r in monitor] == [(r["sample_id"], r["qid"]) for r in evaluated]
    for m, e in zip(monitor, evaluated):
        assert m["prediction"] == e["prediction"] and m["correct"] == e["correct"]
        assert m["gold"] == e["gold"] and m["options"] == e["options"] and m["image_id"] == e["image_id"]
        assert m["probabilities"] == pytest.approx(e["probabilities"], abs=1e-5)
        assert m["nll"] == pytest.approx(e["nll"], abs=1e-4)


def test_monitor_files_match_devision_eval(tmp_path):
    samples = synthetic_samples(tmp_path / "data", 16)
    train_s, val_s, panel = samples[:20], samples[20:26], samples[26:]
    d = tiny_decider(seed=1)
    train(d, train_s, data_root=tmp_path / "data", out_dir=tmp_path / "run", val_samples=val_s,
          eval_sets={"panel": panel}, config=TrainConfig(**CONFIG))
    mon = tmp_path / "run" / "monitor"
    steps = sorted(p.name for p in mon.iterdir())
    assert "step0" in steps and "final" in steps and len(steps) >= 3

    start = tiny_decider(seed=1)     # step 0 is the initial model at temperature 1
    start.cfg.temperature, start.cfg.temperature_by_options = [1.0, 1.0, 1.0], {}
    same(rows(mon / "step0" / "panel.details.jsonl"), evaluate_records(start, panel, tmp_path / "data"))

    # final: the trained model with the fitted temperatures, as decide() answers
    same(rows(mon / "final" / "panel.details.jsonl"), evaluate_records(d, panel, tmp_path / "data"))
    assert (mon / "final" / "val.details.jsonl").exists()


def test_no_monitor_files_unless_asked(tmp_path):
    samples = synthetic_samples(tmp_path / "data", 16)
    train(tiny_decider(seed=1), samples[:20], data_root=tmp_path / "data", out_dir=tmp_path / "run",
          val_samples=samples[20:], config=TrainConfig(**{**CONFIG, "monitor_details": False}))
    assert not (tmp_path / "run" / "monitor").exists()
