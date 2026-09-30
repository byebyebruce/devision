"""End-to-end smoke: convert -> train -> save -> load -> evaluate through `decide`.

Uses the tiny model and synthetic red/blue images so it runs on a laptop CPU. The same code
path trains the real model on the A100 (see CLAUDE.md for the command).
"""
import json

import pytest
from conftest import tiny_decider
from PIL import Image

from devision.data import convert_gqa, select
from devision.decider import Decider
from devision.evaluate import evaluate
from devision.train import TrainConfig, train

COLORS = {"red": (220, 20, 20), "blue": (20, 20, 220)}


def synthetic_gqa(root, n):
    """GQA-shaped records about images that are either red or blue."""
    (root / "gqa" / "images").mkdir(parents=True)
    records = []
    for i in range(n):
        color = "red" if i % 2 else "blue"
        size = (48 + 7 * i, 64) if i % 3 else (64, 30)
        Image.new("RGB", size, COLORS[color]).save(root / "gqa" / "images" / ("%d.jpg" % i))
        records.append(("v%d" % i, {
            "imageId": str(i), "question": "Is the image red?", "answer": "yes" if color == "red" else "no",
            "types": {"structural": "verify"}, "semantic": []}))
        records.append(("c%d" % i, {
            "imageId": str(i), "question": "What color is it, red or blue?", "answer": color,
            "types": {"structural": "choose"},
            "semantic": [{"operation": "choose color", "argument": "red|blue", "dependencies": []}]}))
    return records


@pytest.mark.slow
def test_training_teaches_the_model_what_it_sees(tmp_path):
    root = tmp_path / "data"
    records = synthetic_gqa(root, 32)
    samples = select((convert_gqa(qid, r) for qid, r in records), limit=64, exclude_image_ids=set(), seed=0)
    train_set, val_set = samples[:48], samples[48:]

    report = train(tiny_decider(seed=1), train_set, data_root=root, out_dir=tmp_path / "ckpt",
                   val_samples=val_set,
                   config=TrainConfig(epochs=30, micro_batch=8, lr_new=3e-3, lr_head=3e-3, lr_lora=3e-3,
                                      lora_r=16, lora_alpha=64, seed=0, device="cpu"))

    first, last = report["losses"][:10], report["losses"][-10:]
    assert sum(last) / len(last) < sum(first) / len(first)
    assert len(report["val_accuracy"]) == 30  # one reading per epoch
    assert report["val_accuracy"][-1] > report["val_accuracy"][0]
    assert report["calib_items"] == len(val_set)  # temperatures are fitted on the val set

    decider = Decider.load(tmp_path / "ckpt")
    result = evaluate(decider, samples, data_root=root)

    assert result["accuracy"]["noul"] > 0.8
    assert result["accuracy"]["choice"] > 0.8
    assert 0.0 <= result["ece"] <= 1.0
    assert result["latency_ms"]["p50"] > 0
    assert result["n"] == len(samples)
    json.dumps(result)  # evaluation output is a structured, serialisable file
