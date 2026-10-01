"""End-to-end smoke: samples -> train -> save -> load -> evaluate through `decide`.

Uses the tiny model and synthetic red/blue images so it runs on a laptop CPU. The same code
path trains the real model on the Mac with --device mps (see CLAUDE.md for the command).
"""
import json
import random

import pytest
from conftest import tiny_decider
from PIL import Image

from devision.model import Decider
from devision.train.align import AlignConfig, align
from devision.train.evaluate import evaluate
from devision.train.rlcd import TrainConfig, train

COLORS = {"red": (220, 20, 20), "blue": (20, 20, 220)}


def synthetic_samples(root, n):
    """Samples (in the train JSONL format) about images that are either red or blue."""
    (root / "images").mkdir(parents=True)
    samples = []
    for i in range(n):
        color = "red" if i % 2 else "blue"
        size = (48 + 7 * i, 64) if i % 3 else (64, 30)
        Image.new("RGB", size, COLORS[color]).save(root / "images" / ("%d.jpg" % i))
        image = {"image_id": "syn:%d" % i, "image": "images/%d.jpg" % i, "source": "synthetic"}
        is_red = float(color == "red")
        samples.append(dict(image, id="noul:%d" % i,
                            questions={"q": {"type": "noul", "instructions": "Is the image red?"}},
                            gold={"q": {"probabilities": {"false": 1 - is_red, "true": is_red}}}))
        options = ["red", "blue"] if i % 4 < 2 else ["blue", "red"]
        samples.append(dict(image, id="choice:%d" % i,
                            questions={"q": {"type": "choice", "instructions": "What color is it, red or blue?",
                                             "criteria": {o: None for o in options}}},
                            gold={"q": {"probabilities": {o: float(o == color) for o in options}}}))
    random.Random(0).shuffle(samples)
    return samples


@pytest.mark.slow
def test_training_teaches_the_model_what_it_sees(tmp_path):
    root = tmp_path / "data"
    samples = synthetic_samples(root, 32)
    train_set, val_set = samples[:48], samples[48:]

    report = train(tiny_decider(seed=1), train_set, data_root=root, out_dir=tmp_path / "ckpt",
                   val_samples=val_set,
                   eval_sets={"held": [dict(s, source="other") for s in val_set[:4]] + val_set[4:8]},
                   config=TrainConfig(epochs=40, micro_batch=8, lr_new=3e-3, lr_head=3e-3, lr_lora=3e-3,
                                      lora_r=32, lora_alpha=128, warmup=10, eval_every=25, seed=0, device="cpu"))

    first, last = report["losses"][:10], report["losses"][-10:]
    assert sum(last) / len(last) < sum(first) / len(first)
    assert len(report["val_accuracy"]) == 40  # one reading per epoch
    assert report["val_accuracy"][-1] > report["val_accuracy"][0]
    assert report["calib_items"] == len(val_set)  # temperatures are fitted on the val set
    # Every eval set is tracked from step 0, so progress shows as a curve, not just an end point.
    evals = report["evals"]
    assert evals[0]["step"] == 0 and 25 in [e["step"] for e in evals]
    assert evals[-1]["val/nll"] < evals[0]["val/nll"]
    assert {"held/accuracy", "held/ece"} <= set(evals[-1])
    assert {"held/other/accuracy", "held/synthetic/accuracy"} <= set(evals[-1])  # a mixed set, per source
    assert {"val/accuracy_noul", "val/accuracy_choice", "held/accuracy"} <= set(report["final"])

    decider = Decider.load(tmp_path / "ckpt")
    result = evaluate(decider, samples, data_root=root)

    assert result["accuracy"]["noul"] > 0.8
    # A frozen, randomly initialised tiny encoder learns choice less reliably than noul;
    # "clearly above chance" is what the smoke test asks for.
    assert result["accuracy"]["choice"] > 0.6
    assert 0.0 <= result["ece"] <= 1.0
    assert result["latency_ms"]["p50"] > 0
    assert result["n"] == len(samples)
    assert set(result["accuracy_by_source"]) == {"synthetic"}
    json.dumps(result)  # evaluation output is a structured, serialisable file


def caption_samples(samples):
    """One caption sample (the alignment format) per image of `samples`."""
    out = []
    for image_id, image in sorted({s["image_id"]: s["image"] for s in samples}.items()):
        color = "red" if int(image_id.split(":")[1]) % 2 else "blue"  # as in synthetic_samples
        out.append({"id": "cap:" + image_id, "source": "synthetic", "image_id": image_id, "image": image,
                    "captions": ["the image is %s" % color, "a %s picture" % color]})
    return out


@pytest.mark.slow
def test_training_continues_from_an_aligned_checkpoint(tmp_path):
    root = tmp_path / "data"
    samples = synthetic_samples(root, 32)
    captions = caption_samples(samples)

    report = align(tiny_decider(seed=1), captions[:24], data_root=root, out_dir=tmp_path / "aligned",
                   val_samples=captions[24:],
                   config=AlignConfig(epochs=20, batch=8, lr=3e-3, warmup=5, mlm_head="", eval_every=0,
                                      seed=0, device="cpu"))
    first, last = report["losses"][:5], report["losses"][-5:]
    assert sum(last) / len(last) < sum(first) / len(first)
    assert set(report["val"][-1]) >= {"val/mlm_acc_real", "val/mlm_acc_shuffled"}

    train(Decider.load(tmp_path / "aligned"), samples[:48], data_root=root, out_dir=tmp_path / "ckpt",
          val_samples=samples[48:],
          config=TrainConfig(epochs=40, micro_batch=8, lr_new=3e-3, lr_head=3e-3, lr_lora=3e-3,
                             lora_r=32, lora_alpha=128, warmup=10, unfreeze_top=1, lr_encoder=3e-3,
                             seed=0, device="cpu"))
    result = evaluate(Decider.load(tmp_path / "ckpt"), samples, data_root=root)
    assert result["accuracy"]["noul"] > 0.8
