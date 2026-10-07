"""devision-calibrate: refit the temperatures of an existing checkpoint, then answer through `decide`."""
import json

import pytest
import torch
from conftest import image_b64, tiny_decider
from PIL import Image

from devision.model import Decider
from devision.train.cli import calibrate_main

COLORS = {"red": (220, 20, 20), "blue": (20, 20, 220)}
THREE = ["red", "green", "blue"]


def calibration_set(root, n_noul, n_three, n_two):
    """noul questions (bucket noul:2), 3-option (choice:3-5) and 2-option (choice:2) choice questions. The
    2-option ones are coin flips (gold 50/50), so they want a much softer temperature than the 3-option ones."""
    (root / "images").mkdir(parents=True)
    samples = []
    for i in range(max(n_noul, n_three, n_two)):
        color = "red" if i % 2 else "blue"
        Image.new("RGB", (40, 40), COLORS[color]).save(root / "images" / ("%d.png" % i))
        image = {"image_id": "syn:%d" % i, "image": "images/%d.png" % i, "source": "synthetic"}
        if i < n_noul:
            samples.append(dict(image, id="noul:%d" % i,
                                questions={"q": {"type": "noul", "instructions": "Is the image red?"}},
                                gold={"q": {"probabilities": {"false": float(color != "red"),
                                                              "true": float(color == "red")}}}))
        if i < n_three:
            samples.append(dict(image, id="three:%d" % i,
                                questions={"q": {"type": "choice", "instructions": "What color is it?",
                                                 "criteria": {o: None for o in THREE}}},
                                gold={"q": {"probabilities": {o: float(o == color) for o in THREE}}}))
        if i < n_two:
            samples.append(dict(image, id="two:%d" % i,
                                questions={"q": {"type": "choice", "instructions": "What color is it?",
                                                 "criteria": {"red": None, "blue": None}}},
                                gold={"q": {"probabilities": {"red": 0.5, "blue": 0.5}}}))
    return samples


@pytest.fixture
def checkpoint_and_data(tmp_path):
    root = tmp_path / "data"
    samples = calibration_set(root, n_noul=60, n_three=60, n_two=45)   # choice:2 is below the 50-question floor
    with open(tmp_path / "val.jsonl", "w") as f:
        f.writelines(json.dumps(s) + "\n" for s in samples)
    decider = tiny_decider(seed=3)
    # A random tiny model answers almost exactly 50/50; scale its scorer so it answers with confidence it
    # has not earned, which calibration has to soften.
    with torch.no_grad():
        for p in decider.model.decision.scorer[-1].parameters():   # bias too: the same for every option
            p.mul_(3e4)
    decider.save(tmp_path / "ckpt")
    return tmp_path / "ckpt", tmp_path / "val.jsonl", root


QUESTIONS = {"two": {"type": "choice", "instructions": "What color is it?", "criteria": {"red": None, "blue": None}},
             "three": {"type": "choice", "instructions": "What color is it?", "criteria": {o: None for o in THREE}},
             "noul": {"type": "noul", "instructions": "Is the image red?"}}


def test_calibrate_fits_dense_buckets_and_leaves_sparse_ones_to_the_type_temperature(checkpoint_and_data, tmp_path):
    ckpt, val, root = checkpoint_and_data
    out = tmp_path / "calibrated"

    calibrate_main(["--checkpoint", str(ckpt), "--data", str(val), "--data-root", str(root), "--out", str(out)])

    config = json.loads((out / "config.json").read_text())
    assert set(config["temperature_by_options"]) == {"noul:2", "choice:3-5"}   # no "choice:2": 45 questions
    report = json.loads((out / "calibrate_report.json").read_text())
    assert {"all", "noul:2", "choice:2", "choice:3-5"} <= set(report["metrics_after"])
    assert report["metrics_after"]["all"]["nll"] <= report["metrics_before"]["all"]["nll"] + 1e-6
    # the original checkpoint is untouched
    assert json.loads((ckpt / "config.json").read_text())["temperature_by_options"] == {}

    calibrated = Decider.load(out)
    per_type_only = Decider.load(out)
    per_type_only.cfg.temperature_by_options = {}
    original = Decider.load(ckpt)
    state = [{"type": "image", "base64": image_b64()}]
    answers = calibrated.decide(state, QUESTIONS)["answers"]

    # the sparse bucket answers with the (refitted) choice temperature, the dense ones with their own
    assert answers["two"] == per_type_only.decide(state, QUESTIONS)["answers"]["two"]
    assert answers["two"] != original.decide(state, QUESTIONS)["answers"]["two"]
    assert answers["three"] != per_type_only.decide(state, QUESTIONS)["answers"]["three"]


def test_calibrate_without_a_destination_writes_nothing(checkpoint_and_data):
    ckpt, val, root = checkpoint_and_data
    before = (ckpt / "config.json").read_text()

    calibrate_main(["--checkpoint", str(ckpt), "--data", str(val), "--data-root", str(root)])

    assert (ckpt / "config.json").read_text() == before
    assert not (ckpt / "calibrate_report.json").exists()


def test_calibrate_in_place_rewrites_the_checkpoint_config(checkpoint_and_data):
    ckpt, val, root = checkpoint_and_data

    calibrate_main(["--checkpoint", str(ckpt), "--data", str(val), "--data-root", str(root), "--in-place"])

    assert set(Decider.load(ckpt).cfg.temperature_by_options) == {"noul:2", "choice:3-5"}


@pytest.mark.parametrize("mode", ["out", "in_place"])
def test_calibrate_keeps_every_other_config_entry(checkpoint_and_data, tmp_path, mode):
    """Only the temperatures change: e.g. the "training" record of a published checkpoint survives."""
    ckpt, val, root = checkpoint_and_data
    config = json.loads((ckpt / "config.json").read_text())
    config["training"] = {"round": "v9-x", "stage": "stage2"}
    config["note"] = "kept"
    (ckpt / "config.json").write_text(json.dumps(config))
    before = (ckpt / "config.json").read_text()
    args = ["--checkpoint", str(ckpt), "--data", str(val), "--data-root", str(root)]
    target = tmp_path / "calibrated" if mode == "out" else ckpt
    calibrate_main(args + (["--out", str(target)] if mode == "out" else ["--in-place"]))

    after = json.loads((target / "config.json").read_text())
    changed = {k for k in set(config) | set(after) if config.get(k) != after.get(k)}
    assert changed <= {"temperature", "temperature_by_options"} and "temperature_by_options" in changed
    assert after["training"] == {"round": "v9-x", "stage": "stage2"} and after["note"] == "kept"
    if mode == "out":
        assert (ckpt / "config.json").read_text() == before
    assert set(Decider.load(target).cfg.temperature_by_options) == {"noul:2", "choice:3-5"}
