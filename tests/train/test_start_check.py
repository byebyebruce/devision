"""A run reusing another as its baseline (TrainConfig.start_check, critic EXP18-I01) stops before its first update
unless its step-0 records match the baseline's: same questions, valid probabilities, within the tolerance."""
import json
import math
import shutil

import pytest
from conftest import tiny_decider
from test_pipeline import synthetic_samples

from devision.train import rlcd
from devision.train.compare import same_start
from devision.train.rlcd import START_MISMATCH, StartMismatch, TrainConfig, train

CONFIG = dict(epochs=1, micro_batch=4, lr_new=3e-3, lr_head=3e-3, lr_lora=3e-3, warmup=2, eval_every=0,
              seed=0, device="cpu", monitor_details=True)


def record(sid, p, gold="true"):
    return {"sample_id": sid, "qid": "q", "image_id": "img:" + sid, "type": "noul", "options": ["false", "true"],
            "gold": {"false": float(gold == "false"), "true": float(gold == "true")}, "gold_answer": gold,
            "probabilities": {"false": 1 - p, "true": p}, "correct": p > 0.5}


def write(d, sets):
    d.mkdir(parents=True, exist_ok=True)
    for name, rs in sets.items():
        (d / (name + ".details.jsonl")).write_text("".join(json.dumps(r) + "\n" for r in rs))
    return str(d)


BASE = {"val": [record("a", 0.6), record("b", 0.3)], "panel": [record("c", 0.9)]}


def check(tmp_path, other, tolerance=1e-5):
    return same_start(write(tmp_path / "d", other), write(tmp_path / "b", BASE), tolerance)


def test_the_same_records_pass(tmp_path):
    assert check(tmp_path, BASE)["largest"] == 0.0
    assert check(tmp_path, {**BASE, "val": [record("a", 0.6 + 5e-6), record("b", 0.3)]})["largest"] < 1e-5


def test_an_id_repeated_on_both_sides_is_matched_by_occurrence(tmp_path):
    rep = {**BASE, "val": BASE["val"] + [record("a", 0.2, gold="false")]}
    write(tmp_path / "b2", rep)
    assert same_start(write(tmp_path / "d2", rep), str(tmp_path / "b2"), 1e-5)["largest"] == 0.0
    swapped = {**BASE, "val": [record("a", 0.2, gold="false"), BASE["val"][1], BASE["val"][0]]}
    with pytest.raises(ValueError, match="gold, picture or options"):
        same_start(write(tmp_path / "d3", swapped), str(tmp_path / "b2"), 1e-5)


@pytest.mark.parametrize("other, message", [
    ({**BASE, "val": [record("a", 0.7), record("b", 0.3)]}, "differ by"),
    ({**BASE, "val": [record("a", 0.6)]}, "differ in questions"),
    ({**BASE, "val": [record("a", 0.6), record("a", 0.6), record("b", 0.3)]}, "differ in questions"),
    ({**BASE, "val": []}, "no records"),
    ({**BASE, "val": [record("a", math.nan), record("b", 0.3)]}, "invalid probabilities"),
    ({**BASE, "val": [record("a", 0.6, gold="false"), record("b", 0.3)]}, "gold, picture or options"),
    ({"val": BASE["val"]}, "different or no evaluation sets"),
    ({}, "different or no evaluation sets"),
])
def test_anything_else_is_refused(tmp_path, other, message):
    with pytest.raises(ValueError, match=message):
        check(tmp_path, other)


@pytest.fixture
def samples(tmp_path):
    return synthetic_samples(tmp_path / "data", 8)


def run(tmp_path, name, samples, **over):
    return train(tiny_decider(seed=1), samples[:12], data_root=tmp_path / "data", out_dir=tmp_path / name,
                 val_samples=samples[12:], config=TrainConfig(**{**CONFIG, **over}))


def test_a_matching_start_trains(tmp_path, samples):
    base = run(tmp_path, "base", samples)
    again = run(tmp_path, "again", samples, start_check=str(tmp_path / "base" / "monitor" / "step0"))
    assert again["losses"] == base["losses"]
    assert json.loads((tmp_path / "again" / "start_check.json").read_text())["largest"] == 0.0


def test_a_different_start_stops_before_the_first_update(tmp_path, samples, monkeypatch):
    run(tmp_path, "base", samples)
    step0 = tmp_path / "base" / "monitor" / "step0"
    tampered = tmp_path / "tampered"
    shutil.copytree(step0, tampered)
    rows = [json.loads(l) for l in (tampered / "val.details.jsonl").read_text().splitlines()]
    k = rows[0]["options"][0]
    rows[0]["probabilities"] = {o: float(o == k) for o in rows[0]["options"]}
    (tampered / "val.details.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    updates = []
    real = rlcd._rlcd_loss
    monkeypatch.setattr(rlcd, "_rlcd_loss", lambda *a: (updates.append(1), real(*a))[1])
    with pytest.raises(StartMismatch):
        run(tmp_path, "d", samples, start_check=str(tampered), save_every=1)
    assert updates == []
    assert (tmp_path / "d" / START_MISMATCH).exists()
    assert not (tmp_path / "d" / rlcd.RESUME).exists()


def test_start_check_needs_the_step0_records(tmp_path, samples):
    with pytest.raises(ValueError, match="needs monitor_details"):
        run(tmp_path, "x", samples, monitor_details=False, start_check=str(tmp_path))
