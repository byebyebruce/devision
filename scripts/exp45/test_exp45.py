"""exp45 scripts without a real model: the sidecar gives every training question a box inside its picture for the asked
object and field (the JSONL is not changed); the attention observation runs end to end on a tiny checkpoint with the
module; the report refuses an arm with another region weight or sidecar and gives "incomplete" conditions unless both
arms' four main sets are read."""
import importlib.util
import json
import os
import sys

import pytest
from PIL import Image

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(ROOT, "data")
sys.path.insert(0, os.path.join(ROOT, "tests"))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rep = load("exp45_report", os.path.join(HERE, "report.py"))
attn = load("exp45_attention", os.path.join(HERE, "attention.py"))


def test_sidecar_covers_every_training_question():
    train = [json.loads(x) for x in open(os.path.join(DATA, "exp42", "train.jsonl"))]
    regs = {r["id"]: r for r in (json.loads(x) for x in open(os.path.join(DATA, "exp45", "regions.jsonl")))}
    man = json.load(open(os.path.join(DATA, "exp45", "MANIFEST.json")))
    assert len(regs) == len(train) == 4608 and man["sources"]["train"] == json.load(open(os.path.join(DATA, "exp42", "MANIFEST.json")))["files"]["train"]
    for t in train:
        r = regs[t["id"]]
        (q, v), = t["questions"].items()
        x, y, w, h = r["box"]
        assert 0 <= x and 0 <= y and w > 0 and h > 0 and x + w <= r["width"] and y + h <= r["height"]
        assert r["picture"] == t["image"] and r["field"] == q and r["object"] in v["instructions"].split()[-1]
        assert r["layout"] == ("L" if "/L/" in t["image"] else "original")


def test_attention_observation_on_a_tiny_checkpoint(tmp_path):
    from conftest import tiny_decider  # type: ignore
    d = tiny_decider(seed=1)
    d.model.add_question_conditioning()
    d.cfg.question_conditioning = True
    d.save(tmp_path / "ckpt")
    (tmp_path / "data" / "images").mkdir(parents=True)
    rows, regs = [], []
    for i in range(4):
        Image.new("RGB", (60, 40), (40 * i, 90, 160)).save(tmp_path / "data" / "images" / ("%d.png" % i))
        qs = {"q1": {"type": "choice", "instructions": "What is the mass of each particle in sample A?", "criteria": {"red": None, "blue": None}},
              "q2": {"type": "choice", "instructions": "What is the mass of each particle in sample B?", "criteria": {"red": None, "blue": None}}}
        rows.append({"id": "r%d" % i, "image": "images/%d.png" % i, "image_id": "r%d" % i, "questions": qs,
                     "gold": {q: {"probabilities": {"red": 1.0, "blue": 0.0}} for q in qs}})
        for q, obj, x in (("q1", "A", 0), ("q2", "B", 30)):
            regs.append({"id": "r%d" % i, "set": "x", "qid": q, "box": [x, 20, 30, 10], "width": 60, "height": 40,
                         "picture": "images/%d.png" % i, "object": obj, "field": "mass", "side": "left" if obj == "A" else "right",
                         "layout": "original"})
    (tmp_path / "set.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "regs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in regs))
    attn.main([str(tmp_path / "ckpt"), str(tmp_path / "set.jsonl"), "x", str(tmp_path / "out"), "attn_x", "--regions",
               str(tmp_path / "regs.jsonl"), "--data-root", str(tmp_path / "data"), "--device", "cpu"])
    recs = [json.loads(x) for x in open(tmp_path / "out" / "attn_x.details.jsonl")]
    assert len(recs) == 8 and all(0 <= r["mass_target"] <= 1 and 0 < r["uniform_share"] < 1 for r in recs)
    assert all(0.05 < r["uniform_share"] < 0.25 for r in recs)                  # a 30 x 10 box in a 60 x 40 picture


def test_arm_identity_and_conditions(tmp_path):
    man = json.load(open(os.path.join(DATA, "exp42", "MANIFEST.json")))
    rsha = json.load(open(os.path.join(DATA, "exp45", "MANIFEST.json")))["regions"]["sha256"]

    def make(name, arm, weight, sha=rsha):
        run = tmp_path / name
        (run / "stage2").mkdir(parents=True)
        json.dump({"init": rep.INIT, "init_sha256": rep.INIT_SHA, "data_sha256": man["files"]["train"], "region_weight": weight,
                   "regions_sha256": sha}, open(run / "START.json", "w"))
        json.dump({"config": {"stages": [{"data": "data/exp42/train.jsonl", "init": rep.INIT, "params": {
            "objective": "ce", "question_conditioning": True, "region_file": rep.REGIONS, "region_weight": weight}}]}}, open(run / "run.json", "w"))
        json.dump({"objective": "ce", "region": {"weight": weight, "file_sha256": sha, "smooth": 0.1}}, open(run / "stage2" / "train_report.json", "w"))
        return str(run)
    assert rep.arm_identity("G", make("g", "G", 0.5), man, rsha) == ""
    assert rep.arm_identity("G", make("g0", "G", 0.0), man, rsha).startswith("wrong")         # G with S's weight
    assert rep.arm_identity("S", make("s1", "S", 0.0, sha="0" * 64), man, rsha).startswith("wrong")
    good = {"different": 0.9, "both": 0.5, "twin": 0.5}
    ci = {"value": 0.3, "ci95": [0.1, 0.5]}
    out = {"readout": {n: {"S": good, "G": good} for n in rep.SETS},
           "differences": {"%s|%s" % (n, k): {"both": ci, "twin": ci, "different": ci} for n in rep.SETS for k in ("S-M", "G-M", "G-S")}}
    assert rep.conditions(out) == {"progress_S": True, "progress_G": True, "region_attribution": True}
    del out["readout"]["fit_L"]["G"]
    assert set(rep.conditions(out).values()) == {"incomplete"}


def test_missing_arms_are_incomplete():
    os.chdir(ROOT)
    out = rep.build({"M": "runs/exp42-start", "C": "runs/exp42-c", "S": "/nonexistent/s", "G": "/nonexistent/g"})
    assert out["runs"]["S"].startswith("incomplete") and set(out["conditions"].values()) == {"incomplete"}
    assert out["readout"]["rl_tune"]["C"]["both"] == pytest.approx(0.0128, abs=1e-3)
