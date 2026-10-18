"""exp50 scripts without a model: the geometry report chose the rule the configs and MANIFEST carry and its original
number boxes are all whole in some view; the report refuses an arm with another start, local source, view rule or a
trainable vision tower, accepts the right one, reproduces X's EXP48 readout, and gives "incomplete" conditions until
both arms are read."""
import importlib.util
import json
import os

import pytest
import yaml

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(ROOT, "data")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rep = load("exp50_report", os.path.join(HERE, "report.py"))


def test_rule_geometry_configs_agree():
    geo = json.load(open(os.path.join(DATA, "exp50", "geometry.json")))
    man = json.load(open(os.path.join(DATA, "exp50", "MANIFEST.json")))
    ch = geo["candidates"][geo["chosen"]]
    assert (ch["n"], ch["overlap"]) == (man["rule"]["local_views"], man["rule"]["local_overlap"]) == (rep.VIEWS, rep.OVERLAP)
    assert ch["original"]["whole"] == 1.0 and ch["original"]["missing"] == 0
    t, c = (yaml.safe_load(open(os.path.join(ROOT, "configs", "exp50-%s.yaml" % a))) for a in "tc")
    assert t["stages"][0]["params"].pop("local_source") == "T" and c["stages"][0]["params"].pop("local_source") == "C"
    t.pop("name")
    c.pop("name")
    assert t == c and t["stages"][0]["init"] == rep.X_CKPT and "vision_unfreeze_top" not in t["stages"][0]["params"]


def arm_dir(tmp_path, arm: str = "T", **bad):
    run = tmp_path / ("exp50-" + arm.lower())
    (run / "stage2").mkdir(parents=True)
    man = json.load(open(os.path.join(DATA, "exp42", "MANIFEST.json")))
    q = json.load(open(os.path.join(DATA, "exp48", "MANIFEST.json")))["sha256"]
    start = {"init": rep.X_CKPT, "init_sha256": rep.X_SHA, "data_sha256": man["files"]["train"], "interaction_weight": 1.0,
             "quads_sha256": q, "local_source": arm, **bad.get("start", {})}
    params = {"objective": "ce", "interaction_file": rep.QUADS, "interaction_weight": 1.0, "local_views": 16,
              "local_overlap": 0.5, "local_source": arm, **bad.get("params", {})}
    local = {"views": 16, "overlap": 0.5, "source": arm, "vision_trainable": 0, **bad.get("local", {})}
    (run / "START.json").write_text(json.dumps(start))
    (run / "run.json").write_text(json.dumps({"config": {"stages": [{"data": "data/exp42/train.jsonl", "init": rep.X_CKPT, "params": params}]}}))
    (run / "stage2" / "train_report.json").write_text(json.dumps({"objective": "ce", "interaction": {"weight": 1.0, "file_sha256": q}, "local": local}))
    (run / "stage2" / "config.json").write_text(json.dumps({"local_views": 16, "local_overlap": 0.5, "local_source": bad.get("cfg", arm)}))
    return str(run), man, q


def test_arm_identity_refuses_wrong_arms(tmp_path):
    run, man, q = arm_dir(tmp_path)
    assert rep.arm_identity("T", run, man, q) == ""
    assert rep.arm_identity("C", run, man, q).startswith("wrong")                                  # the other arm's directory
    for i, bad in enumerate(({"start": {"init_sha256": {"model.safetensors": "0"}}}, {"params": {"local_views": 12}},
                             {"params": {"vision_unfreeze_top": 4}}, {"local": {"vision_trainable": 10}},
                             {"local": {"source": "C"}}, {"cfg": "C"}, {"params": {"region_file": "x"}})):
        r, m, qq = arm_dir(tmp_path / str(i), "T", **bad)  # type: ignore[arg-type]
        assert rep.arm_identity("T", r, m, qq).startswith("wrong"), bad


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "runs", "exp48-x", "FROZEN.json")), reason="needs EXP48-X's records")
def test_x_readout_and_incomplete_conditions():
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        out = rep.build({"X": rep.X_RUN, "T": "runs/exp50-none-t", "C": "runs/exp50-none-c"})
    finally:
        os.chdir(cwd)
    assert out["readout"]["rl_tune"]["X"]["both"] == pytest.approx(0.041, abs=5e-4)
    assert out["quads"]["rl_tune"]["X"]["quads"] == 176 and out["quads"]["rl_tune"]["X"]["macro"] == 0.0
    assert out["conditions"] == {"progress_T": "incomplete", "progress_C": "incomplete", "attribution": "incomplete"}
