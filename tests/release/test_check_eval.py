"""scripts/release/check_eval.py: results are used only when sealed under the frozen checkpoint identity and
their content matches the data; same-path weight / temperature changes, results from before the freeze, a wrong
gold, a missing probability or option are refused, a complete fresh result passes."""
import importlib.util
import json
import os
import time

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
spec = importlib.util.spec_from_file_location("check_eval", os.path.join(ROOT, "scripts", "release", "check_eval.py"))
assert spec is not None and spec.loader is not None
ce = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ce)

OPTS = ["red", "blue", "green"]


@pytest.fixture
def env(tmp_path):
    rnd = tmp_path / "runs" / "cand"
    ck = rnd / "stage2"
    ck.mkdir(parents=True)
    (rnd / "run.json").write_text("{}")
    (ck / "model.safetensors").write_bytes(b"weights-1")
    (ck / "config.json").write_text(json.dumps({"temperature": [1.5, 1.0, 1.2], "temperature_by_options": {"choice:3-5": 1.4}}))
    data = tmp_path / "set.jsonl"
    rows = [{"id": "s%d" % i, "image": "img%d.png" % i, "questions": {"q": {"type": "choice", "instructions": "?",
             "criteria": {o: None for o in OPTS}}}, "gold": {"q": {"probabilities": {o: float(o == "blue") for o in OPTS}}}}
            for i in range(3)]
    data.write_text("".join(json.dumps(r) + "\n" for r in rows))
    cfg = {"name": "full", "stages": [], "evaluate": {"checkpoint": str(ck), "sets": {"a": {"path": str(data), "role": "heldout"}}}}
    conf = tmp_path / "c.yaml"
    conf.write_text(yaml.safe_dump(cfg))
    return {"conf": str(conf), "ck": ck, "data": data, "eval": rnd / "eval"}


def write_result(env, mutate=None):
    """What devision-eval writes for set a, optionally damaged by mutate(records)."""
    recs = []
    for i in range(3):
        p = {"red": 0.2, "blue": 0.7, "green": 0.1}
        recs.append({"sample_id": "s%d" % i, "qid": "q", "type": "choice", "options": list(OPTS), "image_used": "img%d.png" % i,
                     "gold": {o: float(o == "blue") for o in OPTS}, "probabilities": p, "prediction": "blue", "correct": True})
    if mutate:
        mutate(recs)
    os.makedirs(env["eval"], exist_ok=True)
    run = {"checkpoint": str(env["ck"]), "data_sha256": ce.sha256(str(env["data"])), "temperature": "fitted",
           "control": "none", "role": "heldout"}
    summary = {"n": len(recs), "accuracy_all": sum(r["correct"] for r in recs) / len(recs), "run": run,
               "temperature": ce.temperature_table(str(env["ck"]))}
    (env["eval"] / "a.json").write_text(json.dumps(summary))
    (env["eval"] / "a.details.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))


def run(*args):
    return ce.main(list(args))


def test_fresh_result_passes(env):
    assert run("freeze", env["conf"]) == 0
    assert run("preflight", env["conf"]) == 0          # nothing yet: missing, to be evaluated
    time.sleep(0.01)
    write_result(env)
    assert run("seal", env["conf"]) == 0 and run("check", env["conf"]) == 0


def test_same_path_weight_or_temperature_change_is_refused(env):
    run("freeze", env["conf"])
    write_result(env)
    run("seal", env["conf"])
    (env["ck"] / "model.safetensors").write_bytes(b"weights-2")
    assert run("check", env["conf"]) == 1 and run("preflight", env["conf"]) == 1
    (env["ck"] / "model.safetensors").write_bytes(b"weights-1")
    (env["ck"] / "config.json").write_text(json.dumps({"temperature": [9.0, 1.0, 1.2]}))
    assert run("check", env["conf"]) == 1
    assert run("freeze", env["conf"]) == 1               # a different identity is never overwritten


def test_result_from_before_the_freeze_is_quarantined(env):
    write_result(env)
    time.sleep(0.01)
    run("freeze", env["conf"])
    assert run("seal", env["conf"]) == 1                 # written before the freeze: not sealed
    assert run("preflight", env["conf"]) == 1
    assert run("preflight", "--quarantine", env["conf"]) == 0
    assert not (env["eval"] / "a.json").exists() and list((env["eval"] / "_invalid").iterdir())


@pytest.mark.parametrize("damage", [
    lambda rs: rs[0].update(gold={"red": 1.0, "blue": 0.0, "green": 0.0}),       # same id, wrong gold
    lambda rs: rs[1]["probabilities"].pop("green"),                              # a probability missing
    lambda rs: rs[2].update(options=["red", "blue"]),                            # an option missing
    lambda rs: rs.pop(),                                                         # a question missing
])
def test_wrong_content_is_refused(env, damage):
    run("freeze", env["conf"])
    write_result(env, damage)
    assert run("seal", env["conf"]) == 1 and run("check", env["conf"]) == 1
