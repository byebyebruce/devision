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
    rows = [{"id": "s%d" % i, "image": "img%d.png" % i, "image_id": "i%d" % i, "questions": {"q": {"type": "choice", "instructions": "?",
             "criteria": {o: None for o in OPTS}}}, "gold": {"q": {"probabilities": {o: float(o == "blue") for o in OPTS}}}}
            for i in range(3)]
    data.write_text("".join(json.dumps(r) + "\n" for r in rows))
    cfg = {"name": "full", "stages": [], "evaluate": {"checkpoint": str(ck), "sets": {"a": {"path": str(data), "role": "heldout"}}}}
    conf = tmp_path / "c.yaml"
    conf.write_text(yaml.safe_dump(cfg))
    return {"conf": str(conf), "ck": ck, "data": data, "eval": rnd / "eval"}


def write_result(env, mutate=None, name="a", control="none", pictures=None, data=None, temperature="fitted", role="heldout"):
    """What devision-eval writes, optionally damaged by mutate(records); pictures: sample index -> image shown."""
    data = data or env["data"]
    recs = []
    for line in open(data):
        row = json.loads(line)
        opts = list(row["questions"]["q"]["criteria"])
        shown = opts[::-1] if control == "reversed" else opts
        gold = row["gold"]["q"]["probabilities"]
        right = max(gold, key=lambda k: gold[k])
        p = {o: (0.7 if o == right else 0.3 / (len(opts) - 1)) for o in shown}
        recs.append({"sample_id": row["id"], "qid": "q", "type": "choice", "options": shown, "image_id": row["image_id"],
                     "image_used": (pictures or {}).get(row["image_id"], row["image"]),
                     "gold": gold, "probabilities": p, "prediction": right, "correct": True})
    if mutate:
        mutate(recs)
    os.makedirs(env["eval"], exist_ok=True)
    run = {"checkpoint": str(env["ck"]), "data_sha256": ce.sha256(str(data)),
           "temperature": "fitted" if temperature == "fitted" else "none (raw)", "control": control, "role": role, "seed": 0}
    table = ce.temperature_table(str(env["ck"])) if temperature == "fitted" else ce.RAW_TEMPERATURE
    summary = {"n": len(recs), "accuracy_all": sum(r["correct"] for r in recs) / len(recs), "run": run, "temperature": table}
    (env["eval"] / ("%s.json" % name)).write_text(json.dumps(summary))
    (env["eval"] / ("%s.details.jsonl" % name)).write_text("".join(json.dumps(r) + "\n" for r in recs))


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
    lambda rs: rs[0].pop("image_used"),                                          # no picture recorded
    lambda rs: rs[0]["probabilities"].update(red=-0.2, blue=1.2, green=0.0),     # sums to 1, out of range
    lambda rs: rs[0].pop("image_id"),                                            # picture identity missing
    lambda rs: rs[1].update(image_id="i0"),                                      # another picture's identity
])
def test_wrong_content_is_refused(env, damage):
    run("freeze", env["conf"])
    write_result(env, damage)
    assert run("seal", env["conf"]) == 1 and run("check", env["conf"]) == 1


def test_quarantine_then_reseal(env):
    """seal -> damage -> quarantine (the old seal revoked) -> evaluate again -> seal -> check passes."""
    run("freeze", env["conf"])
    write_result(env)
    assert run("seal", env["conf"]) == 0
    det = env["eval"] / "a.details.jsonl"
    det.write_text(det.read_text().replace('"blue", "correct"', '"red", "correct"', 1))
    assert run("check", env["conf"]) == 1
    assert run("preflight", "--quarantine", env["conf"]) == 0
    assert json.load(open(env["eval"] / "RESULTS.json")) == {}
    time.sleep(0.01)
    write_result(env)
    assert run("seal", env["conf"]) == 0 and run("check", env["conf"]) == 0


def test_repeated_ids_match_by_copy(env, tmp_path):
    """An id held twice with different options (as in data/r13b/val.jsonl): both copies must be there."""
    data = tmp_path / "dup.jsonl"
    rows = [json.loads(x) for x in open(env["data"])]
    twin = json.loads(json.dumps(rows[0]))
    twin["questions"]["q"]["criteria"] = {"blue": None, "red": None}
    twin["gold"]["q"]["probabilities"] = {"blue": 1.0, "red": 0.0}
    data.write_text("".join(json.dumps(r) + "\n" for r in rows + [twin]))
    extras = tmp_path / "extras.yaml"
    extras.write_text(yaml.safe_dump([{"name": "dup", "data": str(data), "temperature": "raw", "role": "calibration_fit"}]))
    run("freeze", env["conf"])
    write_result(env)
    time.sleep(0.01)
    write_result(env, name="extra_dup", data=data, temperature="raw", role="calibration_fit")
    assert run("seal", "--extras", str(extras), env["conf"]) == 0
    assert run("check", "--extras", str(extras), env["conf"]) == 0
    assert set(ce.verified(env["conf"], str(extras), ["dup", "a"])) == {"dup", "a"}
    write_result(env, lambda rs: rs.pop(), name="extra_dup", data=data, temperature="raw", role="calibration_fit")
    assert run("check", "--extras", str(extras), env["conf"]) == 1
    with pytest.raises(SystemExit):
        ce.verified(env["conf"], str(extras), ["dup"])


def test_mismatched_must_follow_the_fixed_pairing(env):
    cfg = yaml.safe_load(open(env["conf"]))
    cfg["evaluate"]["controls"] = True
    open(env["conf"], "w").write(yaml.safe_dump(cfg))
    from devision.train.evaluate import picture_pairs
    from devision.train.samples import read_decision_jsonl
    pairs = picture_pairs(read_decision_jsonl(str(env["data"])), seed=0)
    run("freeze", env["conf"])
    write_result(env)
    write_result(env, name="a.reversed", control="reversed")
    write_result(env, name="a.mismatched", control="mismatched", pictures=pairs)
    assert run("seal", env["conf"]) == 0 and run("check", env["conf"]) == 0
    wrong = {k: v for k, v in zip(pairs, list(pairs.values())[1:] + list(pairs.values())[:1])}   # another derangement
    write_result(env, name="a.mismatched", control="mismatched", pictures=wrong)
    assert run("check", env["conf"]) == 1
