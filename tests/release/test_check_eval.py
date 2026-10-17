"""scripts/release/check_eval.py: reused results must come from the YAML's checkpoint, the current data bytes and
cover every question; a missing record or another checkpoint's file is reported."""
import importlib.util
import json
import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
spec = importlib.util.spec_from_file_location("check_eval", os.path.join(ROOT, "scripts", "release", "check_eval.py"))
assert spec is not None and spec.loader is not None
check_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_eval)


def setup(tmp_path, records=None, checkpoint=None):
    rnd = tmp_path / "runs" / "cand"
    (rnd / "stage2").mkdir(parents=True)
    (rnd / "eval").mkdir()
    (rnd / "run.json").write_text("{}")
    data = tmp_path / "set.jsonl"
    rows = [{"id": "s%d" % i, "image": "x.png", "questions": {"q": {"type": "noul", "instructions": "?"}},
             "gold": {"q": {"probabilities": {"true": 1.0, "false": 0.0}}}} for i in range(3)]
    data.write_text("".join(json.dumps(r) + "\n" for r in rows))
    ckpt = str(rnd / "stage2")
    cfg = {"name": "full", "stages": [], "evaluate": {"checkpoint": ckpt, "sets": {"a": {"path": str(data), "role": "heldout"}}}}
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg))
    run = {"checkpoint": checkpoint or ckpt, "data_sha256": check_eval.sha256(str(data)), "temperature": "fitted",
           "control": "none", "role": "heldout"}
    (rnd / "eval" / "a.json").write_text(json.dumps({"run": run}))
    recs = records if records is not None else ["s0", "s1", "s2"]
    (rnd / "eval" / "a.details.jsonl").write_text("".join(json.dumps({"sample_id": s, "qid": "q"}) + "\n" for s in recs))
    return str(tmp_path / "c.yaml")


def test_complete(tmp_path):
    assert check_eval.check(setup(tmp_path)) == []


def test_missing_record_and_other_checkpoint(tmp_path):
    problems = check_eval.check(setup(tmp_path, records=["s0", "s1", "s1"], checkpoint="runs/other/stage2"))
    assert any("run.checkpoint" in p for p in problems)
    assert any("duplicate" in p for p in problems) and any("1 of 3 questions missing" in p for p in problems)
