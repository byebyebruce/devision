"""Candidate B screen without a model: missing training or any required set gives "incomplete" (never a pass), and
an evaluation of another checkpoint is refused."""
import importlib.util
import json
import os

import pytest

HERE = os.path.dirname(__file__)
spec = importlib.util.spec_from_file_location("v04b_report", os.path.join(HERE, "report.py"))
assert spec is not None and spec.loader is not None
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def setup(tmp_path, statuses):
    run = tmp_path / "v04b-fit"
    (run / "stage2").mkdir(parents=True)
    (run / "stage2" / "model.safetensors").write_bytes(b"w")
    (run / "stage2" / "config.json").write_text("{}")
    (run / "eval").mkdir()
    st = tmp_path / "status.txt"
    st.write_text("".join("%s %s\n" % x for x in statuses))
    return str(run), str(st)


def test_incomplete(tmp_path):
    run, st = setup(tmp_path, [("train", "failed")])
    assert r.build(run, "abc", st)["outcome"].startswith("incomplete: train")
    run, st = setup(tmp_path / "b", [("train", "completed")] + [("eval:" + n, "completed") for n in r.REQUIRED if n != "syn_erased"])
    out = r.build(run, "abc", st)
    assert out["outcome"] == "incomplete: syn_erased" and out["identity"]["start_model_sha256"] == "abc"


def test_other_checkpoint_refused(tmp_path):
    run, st = setup(tmp_path, [("train", "completed")] + [("eval:" + n, "completed") for n in r.REQUIRED])
    for n in r.REQUIRED:
        json.dump({"run": {"checkpoint": "runs/other/stage2", "temperature": "fitted"}}, open(os.path.join(run, "eval", n + ".json"), "w"))
    with pytest.raises(SystemExit):
        r.build(run, "abc", st)
