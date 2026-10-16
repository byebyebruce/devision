"""exp25 data and readout without a model: text-comparison questions carry the two values and the right answer,
on pictures without numbers, never a dev family; each outcome of the readout at step 5000."""
import importlib.util
import json
import os
import sys

import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "exp21"))
sys.path.insert(0, os.path.join(HERE, "..", "data"))
from exp21_reading import EQUAL  # type: ignore  # noqa: E402
from exp21_reading import build as exp21_build  # type: ignore  # noqa: E402
from test_readout import control, syn_records, write  # type: ignore  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


build = _load("exp25_build", os.path.join(HERE, "build.py"))
report = _load("exp25_report", os.path.join(HERE, "report.py"))
exp22_build = _load("exp22_build_for25", os.path.join(HERE, "..", "exp22", "build.py"))


def test_text_comparison_questions(tmp_path):
    replay = tmp_path / "r.jsonl"
    replay.write_text(json.dumps({"id": "r", "source": "r14b", "image_id": "x", "image": "x.png",
                                  "questions": {"q": {"type": "noul", "instructions": "?"}},
                                  "gold": {"q": {"probabilities": {"false": 0.0, "true": 1.0}}}}) + "\n")
    exp21_build(str(tmp_path), str(replay), n_train=20, n_dev=10, n_controls=2, total=10)
    exp22_build.build(str(tmp_path), n_train=20, n_fit=5)
    m = build.build(str(tmp_path), n=30)
    rows = [json.loads(x) for x in open(tmp_path / "exp25" / "text_cmp.jsonl")]
    assert len(rows) == 60 and all(list(r["questions"]) == ["cmp"] for r in rows)
    for r in rows:
        nums = [int(t.split(" of ")[1].split()[0].replace(",", "")) for t in r["state_text"].split(". ")[:2]]
        gold = max(r["gold"]["cmp"]["probabilities"], key=r["gold"]["cmp"]["probabilities"].get)
        assert gold == ("A" if nums[0] > nums[1] else "B" if nums[1] > nums[0] else EQUAL)
        assert r["image"].endswith("_erased.png") and os.path.exists(tmp_path / r["image"])
    dev = {(s["questions"]["cmp"]["instructions"], tuple(sorted(s["questions"]["readA"]["criteria"])))
           for s in map(json.loads, open(tmp_path / "exp21" / "dev_syn.jsonl"))}
    assert not any((r["questions"]["cmp"]["instructions"], tuple(r["candidates"])) in dev for r in rows)
    assert m["files"]["train"]["questions"] == 20 * 2 * 3 + 60


def run_dir(tmp_path, read, cmp, text_cmp, erased=lambda q: False):
    run = str(tmp_path / "run")
    for step in report.STEPS:
        d = os.path.join(run, "stage2", "monitor", "step%d" % step)
        last = step == report.LAST
        right = lambda f, v, h, q: last and (read if q != "cmp" else cmp)   # noqa: E731
        write(d, "val", syn_records(right, heights=(16,)))
        write(d, "dev16", syn_records(right, heights=(16,)))
        write(d, "dev", syn_records(right))
        write(d, "erased", control(erased, "erased"))
        write(d, "text", control(lambda q: q != "cmp" or text_cmp, "text"))
    return run


@pytest.mark.parametrize("read,cmp,text_cmp,erased,start", [
    (True, True, True, lambda q: False, "reading and comparison both pass"),
    (True, False, True, lambda q: False, "reads and compares text, but picture comparison does not pass"),
    (True, False, False, lambda q: False, "text comparison does not pass"),
    (False, False, True, lambda q: False, "text comparison passes, reading does not"),
    (True, True, True, lambda q: q == "readA", "invalid"),
])
def test_outcomes(tmp_path, capsys, read, cmp, text_cmp, erased, start):
    run = run_dir(tmp_path, read, cmp, text_cmp, erased)
    report.main([run])
    out = json.load(open(os.path.join(run, "report.json")))
    assert out["outcome"].startswith(start) and out["step"] == 5000
    assert set(out["trajectory"]) == {str(s) for s in report.STEPS}
    assert "outcome:" in capsys.readouterr().out
