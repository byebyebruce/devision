"""exp24 gate and phase-B readout without a model: the gate's threshold and exits, the paired comparison with
exp22 on the same questions, and the report command end to end on fabricated runs."""
import importlib.util
import json
import os
import sys

import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "exp21"))
from test_readout import control, syn_records, write  # type: ignore  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location("exp24_" + name, os.path.join(HERE, name + ".py"))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gate, report = _load("gate"), _load("report")


def row(step, real, shuffled):
    return {"step": step, "val/mlm_acc_real": real, "val/mlm_acc_shuffled": shuffled, "val/mlm_acc_gap": real - shuffled}


@pytest.mark.parametrize("end,run", [((0.70, 0.40), True), ((0.65, 0.35), True), ((0.64, 0.35), False), ((0.2, 0.2), False)])
def test_gate(end, run, tmp_path):
    rep = {"val": [row(0, 0.1, 0.1), row(2500, *end)]}
    assert gate.decide(rep)["run_phase_b"] is run
    p = tmp_path / "a.json"
    p.write_text(json.dumps(rep))
    with pytest.raises(SystemExit) as e:
        gate.main([str(p)])
    assert e.value.code == (0 if run else 10)


def test_gate_refuses_a_report_without_validation():
    with pytest.raises(ValueError):
        gate.decide({"val": []})


def run_dir(tmp_path, name, right):
    run = str(tmp_path / name)
    for step in (0, 500, 1000, 1500, 2000):
        d = os.path.join(run, "stage2", "monitor", "step%d" % step)
        last = step == 2000
        write(d, "val", syn_records(lambda f, v, h, q: last and right(f, q), heights=(16,)))
        write(d, "dev16", syn_records(lambda f, v, h, q: last and right(f, q), heights=(16,)))
        write(d, "dev", syn_records(lambda f, v, h, q: last and right(f, q) and h == 16))
        write(d, "erased", control(lambda q: False, "erased"))
        write(d, "text", control(lambda q: True, "text"))
    return run


def test_report_against_exp22(tmp_path, capsys):
    exp22 = run_dir(tmp_path, "exp22", lambda f, q: False)
    exp24 = run_dir(tmp_path, "exp24", lambda f, q: True)
    report.main([exp24, exp22])
    text = capsys.readouterr().out
    out = json.load(open(os.path.join(exp24, "report.json")))
    assert out["label"].startswith("fitted and generalizes")
    assert out["vs_exp22"]["val"]["value"] == 1.0 and out["vs_exp22"]["val"]["n"] == 40 * 2 * 2
    assert "vs exp22, 16 px read fit" in text


def test_report_refuses_different_questions(tmp_path):
    a = run_dir(tmp_path, "a", lambda f, q: False)
    b = str(tmp_path / "b")
    write(os.path.join(b, "stage2", "monitor", "step2000"), "val", syn_records(lambda f, v, h, q: True, families=5, heights=(16,)))
    with pytest.raises(ValueError):
        report.paired_read(os.path.join(a, "stage2", "monitor", "step2000", "val.details.jsonl"),
                           os.path.join(b, "stage2", "monitor", "step2000", "val.details.jsonl"))
