"""exp38 without a model: the report (H - C, the four conditions, identity checks, incomplete main readouts, partly
evaluated protection) and the step control (a deadline that falls inside the report-only evaluations keeps the main
results, records what is missing, stops the child within its bound, still writes the final report; a failing report
makes the chain fail)."""
import importlib.util
import json
import os
import subprocess
import textwrap
import time

import pytest

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.dirname(HERE))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


INPUT = {"H": {"image_size": 512, "visual_tokens": 256, "interpolate_pos_encoding": True},
         "C": {"image_size": 256, "visual_tokens": 64, "interpolate_pos_encoding": False}}


@pytest.fixture
def runs(tmp_path):
    t36 = load("exp36_test", os.path.join(HERE, "..", "exp36", "test_exp36.py"))
    out = {}
    for arm, right in (("H", lambda f, x: True), ("C", lambda f, x: False)):
        for split in ("tune", "fit"):
            d = tmp_path / arm / ("numbers_" + split)
            d.mkdir(parents=True)
            with open(d / "numbers.jsonl", "w") as fh:
                for x in t36.records(right, lambda f, x: False):
                    fh.write(json.dumps(x) + "\n")
            (d / "START.json").write_text(json.dumps({"sha256": {"model.safetensors": "same"},
                                                      "input": dict(INPUT[arm], config_sha256=arm)}))
        out[arm] = str(tmp_path / arm)
    return out


def write_status(path, lines):
    path.write_text("".join("%s %s\n" % x for x in lines))


MAIN = [("readout:%s:%s" % (a, s), "completed") for a in "hc" for s in ("tune", "fit")]


def test_report_conditions_and_partial_protection(runs, tmp_path, monkeypatch):
    r = load("exp38_report", os.path.join(HERE, "report.py"))
    monkeypatch.setattr(r.r36._exp30, "recs", lambda d, name: [])
    monkeypatch.setattr(r.r36._exp30, "metric", lambda a, b, p: {"difference": -0.05, "ci95": [-0.08, -0.02]})
    for arm in "HC":
        os.makedirs(os.path.join(runs[arm], "eval"))
    st = tmp_path / "status.txt"
    write_status(st, MAIN + [("eval:h:exist", "completed"), ("eval:h:science", "interrupted"), ("eval:c:exist", "missing")])
    out = r.build(runs, str(st))
    assert out["invest_in_512"] and out["identity"]["H"]["image_size"] == 512
    h = out["report_only"]["H"]
    assert list(h["protection"]) == ["exist"] and not h["protection_complete"] and h["costs"] == ["exist"]
    assert h["costs_meaning"].startswith("protection not completely evaluated") and "eval:h:science" in h["missing"]


def test_incomplete_main_and_swapped_identity(runs, tmp_path):
    r = load("exp38_report", os.path.join(HERE, "report.py"))
    st = tmp_path / "status.txt"
    write_status(st, MAIN[:-1] + [("readout:c:fit", "interrupted")])
    assert r.build(runs, str(st))["outcome"].startswith("incomplete")
    write_status(st, MAIN)
    start = os.path.join(runs["H"], "numbers_fit", "START.json")
    json.dump({"sha256": {"model.safetensors": "same"}, "input": dict(INPUT["C"], config_sha256="C")}, open(start, "w"))
    with pytest.raises(SystemExit):                      # H's fit read at C's setting
        r.build(runs, str(st))
    json.dump({"sha256": {"model.safetensors": "same"}, "input": dict(INPUT["C"], config_sha256="C")},
              open(os.path.join(runs["H"], "numbers_tune", "START.json"), "w"))
    with pytest.raises(SystemExit):                      # both H readouts at C's setting
        r.build(runs, str(st))


def run_steps(tmp_path, body, deadline_in):
    script = tmp_path / "t.sh"
    script.write_text(textwrap.dedent("""
        set -u
        log() { echo "== $*"; }
        . %s/scripts/exp35/guard.sh
        DEADLINE=$(( $(date +%%s) + %d )); STATUS=%s/status.txt; RESERVE=8
        . %s/scripts/exp38/steps.sh
        """ % (ROOT, deadline_in, tmp_path, ROOT)) + textwrap.dedent(body))
    t0 = time.time()
    p = subprocess.run(["bash", str(script)], capture_output=True, text=True)
    return p, time.time() - t0, (tmp_path / "status.txt").read_text() if (tmp_path / "status.txt").exists() else ""


def test_deadline_inside_report_only_steps(tmp_path):
    p, took, status = run_steps(tmp_path, """
        step main true || exit 1
        echo main-result > %s/main.txt
        step extra1 bash -c 'trap "" TERM; sleep 60'
        step extra2 true
        final_report bash -c 'echo final > %s/final.txt'
        exit $?
        """ % (tmp_path, tmp_path), 30)
    assert p.returncode == 0 and took < 31
    assert "main completed" in status and "extra1 interrupted" in status and "extra2 missing" in status
    assert (tmp_path / "main.txt").exists() and (tmp_path / "final.txt").exists()


def test_failing_report_fails_the_chain(tmp_path):
    p, _, _ = run_steps(tmp_path, """
        final_report bash -c 'exit 3'
        exit $?
        """, 30)
    assert p.returncode == 3
