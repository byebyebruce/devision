"""exp55 without the real model: the separator function (thousands, fractions, unreadable, refusals, round trip); P / G / D
change only table values and E is EXP54's byte for byte; gold / options never reach the formatting; the report keeps
EXP54's read-right set and donor counts (separators removed for matching); paired details on other questions are refused;
an unreproduced control withholds the format attribution; the run finalizes inside its limit through the real watch."""
import copy
import importlib.util
import json
import os
import sys

import pytest

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tests"))
spec = importlib.util.spec_from_file_location("exp55", os.path.join(HERE, "exp55.py"))
assert spec is not None and spec.loader is not None
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
e54 = m.e54


def test_format_function():
    cases = {"1360": "1,360", "14": "14", "0.75": "0.75", "999": "999", "1000": "1,000", "1234567.25": "1,234,567.25",
             "12345.0": "12,345.0", m.UNREADABLE: m.UNREADABLE}
    for a, b in cases.items():
        assert m.fmt(a) == b and m.unfmt(m.fmt(a)) == a
    for bad in ("1,360", "abc", "", "-5", "1.2.3", " 14", "1e3"):
        with pytest.raises(ValueError):
            m.fmt(bad)


@pytest.fixture(scope="module")
def old_inputs():
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        rows = {r["id"]: r for r in map(json.loads, open(e54.TUNE))}
        return rows, {c: e54.load_inputs(m.E54_OUT, c) for c in m.CONDS}
    finally:
        os.chdir(cwd)


def test_only_table_values_change(old_inputs):
    rows, old = old_inputs
    new, counts = m.new_inputs(old, rows)
    assert new["E"] == old["E"] and counts["E"]["requests_changed"] == 0
    assert counts["G"]["values_changed"] == 389 and counts["P"]["values_changed"] == counts["D"]["values_changed"] == 386
    for c in ("P", "G", "D"):
        for rid, v in new[c].items():
            assert [m.unfmt(x) for _, _, x in v["table"]] == [x for _, _, x in old[c][rid]["table"]]
            assert [r[:2] for r in v["table"]] == [r[:2] for r in old[c][rid]["table"]]
            assert v["state"].replace(",", "") == old[c][rid]["state"].replace(",", "")      # nothing else moved
            assert v.get("twin_row") == old[c][rid].get("twin_row")


def test_gold_and_options_never_reach_the_formatting(old_inputs):
    rows, old = old_inputs
    rows2 = copy.deepcopy(rows)
    for r in rows2.values():
        for qid, q in r["questions"].items():
            q["criteria"] = dict(reversed(list(q["criteria"].items())))
            r["gold"][qid]["probabilities"] = {o: 1.0 / len(q["criteria"]) for o in q["criteria"]}
    assert m.new_inputs(old, rows2)[0] == m.new_inputs(old, rows)[0]


def test_report_keeps_read_right_and_donor_counts(old_inputs, tmp_path):
    rows, old = old_inputs
    new, _ = m.new_inputs(old, rows)
    out = tmp_path / "exp55"
    out.mkdir()
    files = {}
    for c in m.CONDS:
        f = "inputs_%s.jsonl" % c
        (out / f).write_text("".join(json.dumps({"id": rid, **{k: v for k, v in new[c][rid].items() if k != "id"}}) + "\n"
                                     for rid in sorted(new[c])))
        files[f] = e54.sha(str(out / f))
    e54.register(str(out), "inputs", files)
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        qs53 = {q["key"]: q for q in map(json.loads, open(e54.QUESTIONS)) if q["set"] == "rl_tune"}
        reading = [{"key": k, "prediction": None} for k in qs53]
        plain = m.plain_inputs(str(out), "D")
        st = e54.donor_stats(reading, rows, qs53, plain)
        raw = e54.donor_stats(reading, rows, qs53, new["D"])             # matching against "1,360" would miss
    finally:
        os.chdir(cwd)
    assert (st["changed"], st["actual_differs_from_twin_gold"], st["actual_unreadable"], st["actual_input_in_options"]) == (828, 131, 24, 702)
    assert raw["actual_input_in_options"] < 702


def rec(sid, qid, probs, gold="a"):
    pred = max(probs, key=probs.get)
    return {"sample_id": sid, "qid": qid, "probabilities": probs, "prediction": pred, "correct": pred == gold, "nll": 0.1}


def test_pairing_and_reproduction():
    old = [rec("r1", "q", {"a": 0.7, "b": 0.3}), rec("r2", "q", {"a": 0.4, "b": 0.6})]
    new = [rec("r1", "q", {"a": 0.7, "b": 0.3}), rec("r2", "q", {"a": 0.8, "b": 0.2})]
    p = m.paired(new, old, {}, changed_requests={"r2"})
    assert p["reproduced"] and p["transitions"] == {"wrong_to_right": 1, "right_to_wrong": 0}
    p2 = m.paired([rec("r1", "q", {"a": 0.69, "b": 0.31}), new[1]], old, {}, changed_requests={"r2"})
    assert not p2["reproduced"]                                              # an unchanged request moved
    with pytest.raises(SystemExit):
        m.paired(new[:1], old, {}, changed_requests=set())                    # other questions


def test_unreproduced_control_withholds_the_attribution():
    assert m.final_branch("G below its line: x", False).startswith("control not reproduced")
    assert m.final_branch("G below its line: x", True) == "G below its line: x"
    assert m.final_branch("incomplete: ['P']", False).startswith("incomplete")


def test_e54_files_are_bound(monkeypatch, tmp_path):
    monkeypatch.setattr(m, "E54_OUT", str(tmp_path))
    with pytest.raises((SystemExit, FileNotFoundError)):
        m.e54_bound()


STAGE = """import signal, subprocess, sys, time
child = subprocess.Popen([sys.executable, '-c', 'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(300)'])
open('runs/exp55/child.pid', 'w').write(str(child.pid))
time.sleep(300)
"""
STUB = """#!/bin/bash
echo "uv $*" >> "${MARK}"
case "$*" in
  *"watch.py"*) shift 2; exec python3 "$@";;
  *"exp55.py start"*) exit 0;;
  *"exp55.py decide"*) exec python3 "${STAGE}";;
  *"exp55.py report --finalize"*) echo final > runs/exp55/report.txt; exit 0;;
esac
exit 9
"""


def test_run_stops_at_the_watch_deadline_and_finalizes(tmp_path):
    import shutil
    import stat
    import subprocess
    import time
    root = tmp_path / "repo"
    for rel in ("scripts/exp55/run.sh", "scripts/exp53/watch.py", "scripts/exp35/guard.sh", "scripts/exp38/steps.sh"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join(ROOT, rel), root / rel)
    (tmp_path / "stage.py").write_text(STAGE)
    b = tmp_path / "bin"
    b.mkdir()
    for name, body in {"uv": STUB, "pgrep": '#!/bin/bash\n[ "$1" = "-fl" ] && exit 1\nexec /usr/bin/pgrep "$@"\n'}.items():
        (b / name).write_text(body)
        (b / name).chmod((b / name).stat().st_mode | stat.S_IEXEC)
    env = {**os.environ, "PATH": "%s:%s" % (b, os.environ["PATH"]), "MARK": str(tmp_path / "calls"), "STAGE": str(tmp_path / "stage.py"),
           "EXP55_LIMIT": "60", "EXP55_RESERVE": "4"}
    t = time.time()
    p = subprocess.run(["bash", str(root / "scripts/exp55/run.sh")], env=env, capture_output=True, text=True, timeout=120)
    assert p.returncode == 124 and time.time() - t < 60, p.stdout + p.stderr
    assert "report --finalize" in (tmp_path / "calls").read_text()
    child = int((root / "runs/exp55/child.pid").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(child, 0)
