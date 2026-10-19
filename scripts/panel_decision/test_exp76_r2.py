"""EXP76 r2 preparation (EXP76-RETRY-PREP-01), CPU, fake processes only:
  - the r2 config equals EXP76's but for the name; preflight_r2 passes on the real files (records written elsewhere),
    refuses changed retry-of evidence or an r2 path that exists;
  - guarded.py: a GPU line in a log created later ends the group (TERM, then KILL of a child that ignores TERM), exit 70;
    a GPU line found only after a clean exit still gives 70; the limit gives 124; a clean run returns its own code; a
    bystander process outside the group survives;
  - report_r2: a GPU error (even with exit 0) or the limit voids the outcome; a clean record leaves the D2 report as is;
  - run_r2.sh end to end with a fake pipeline: the runner survives the guard, the report is written, one start."""
import hashlib
import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
import time

import pytest
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
GUARD = os.path.join(HERE, "guarded.py")


def load(name):
    spec = importlib.util.spec_from_file_location("r2_" + name, os.path.join(HERE, name + ".py"))
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_config_is_exp76_but_for_the_name():
    a, b = (yaml.safe_load(open(os.path.join(ROOT, c))) for c in ("configs/exp76-panel-decision.yaml", "configs/exp76-panel-decision-r2.yaml"))
    assert b.pop("name") == "exp76-panel-decision-r2" and a.pop("name") == "exp76-panel-decision" and a == b


@pytest.fixture
def pf(tmp_path, monkeypatch):
    m = load("preflight_r2")
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(m, "OUT", str(tmp_path / "pf.json"))
    monkeypatch.setattr(m, "START", str(tmp_path / "START.json"))
    return m, tmp_path


def test_preflight_r2_on_the_real_files(pf):
    m, tmp = pf
    assert m.problems() == []
    assert m.main() == 0 and json.load(open(tmp / "START.json"))["retry_of"]["evidence"] == m.RETRY_OF


def test_preflight_r2_refusals(pf, monkeypatch):
    m, tmp = pf
    monkeypatch.setattr(m, "RETRY_OF", dict(m.RETRY_OF, **{"runs/exp76-panel-decision/report.json": "0" * 64}))
    assert any("retry-of evidence" in x for x in m.problems())
    assert m.main() == 12 and not (tmp / "START.json").exists()
    monkeypatch.setattr(m, "RETRY_OF", {})
    monkeypatch.setattr(m, "FRESH", (str(tmp),))           # an r2 path that already exists
    assert any("exists (fresh only)" in x for x in m.problems())


FAKE = r'''
import os, signal, subprocess, sys, time
mode, logs = sys.argv[1], sys.argv[2]
child = subprocess.Popen([sys.executable, "-c", "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(120)"])
print("fake pipeline", flush=True)
if mode == "gpu_late_log":
    time.sleep(1.5)
    os.makedirs(logs, exist_ok=True)
    open(os.path.join(logs, "d2.log"), "w").write("step 0 ...\nError: command buffer exited with error status.\n")
    time.sleep(120)
if mode == "gpu_then_exit0":
    os.makedirs(logs, exist_ok=True)
    open(os.path.join(logs, "d2.log"), "w").write("Discarded (victim of GPU error/recovery) (00000005:kIOGPUCommandBufferCallbackErrorInnocentVictim)\n")
    child.kill()
    sys.exit(0)
if mode == "slow":
    time.sleep(120)
child.kill()
sys.exit(3 if mode == "fail" else 0)
'''


def guard(tmp_path, mode, limit=60):
    fake = tmp_path / "fake.py"
    fake.write_text(FAKE)
    env = {**os.environ, "GUARD_GRACE": "2", "GUARD_POLL": "0.3"}
    t0 = time.time()
    p = subprocess.run([sys.executable, GUARD, "--record", str(tmp_path / "rec.json"), "--watch", str(tmp_path / "logs"),
                        "--out", str(tmp_path / "out.txt"), "--limit", str(limit), "--", sys.executable, str(fake), mode, str(tmp_path / "logs")],
                       env=env, capture_output=True, text=True, timeout=90)
    return p, json.load(open(tmp_path / "rec.json")), time.time() - t0


def group_gone(pgid):
    return not load("guarded").members(pgid)


def test_gpu_error_in_a_late_log_ends_the_group(tmp_path):
    bystander = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        p, rec, took = guard(tmp_path, "gpu_late_log")
        assert p.returncode == 70 and "command buffer exited" in rec["gpu_error"]["line"] and rec["gpu_error"]["file"].endswith("d2.log")
        assert rec["end"]["kill"] and took < 30 and group_gone(rec["pgid"])            # the TERM-ignoring child was KILLed
        assert bystander.poll() is None                                                 # outside the group: untouched
    finally:
        bystander.kill()


def test_gpu_error_found_after_a_clean_exit(tmp_path):
    p, rec, _ = guard(tmp_path, "gpu_then_exit0")
    assert p.returncode == 70 and rec["child_returncode"] == 0 and rec["gpu_error"]["after_exit"]


def test_limit_and_clean_runs(tmp_path):
    p, rec, took = guard(tmp_path / "a" if (tmp_path / "a").mkdir() is None else tmp_path, "slow", limit=3)
    assert p.returncode == 124 and rec["end"]["why"] == "limit" and took < 30 and group_gone(rec["pgid"])
    (tmp_path / "b").mkdir()
    p, rec, _ = guard(tmp_path / "b", "fail")
    assert p.returncode == 3 and "gpu_error" not in rec and group_gone(rec["pgid"])        # the leftover child was cleared too


@pytest.mark.parametrize("rec,valid", [({"exit": 0}, True), ({"exit": 70, "gpu_error": {"line": "x"}}, False),
                                       ({"exit": 0, "gpu_error": {"line": "x", "after_exit": True}}, False), ({"exit": 124}, False), (None, False)])
def test_report_execution_first(tmp_path, monkeypatch, rec, valid):
    m = load("report_r2")
    g = tmp_path / "guard.json"
    if rec is not None:
        g.write_text(json.dumps(rec))

    class D2:
        @staticmethod
        def d2_report(run, sealed, config):
            return {"outcome": "A: fit learned, protection kept", "parts": {"x": 1}}
    monkeypatch.setattr(m, "d2", lambda: D2)
    out = m.report(str(tmp_path / "run"), str(g))
    if valid:
        assert out["outcome"].startswith("A:") and out["execution"]["valid"]
    else:
        assert out["outcome"].startswith("incomplete: execution failed") and out["voided"]["outcome"].startswith("A:")
    assert json.load(open(tmp_path / "run" / "report.json"))["outcome"] == out["outcome"]


STUB = r'''#!/bin/bash
echo "uv $*" >> "${MARK_CALLS}"
case "$*" in
  *"preflight_r2.py"*) echo '{"ok": true}' > runs/exp76-panel-decision-r2-preflight.json; echo '{}' > runs/exp76-panel-decision-r2.START.json; exit 0;;
  *"guarded.py"*) shift 2; exec python3 "$@";;
  *"devision-pipeline"*) exec python3 "${FAKE_PY}" "${FAKE_MODE}" runs/exp76-panel-decision-r2/logs;;
  *"report_r2.py"*) shift 2; exec python3 "$@";;
esac
exit 9
'''


@pytest.mark.parametrize("mode,limit,reserve,rc,want", [("gpu_late_log", "90", "20", 70, "incomplete: execution failed (GPU execution error)"),
                                                        ("slow", "40", "10", 124, "incomplete: execution failed (stopped at the limit)"),
                                                        ("clean", "90", "20", 1, "incomplete: the D2 report could not be built")])
def test_run_r2_end_to_end(tmp_path, mode, limit, reserve, rc, want):
    root = tmp_path / "repo"
    for rel in ("scripts/panel_decision/run_r2.sh", "scripts/panel_decision/preflight_r2.py", "scripts/panel_decision/guarded.py",
                "scripts/panel_decision/report_r2.py", "scripts/panel_learn/report.py", "configs/exp76-panel-decision-r2.yaml",
                "scripts/exp35/guard.sh", "scripts/exp38/steps.sh"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join(ROOT, rel), root / rel)
    (root / "runs").mkdir()
    (tmp_path / "fake.py").write_text(FAKE)
    b = tmp_path / "bin"
    b.mkdir()
    for name, body in {"uv": STUB, "pgrep": "#!/bin/bash\nexit 1\n", "ps": "#!/bin/bash\nexit 0\n"}.items():
        (b / name).write_text(body)
        (b / name).chmod((b / name).stat().st_mode | stat.S_IEXEC)
    env = {**os.environ, "PATH": "%s:%s" % (b, os.environ["PATH"]), "MARK_CALLS": str(tmp_path / "calls"), "FAKE_PY": str(tmp_path / "fake.py"),
           "FAKE_MODE": mode, "EXP76_LIMIT": limit, "EXP76_RESERVE": reserve, "GUARD_GRACE": "2", "GUARD_POLL": "0.3"}
    t0 = time.time()
    p = subprocess.run(["bash", str(root / "scripts/panel_decision/run_r2.sh")], env=env, capture_output=True, text=True, timeout=120, cwd=str(root))
    assert p.returncode == rc, p.stdout + p.stderr
    assert time.time() - t0 < int(limit)                                                              # inside the total limit
    rep = json.load(open(root / "runs/exp76-panel-decision-r2/report.json"))
    assert rep["outcome"].startswith(want), rep["outcome"]
    assert rep["execution"]["valid"] == (mode == "clean")                                            # a clean run keeps the D2 report's own verdict
    assert "report exit 1" in p.stdout                                                                # the runner survived and reported
    if mode != "clean":
        assert "voided" in rep
    p2 = subprocess.run(["bash", str(root / "scripts/panel_decision/run_r2.sh")], env=env, capture_output=True, text=True, cwd=str(root))
    assert p2.returncode == 1 and "already started once" in p2.stdout
    s = open(os.path.join(HERE, "run_r2.sh")).read()
    assert "EXP76_LIMIT:-5700" in s and "EXP76_RESERVE:-120" in s and "GUARD_GRACE:-15" in s
