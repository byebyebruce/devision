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
    env = {**os.environ, "GUARD_GRACE": "2", "GUARD_POLL": "0.3", "GUARD_KILL_CHECK": "1"}
    t0 = time.time()
    p = subprocess.run([sys.executable, GUARD, "--record", str(tmp_path / "rec.json"), "--watch", str(tmp_path / "logs"),
                        "--out", str(tmp_path / "out.txt"), "--deadline", str(time.time() + limit), "--", sys.executable, str(fake), mode, str(tmp_path / "logs")],
                       env=env, capture_output=True, text=True, timeout=90)
    return p, json.load(open(tmp_path / "rec.json")), time.time() - t0


def group_gone(pgid):
    return not load("guarded").members(pgid)


def test_gpu_error_in_a_late_log_ends_the_group(tmp_path):
    bystander = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        p, rec, took = guard(tmp_path, "gpu_late_log")
        assert p.returncode == 70 and "command buffer exited" in rec["gpu_error"]["line"] and rec["gpu_error"]["file"].endswith("d2.log")
        assert rec["end"]["kill"] and rec["end"]["confirmed"] and rec["cleanup_confirmed"] and took < 30 and group_gone(rec["pgid"])
        assert bystander.poll() is None                                                 # outside the group: untouched
    finally:
        bystander.kill()


def test_gpu_error_found_after_a_clean_exit(tmp_path):
    p, rec, _ = guard(tmp_path, "gpu_then_exit0")
    assert p.returncode == 70 and rec["child_returncode"] == 0 and rec["gpu_error"]["after_exit"]


def test_limit_and_clean_runs(tmp_path):
    p, rec, took = guard(tmp_path / "a" if (tmp_path / "a").mkdir() is None else tmp_path, "slow", limit=8)
    assert p.returncode == 124 and rec["end"]["why"] == "limit" and took < 8 and group_gone(rec["pgid"])   # stopped AND cleaned by the deadline
    (tmp_path / "b").mkdir()
    p, rec, _ = guard(tmp_path / "b", "fail")
    assert p.returncode == 3 and "gpu_error" not in rec and group_gone(rec["pgid"])        # the leftover child was cleared too


def test_an_unreapable_group_still_returns_by_the_deadline(tmp_path, monkeypatch):
    """EXP76-RETRY-R1-01: a command that never gets reaped and a member that never leaves (CPU stand-ins -- no real
    unkillable process): the guard stops trying by its deadline, records the PIDs left and "not confirmed", exit 71."""
    g = load("guarded")
    monkeypatch.setattr(g, "GRACE", 1.0)
    monkeypatch.setattr(g, "KILL_CHECK", 1.0)
    monkeypatch.setattr(g, "POLL", 0.2)

    class Stuck:
        pid = 2 ** 22 + 7                                   # no such process / group: every signal is a no-op
        returncode = None

        def __init__(self, *a, **k):
            pass

        def wait(self, timeout=None):
            if timeout is None:
                raise AssertionError("an unbounded wait")
            time.sleep(timeout)
            raise subprocess.TimeoutExpired("stuck", timeout)
    monkeypatch.setattr(g.subprocess, "Popen", Stuck)
    monkeypatch.setattr(g, "members", lambda pgid: [pgid + 1])
    deadline = time.time() + 6
    rc = g.main(["--record", str(tmp_path / "rec.json"), "--out", str(tmp_path / "out.txt"), "--deadline", str(deadline), "--", "x"])
    assert rc == 71 and time.time() <= deadline
    rec = json.load(open(tmp_path / "rec.json"))
    assert rec["cleanup_confirmed"] is False and rec["end"]["why"] == "limit" and rec["end"]["left"] == [Stuck.pid + 1]
    assert rec["end"]["child_reaped"] is False and rec["end"]["kill"]


@pytest.mark.parametrize("rec,valid", [({"exit": 0, "cleanup_confirmed": True}, True), ({"exit": 70, "gpu_error": {"line": "x"}, "cleanup_confirmed": True}, False),
                                       ({"exit": 0, "gpu_error": {"line": "x", "after_exit": True}, "cleanup_confirmed": True}, False),
                                       ({"exit": 124, "cleanup_confirmed": True}, False), (None, False),
                                       ({"exit": 71, "cleanup_confirmed": False, "end": {"left": [5], "child_reaped": False}}, False),
                                       ({"exit": 0}, False), ({"exit": 3, "cleanup_confirmed": True}, False)])
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
    assert "EXP76_LIMIT:-5700" in s and "EXP76_RESERVE:-120" in s


STUCK_GUARD = r'''
import json, sys, time
a = sys.argv[1:]
rec, deadline = a[a.index("--record") + 1], float(a[a.index("--deadline") + 1])
time.sleep(max(0, deadline - time.time() - 0.5))          # a cleanup that uses its whole budget and still fails
json.dump({"exit": 71, "cleanup_confirmed": False, "end": {"why": "limit", "left": [123], "child_reaped": False}}, open(rec, "w"))
print("== stuck guard: exit 71", flush=True)
sys.exit(71)
'''


def test_run_r2_keeps_the_report_reserve_when_cleanup_fails(tmp_path):
    root = tmp_path / "repo"
    for rel in ("scripts/panel_decision/run_r2.sh", "scripts/panel_decision/preflight_r2.py", "scripts/panel_decision/report_r2.py",
                "scripts/panel_learn/report.py", "configs/exp76-panel-decision-r2.yaml", "scripts/exp35/guard.sh", "scripts/exp38/steps.sh"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join(ROOT, rel), root / rel)
    (root / "scripts/panel_decision/guarded.py").write_text(STUCK_GUARD)        # the guard's failed-cleanup outcome, stood in
    (root / "runs").mkdir()
    b = tmp_path / "bin"
    b.mkdir()
    for name, body in {"uv": STUB, "pgrep": "#!/bin/bash\nexit 1\n", "ps": "#!/bin/bash\nexit 0\n"}.items():
        (b / name).write_text(body)
        (b / name).chmod((b / name).stat().st_mode | stat.S_IEXEC)
    env = {**os.environ, "PATH": "%s:%s" % (b, os.environ["PATH"]), "MARK_CALLS": str(tmp_path / "calls"), "FAKE_PY": "-", "FAKE_MODE": "-",
           "EXP76_LIMIT": "40", "EXP76_RESERVE": "20"}
    p = subprocess.run(["bash", str(root / "scripts/panel_decision/run_r2.sh")], env=env, capture_output=True, text=True, timeout=60, cwd=str(root))
    assert p.returncode == 71, p.stdout + p.stderr
    deadline = int(open(root / "runs/exp76-panel-decision-r2.started").readline().split()[1])
    guard_end = time.mktime(time.strptime(next(x for x in p.stdout.splitlines() if "guarded pipeline exit" in x)[3:22], "%Y-%m-%d %H:%M:%S"))
    assert deadline - guard_end >= 20 - 1                                    # the full report reserve is left (1 s clock granularity)
    assert "no further model is started" in p.stdout and "report exit 1" in p.stdout
    rep = json.load(open(root / "runs/exp76-panel-decision-r2/report.json"))
    assert rep["outcome"].startswith("incomplete: execution failed (cleanup not confirmed") and rep["execution"]["valid"] is False
