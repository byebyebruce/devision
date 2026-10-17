"""Candidate B watch rules and process handling on fabricated windows / processes (no model)."""
import importlib.util
import os
import subprocess
import sys
import time

HERE = os.path.dirname(__file__)
spec = importlib.util.spec_from_file_location("v04b_watch", os.path.join(HERE, "watch.py"))
assert spec is not None and spec.loader is not None
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


def wins(**kw):
    return {int(k[1:]): v for k, v in kw.items()}


def test_rules():
    assert w.decide(wins(s100=6.5), {}, {}, 0).startswith("first window")
    assert w.decide(wins(s100=5, s200=6.2, s300=6.3), {}, {}, 0).startswith("two windows")
    assert w.decide(wins(s100=5, s1600=5, s1700=30, s1800=6.5), {}, {}, 0) == ""          # 1,700 holds the validation
    assert w.decide(wins(s100=5, s200=5), {200: 16.5}, {200: 1.5}, 0).startswith("window 200")  # any window
    assert w.decide({}, {}, {}, 2600) == "" and w.decide({}, {}, {}, 2800).startswith("no new window")
    assert w.decide(wins(s100=5, s200=5), {}, {}, 1000).startswith("no new window")           # 15 min otherwise
    assert w.decide(wins(s100=5, s3200=5), {}, {}, 2600) == ""                                # after 3,200: the end
    assert w.decide(wins(s100=5, s3200=5), {}, {}, 2800).startswith("no new window")


def run(tmp_path, cmd, limits, poll=0.2):
    code = ("import sys; sys.path.insert(0, %r); import watch; sys.exit(watch.main(sys.argv[1:], limits=%r, poll=%r))"
            % (HERE, limits, poll))
    return subprocess.Popen([sys.executable, "-c", code, str(tmp_path / "stage"), str(tmp_path / "log.txt"), "--"] + cmd,
                            cwd=str(tmp_path))


def test_processes(tmp_path):
    """Normal completion passes the status through; a stall with no cache stops only the owned group (a child that
    ignores TERM is killed), a bystander lives on, nothing is left; a TERM to the watch stops the training."""
    p = run(tmp_path, [sys.executable, "-c", "import sys; sys.exit(0)"], {"first": 60.0, "val": 60.0, "other": 60.0})
    assert p.wait(20) == 0
    tag = "v04bfake%d" % os.getpid()
    sleeper = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(300)"
    bystander = subprocess.Popen([sys.executable, "-c", sleeper, tag])
    p = run(tmp_path, [sys.executable, "-c", sleeper, tag], {"first": 2.0, "val": 2.0, "other": 2.0})
    t0 = time.time()
    assert p.wait(40) == 3 and time.time() - t0 < 25
    time.sleep(1)
    assert bystander.poll() is None
    assert subprocess.run(["pgrep", "-f", tag], capture_output=True, text=True).stdout.split() == [str(bystander.pid)]
    bystander.kill()
    p = run(tmp_path, [sys.executable, "-c", "import time; time.sleep(300)", tag + "t"], {"first": 600.0, "val": 600.0, "other": 600.0})
    time.sleep(1.5)
    p.terminate()
    assert p.wait(30) == 3
    time.sleep(0.5)
    assert subprocess.run(["pgrep", "-f", tag + "t"], capture_output=True).returncode != 0
