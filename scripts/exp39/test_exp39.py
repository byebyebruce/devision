"""exp39 resource watch rules on fabricated windows (no model)."""
import importlib.util
import os

spec = importlib.util.spec_from_file_location("exp39_watch", os.path.join(os.path.dirname(__file__), "watch.py"))
assert spec is not None and spec.loader is not None
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


def test_rules():
    assert w.decide({50: 9.5}, {}, {}, 0).startswith("first window")
    assert w.decide({50: 7.0}, {50: 17.0}, {50: 1.5}, 0).startswith("first window MPS")
    assert w.decide({50: 7.0, 100: 13.0, 150: 13.0}, {}, {}, 0).startswith("two windows")
    assert w.decide({50: 7, 500: 7, 550: 30, 600: 13}, {}, {}, 0) == ""          # the window after val is not counted
    assert w.decide({50: 7, 100: 7}, {}, {}, 700).startswith("no new window")
    assert w.decide({50: 7, 500: 7}, {}, {}, 700) == ""                            # val after step 500: longer allowance
    assert w.decide({50: 7, 100: 7.5}, {50: 15}, {50: 3}, 100) == ""


def test_first_window_allowance():
    assert w.decide({}, {}, {}, 1500) == ""                                         # step-0 validation + 50 steps
    assert w.decide({}, {}, {}, 1900).startswith("no new window")


import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

HERE = os.path.dirname(__file__)


def test_report_incomplete_without_any_h_output(tmp_path):
    """An early stop leaves no START / run.json for H: the report says incomplete, with null identities, no crash."""
    spec = importlib.util.spec_from_file_location("exp39_report", os.path.join(HERE, "report.py"))
    assert spec is not None and spec.loader is not None
    r = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r)
    st = tmp_path / "s39.txt"
    st.write_text("train:h failed\n")
    out = r.build(str(tmp_path / "nothing-h"), str(tmp_path / "nothing-c"), (str(st),), str(tmp_path / "merged.txt"))
    assert out["outcome"].startswith("incomplete")
    assert out["sources"]["H"] == {"micro_batch": None, "init": None, "data": None, "readout_forward_rows": None}


def watch(tmp_path, cmd, limits, poll=0.2):
    code = ("import sys; sys.path.insert(0, %r); import watch; "
            "sys.exit(watch.main(sys.argv[1:], limits=%r, poll=%r))" % (HERE, limits, poll))
    return subprocess.Popen([sys.executable, "-c", code, str(tmp_path / "stage"), str(tmp_path / "log.txt"), "--"] + cmd,
                            cwd=str(tmp_path))


def alive(pattern):
    return subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode == 0


def test_stall_without_any_cache_stops_only_the_owned_group(tmp_path):
    """No run id / cache ever appears: the stall clock runs from the start; the owned training (a child that ignores
    TERM and has its own child) is stopped within the grace; a bystander with the same command line lives on."""
    tag = "exp39fake%d" % os.getpid()
    sleeper = "import signal, sys, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(300)"
    bystander = subprocess.Popen([sys.executable, "-c", sleeper, tag])
    inner = "%s -c '%s' %s" % (sys.executable, sleeper, tag)
    p = watch(tmp_path, ["bash", "-c", "trap '' TERM; %s & %s" % (inner, inner)], {"first": 2.0, "val": 2.0, "other": 2.0})
    t0 = time.time()
    rc = p.wait(40)
    assert rc == 3 and time.time() - t0 < 25                            # stall at 2 s, TERM ignored, KILL at +15 s
    time.sleep(1)
    assert bystander.poll() is None                                     # not touched
    out = subprocess.run(["pgrep", "-f", tag], capture_output=True, text=True).stdout.split()
    assert out == [str(bystander.pid)]                                   # nothing of the owned group is left
    bystander.kill()
    assert "stop: no new window" in (tmp_path / "log.txt").read_text()


def test_normal_exit_and_forwarded_term(tmp_path):
    p = watch(tmp_path, ["bash", "-c", "exit 0"], {"first": 60.0, "val": 60.0, "other": 60.0})
    assert p.wait(20) == 0 and "training exited 0" in (tmp_path / "log.txt").read_text()
    tag = "exp39term%d" % os.getpid()
    p = watch(tmp_path, [sys.executable, "-c", "import time; time.sleep(300)", tag], {"first": 600.0, "val": 600.0, "other": 600.0})
    time.sleep(1.5)
    p.terminate()                                                        # the chain's deadline
    assert p.wait(30) == 3
    time.sleep(0.5)
    assert not alive(tag)


def test_half_written_cache_is_not_progress(tmp_path):
    """A cache that holds only a truncated record gives no window: the stall clock still runs."""
    (tmp_path / "stage").mkdir()
    (tmp_path / "stage" / "swanlab_run.json").write_text(json.dumps({"id": "abc"}))
    d = tmp_path / "swanlog" / "run-1-abc"
    d.mkdir(parents=True)
    (d / "run-abc.swanlab").write_bytes(b"\n\x0bperf/step_s\x10")      # cut in the middle of a record
    p = watch(tmp_path, ["bash", "-c", "sleep 300"], {"first": 2.0, "val": 2.0, "other": 2.0})
    assert p.wait(40) == 3
    assert "stop: no new window" in (tmp_path / "log.txt").read_text()
