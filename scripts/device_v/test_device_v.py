"""EXP76 device V stage (DEVICE-V-IMPL-PREP-01), CPU only: pure-data fixtures and fake processes; no weights, no
forward / backward, no V run.
  - vcheck: the fixed reference (changed -> refused), missing / duplicate / changed-input questions, the 1e-6 line and
    the prediction, the eight divergent questions judged and listed, the first attempt only described; the 64 V3
    questions; gpuEvent timestamps against the window; the boot cycle; when V3 may start; the one outcome;
  - preflight_v on the real files: everything holds but the boot cycle (still the EXP76 one) -> exit 12;
  - vrun with a fake training process: the planned stop after >= 250 updates (not the limit), then V3; a V1 mismatch,
    a GPU line, a non-finite loss, a system event in the window, the limit, an early exit, an unconfirmed cleanup each
    end without V3; a TERM-ignoring process is KILLed and confirmed;
  - run_v.sh end to end with stubs: one start, the report, the reserves; a refused preflight runs nothing."""
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
sys.path.insert(0, HERE)
import vcheck  # noqa: E402
import vrun  # noqa: E402

REF = os.path.join(ROOT, vcheck.REF_DIR)


def test_config_is_r2_but_for_the_name():
    a, b = (yaml.safe_load(open(os.path.join(ROOT, c))) for c in ("configs/exp76-panel-decision-r2.yaml", "configs/exp76-device-v.yaml"))
    assert b.pop("name") == "exp76-device-v" and a.pop("name") == "exp76-panel-decision-r2" and a == b


def copy_ref(dst, change=None):
    os.makedirs(dst, exist_ok=True)
    for n in vcheck.REF:
        rs = vcheck.rows(os.path.join(REF, n + ".details.jsonl"))
        if change and n == change[0]:
            rs = change[1](rs)
        with open(os.path.join(dst, n + ".details.jsonl"), "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rs)


def nudge(i, by):
    def f(rs):
        o = next(iter(rs[i]["probabilities"]))
        rs[i]["probabilities"][o] += by
        return rs
    return f


def test_reference_identical_and_first_attempt_described(tmp_path):
    r = vcheck.step0_consistency(REF, REF, first_dir=os.path.join(ROOT, vcheck.FIRST_DIR))
    assert r["ok"] and r["questions"] == 3562 and not r["refused"]
    assert r["sets"]["dev16"]["first_attempt_described"]["differing"] == 8          # described, not a pass rule
    assert [d["difference"] for d in r["divergent8"]] == [0.0] * 8


def test_first_attempt_fails_on_the_eight(tmp_path):
    r = vcheck.step0_consistency(REF, os.path.join(ROOT, vcheck.FIRST_DIR))
    assert not r["ok"] and len(r["problems"]) == 8 and all(p.startswith("dev16:") for p in r["problems"])
    assert all(d["difference"] > 1e-6 for d in r["divergent8"])                   # not exempt


@pytest.mark.parametrize("change,want", [
    (("val", nudge(5, 2e-6)), "probability differs"),
    (("val", nudge(5, 5e-7)), None),                                               # inside 1e-6
    (("dev16", nudge(995, 1e-3)), "probability differs"),                           # one of the eight, judged like any
    (("k_text", lambda rs: rs[:-1]), "missing"),
    (("erased", lambda rs: rs + rs[:1]), "twice"),
    (("rs_temp", lambda rs: [dict(r, gold_answer="x") if i == 0 else r for i, r in enumerate(rs)]), "inputs or gold differ"),
    (("rs_force", lambda rs: [dict(r, prediction="zz") if i == 0 else r for i, r in enumerate(rs)]), "prediction"),
    (("val", lambda rs: [dict(r, probabilities={k: float("nan") for k in r["probabilities"]}) if i == 0 else r for i, r in enumerate(rs)]), "not finite")])
def test_consistency_rules(tmp_path, change, want):
    copy_ref(tmp_path / "new", change)
    r = vcheck.step0_consistency(REF, str(tmp_path / "new"))
    assert r["ok"] == (want is None)
    if want:
        assert any(want in p for p in r["problems"]), r["problems"][:3]


def test_a_changed_reference_is_a_refusal(tmp_path):
    copy_ref(tmp_path / "ref", ("val", nudge(0, 1e-9)))
    r = vcheck.step0_consistency(str(tmp_path / "ref"), REF)
    assert r["refused"] and not r["ok"] and "reference val is missing or changed" in r["problems"]


def test_v3_questions():
    q = vcheck.v3_questions(vcheck.rows(os.path.join(REF, "dev16.details.jsonl")))
    assert len(q) == 64 == len(set(q)) and q[:8] == vcheck.DIVERGENT
    assert q[8:] == sorted(q[8:])
    with pytest.raises(ValueError):
        vcheck.v3_questions([{"sample_id": "a", "qid": "b"}])


def ips(path, when, reason):
    head = {"bug_type": "284", "timestamp": time.strftime("%Y-%m-%d %H:%M:%S.00 %z", time.localtime(when))}
    open(path, "w").write(json.dumps(head) + "\n" + '{\n  "process_name" : "WindowServer",\n  "analysis" : {"restart_reason_desc":"%s"}\n}\n' % reason)


def test_events_by_their_own_time(tmp_path):
    now = time.time()
    ips(tmp_path / "gpuEvent-WindowServer-a.ips", now - 3600, "MMU interrupt")          # before the window
    ips(tmp_path / "gpuEvent-WindowServer-b.ips", now - 10, "something else")            # other reason
    ips(tmp_path / "other-b.ips", now - 10, "MMU interrupt")                             # not a gpuEvent file
    assert vcheck.events_in_window(str(tmp_path), now - 60, now) == []
    ips(tmp_path / "gpuEvent-WindowServer-c.ips", now - 5, "BIF0 page fault")
    open(tmp_path / "gpuEvent-WindowServer-d.ips", "w").write("not json")                 # unreadable, written now
    hits = vcheck.events_in_window(str(tmp_path), now - 60, now + 5)
    assert sorted(h["file"] for h in hits) == ["gpuEvent-WindowServer-c.ips", "gpuEvent-WindowServer-d.ips"]
    real = "/Library/Logs/DiagnosticReports/gpuEvent-WindowServer-2026-10-20-025148.ips"
    if os.path.exists(real):
        e = vcheck.ips_event(real)
        assert e["reason"] == "BIF0 page fault" and time.strftime("%F %T", time.localtime(e["time"])) == "2026-10-20 02:51:48"


def test_boot_cycle():
    assert vcheck.boot_time("{ sec = 1786082443, usec = 273183 } Fri Aug  7 14:00:43 2026") == vcheck.EXP76_BOOT
    assert vcheck.boot_problems(vcheck.EXP76_BOOT, 1792445418)
    assert vcheck.boot_problems(1792445418 - 1, 1792445418)                             # booted before the last fault
    assert vcheck.boot_problems(None, 1792445418)
    assert vcheck.boot_problems(1792450000, 1792445418) == []


def test_v3_allowed_and_outcome():
    ok = {"end": {"why": "planned_stop", "initiated": True}, "v1": {"ok": True}, "cleanup_confirmed": True, "updates": 250}
    assert vcheck.v3_allowed(ok, 100, 600, 300, 120) == []
    assert any("needed" in w for w in vcheck.v3_allowed(ok, 200, 600, 300, 120))     # 400 s left < 420
    for k, v in (("end", {"why": "limit"}), ("v1", {"ok": False}), ("cleanup_confirmed", False), ("gpu_error", {"x": 1})):
        assert vcheck.v3_allowed(dict(ok, **{k: v}), 100, 600, 300, 120)
    good = {"start_epoch": time.time() - 100, "deadline_epoch": time.time() + 2600, "training": ok, "v3": {"exit": 0, "result": {"ok": True}, "cleanup_confirmed": True}}
    assert vcheck.outcome(good, []) == "pass"
    assert vcheck.outcome(good, [{"file": "x"}]).startswith("execution-unverified")      # the report's last look counts
    assert vcheck.outcome(dict(good, training=dict(ok, updates=225)), []).startswith("failed: planned stop recorded with fewer")
    assert vcheck.outcome(dict(good, v3={"exit": 1, "result": {"ok": False}}), []).startswith("failed: V3")
    assert vcheck.outcome(dict(good, v3={"exit": 0, "result": {"ok": False}}), []) == "consistency not established (V3 save / load)"
    assert vcheck.outcome(None, []).startswith("incomplete")


def test_preflight_v_refuses_this_boot_cycle(tmp_path, monkeypatch):
    import preflight_v as m
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(m, "OUT", str(tmp_path / "pf.json"))
    monkeypatch.setattr(m, "START", str(tmp_path / "START.json"))
    assert m.problems(boot=1792450000) == []                                         # everything else holds on the real files
    now = m.problems()
    assert len(now) == 1 and "boot cycle of the EXP76 faults" in now[0]               # today: no restart yet
    assert m.main() == 12 and not (tmp_path / "START.json").exists()
    copy_ref(tmp_path / "ref", ("dev16", nudge(0, 1e-9)))
    monkeypatch.setattr(vcheck, "REF_DIR", str(tmp_path / "ref"))
    assert any("V1 reference dev16" in p for p in m.problems(boot=1792450000))


FAKE = r'''
import json, os, signal, sys, time
mode, run = sys.argv[1], sys.argv[2]
ref, diag = os.environ["FAKE_REF"], os.environ.get("FAKE_DIAG", "")
if mode in ("ignore_term", "badjson"):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
os.makedirs(run + "/logs", exist_ok=True)
log = open(run + "/logs/d2.log", "a", buffering=1)
s0 = run + "/d2/monitor/step0"
os.makedirs(s0, exist_ok=True)
for f in sorted(os.listdir(ref)):
    rows = [json.loads(x) for x in open(os.path.join(ref, f))]
    if mode == "mismatch" and f.startswith("dev16"):
        o = next(iter(rows[992]["probabilities"])); rows[992]["probabilities"][o] += 0.1
    open(os.path.join(s0, f), "w").writelines(json.dumps(r) + "\n" for r in rows)
    time.sleep(0.02)
if mode == "badjson":
    open(os.path.join(s0, "dev16.details.jsonl"), "a").write('{"truncated\n')
log.write("step 0 {'val/accuracy': 0.5067}\n")
dt = float(os.environ.get("FAKE_DT", "0.05"))
for n in range(25, 1055, 25):
    time.sleep(dt)
    if mode == "gpu" and n == 100:
        log.write("Error: command buffer exited with error status.\n")
    if mode == "ips" and n == 100:
        h = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S.00 %z")}
        open(os.path.join(diag, "gpuEvent-WindowServer-x.ips"), "w").write(json.dumps(h) + '\n{"restart_reason_desc":"MMU interrupt"}\n')
    if mode == "exit" and n == 100:
        sys.exit(3)
    loss = "nan" if (mode == "nan" and n == 125) else "0.8065"
    log.write("epoch 1 step %d/1054 loss %s nll %s acc 0.625 step_s 3.249\n" % (n, loss, loss))
    if mode == "exit250" and n == 250:
        time.sleep(0.5)                                     # seen by a poll while alive, gone before the held stop
        sys.exit(3)
open(run + "/finished", "w").write("all 1054")
'''

FAKE_V3 = r'''
import json, sys
out = sys.argv[sys.argv.index("--out") + 1]
ok = sys.argv[1] == "ok"
json.dump({"ok": ok, "comparison": {"n": 64, "largest": 0.0 if ok else 0.1, "problems": [] if ok else ["x"]}}, open(out, "w"))
sys.exit(0 if ok else 1)
'''


@pytest.fixture
def fast(monkeypatch, tmp_path):
    monkeypatch.setattr(vrun.guarded, "GRACE", 1.0)
    monkeypatch.setattr(vrun.guarded, "POLL", 0.2)
    monkeypatch.setattr(vrun.guarded, "KILL_CHECK", 1.0)
    monkeypatch.setattr(vrun, "MARGIN", 3)
    (tmp_path / "fake.py").write_text(FAKE)
    (tmp_path / "v3.py").write_text(FAKE_V3)
    (tmp_path / "diag").mkdir()
    return tmp_path


def run_v(tmp, mode, total=40, dt="0.05", v3="ok", ref=REF, monkeypatch=None):
    os.environ.update(FAKE_REF=REF, FAKE_DIAG=str(tmp / "diag"), FAKE_DT=dt)
    run = str(tmp / "run")
    t0 = time.time()
    rc = vrun.main(["--record", str(tmp / "rec.json"), "--run-dir", run, "--start", str(t0), "--total", str(total),
                    "--v3-reserve", "6", "--report-reserve", "4", "--ref-dir", ref, "--first-dir", str(tmp / "none"), "--diag", str(tmp / "diag"),
                    "--", sys.executable, str(tmp / "fake.py"), mode, run, "--", sys.executable, str(tmp / "v3.py"), v3])
    rec = json.load(open(tmp / "rec.json"))
    return rc, rec, time.time() - t0


def test_planned_stop_after_250_then_v3(fast):
    rc, rec, took = run_v(fast, "ok")
    tr = rec["training"]
    assert rc == 0 and rec["outcome_at_end"] == "pass"
    assert tr["end"]["why"] == "planned_stop" and tr["cleanup_confirmed"] and 250 <= tr["updates"] < 1054
    assert tr["v1"]["ok"] and tr["v1"]["questions"] == 3562
    assert not (fast / "run" / "finished").exists()                                  # stopped, not run to the end
    assert rec["v3"]["exit"] == 0 and rec["v3"]["result"]["ok"] and took < 40 - 4


@pytest.mark.parametrize("mode,dt,why,want", [
    ("mismatch", "0.05", "v1_mismatch", "consistency not established (V1)"),
    ("gpu", "0.05", "gpu_error", "failed: GPU execution error"),
    ("nan", "0.05", "nonfinite", "failed: a non-finite training value"),
    ("ips", "0.2", "system_event", "execution-unverified"),
    ("slow", "2.0", "limit", "incomplete: the limit came before the planned stop"),
    ("exit", "0.05", None, "failed: the training process ended by itself (exit 3)")])
def test_every_other_end_has_no_v3(fast, mode, dt, why, want):
    rc, rec, took = run_v(fast, mode, total=30, dt=dt)
    tr = rec["training"]
    assert rc == 1 and rec["outcome_at_end"].startswith(want), rec["outcome_at_end"]
    assert (tr.get("end") or {}).get("why") == why and tr["cleanup_confirmed"]
    assert "v3" not in rec and rec["v3_not_started"]
    assert took < 30 - 4 - 6                                                          # inside the training phase's end
    if mode == "slow":
        assert tr.get("updates", 0) < 250


def test_a_changed_reference_refuses_during_the_run(fast, tmp_path):
    copy_ref(tmp_path / "ref", ("val", nudge(0, 1e-9)))
    rc, rec, _ = run_v(fast, "ok", total=30, ref=str(tmp_path / "ref"))
    assert rc == 1 and rec["training"]["end"]["why"] == "v1_refused" and rec["outcome_at_end"].startswith("refused")


def test_term_ignored_is_killed_and_confirmed(fast):
    rc, rec, _ = run_v(fast, "ignore_term")
    tr = rec["training"]
    assert tr["end"]["why"] == "planned_stop" and "kill" in tr["end"] and tr["cleanup_confirmed"] and rc == 0


def test_unconfirmed_cleanup_starts_no_v3(fast, monkeypatch):
    real = vrun.guarded.end_group

    def stuck(child, pgid, why, log, deadline):
        t = real(child, pgid, why, log, deadline)
        return dict(t, left=[99999], confirmed=False)                                # a member left behind, stood in
    monkeypatch.setattr(vrun.guarded, "end_group", stuck)
    rc, rec, _ = run_v(fast, "ok")
    assert rc == 71 and rec["training"]["cleanup_confirmed"] is False and "v3" not in rec
    assert rec["outcome_at_end"].startswith("failed: cleanup not confirmed")
    assert any("cleanup not confirmed" in w for w in rec["v3_not_started"])


def test_v3_inconsistent(fast):
    rc, rec, _ = run_v(fast, "ok", v3="bad")
    assert rc == 1 and rec["outcome_at_end"].startswith("failed: V3")


STUB = r'''#!/bin/bash
echo "uv $*" >> "${MARK_CALLS}"
case "$*" in
  *"preflight_v.py"*) [ "${FAKE_PREFLIGHT}" = ok ] || exit 12; echo '{"ok": true}' > runs/exp76-device-v-preflight.json; echo '{}' > runs/exp76-device-v.START.json; exit 0;;
  *"vrun.py"*) shift 2; exec python3 "$@";;
  *"devision-pipeline"*) exec python3 "${FAKE_PY}" ok runs/exp76-device-v;;
  *"v3.py"*) exec python3 "${FAKE_V3}" ok "$@";;
  *"report_v.py"*) shift 2; exec python3 "$@";;
esac
exit 9
'''


def repo(tmp_path):
    root = tmp_path / "repo"
    for rel in ("scripts/device_v/run_v.sh", "scripts/device_v/preflight_v.py", "scripts/device_v/vcheck.py", "scripts/device_v/vrun.py",
                "scripts/device_v/v3.py", "scripts/device_v/report_v.py", "scripts/panel_decision/guarded.py", "configs/exp76-device-v.yaml",
                "scripts/exp35/guard.sh", "scripts/exp38/steps.sh"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join(ROOT, rel), root / rel)
    copy_ref(root / vcheck.REF_DIR)
    (tmp_path / "fake.py").write_text(FAKE)
    (tmp_path / "v3.py").write_text(FAKE_V3)
    b = tmp_path / "bin"
    b.mkdir()
    for name, body in {"uv": STUB, "pgrep": "#!/bin/bash\nexit 1\n", "ps": "#!/bin/bash\nexit 0\n"}.items():
        (b / name).write_text(body)
        (b / name).chmod((b / name).stat().st_mode | stat.S_IEXEC)
    env = {**os.environ, "PATH": "%s:%s" % (b, os.environ["PATH"]), "MARK_CALLS": str(tmp_path / "calls"), "FAKE_PY": str(tmp_path / "fake.py"),
           "FAKE_V3": str(tmp_path / "v3.py"), "FAKE_REF": str(root / vcheck.REF_DIR), "FAKE_DT": "0.05",
           "V_LIMIT": "60", "V_V3_RESERVE": "8", "V_RESERVE": "6", "GUARD_GRACE": "1", "GUARD_POLL": "0.2", "GUARD_KILL_CHECK": "1"}
    return root, env


def test_run_v_end_to_end(tmp_path):
    root, env = repo(tmp_path)
    env["FAKE_PREFLIGHT"] = "ok"
    t0 = time.time()
    p = subprocess.run(["bash", str(root / "scripts/device_v/run_v.sh")], env=env, capture_output=True, text=True, timeout=120, cwd=str(root))
    assert p.returncode == 0, p.stdout + p.stderr
    assert time.time() - t0 < 60
    rep = json.load(open(root / "runs/exp76-device-v/report.json"))
    assert rep["outcome"] == "pass" and rep["duration_s"] < 60 and "report exit 0" in p.stdout
    rec = rep["record"]
    assert rec["training"]["end"]["why"] == "planned_stop" and rec["reserves"] == {"v3": 8.0, "report": 6.0, "guard_margin": 28}
    p2 = subprocess.run(["bash", str(root / "scripts/device_v/run_v.sh")], env=env, capture_output=True, text=True, cwd=str(root))
    assert p2.returncode == 1 and "already started once" in p2.stdout
    s = open(os.path.join(HERE, "run_v.sh")).read()
    assert "V_LIMIT:-2700" in s and "V_V3_RESERVE:-300" in s and "V_RESERVE:-120" in s
    assert not any(x in s for x in ("cc-connect", "crontab", "launchctl", "exp76-panel-decision"))      # nothing chained


def test_run_v_refused_preflight_runs_nothing(tmp_path):
    root, env = repo(tmp_path)
    env["FAKE_PREFLIGHT"] = "no"
    p = subprocess.run(["bash", str(root / "scripts/device_v/run_v.sh")], env=env, capture_output=True, text=True, timeout=60, cwd=str(root))
    assert p.returncode == 12 and "preflight refused" in p.stdout
    calls = open(tmp_path / "calls").read()
    assert "devision-pipeline" not in calls and not (root / "runs/exp76-device-v.started").exists()


@pytest.mark.parametrize("ref_p,new_p", [({"a": 0.5, "b": 0.3, "c": 0.2}, {"a": 0.5, "b": float("nan"), "c": 0.2}),   # second option
                                         ({"a": 0.5, "b": 0.3, "c": 0.2}, {"a": 0.5, "b": 0.3, "c": float("nan")}),   # last option
                                         ({"a": 0.5, "b": 0.3, "c": float("inf")}, {"a": 0.5, "b": 0.3, "c": 0.2}),   # Inf, reference side
                                         ({"a": 0.5, "b": 0.3, "c": 0.2}, {"a": float("-inf"), "b": 0.3, "c": 0.2})])
def test_non_finite_option_is_rejected(ref_p, new_p):
    """R1-01: checked per option before any max."""
    row = {"sample_id": "s", "qid": "q", "prediction": "a"}
    c = vcheck.compare_rows([dict(row, probabilities=ref_p)], [dict(row, probabilities=new_p)])
    assert c["problems"] and "not finite" in c["problems"][0] and c["largest"] == float("inf")
    fin = vcheck.compare_rows([dict(row, probabilities={"a": 0.5, "b": 0.5})], [dict(row, probabilities={"a": 0.5 + 5e-7, "b": 0.5})])
    assert fin["problems"] == []
    out = vcheck.compare_rows([dict(row, probabilities={"a": 0.5, "b": 0.5})], [dict(row, probabilities={"a": 0.5 + 2e-6, "b": 0.5})])
    assert out["problems"]


T0 = time.time() - 100                                     # a real-format window: started 100 s ago, 2,700 s long


def good_record():
    return {"start_epoch": T0, "deadline_epoch": T0 + 2700, "start": vcheck.time_str(T0), "deadline": vcheck.time_str(T0 + 2700),
            "training": {"end": {"why": "planned_stop", "initiated": True, "confirmed": True}, "v1": {"ok": True},
                         "cleanup_confirmed": True, "updates": 250, "child_returncode": -15},
            "v3": {"exit": 0, "result": {"ok": True}, "cleanup_confirmed": True}}


@pytest.mark.parametrize("drop", ["training.v1", "training.cleanup_confirmed", "v3.cleanup_confirmed", "start_epoch", "deadline_epoch",
                                  "training.end.initiated"])
def test_missing_evidence_never_passes(drop):
    """R1-02: each piece removed from a passing record."""
    assert vcheck.outcome(good_record(), []) == "pass"
    rec = good_record()
    *path, last = drop.split(".")
    d = rec
    for k in path:
        d = d[k]
    del d[last]
    assert vcheck.outcome(rec, []) != "pass"


@pytest.fixture
def report_dir(tmp_path, monkeypatch):
    import report_v
    monkeypatch.chdir(tmp_path)
    (tmp_path / "runs").mkdir()
    monkeypatch.setattr(report_v, "DIAG", str(tmp_path / "diag"))
    return report_v, tmp_path


def test_report_requires_start_and_a_readable_record(report_dir):
    m, tmp = report_dir
    json.dump(good_record(), open(tmp / m.RECORD, "w"))
    assert m.main() == 1                                                             # no started marker: not pass
    rep = json.load(open(tmp / m.OUT))
    assert rep["outcome"].startswith("incomplete") and "window cannot be checked" in rep["outcome"]
    open(tmp / m.STARTED, "w").write("start %r\ndeadline %r\nstarted x\n" % (T0, T0 + 2700))
    assert m.main() == 0 and json.load(open(tmp / m.OUT))["outcome"] == "pass"
    open(tmp / m.RECORD, "w").write('{"training": {"end"')                           # truncated record
    assert m.main() == 1
    rep = json.load(open(tmp / m.OUT))
    assert rep["outcome"].startswith("incomplete") and "unreadable" in rep["outcome"]
    open(tmp / m.STARTED, "w").write("start x\n")                                   # truncated / bad marker
    json.dump(dict(good_record(), training=dict(good_record()["training"], gpu_error={"line": "x"})), open(tmp / m.RECORD, "w"))
    assert m.main() == 1
    assert json.load(open(tmp / m.OUT))["outcome"].startswith("failed: GPU execution error; also incomplete")


def test_supervisor_exception_ends_its_group(fast):
    """R1-03: a corrupt step 0 file makes the comparison raise after the group started; the TERM-ignoring group is
    KILLed within the deadline, a bystander survives, no V3, the record is written."""
    by = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        rc, rec, took = run_v(fast, "badjson", total=30)
        tr = rec["training"]
        assert rc == 1 and tr["exception"].startswith("JSONDecodeError") and tr["end"]["why"] == "exception"
        assert "kill" in tr["end"] and tr["cleanup_confirmed"] is True and not vrun.guarded.members(tr["pgid"])
        assert "v3" not in rec and rec["outcome_at_end"].startswith("failed: the V supervisor raised")
        assert took < 30 - 4 - 6 and by.poll() is None
    finally:
        by.kill()


def test_supervisor_exception_with_unconfirmed_cleanup_returns_71(fast, monkeypatch):
    real = vrun.guarded.end_group
    monkeypatch.setattr(vrun.guarded, "end_group", lambda *a: dict(real(*a), left=[99999], confirmed=False))
    rc, rec, took = run_v(fast, "badjson", total=30)
    assert rc == 71 and rec["training"]["cleanup_confirmed"] is False and "v3" not in rec and took < 30 - 4


def test_own_failure_after_250_is_not_a_planned_stop(fast, monkeypatch):
    """R1-04: the 250th-update line is written, then the process exits 3 before the supervisor's stop (a fixed
    synchronisation: the poll's planned_stop answer is held 1 s, the fake exits at once)."""
    real = vrun.training_poll

    def held(*a):
        poll = real(*a)

        def p():
            why = poll()
            if why == "planned_stop":
                time.sleep(1.0)
            return why
        return p
    monkeypatch.setattr(vrun, "training_poll", held)
    rc, rec, _ = run_v(fast, "exit250", total=30)
    tr = rec["training"]
    assert rc == 1 and "end" not in tr and tr["ended_before_stop"] == {"wanted": "planned_stop", "returncode": 3}
    assert tr["child_returncode"] == 3 and tr["updates"] == 250 and "v3" not in rec
    assert rec["outcome_at_end"] == "failed: the training process ended by itself (exit 3) before the planned stop"


@pytest.mark.parametrize("start,deadline,rec_window,want", [
    (time.time() + 600, time.time() + 3300, None, "starts in the future"),
    (float("nan"), T0 + 2700, None, "not two finite times"),
    (float("inf"), T0 + 2700, None, "not two finite times"),
    (float("-inf"), T0 + 2700, None, "not two finite times"),
    (T0, T0 - 1, None, "ends before it starts"),
    (T0, T0 + 2700, (T0 - 50, T0 + 2650), "different windows"),
    (T0, T0 + 2700, (float("nan"), T0 + 2700), "different windows")])
def test_report_rejects_bad_windows(report_dir, start, deadline, rec_window, want):
    """R2-01: started / record windows parsed, finite, ordered and the same; none of these passes, each is reported
    with a null duration and no exception."""
    m, tmp = report_dir
    rec = good_record()
    if rec_window is None:                                                           # the record names the same (bad) window
        rec["start_epoch"], rec["deadline_epoch"] = start, deadline
    else:
        rec["start_epoch"], rec["deadline_epoch"] = rec_window
    json.dump(rec, open(tmp / m.RECORD, "w"))
    open(tmp / m.STARTED, "w").write("start %r\ndeadline %r\n" % (start, deadline))
    assert m.main() == 1
    rep = json.load(open(tmp / m.OUT))
    assert rep["outcome"].startswith("incomplete") and want in rep["outcome"], rep["outcome"]
    assert rep["duration_s"] is None and rep["system_events_in_window"] == []


def test_bad_window_keeps_a_recorded_failure(report_dir):
    m, tmp = report_dir
    json.dump(dict(good_record(), training=dict(good_record()["training"], gpu_error={"line": "x"})), open(tmp / m.RECORD, "w"))
    open(tmp / m.STARTED, "w").write("start nan\ndeadline 1\n")
    assert m.main() == 1
    rep = json.load(open(tmp / m.OUT))
    assert rep["outcome"].startswith("failed: GPU execution error; also incomplete") and rep["duration_s"] is None


def test_outcome_window_rules():
    for start, deadline in ((time.time() + 60, time.time() + 2760), (float("nan"), 1.0), (5.0, 4.0), ("s", "d")):
        assert vcheck.outcome(dict(good_record(), start_epoch=start, deadline_epoch=deadline), []) == "incomplete: no valid V window recorded"
