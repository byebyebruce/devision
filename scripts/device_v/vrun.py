"""EXP76 device V orchestration (DEVICE-V-IMPL-PREP-01; device plan section 2). One training process (V1 + V2) and,
only when allowed, one V3 process, each in its own process group under the accepted guard's pieces
(scripts/panel_decision/guarded.py: Watch for GPU lines, members / end_group for a bounded TERM -> KILL with confirmed
cleanup); guarded.py itself is not changed. Every poll (about 2 s) of the training process:
  - a GPU error line in its output / logs -> end, gpu_error;
  - a system gpuEvent (BIF0 / MMU) timed inside the window -> end, system_event;
  - all eight step 0 files written and the "step 0 {" line logged -> V1: the fixed comparison; not consistent -> end, v1_mismatch;
  - a non-finite training value in a window line -> end, nonfinite;
  - V1 passed and a window line of >= 250 updates logged -> the planned diagnostic stop, planned_stop;
  - the training phase's stop time -> end, limit; the process ending by itself -> its exit code.
V3 runs only after planned_stop with V1 passed, cleanup confirmed, no event, and V3's reserve plus the report's left.
Writes <record>; exit 0 only for a V record without failure (the report decides "pass"), 71 when any cleanup is not
confirmed (no further model may start), 1 otherwise. Nothing is retried or chained.

    python scripts/device_v/vrun.py --record R --run-dir runs/exp76-device-v --start T0 --total 2700 -- train cmd ... -- v3 cmd ...
"""
import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vcheck  # noqa: E402

_spec = importlib.util.spec_from_file_location("guarded", os.path.join(HERE, "..", "panel_decision", "guarded.py"))
assert _spec and _spec.loader
guarded = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(guarded)

PLANNED = 250
WINDOW = re.compile(r"^epoch \d+ step (\d+)/(\d+) loss (\S+) nll (\S+)")
MARGIN = 28                     # the guard's own stop margin: GRACE 15 + KILL_CHECK 5 + POLL 2 + 1, + 5
REPORT_RESERVE = 120
V3_RESERVE = 300
DIAG = "/Library/Logs/DiagnosticReports"


def log(msg):
    print("== %s V: %s" % (time.strftime("%F %T"), msg), flush=True)


def supervise(cmd, out_path, watch_dirs, deadline, on_poll, rec):
    """Run `cmd` in its own group until it ends, `on_poll()` names a stop reason, a GPU line appears, or the stop
    time; every wait bounded by `deadline` (the guard's rule). A stop counts as initiated by this supervisor only when
    the command was still running when it was sent (R1-04): one that already ended keeps its own exit code. Any
    exception after the group started ends that group the same bounded way and is recorded (R1-03). Fills `rec`."""
    stop_at = deadline - guarded.GRACE - guarded.KILL_CHECK - guarded.POLL - guarded.STOP_MARGIN
    with open(out_path, "ab") as out:
        child = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    pgid = child.pid
    rec.update(pgid=pgid, cmd=cmd, started=time.strftime("%F %T"), deadline=deadline)

    def stop(why):
        if child.poll() is not None:                       # ended by itself before the stop: not a stop of ours
            rec["ended_before_stop"] = {"wanted": why, "returncode": child.returncode}
            log("the command had already ended (%s) before the %s stop" % (child.returncode, why))
            return
        t = guarded.end_group(child, pgid, why, log, deadline - guarded.STOP_MARGIN)
        t["initiated"] = not t.get("gone_before_term")
        rec["end"] = t

    try:
        log("started pid / pgid %d: %s (stop by %s)" % (pgid, " ".join(cmd), vcheck.time_str(stop_at)))
        watch = guarded.Watch([out_path], watch_dirs)
        while True:
            hit = watch.scan()
            if hit:
                rec["gpu_error"] = dict(hit, seen=time.strftime("%F %T"))
                log("GPU error in %s: %s" % (hit["file"], hit["line"]))
                stop("gpu_error")
                break
            why = on_poll()
            if why:
                log("stop: %s" % why)
                stop(why)
                break
            if time.time() >= stop_at:
                log("limit reached")
                stop("limit")
                break
            try:
                child.wait(timeout=max(0.0, min(guarded.POLL, stop_at - time.time())))
                break
            except subprocess.TimeoutExpired:
                pass
        if "gpu_error" not in rec:
            hit = watch.scan()
            if hit:
                rec["gpu_error"] = dict(hit, seen=time.strftime("%F %T"), after_stop=True)
                log("GPU error found after the stop in %s: %s" % (hit["file"], hit["line"]))
    except Exception as e:                                 # the supervisor's own failure: end the group it owns
        rec["exception"] = "%s: %s" % (type(e).__name__, e)
        log("supervisor exception (%s): ending the model group" % rec["exception"])
        rec.pop("end", None)
        try:
            rec["end"] = dict(guarded.end_group(child, pgid, "exception", log, deadline - guarded.STOP_MARGIN), initiated=True)
        except Exception as e2:
            rec["end"] = {"why": "exception", "confirmed": False, "error": "%s: %s" % (type(e2).__name__, e2)}
    if "end" not in rec and guarded.members(pgid):
        rec["leftovers"] = guarded.end_group(child, pgid, "leftovers", log, deadline - guarded.STOP_MARGIN)
    stop_rec = rec.get("end") or rec.get("leftovers")
    rec["cleanup_confirmed"] = bool(stop_rec["confirmed"]) if stop_rec else (child.poll() is not None and not guarded.members(pgid))
    rec.update(child_returncode=child.returncode, ended=time.strftime("%F %T"))
    return rec


def training_poll(rec, run_dir, step0_dir, ref_dir, first_dir, diag, t0):
    log_path = os.path.join(run_dir, "logs", "d2.log")
    seen = {"pos": 0, "step0": False}

    def poll():
        if os.path.exists(log_path):
            with open(log_path, "rb") as fh:
                fh.seek(seen["pos"])
                data = fh.read()
            cut = data.rfind(b"\n") + 1
            seen["pos"] += cut
            for line in data[:cut].decode("utf-8", "replace").splitlines():
                if line.startswith("step 0 {"):
                    seen["step0"] = True
                m = WINDOW.match(line)
                if m:
                    rec["updates"] = int(m.group(1))
                    rec["windows"] = rec.get("windows", []) + [{"updates": int(m.group(1)), "loss": m.group(3), "seen": time.strftime("%F %T")}]
                    if any(x.lower() in ("nan", "inf", "-inf") for x in (m.group(3), m.group(4))):
                        return "nonfinite"
        ev = vcheck.events_in_window(diag, t0, time.time() + 1)
        if ev:
            rec["system_events"] = ev
            return "system_event"
        if "v1" not in rec and seen["step0"] and all(os.path.exists(os.path.join(step0_dir, n + ".details.jsonl")) for n in vcheck.REF):
            rec["v1"] = vcheck.step0_consistency(ref_dir, step0_dir, first_dir=first_dir)
            rec["v1"]["compared_at"] = time.strftime("%F %T")
            rec["v1"]["updates_when_compared"] = rec.get("updates", 0)
            log("V1 %s (%d problems, largest %.3g)" % ("consistent" if rec["v1"]["ok"] else "NOT consistent", len(rec["v1"]["problems"]),
                                                      max([s["largest"] for s in rec["v1"]["sets"].values()] or [0])))
            if not rec["v1"]["ok"]:
                return "v1_refused" if rec["v1"].get("refused") else "v1_mismatch"
        if rec.get("v1", {}).get("ok") and rec.get("updates", 0) >= PLANNED:
            return "planned_stop"
        return None
    return poll


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--start", type=float, required=True, help="epoch seconds the V window began (the runner's start)")
    ap.add_argument("--total", type=float, default=2700)
    ap.add_argument("--v3-reserve", type=float, default=V3_RESERVE)
    ap.add_argument("--report-reserve", type=float, default=REPORT_RESERVE)
    ap.add_argument("--ref-dir", default=vcheck.REF_DIR)
    ap.add_argument("--first-dir", default=vcheck.FIRST_DIR)
    ap.add_argument("--diag", default=DIAG)
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    rest = a.rest[1:] if a.rest[:1] == ["--"] else a.rest
    if "--" not in rest:
        ap.error("two commands are needed: -- training cmd -- v3 cmd")
    i = rest.index("--")
    train_cmd, v3_cmd = rest[:i], rest[i + 1:]
    deadline = a.start + a.total
    train_end = deadline - a.report_reserve - a.v3_reserve - MARGIN
    rec = {"start_epoch": a.start, "deadline_epoch": deadline, "start": vcheck.time_str(a.start), "deadline": vcheck.time_str(deadline), "training_end": vcheck.time_str(train_end),
           "reserves": {"v3": a.v3_reserve, "report": a.report_reserve, "guard_margin": MARGIN}, "planned_updates": PLANNED}

    def save():
        tmp = a.record + ".tmp"
        json.dump(rec, open(tmp, "w"), indent=1)
        os.replace(tmp, a.record)

    tr: dict = {}
    rec["training"] = tr
    try:
        rec["events_at_start"] = vcheck.events_in_window(a.diag, a.start, time.time())
        if rec["events_at_start"]:
            tr["system_events"] = rec["events_at_start"]
            log("a system GPU event inside the window before training: not started")
        else:
            supervise(train_cmd, a.run_dir + "-pipeline.out", [os.path.join(a.run_dir, "logs")], train_end,
                      training_poll(tr, a.run_dir, os.path.join(a.run_dir, "d2", "monitor", "step0"), a.ref_dir, a.first_dir, a.diag, a.start), tr)
            ev = vcheck.events_in_window(a.diag, a.start, time.time())
            if ev:
                tr["system_events"] = ev
        save()
        why = vcheck.v3_allowed(tr, time.time(), deadline, a.v3_reserve, a.report_reserve)
        if why:
            rec["v3_not_started"] = why
            log("V3 not started: %s" % "; ".join(why))
        else:
            v3: dict = {}
            rec["v3"] = v3
            res_path = a.run_dir + "-v3.json"

            def v3_poll():
                ev = vcheck.events_in_window(a.diag, a.start, time.time() + 1)
                if ev:
                    v3["system_events"] = ev
                    return "system_event"
                return None
            supervise(v3_cmd + ["--out", res_path], a.run_dir + "-v3.out", [], deadline - a.report_reserve - 5, v3_poll, v3)
            v3["exit"] = v3["child_returncode"]
            v3["result"] = json.load(open(res_path)) if os.path.exists(res_path) else None
            ev = vcheck.events_in_window(a.diag, a.start, time.time())
            if ev:
                v3["system_events"] = ev
    except Exception as e:                                 # no model group is left running here: supervise ended it
        rec["exception"] = "%s: %s" % (type(e).__name__, e)
        log("V orchestration exception: %s" % rec["exception"])
    rec["ended"] = time.strftime("%F %T")
    rec["outcome_at_end"] = vcheck.outcome(rec, [])
    save()
    log("outcome at the end of the model steps: %s" % rec["outcome_at_end"])
    if any(p.get("cleanup_confirmed") is False for p in (tr, rec.get("v3") or {})):
        log("cleanup not confirmed: no further model is started")
        return 71
    return 0 if rec["outcome_at_end"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
