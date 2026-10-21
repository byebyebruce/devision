"""Serial, finite-budget controller for the authorised post-r15 experiment.

Run detached with the repository Python; the hourly cc-connect task observes this controller.
No publishing, no hyperparameter search. A -> B -> report; only a passing B continues to
seed 2 -> replication check -> final evaluation. All stops/errors are explicit in state.json.
"""
import datetime
import fcntl
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from build import sha
from report import COUNTS, GUARDS, RULES, clear_losses, complete, final_report, pair, same_start

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/post-r15-order"
STATE = RUN / "state.json"
ACTIVE = None


def save(state):
    state.update(updated_at=datetime.datetime.now().astimezone().isoformat(), controller_pid=os.getpid())
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(STATE)
    print(state["updated_at"], state.get("status"), state.get("arm", ""), state.get("message", ""), flush=True)


def check_frozen():
    frozen = json.loads((RUN / "frozen.json").read_text())
    for path, expected in frozen["file_sha256"].items():
        if sha(path) != expected:
            raise ValueError("frozen input changed: %s" % path)
    if sha(Path(frozen["checkpoint"]) / "model.safetensors") != frozen["checkpoint_sha256"]:
        raise ValueError("initial checkpoint changed")
    return frozen


def stop_process(proc):
    if proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=20)


def interrupted(sig, frame):
    if ACTIVE is not None:
        stop_process(ACTIVE)
    save(dict(status="interrupted", message="controller received signal %d" % sig, finished=True))
    raise SystemExit(128 + sig)


def run_arm(arm):
    global ACTIVE
    check_frozen()
    stage = ROOT / ("runs/post-r15-order-%s/stage2" % arm)
    if (stage / "train_report.json").exists():
        if not complete(stage / "monitor/step2000"):
            raise ValueError("completed arm lacks final monitor: %s" % arm)
        return True
    if (stage / "START_MISMATCH").exists() or (stage / "STOPPED_BY_RULE").exists():
        save(dict(status="blocked_by_existing_stop", arm=arm, finished=True))
        return False
    config = "configs/post-r15-order-%s.yaml" % arm
    for attempt in (1, 2):
        command = [sys.executable, "-u", "-c",
                   "from devision.train.pipeline import pipeline_main; pipeline_main()", config]
        with (RUN / (arm + ".pipeline.log")).open("a") as log:
            ACTIVE = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, PYTHONUNBUFFERED="1"),
                                      stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            proc = ACTIVE
            save(dict(status="training", arm=arm, attempt=attempt, pipeline_pid=proc.pid,
                      log="runs/post-r15-order-%s/logs/stage2.log" % arm, finished=False))
            checked = False
            while proc.poll() is None:
                d = stage / "monitor/step1000"
                if not checked and complete(d):
                    result = pair(stage / "monitor/step0", d)
                    losses = clear_losses(result, early=True)
                    (RUN / (arm + "-step1000.json")).write_text(json.dumps({"comparison": result, "stop_losses": losses}, indent=2))
                    checked = True
                    if losses:
                        (stage / "STOPPED_BY_RULE").write_text(json.dumps(losses))
                        stop_process(proc)
                        ACTIVE = None
                        save(dict(status="stopped_by_rule", arm=arm, losses=losses, finished=True))
                        return False
                time.sleep(20)
            rc = proc.wait()
        ACTIVE = None
        if (stage / "START_MISMATCH").exists():
            save(dict(status="start_mismatch", arm=arm, message=(stage / "START_MISMATCH").read_text(), finished=True))
            return False
        if rc == 0 and (stage / "train_report.json").exists() and complete(stage / "monitor/step2000"):
            save(dict(status="arm_complete", arm=arm, finished=False))
            return True
        logfile = ROOT / ("runs/post-r15-order-%s/logs/stage2.log" % arm)
        tail = logfile.read_text()[-12000:] if logfile.exists() else ""
        transient = any(s in tail.lower() for s in ("mps backend", "mps device", "mpsgraph", "metal",
                                                    "command buffer", "device is returning invalid results"))
        if attempt == 1 and transient and (stage / "resume.pt").exists():
            save(dict(status="retrying", arm=arm, attempt=attempt, exit_code=rc, finished=False,
                      message="One retry from the saved state after a device-related failure."))
            time.sleep(30)
            continue
        save(dict(status="failed", arm=arm, exit_code=rc, finished=True,
                  message="No blind retry: inspect the pipeline and stage log."))
        return False
    return False


def replication():
    a = ROOT / "runs/post-r15-order-a/stage2/monitor"
    b = ROOT / "runs/post-r15-order-b-seed2/stage2/monitor"
    start = same_start(str(a / "step0"), str(b / "step0"), 1e-5)
    result = pair(b / "step0", b / "step2000")
    f = result["count_wide"]["flip"]
    counts = [result[p][m] for p, m in (("count_wide", "accuracy"), ("count_wide_reverse", "accuracy"),
                                       ("count_wide", "joint"), ("count_narrow", "accuracy"))]
    losses = clear_losses(result)
    warnings = [p for p, r in result.items() if r["nll"]["difference"] > .02 and r["nll"]["ci95"][0] > 0]
    passed = f["ci95"][1] < 0 and all(r["ci95"][0] > -.02 for r in counts) and not losses and not warnings
    report = dict(same_start=start, B_seed2_minus_r15=result, replication_pass=passed,
                  clear_losses=losses, nll_warnings=warnings,
                  scope="No A seed2: replicates the B recipe, not the B-minus-A causal effect.")
    (RUN / "replication.json").write_text(json.dumps(report, indent=2) + "\n")
    return passed


def final_evaluation():
    global ACTIVE
    check_frozen()
    save(dict(status="final_evaluation", arm="b", finished=False,
              message="Fixed seed1 candidate; seed2 is replication, never best-seed selection."))
    with (RUN / "final-evaluation.log").open("a") as log:
        ACTIVE = subprocess.Popen([sys.executable, "-u", "-c",
                                   "from devision.train.pipeline import pipeline_main; pipeline_main()",
                                   "configs/post-r15-order-final.yaml"], cwd=ROOT,
                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                  env=dict(os.environ, PYTHONUNBUFFERED="1"))
        rc = ACTIVE.wait()
    ACTIVE = None
    if rc:
        save(dict(status="evaluation_failed", exit_code=rc, finished=True))
        return
    spec = importlib.util.spec_from_file_location("r14_accept", ROOT / "scripts/r14/accept.py")
    accept = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(accept)
    from devision.train.compare import load_records, paired_metric
    results = {}
    for name in tuple(dict.fromkeys(accept.PROTECTED + ("bench_countbench", "bench_whatsup"))):
        a = load_records(str(ROOT / "runs/r15-vision/eval" / (name + ".details.jsonl")))
        b = load_records(str(ROOT / "runs/post-r15-order-b/eval" / (name + ".details.jsonl")))
        # Preserve canonical picture clusters, including mirror families, for final compatibility checks.
        from report import normalise
        a, b = normalise(a), normalise(b)
        results[name] = {m: paired_metric(a, b, m, resamples=10000, seed=1518) for m in ("accuracy", "nll")}
    not_held = [p for p, r in results.items() if r["accuracy"]["ci95"][0] <= -.02]
    warnings = [p for p, r in results.items() if r["nll"]["difference"] > .02 and r["nll"]["ci95"][0] > 0]
    (RUN / "final_protection.json").write_text(json.dumps(dict(sets=results, not_held=not_held,
        nll_warnings=warnings, protection_pass=not not_held and not warnings, published=False), indent=2) + "\n")
    save(dict(status="completed", finished=True, published=False, not_held=not_held, nll_warnings=warnings))


def main():
    os.chdir(ROOT)
    RUN.mkdir(exist_ok=True, parents=True)
    with (RUN / "controller.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("this experiment already has a controller")
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, interrupted)
        if STATE.exists() and json.loads(STATE.read_text()).get("finished"):
            raise SystemExit("experiment already ended; inspect state.json instead of restarting")
        try:
            for arm in ("a", "b"):
                if not run_arm(arm):
                    return
            save(dict(status="screening", finished=False))
            report = final_report(ROOT / "runs/post-r15-order-a/stage2/monitor",
                                  ROOT / "runs/post-r15-order-b/stage2/monitor", RUN / "report.json")
            if not report["screen_pass"]:
                save(dict(status="screen_failed", finished=True, message="Keep v0.3; no extra training or evaluation."))
                return
            if not run_arm("b-seed2"):
                return
            if not replication():
                save(dict(status="replication_failed", finished=True))
                return
            final_evaluation()
        except Exception as e:
            if ACTIVE is not None:
                stop_process(ACTIVE)
            save(dict(status="failed", message=str(e), finished=True))
            raise


if __name__ == "__main__":
    main()
