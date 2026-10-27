"""Bounded, single-GPU exp80 orchestration. Logs survive the interactive session."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import sysconfig
import time

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "runs/exp80"
CURRENT = None


def status(**kwargs):
    tmp = WORK / "STATUS.tmp"
    tmp.write_text(json.dumps({"updated": time.strftime("%Y-%m-%d %H:%M:%S %z"), **kwargs}, indent=2) + "\n")
    tmp.replace(WORK / "STATUS.json")


def stop_child():
    if CURRENT is not None:
        try:
            os.killpg(CURRENT.pid, signal.SIGTERM)
            CURRENT.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(CURRENT.pid, signal.SIGKILL)
            CURRENT.wait(timeout=10)
        except ProcessLookupError:
            pass


def interrupted(signum, frame):
    stop_child()
    status(state="interrupted", signal=signum)
    raise SystemExit(128 + signum)


def run(args, label, timeout):
    global CURRENT
    if timeout < 1:
        raise TimeoutError(label)
    print(time.strftime("%F %T"), label, "budget_s", round(timeout), flush=True)
    status(state="running", command=label, budget_seconds=timeout)
    with (WORK / (label + ".log")).open("w") as log:
        CURRENT = subprocess.Popen(["uv", "run", "--no-sync", *args], cwd=ROOT, env=os.environ,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = CURRENT.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            stop_child()
            status(state="incomplete", command=label, reason="wallclock_limit")
            raise TimeoutError(label)
        finally:
            CURRENT = None
    if rc:
        status(state="execution_failed", command=label, exit_code=rc)
        raise RuntimeError(f"{label} exited {rc}")


def evaluate(checkpoint, dest, budget):
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("chart", "fit", "protect"):
        dataset = "chart_dev" if name == "chart" else name
        for control in (("real", "mismatched", "reversed") if name == "chart" else ("real",)):
            args = ["devision-eval", "--checkpoint", checkpoint,
                    "--data", f"data/v05/exp80v2/{dataset}.jsonl", "--data-root", "data",
                    "--device", "cuda", "--no-temperature", "--role", "monitoring",
                    "--out", str(dest / f"{name}.{control}.json")]
            if control != "real":
                args += ["--control", control]
            run(args, dest.parent.name + "_" + name + "_" + control, budget - time.monotonic())


def main():
    os.chdir(ROOT)
    WORK.mkdir(parents=True, exist_ok=True)
    with (WORK / "STARTED").open("x") as f:
        f.write(time.strftime("%F %T %z") + "\n")
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    os.environ.update(UV_NO_SYNC="1", HF_HUB_OFFLINE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                      TOKENIZERS_PARALLELISM="false")
    include = sysconfig.get_path("include")
    if not (Path(include) / "Python.h").exists():
        managed = sorted((Path.home() / ".local/share/uv/python").glob(
            f"cpython-{sys.version_info.major}.{sys.version_info.minor}.*-*/include/python{sys.version_info.major}.{sys.version_info.minor}/Python.h"))
        if not managed:
            raise RuntimeError("Python.h unavailable for Triton")
        include = str(managed[-1].parent)
    os.environ["CPATH"] = include + (":" + os.environ["CPATH"] if os.environ.get("CPATH") else "")
    def file_sha(path):
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    frozen = [ROOT / "runs/r21-soft/train/model.safetensors", ROOT / "runs/r21-soft/train/config.json",
              *sorted((ROOT / "scripts/v05").glob("*.py")),
              ROOT / "configs/exp80-chart-r.yaml", ROOT / "configs/exp80-chart-c.yaml",
              ROOT / "src/devision/train/rlcd.py", ROOT / "src/devision/infer.py"]
    identity = {str(p.relative_to(ROOT)): file_sha(p) for p in frozen}
    (WORK / "IDENTITY.json").write_text(json.dumps({"sha256": identity, "python_include": include,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], text=True)}, indent=2) + "\n")
    manifest = json.loads((ROOT / "data/v05/exp80v2/MANIFEST.json").read_text())
    audit = json.loads((ROOT / "runs/v05-prep/data_audit_v2.json").read_text())
    if not audit["pass"] or file_sha(ROOT / "data/v05/exp80v2/MANIFEST.json") != audit["manifest_sha256"]:
        raise RuntimeError("Data isolation audit missing or stale")
    for name, meta in manifest["files"].items():
        actual = hashlib.sha256((ROOT / f"data/v05/exp80v2/{name}.jsonl").read_bytes()).hexdigest()
        if actual != meta["sha256"]:
            raise RuntimeError("Data changed: " + name)
    # Duplicate output directories would silently resume a different diagnostic.
    for arm in ("r", "c"):
        if (ROOT / f"runs/exp80-chart-{arm}").exists():
            raise RuntimeError("Arm already exists: " + arm)
        run(["devision-pipeline", f"configs/exp80-chart-{arm}.yaml", "--dry-run"], "preflight_" + arm, 120)
    evaluate("runs/r21-soft/train", WORK / "baseline/raw", time.monotonic() + 1800)
    for arm in ("r", "c"):
        if any(file_sha(ROOT / p) != h for p, h in identity.items()):
            raise RuntimeError("Frozen implementation or checkpoint changed")
        deadline = time.monotonic() + 2700
        run(["devision-pipeline", f"configs/exp80-chart-{arm}.yaml"], "pipeline_" + arm, deadline - time.monotonic())
        report = json.loads((ROOT / f"runs/exp80-chart-{arm}/train/train_report.json").read_text())
        if len(report["losses"]) != 2000 or report["train_items"] != 16000:
            raise RuntimeError("Unexpected actual training budget")
        evaluate(f"runs/exp80-chart-{arm}/train", WORK / f"{arm}/raw", deadline)
    run(["python", "scripts/v05/report_exp80.py"], "report", 180)
    status(state="completed", report="runs/exp80/report.json")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(type(exc).__name__, str(exc), flush=True)
        # Preserve the more specific timeout/failure state, if already recorded.
        p = WORK / "STATUS.json"
        if not p.exists() or json.loads(p.read_text()).get("state") == "running":
            status(state="execution_failed", reason=str(exc))
        raise
