"""EXP76 r2 report (EXP76-RETRY-PREP-01), no model: execution validity first, then the accepted D2 report.

The guard's record (runs/exp76-panel-decision-r2-guard.json) decides whether the run was a valid execution: a GPU
command-buffer error / recovery line anywhere in the run's output or logs, a limit stop, or a missing / unreadable
record makes the execution failed -- whatever the command returned and whatever files exist -- and the result
incomplete with no A / B / C / D; the ordinary report is kept under "voided" as a record only. Otherwise the accepted
d2_report (scripts/panel_learn/report.py) decides, unchanged.

    python scripts/panel_decision/report_r2.py
"""
import importlib.util
import json
import os
import sys

NAME = "exp76-panel-decision-r2"
RUN = "runs/" + NAME
GUARD = "runs/%s-guard.json" % NAME
CONFIG = "configs/%s.yaml" % NAME
SEALED = "runs/v04-k/eval"


def d2():
    spec = importlib.util.spec_from_file_location("pl_report", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "panel_learn", "report.py"))
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def execution(guard: str = GUARD) -> dict:
    try:
        rec = json.load(open(guard))
    except (OSError, ValueError) as e:
        return {"valid": False, "why": "no readable guard record (%s)" % e.__class__.__name__}
    if rec.get("gpu_error"):
        return {"valid": False, "why": "GPU execution error", "gpu_error": rec["gpu_error"], "end": rec.get("end")}
    if rec.get("exit") == 124:
        return {"valid": False, "why": "stopped at the limit", "end": rec.get("end")}
    return {"valid": True, "exit": rec.get("exit")}


def report(run: str = RUN, guard: str = GUARD, sealed: str = SEALED, config: str = CONFIG) -> dict:
    ex = execution(guard)
    try:
        inner = d2().d2_report(run, sealed, config)
    except Exception as e:                       # the run's files may be partial after a stop: still a written report
        inner = {"outcome": "incomplete: the D2 report could not be built (%s)" % e.__class__.__name__}
    if ex["valid"]:
        out = dict(inner, execution=ex)
    else:
        out = {"execution": ex, "outcome": "incomplete: execution failed (%s) -- no scientific outcome" % ex["why"],
               "voided": inner}
    os.makedirs(run, exist_ok=True)
    tmp = os.path.join(run, "report.json.tmp")
    json.dump(out, open(tmp, "w"), indent=1)
    os.replace(tmp, os.path.join(run, "report.json"))
    return out


def main() -> int:
    out = report()
    print(json.dumps({k: v for k, v in out.items() if k != "voided"}, indent=1))
    return 0 if out["outcome"][:2] in ("A:", "B:", "C:", "D:") else 1


if __name__ == "__main__":
    sys.exit(main())
