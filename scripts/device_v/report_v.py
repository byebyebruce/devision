"""EXP76 device V report (no model): the V record's one outcome, after a last look for system GPU events over the
whole window (start .. now). Writes runs/exp76-device-v/report.json; exit 0 only for "pass". The actual duration is
reported; the 2,700 s limit is not a duration. A pass says only that this one run on this device met the V rules
and matched the record written before the fault -- not that long runs are safe.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vcheck  # noqa: E402

NAME = "exp76-device-v"
RECORD = "runs/%s-record.json" % NAME
STARTED = "runs/%s.started" % NAME
OUT = "runs/%s/report.json" % NAME
DIAG = "/Library/Logs/DiagnosticReports"


def load_parts():
    """(t0 or None, record or None, problems): an unreadable / truncated started marker or record is a problem, not an
    empty window or a missing record that could pass (R1-02)."""
    probs, t0, rec = [], None, None
    try:
        t0 = float(next(x for x in open(STARTED).read().splitlines() if x.startswith("start ")).split()[1])
    except (OSError, StopIteration, ValueError, IndexError) as e:
        probs.append("no valid start in %s (%s): the window cannot be checked" % (STARTED, type(e).__name__))
    try:
        rec = json.load(open(RECORD))
    except (OSError, ValueError) as e:
        probs.append("the V record %s is missing or unreadable (%s)" % (RECORD, type(e).__name__))
    return t0, rec, probs


def main() -> int:
    t0, rec, probs = load_parts()
    events: list = []
    if t0 is not None:
        try:
            events = vcheck.events_in_window(DIAG, t0, time.time())
        except Exception as e:
            probs.append("the system event check failed (%s: %s)" % (type(e).__name__, e))
    out = vcheck.outcome(rec, events) if rec else None
    if probs:                                              # missing evidence never passes; a recorded failure stays named
        out = ("incomplete: " if out in (None, "pass") else out + "; also incomplete: ") + "; ".join(probs)
    rep = {"outcome": out, "record": rec, "problems": probs, "system_events_in_window": events,
           "duration_s": round(time.time() - t0) if t0 is not None else None,
           "note": "diagnostic only: no scientific outcome; a pass is not a proof that long runs are safe"}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(rep, open(OUT + ".tmp", "w"), indent=1)
    os.replace(OUT + ".tmp", OUT)
    print(json.dumps({"outcome": rep["outcome"], "duration_s": rep["duration_s"]}, indent=1))
    return 0 if rep["outcome"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
