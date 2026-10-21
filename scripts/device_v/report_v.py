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
    """(start, deadline, record, problems) -- the started marker's window as written, the V record or None. An
    unreadable / truncated marker or record is a problem, not an empty window or a missing record that could pass."""
    probs, start, deadline, rec = [], None, None, None
    try:
        lines = dict(x.split(None, 1) for x in open(STARTED).read().splitlines() if x.strip() and len(x.split(None, 1)) == 2)
        start, deadline = float(lines["start"]), float(lines["deadline"])
    except (OSError, KeyError, ValueError) as e:
        probs.append("no valid start / deadline in %s (%s): the window cannot be checked" % (STARTED, type(e).__name__))
    try:
        rec = json.load(open(RECORD))
    except (OSError, ValueError) as e:
        probs.append("the V record %s is missing or unreadable (%s)" % (RECORD, type(e).__name__))
    return start, deadline, rec, probs


def main() -> int:
    start, deadline, rec, probs = load_parts()
    now = time.time()
    window_ok = False
    if start is not None:
        other = (rec.get("start_epoch"), rec.get("deadline_epoch")) if isinstance(rec, dict) else None
        bad = vcheck.window_problems(start, deadline, now, other)
        probs += bad
        window_ok = not bad
    events: list = []
    if window_ok and start is not None:                    # events only over a valid window: never an empty query
        try:
            events = vcheck.events_in_window(DIAG, start, now)
        except Exception as e:
            probs.append("the system event check failed (%s: %s)" % (type(e).__name__, e))
    out = vcheck.outcome(rec, events) if isinstance(rec, dict) else None
    if probs:                                              # missing evidence never passes; a recorded failure stays named
        out = ("incomplete: " if out in (None, "pass") else out + "; also incomplete: ") + "; ".join(probs)
    rep = {"outcome": out, "record": rec, "problems": probs, "system_events_in_window": events,
           "duration_s": round(now - start) if window_ok and start is not None else None,
           "note": "diagnostic only: no scientific outcome; a pass is not a proof that long runs are safe"}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(rep, open(OUT + ".tmp", "w"), indent=1)
    os.replace(OUT + ".tmp", OUT)
    print(json.dumps({"outcome": rep["outcome"], "duration_s": rep["duration_s"]}, indent=1))
    return 0 if rep["outcome"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
