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


def main() -> int:
    t0 = float(open(STARTED).readline().split()[1]) if os.path.exists(STARTED) else None
    rec = json.load(open(RECORD)) if os.path.exists(RECORD) else None
    events = vcheck.events_in_window(DIAG, t0, time.time()) if t0 is not None else []
    rep = {"outcome": vcheck.outcome(rec, events), "record": rec, "system_events_in_window": events,
           "duration_s": round(time.time() - t0) if t0 is not None else None,
           "note": "diagnostic only: no scientific outcome; a pass is not a proof that long runs are safe"}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(rep, open(OUT + ".tmp", "w"), indent=1)
    os.replace(OUT + ".tmp", OUT)
    print(json.dumps({"outcome": rep["outcome"], "duration_s": rep["duration_s"]}, indent=1))
    return 0 if rep["outcome"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
