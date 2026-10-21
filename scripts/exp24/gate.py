"""exp24 gate between phase A (reading alignment) and phase B (exp22's fit test from the aligned checkpoint), rule
written before running: on the 1,000 held-out text pictures, masked-token accuracy with the right picture minus
with another picture (align.py's val/mlm_acc_gap) at the end of phase A must be >= 0.30; otherwise phase B is not
run (exit 10). Exit 0 = run phase B.

    uv run python scripts/exp24/gate.py runs/exp24-align/stage1/align_report.json
"""
import json
import sys

GAP = 0.30


def decide(report: dict) -> dict:
    val = report.get("val") or []
    if not val:
        raise ValueError("the alignment report has no validation rows")
    first, last = val[0], val[-1]
    keys = ("val/mlm_acc_real", "val/mlm_acc_shuffled", "val/mlm_acc_gap")
    out = {"start": {k: first[k] for k in keys}, "end": {k: last[k] for k in keys}, "end_step": last["step"],
           "threshold": GAP}
    out["run_phase_b"] = round(last["val/mlm_acc_gap"], 9) >= GAP   # exactly 0.30 passes (no float noise)
    return out


def main(argv=None):
    path = (argv if argv is not None else sys.argv[1:])[0]
    out = decide(json.load(open(path)))
    print(json.dumps(out, indent=1))
    sys.exit(0 if out["run_phase_b"] else 10)


if __name__ == "__main__":
    main()
