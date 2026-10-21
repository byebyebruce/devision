"""exp42 resource watch (critic EXP42-PREP-01; rules written before the run): candidate B's watch (scripts/v04b/watch.py
-- process group ownership, cache reading, memory rule) with limits for this 576-update run at 256. A window every 100
updates (100 .. 500); validations (val_sub, 973 questions, about 2 minutes) at step 0 and at the end, then once more
after merging. Stop when:
  - the first window (step 100) is slower than 3 s / update, or two consecutive windows are (exp40: 1.45-1.53);
  - any window shows MPS driver memory > 16 GB with system memory available < 2 GB;
  - no new window for 20 minutes before the first one (loading + step-0 validation + 100 updates, about 5 expected),
    20 minutes after the 500 window (the last 76 updates, the end validation, the merged pass, temperatures and
    saving, about 8 expected), 10 minutes otherwise.
The chain's absolute deadline applies on top. Exit status: the training's, or 3 when the watch stopped it.

    uv run python scripts/exp42/watch.py runs/exp42-r/stage2 runs/exp42-r-watch.txt -- uv run devision-pipeline configs/exp42-r.yaml
"""
import importlib.util
import os
import sys

_spec = importlib.util.spec_from_file_location("v04b_watch", os.path.join(os.path.dirname(__file__), "..", "v04b", "watch.py"))
assert _spec is not None and _spec.loader is not None
wb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wb)

LIMITS = {"first": 1200.0, "val": 1200.0, "other": 600.0}
SLOW = 3.0
VAL_WINDOWS = (500,)


def decide(windows: dict, mps: dict, ram: dict, since_last: float, limits=LIMITS) -> str:
    return wb.decide(windows, mps, ram, since_last, limits=limits, slow=SLOW, val_windows=VAL_WINDOWS)


def main(argv=None, limits=LIMITS, poll: float = 30.0) -> int:
    return wb.main(argv, limits=limits, poll=poll, slow=SLOW, val_windows=VAL_WINDOWS)


if __name__ == "__main__":
    sys.exit(main())
