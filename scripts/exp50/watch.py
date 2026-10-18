"""exp50 resource watch (critic EXP50-PREP-03; rules written before the run): candidate B's watch (scripts/v04b/watch.py
-- process group ownership, cache reading, memory rule) for each 576-update arm, the vision tower frozen, each micro
batch's 4 pictures encoded with their 12 views (52 images, no gradient). A window every 100 updates (100 .. 500);
validations (data/exp50/val_quarter.jsonl, 244 questions x 13 images, about 1 minute) at step 0, at the end and after merging.
Stop when:
  - the first window (step 100) is slower than 2.6 s / update, or two consecutive windows are (estimate about 2.1: EXP48-X's
    1.43 plus 4 pictures x 12 views x about 12 ms plus the 3,328-patch keys / values' projection and attention backward; above 2.6 the two arms no longer fit the 90 minutes);
  - any window shows MPS driver memory > 16 GB with system memory available < 2 GB;
  - no new window for 15 minutes before the first one (loading + step-0 validation + 100 updates, about 6 expected),
    15 minutes after the 500 window (the last 76 updates, the end validation, temperatures and saving, about 6
    expected), 10 minutes otherwise.
The chain's absolute deadline applies on top. Exit status: the training's, or 3 when the watch stopped it.

    uv run python scripts/exp50/watch.py runs/exp50-t/stage2 runs/exp50-t-watch.txt -- uv run devision-pipeline configs/exp50-t.yaml
"""
import importlib.util
import os
import sys

_spec = importlib.util.spec_from_file_location("v04b_watch", os.path.join(os.path.dirname(__file__), "..", "v04b", "watch.py"))
assert _spec is not None and _spec.loader is not None
wb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wb)

LIMITS = {"first": 900.0, "val": 900.0, "other": 600.0}
SLOW = 2.6
VAL_WINDOWS = (500,)


def decide(windows: dict, mps: dict, ram: dict, since_last: float, limits=LIMITS) -> str:
    return wb.decide(windows, mps, ram, since_last, limits=limits, slow=SLOW, val_windows=VAL_WINDOWS)


def main(argv=None, limits=LIMITS, poll: float = 30.0) -> int:
    return wb.main(argv, limits=limits, poll=poll, slow=SLOW, val_windows=VAL_WINDOWS)


if __name__ == "__main__":
    sys.exit(main())
