"""exp52 resource watch (critic EXP52-PREP-01; the exp50 watch with the reader; rules written before the run): candidate B's watch (scripts/v04b/watch.py
-- process group ownership, cache reading, memory rule) for each 576-update arm, the vision tower frozen, each micro
batch's 4 pictures encoded with their 12 views (52 images, no gradient). A window every 100 updates (100 .. 500);
validations (data/exp50/val_quarter.jsonl, 244 questions x 13 images, about 1 minute) at step 0, at the end and after merging.
Stop when:
  - the first window (step 100) is slower than 2.7 s / update, or two consecutive windows are (EXP50-T measured 2.31-2.37
    with the same pass; the reader adds two small attentions over 5 slots, estimate about 2.45; above 2.7 the two arms and
    their required readout no longer fit the 120 minutes);
  - any window shows MPS driver memory > 16 GB with system memory available < 2 GB;
  - no new window for 15 minutes before the first one (loading + step-0 validation + 100 updates, about 6 expected),
    15 minutes after the 500 window (the last 76 updates, the end validation, temperatures and saving, about 6
    expected), 10 minutes otherwise.
The chain's absolute deadline applies on top. Exit status: the training's, or 3 when the watch stopped it.

    uv run python scripts/exp52/watch.py runs/exp52-n/stage2 runs/exp52-n-watch.txt -- uv run devision-pipeline configs/exp52-n.yaml
"""
import importlib.util
import os
import sys

_spec = importlib.util.spec_from_file_location("v04b_watch", os.path.join(os.path.dirname(__file__), "..", "v04b", "watch.py"))
assert _spec is not None and _spec.loader is not None
wb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wb)

LIMITS = {"first": 900.0, "val": 900.0, "other": 600.0}
SLOW = 2.7
VAL_WINDOWS = (500,)


def decide(windows: dict, mps: dict, ram: dict, since_last: float, limits=LIMITS) -> str:
    return wb.decide(windows, mps, ram, since_last, limits=limits, slow=SLOW, val_windows=VAL_WINDOWS)


def main(argv=None, limits=LIMITS, poll: float = 30.0) -> int:
    return wb.main(argv, limits=limits, poll=poll, slow=SLOW, val_windows=VAL_WINDOWS)


if __name__ == "__main__":
    sys.exit(main())
