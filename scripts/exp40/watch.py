"""exp40 resource watch (critic V04-TIME-04; rules written before the run): candidate B's watch (scripts/v04b/watch.py
-- process group ownership, cache reading, memory rule, window bookkeeping) with limits for 256 input. exp32 B, the
same decision training at 256, ran 1.39-1.46 s / update, its 1,700 window (with a validation) 7.8 s. A window every
100 updates (last 3,200 of 3,298); validations at 0, 1,650 (inside the 1,700 window), the end (3,298), then all sets
after merging (eval_every 1,650: B's near-duplicate 3,200 round is not run). Stop when:
  - the first window (step 100) is slower than 3 s / update, or two consecutive training windows are (the 1,700
    window holds the 1,650 validation and is not a training window);
  - any window shows MPS driver memory > 16 GB with system memory available < 2 GB;
  - no new window for 30 minutes before the first one (loading + the step-0 validation, about 12 minutes expected),
    40 minutes after the 1,600 window (the 1,650 validation) and after 3,200 (last updates, final validation, merged evaluation,
    temperatures, saving, about 28 minutes expected), 10 minutes otherwise.
The chain's absolute deadline applies on top. Exit status: the training's, or 3 when the watch stopped it.

    uv run python scripts/exp40/watch.py runs/exp40-m/stage2 runs/exp40-m-watch.txt -- uv run devision-pipeline configs/exp40-m.yaml
"""
import importlib.util
import os
import sys

_spec = importlib.util.spec_from_file_location("v04b_watch", os.path.join(os.path.dirname(__file__), "..", "v04b", "watch.py"))
assert _spec is not None and _spec.loader is not None
wb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wb)

LIMITS = {"first": 1800.0, "val": 2400.0, "other": 600.0}
SLOW = 3.0


def decide(windows: dict, mps: dict, ram: dict, since_last: float, limits=LIMITS) -> str:
    return wb.decide(windows, mps, ram, since_last, limits=limits, slow=SLOW)


def main(argv=None, limits=LIMITS, poll: float = 30.0) -> int:
    return wb.main(argv, limits=limits, poll=poll, slow=SLOW)


if __name__ == "__main__":
    sys.exit(main())
