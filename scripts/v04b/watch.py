"""Candidate B resource watch (critic V04-B-R1-01; rules written before the run). Like scripts/exp39/watch.py (whose
cache reading and process-group handling it reuses, unchanged) it STARTS the training in its own process group and
stops only that group (TERM, KILL 15 s later); a TERM to the watch (the chain's deadline) is passed on. B logs a speed
window every 100 updates (last one at 3,200 of 3,298); validations run at 0, 1,600, 3,200 and 3,298, then every set
again after LoRA is merged, the temperatures are fitted and the checkpoint is saved. Stop when:
  - the first window (step 100) is slower than 6 s / update, or any two consecutive training windows are;
    the 1,700 window contains the 1,600 validation and is not a training window;
  - any window shows MPS driver memory > 16 GB with system memory available < 2 GB;
  - no new window for 45 minutes before the first one, 45 minutes after the 1,600 / 3,200 windows (validation; after
    3,200 also the last updates, the final validation, merged evaluation, temperatures and saving), 15 otherwise.
The chain's absolute deadline still applies on top. Exit status: the training's, or 3 when the watch stopped it.

    uv run python scripts/v04b/watch.py runs/v04b-fit/stage2 runs/v04b-watch.txt -- uv run devision-pipeline configs/v04b-fit.yaml
"""
import importlib.util
import os
import signal
import struct
import subprocess
import sys
import time

_spec = importlib.util.spec_from_file_location("exp39_watch", os.path.join(os.path.dirname(__file__), "..", "exp39", "watch.py"))
assert _spec is not None and _spec.loader is not None
w39 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(w39)

LIMITS = {"first": 2700.0, "val": 2700.0, "other": 900.0}
VAL_WINDOWS = (1600, 3200)      # windows after which a validation (or, after 3,200, the end) runs


def decide(windows: dict, mps: dict, ram: dict, since_last: float, limits=LIMITS, slow: float = 6.0,
           val_windows=VAL_WINDOWS) -> str:
    """'' to go on, else the reason to stop (`slow`: the s / update limit; exp40 reuses this at 256 with its own)."""
    steps = sorted(windows)
    for s in steps:
        if mps.get(s, 0) > 16 and ram.get(s, 99) < 2:
            return "window %d: MPS %.1f GB with %.1f GB available" % (s, mps[s], ram[s])
    if 100 in windows and windows[100] > slow:
        return "first window %.2f s/update > %g" % (windows[100], slow)
    train = [s for s in steps if not any(v < s <= v + 100 for v in val_windows)]
    for a, b in zip(train, train[1:]):
        if b - a == 100 and windows[a] > slow and windows[b] > slow:
            return "two windows > %g s/update at %d / %d" % (slow, a, b)
    last = steps[-1] if steps else 0
    limit = limits["first"] if not steps else limits["val"] if last in val_windows else limits["other"]
    if since_last > limit:
        return "no new window for %.0f s after step %d" % (since_last, last)
    return ""


def main(argv=None, limits=LIMITS, poll: float = 30.0, slow: float = 6.0, val_windows=VAL_WINDOWS) -> int:
    argv = sys.argv[1:] if argv is None else argv
    stage_dir, log = argv[0], argv[1]
    cmd = argv[argv.index("--") + 1:]
    child = subprocess.Popen(cmd, start_new_session=True)
    stopping = {"why": ""}

    def on_term(signum, frame):
        stopping["why"] = "the watch was told to stop (signal %d)" % signum
    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)
    seen, last_change, unreadable = 0, time.time(), 0
    win: dict = {}
    mps: dict = {}
    ram: dict = {}
    with open(log, "a") as out:
        out.write("%s started pid %d: %s\n" % (time.strftime("%T"), child.pid, " ".join(cmd)))
        out.flush()
        while child.poll() is None:
            why = stopping["why"]
            cache = w39.cache_of(stage_dir)
            if cache and not why:
                try:
                    nw, nm, nr = (w39.series(cache, k) for k in ("perf/step_s", "sys/torch_mps_gb", "sys/ram_available_gb"))
                    if len(nw) >= len(win):
                        win, mps, ram = nw, nm, nr
                except (OSError, IndexError, struct.error):
                    unreadable += 1
                if len(win) > seen:
                    seen, last_change = len(win), time.time()
                    out.write("%s windows %s\n" % (time.strftime("%T"), {k: round(v, 2) for k, v in sorted(win.items())[-3:]}))
                    out.flush()
            if not why:
                why = decide(win, mps, ram, time.time() - last_change, limits=limits, slow=slow, val_windows=val_windows)
            if why:
                out.write("%s stop: %s (unreadable cache reads %d)\n" % (time.strftime("%T"), why, unreadable))
                out.flush()
                w39.stop_group(child)
                return 3
            deadline = time.time() + poll
            while time.time() < deadline and child.poll() is None and not stopping["why"]:
                time.sleep(min(1.0, poll))
        out.write("%s training exited %s\n" % (time.strftime("%T"), child.returncode))
    return int(child.returncode or 0)


if __name__ == "__main__":
    sys.exit(main())
