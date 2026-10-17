"""exp39 resource watch (docs/experiments/2026-10-17-exp39-lowmem-plan.md; critic EXP39-R2-02). The watch STARTS the
training command itself, in its own process group, and owns it: it reads that run's SwanLab cache and stops exactly
that process group (TERM, then KILL 15 s later) on the pre-written rules --
  - first window (step 50): step_s > 9, or MPS driver memory > 16 GB with system memory available < 2 GB;
  - two consecutive training windows > 12 s / step (the window after the step-500 validation is not counted);
  - a stall, timed from the start of the training even while no run id / cache exists yet: no new window for
    30 minutes before the first one (the step-0 validation at 512 comes first), 20 minutes after a validation step,
    10 minutes otherwise.
A cache that cannot be read (being written) counts as "no new window" -- the stall clock keeps running, it is never
read as progress. A TERM to the watch itself (the chain's deadline) is passed on to the training group the same way.
Exit status: the training's own status, or 3 when the watch stopped it. What it saw and why it stopped: the log.

    uv run python scripts/exp39/watch.py runs/exp39-h/stage1 runs/exp39-watch.txt -- uv run devision-pipeline configs/exp39-h.yaml
"""
import glob
import json
import os
import signal
import struct
import subprocess
import sys
import time

LIMITS = {"first": 1800.0, "val": 1200.0, "other": 600.0}


def series(path: str, key: str) -> dict:
    b = open(path, "rb").read()

    def varint(i):
        v = s = 0
        while True:
            c = b[i]
            v |= (c & 0x7F) << s
            i += 1
            s += 7
            if c < 0x80:
                return v, i
    k = b"\n" + bytes([len(key)]) + key.encode() + b"\x10"
    out, i = {}, 0
    while True:
        i = b.find(k, i)
        if i < 0:
            return out
        step, j = varint(i + len(k))
        m = b.find(b"*\t\t", j, j + 60)
        if m > 0 and m + 11 <= len(b):
            out[step] = struct.unpack("<d", b[m + 3:m + 11])[0]
        i = j


def decide(windows: dict, mps: dict, ram: dict, since_last: float, val_steps=(500,), limits=LIMITS) -> str:
    """'' to go on, else the reason to stop."""
    steps = sorted(windows)
    if 50 in windows:
        if windows[50] > 9:
            return "first window %.2f s/step > 9" % windows[50]
        if mps.get(50, 0) > 16 and ram.get(50, 99) < 2:
            return "first window MPS %.1f GB with %.1f GB available" % (mps[50], ram[50])
    train = [s for s in steps if not any(v < s <= v + 50 for v in val_steps)]
    for a, b in zip(train, train[1:]):
        if b - a == 50 and windows[a] > 12 and windows[b] > 12:
            return "two windows > 12 s/step at %d / %d" % (a, b)
    last = steps[-1] if steps else 0
    limit = limits["first"] if not steps else limits["val"] if any(v <= last < v + 50 for v in val_steps) else limits["other"]
    if since_last > limit:
        return "no new window for %.0f s after step %d" % (since_last, last)
    return ""


def stop_group(child: subprocess.Popen, grace: float = 15.0) -> None:
    """TERM the training's process group, KILL it `grace` seconds later if anything is left."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            return
        if sig == signal.SIGTERM:
            try:
                child.wait(grace)
                os.killpg(child.pid, 0)       # the leader is gone; anything left in the group?
            except subprocess.TimeoutExpired:
                pass
            except ProcessLookupError:
                return
            time.sleep(0.5)


def cache_of(stage_dir: str):
    f = os.path.join(stage_dir, "swanlab_run.json")
    try:
        run_id = json.load(open(f)).get("id")
    except (OSError, ValueError):
        return None
    return next(iter(glob.glob("swanlog/run-*-%s/run-%s.swanlab" % (run_id, run_id))), None) if run_id else None


def main(argv=None, limits=LIMITS, poll: float = 30.0) -> int:
    argv = sys.argv[1:] if argv is None else argv
    stage_dir, log = argv[0], argv[1]
    cmd = argv[argv.index("--") + 1:]
    child = subprocess.Popen(cmd, start_new_session=True)       # its own process group: nothing else is touched
    stopping = {"why": ""}

    def on_term(signum, frame):
        stopping["why"] = "the watch was told to stop (signal %d)" % signum
    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)
    seen, last_change, unreadable = 0, time.time(), 0
    w: dict = {}
    mps: dict = {}
    ram: dict = {}
    with open(log, "a") as out:
        out.write("%s started pid %d: %s\n" % (time.strftime("%T"), child.pid, " ".join(cmd)))
        out.flush()
        while child.poll() is None:
            why = stopping["why"]
            cache = cache_of(stage_dir)
            if cache and not why:
                try:
                    nw, nm, nr = (series(cache, k) for k in ("perf/step_s", "sys/torch_mps_gb", "sys/ram_available_gb"))
                    if len(nw) >= len(w):          # a half-written read never removes what was seen
                        w, mps, ram = nw, nm, nr
                except (OSError, IndexError, struct.error):
                    unreadable += 1
                if len(w) > seen:
                    seen, last_change = len(w), time.time()
                    out.write("%s windows %s\n" % (time.strftime("%T"), {k: round(v, 2) for k, v in sorted(w.items())[-3:]}))
                    out.flush()
            if not why:
                why = decide(w, mps, ram, time.time() - last_change, limits=limits)
            if why:
                out.write("%s stop: %s (unreadable cache reads %d)\n" % (time.strftime("%T"), why, unreadable))
                out.flush()
                stop_group(child)
                return 3
            deadline = time.time() + poll
            while time.time() < deadline and child.poll() is None and not stopping["why"]:
                time.sleep(min(1.0, poll))
        out.write("%s training exited %s\n" % (time.strftime("%T"), child.returncode))
    return int(child.returncode or 0)


if __name__ == "__main__":
    sys.exit(main())
