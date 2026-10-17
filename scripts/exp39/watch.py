"""exp39 resource watch (docs/experiments/2026-10-17-exp39-lowmem-plan.md): reads the training run's SwanLab cache and
stops the run (TERM to the given pattern's processes, KILL 15 s later) on the pre-written rules:
  - first window (step 50): step_s > 9, or MPS driver memory > 16 GB with system memory available < 2 GB;
  - two consecutive training windows > 12 s / step (windows that contain a validation -- the one after step 500 --
    are not counted);
  - a stall: no new window for 10 minutes (20 minutes after a validation step).
Writes what it saw and why it stopped to the given log.

    uv run python scripts/exp39/watch.py runs/exp39-h/stage1 "align_main.*exp39-h|configs/exp39-h.yaml" runs/exp39-watch.txt
"""
import glob
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))


def series(path: str, key: str):
    import struct
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
        if m > 0:
            out[step] = struct.unpack("<d", b[m + 3:m + 11])[0]
        i = j


def decide(windows: dict, mps: dict, ram: dict, since_last: float, val_steps=(500,)) -> str:
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
    limit = 1200 if any(v <= last < v + 50 for v in val_steps) else 600
    if since_last > limit:
        return "no new window for %.0f s after step %d" % (since_last, last)
    return ""


def training_pids(pattern: str):
    """pids matching `pattern`, never this watcher itself."""
    out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True).stdout.split()
    return [int(p) for p in out if int(p) != os.getpid()]


def stop(pattern: str) -> None:
    for sig in ("TERM", "KILL"):
        for p in training_pids(pattern):
            try:
                os.kill(p, getattr(__import__("signal"), "SIG" + sig))
            except ProcessLookupError:
                pass
        if sig == "TERM":
            time.sleep(15)


def main(stage_dir: str, pattern: str, log: str):
    run_id = None
    while run_id is None:
        f = os.path.join(stage_dir, "swanlab_run.json")
        if os.path.exists(f):
            run_id = json.load(open(f)).get("id") or json.load(open(f)).get("run_id")
        time.sleep(10)
    cache = None
    seen, last_change = 0, time.time()
    with open(log, "a") as out:
        while training_pids(pattern):
            cache = cache or next(iter(glob.glob("swanlog/run-*-%s/run-%s.swanlab" % (run_id, run_id))), None)
            if cache:
                w = series(cache, "perf/step_s")
                if len(w) != seen:
                    seen, last_change = len(w), time.time()
                    out.write("%s windows %s\n" % (time.strftime("%T"), {k: round(v, 2) for k, v in sorted(w.items())[-3:]}))
                    out.flush()
                why = decide(w, series(cache, "sys/torch_mps_gb"), series(cache, "sys/ram_available_gb"), time.time() - last_change)
                if why:
                    out.write("%s stop: %s\n" % (time.strftime("%T"), why))
                    out.flush()
                    stop(pattern)
                    return 3
            time.sleep(30)
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
