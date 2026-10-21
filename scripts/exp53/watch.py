"""exp53 stage watch (critic EXP53-R1-04; rules written before any run). Runs one stage command in its own process group
(start_new_session) and stops that group -- TERM, then KILL GRACE seconds later; nothing outside the group is touched --
when:
  - no progress line within FIRST seconds of the start (default 300), or none for STALL seconds after the last one
    (default 180) -- a progress line is one the stage appends to the progress file (scripts/exp53/stages.progress);
  - available system memory stays below MEM_FLOOR GiB (default 1.5) for MEM_SECONDS consecutive seconds (default 30);
    swap use alone never stops it;
  - the deadline (epoch seconds, the run's end minus its report reserve) is reached;
  - at two consecutive checks (every CHECK seconds, default 60), the stage's own rate -- only once there are two
    progress lines with different counts and a positive rate; before that the stall rules alone apply -- puts its end
    after the deadline.
A SIGTERM / SIGINT to the watch (e.g. the outer run_capped) is forwarded to the group the same way, and any exception
inside the watch also stops the group (finally). The run passes a deadline early enough that the group's TERM, GRACE and
KILL end before the outer bound's TERM.
The first progress window's speed and the stage's ETA are logged. Exit status: the command's own, or 124 when the watch
stopped it (with the reason on stderr and in the log).

    uv run python scripts/exp53/watch.py --deadline EPOCH --progress runs/exp53/progress-local.txt --log runs/exp53/watch-local.txt -- CMD...
"""
import argparse
import os
import signal
import subprocess
import sys
import time


def available_gib() -> float:
    try:
        import psutil
        return psutil.virtual_memory().available / 2 ** 30
    except ImportError:
        return float("inf")


def read_progress(path: str):
    """(last done, total, time of the last line, time of the first line, first done) or None.
    Lines: '<epoch> <done> <total>'."""
    if not os.path.exists(path):
        return None
    rows = [x.split() for x in open(path).read().splitlines() if len(x.split()) == 3]
    if not rows:
        return None
    t0, (t, d, n) = float(rows[0][0]), rows[-1]
    return int(d), int(n), float(t), t0, int(rows[0][1])


def stop(proc, grace: float, log) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    end = time.time() + grace
    while time.time() < end and proc.poll() is None:
        time.sleep(0.2)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
        log("KILL after %.0f s" % grace)
    except ProcessLookupError:
        pass
    proc.wait()


class Stop(Exception):
    pass


def watch(cmd, deadline: float, progress: str, logpath: str, first: float = 300, stall: float = 180, grace: float = 15,
          mem_floor: float = 1.5, mem_seconds: float = 30, check: float = 60, poll: float = 1.0, mem=available_gib) -> int:
    def log(msg):
        with open(logpath, "a") as fh:
            fh.write("%s %s\n" % (time.strftime("%T"), msg))

    def on_signal(signum, frame):
        raise Stop("signal %d" % signum)
    old = {sg: signal.signal(sg, on_signal) for sg in (signal.SIGTERM, signal.SIGINT)}
    start = time.time()
    proc = subprocess.Popen(cmd, start_new_session=True)
    log("started pgid %d: %s" % (proc.pid, " ".join(cmd)))
    low_since, late, next_check, first_logged = None, 0, start + check, False
    try:
        while proc.poll() is None:
            now = time.time()
            p = read_progress(progress)
            why = None
            if now >= deadline:
                why = "deadline"
            elif p is None and now - start > first:
                why = "no progress within %.0f s" % first
            elif p is not None and now - p[2] > stall:
                why = "no progress for %.0f s" % stall
            if mem() < mem_floor:
                low_since = low_since or now
                if now - low_since >= mem_seconds:
                    why = "available memory below %.1f GiB for %.0f s" % (mem_floor, mem_seconds)
            else:
                low_since = None
            if why is None and p is not None and now >= next_check:
                next_check = now + check
                done, total, t_last, t0, d0 = p
                if done > d0 and t_last > t0:      # two progress points and a positive rate: an ETA exists
                    rate = (done - d0) / (t_last - t0)
                    eta = now + (total - done) / rate
                    if not first_logged:
                        log("first window: %.3f units/s, ETA %s" % (rate, time.strftime("%T", time.localtime(eta))))
                        first_logged = True
                    late = late + 1 if eta > deadline else 0
                    if late >= 2:
                        why = "two checks project the stage's end after the deadline"
            if why:
                raise Stop(why)
            time.sleep(poll)
    except Stop as e:
        log("stopping: %s" % e)
        print("watch: %s" % e, file=sys.stderr, flush=True)
        stop(proc, grace, log)
        return 124
    finally:
        if proc.poll() is None:            # any other exit of the watch: the group never outlives it
            log("watch ending abnormally: stopping the group")
            stop(proc, grace, log)
        for sg, h in old.items():
            signal.signal(sg, h)
    log("exited %d" % proc.returncode)
    return proc.returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deadline", type=float, required=True)
    ap.add_argument("--progress", required=True)
    ap.add_argument("--log", required=True)
    for k, v in (("first", 300), ("stall", 180), ("grace", 15), ("mem-floor", 1.5), ("mem-seconds", 30), ("check", 60)):
        ap.add_argument("--" + k, type=float, default=v)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    return watch(cmd, a.deadline, a.progress, a.log, a.first, a.stall, a.grace, a.mem_floor, a.mem_seconds, a.check)


if __name__ == "__main__":
    sys.exit(main())
