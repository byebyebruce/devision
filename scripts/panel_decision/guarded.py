"""EXP76 r2 execution guard (EXP76-RETRY-PREP-01): runs one command in its own process group (the calling runner is
not in it), keeps the command's stdout / stderr in a file, and watches that file and every file under the run's logs
directory (files created later are read from their start) about every 2 s, and once more after the command ends. On the
first line naming a GPU command-buffer error / GPU recovery (PATTERNS) it records the time, file and line to
<record>, sends TERM to the group, and KILL to whatever of the group is left after GRACE s. A limit ends the group the
same way (recorded as "limit"). Only this group is signalled; nothing is killed by name.

    python scripts/panel_decision/guarded.py --record R --watch DIR --out FILE --deadline EPOCH -- cmd ...
Every wait -- polling, the TERM grace, the KILL check, waiting for the command -- is bounded by one absolute deadline
(EXP76-RETRY-R1-01): a stop starts early enough to end by it, and the guard writes its record and returns by then even
if the group cannot be reaped. Exit: the command's own code when it ended by itself without a GPU error and the group is
gone; 70 after a GPU error (whatever the command returned); 124 at the limit; 71 when the cleanup is not confirmed
(a live member or an unreaped command is left; no further model may start).
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time

PATTERNS = ("command buffer exited with error status", "victim of GPU error/recovery", "kIOGPUCommandBufferCallbackError", "InnocentVictim")
GRACE = float(os.environ.get("GUARD_GRACE", "15"))
POLL = float(os.environ.get("GUARD_POLL", "2"))


class Watch:
    def __init__(self, files, dirs):
        self.files, self.dirs, self.pos = list(files), list(dirs), {}

    def scan(self):
        paths = list(self.files)
        for d in self.dirs:
            if os.path.isdir(d):
                paths += [os.path.join(d, f) for f in sorted(os.listdir(d))]
        for p in paths:
            if not os.path.isfile(p):
                continue
            with open(p, "rb") as fh:
                fh.seek(self.pos.get(p, 0))
                data = fh.read()
            if not data:
                continue
            cut = data.rfind(b"\n") + 1               # whole lines only; the rest is read next time
            if cut == 0:
                continue
            self.pos[p] = self.pos.get(p, 0) + cut
            for line in data[:cut].decode("utf-8", "replace").splitlines():
                if any(x in line for x in PATTERNS):
                    return {"file": p, "line": line.strip()}
        return None


def members(pgid: int) -> list:
    """Live (not zombie) processes of the group, from ps -- os.killpg(pgid, 0) answers EPERM on macOS once only
    exited / zombie members are left, so it cannot tell a live group from a dead one."""
    out = subprocess.run(["/bin/ps", "-axo", "pid=,pgid=,stat="], capture_output=True, text=True).stdout
    return [int(f[0]) for f in (line.split() for line in out.splitlines()) if len(f) >= 3 and f[1] == str(pgid) and not f[2].startswith("Z")]


def signal_group(pgid: int, sig) -> None:
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        pass
    for pid in members(pgid):                      # whatever is still in the group, by its own pid
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError):
            pass


KILL_CHECK = float(os.environ.get("GUARD_KILL_CHECK", "5"))   # the longest wait after KILL before reporting what is left
STOP_MARGIN = 1.0                                               # scan / bookkeeping time kept free before the deadline


def end_group(child, pgid: int, why: str, log, deadline: float) -> dict:
    """TERM the group, KILL whatever is left after GRACE s, then wait at most KILL_CHECK s -- every wait bounded by the
    absolute `deadline`. Returns what happened, with "confirmed": True only when the group has no live member and the
    command itself was reaped; otherwise the PIDs still there (a KILL sent is not a process gone)."""
    t: dict = {"why": why, "term": time.strftime("%F %T")}
    if members(pgid):
        signal_group(pgid, signal.SIGTERM)
        until = min(time.time() + GRACE, deadline - KILL_CHECK)
        while time.time() < until and members(pgid):
            time.sleep(0.2)
        left = members(pgid)
        if left:
            signal_group(pgid, signal.SIGKILL)
            t.update(kill=time.strftime("%F %T"), killed=left)
            log("KILL: %s" % left)
    else:
        t["gone_before_term"] = True
    until = deadline
    while members(pgid) and time.time() < until:
        time.sleep(0.2)
    try:
        child.wait(timeout=max(0.0, until - time.time()))
        reaped = True
    except subprocess.TimeoutExpired:
        reaped = False
    left = members(pgid)
    t.update(left=left, child_reaped=reaped, confirmed=reaped and not left, ended=time.strftime("%F %T"))
    if not t["confirmed"]:
        log("cleanup not confirmed: still %s, command reaped %s -- no further model may start" % (left, reaped))
    return t


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record", required=True)
    ap.add_argument("--watch", action="append", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--deadline", type=float, required=True, help="absolute epoch seconds: the guard has returned by then")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd

    def log(msg):
        print("== %s guard: %s" % (time.strftime("%F %T"), msg), flush=True)
    deadline = a.deadline
    stop_at = deadline - GRACE - KILL_CHECK - POLL - STOP_MARGIN     # a stop started then still ends by the deadline
    with open(a.out, "ab") as out:
        child = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    pgid = child.pid
    log("started pid / pgid %d: %s (stop by %s, return by %s)" % (pgid, " ".join(cmd), time.strftime("%T", time.localtime(stop_at)),
                                                                  time.strftime("%T", time.localtime(deadline))))
    watch = Watch([a.out], a.watch)
    rec: dict = {"pgid": pgid, "cmd": cmd, "started": time.strftime("%F %T"), "deadline": deadline}
    code = None
    while code is None:
        hit = watch.scan()
        if hit:
            rec.update(gpu_error=dict(hit, seen=time.strftime("%F %T")))
            log("GPU error in %s: %s" % (hit["file"], hit["line"]))
            rec["end"] = end_group(child, pgid, "gpu_error", log, deadline - STOP_MARGIN)
            code = 70
            break
        if time.time() >= stop_at:
            log("limit reached")
            rec["end"] = end_group(child, pgid, "limit", log, deadline - STOP_MARGIN)
            code = 124
            break
        try:
            code = child.wait(timeout=max(0.0, min(POLL, stop_at - time.time())))
        except subprocess.TimeoutExpired:
            code = None
    if "gpu_error" not in rec:                        # the last scan after the command ended
        hit = watch.scan()
        if hit:
            rec.update(gpu_error=dict(hit, seen=time.strftime("%F %T"), after_exit=True))
            log("GPU error found after exit in %s: %s" % (hit["file"], hit["line"]))
            code = 70
    if "end" not in rec and members(pgid):              # leftovers of the group after the command itself ended
        rec["leftovers"] = end_group(child, pgid, "leftovers", log, deadline - STOP_MARGIN)
    stop = rec.get("end") or rec.get("leftovers")
    rec["cleanup_confirmed"] = stop["confirmed"] if stop else (child.returncode is not None and not members(pgid))
    if not rec["cleanup_confirmed"]:
        code = 71
    rec.update(exit=code, child_returncode=child.returncode, ended=time.strftime("%F %T"))
    tmp = a.record + ".tmp"
    json.dump(rec, open(tmp, "w"), indent=1)
    os.replace(tmp, a.record)
    log("exit %s" % code)
    return code


if __name__ == "__main__":
    sys.exit(main())
