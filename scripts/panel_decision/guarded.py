"""EXP76 r2 execution guard (EXP76-RETRY-PREP-01): runs one command in its own process group (the calling runner is
not in it), keeps the command's stdout / stderr in a file, and watches that file and every file under the run's logs
directory (files created later are read from their start) about every 2 s, and once more after the command ends. On the
first line naming a GPU command-buffer error / GPU recovery (PATTERNS) it records the time, file and line to
<record>, sends TERM to the group, and KILL to whatever of the group is left after GRACE s. A limit ends the group the
same way (recorded as "limit"). Only this group is signalled; nothing is killed by name.

    python scripts/panel_decision/guarded.py --record R --watch DIR --out FILE --limit SECONDS -- cmd ...
Exit: the command's own code when it ended by itself without a GPU error; 70 after a GPU error (whatever the command
returned); 124 at the limit.
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


def end_group(pgid: int, why: str, log) -> dict:
    t: dict = {"why": why, "term": time.strftime("%F %T")}
    if not members(pgid):
        return dict(t, gone_before_term=True)
    signal_group(pgid, signal.SIGTERM)
    end = time.time() + GRACE
    while time.time() < end:
        if not members(pgid):
            return dict(t, ended=time.strftime("%F %T"), kill=False)
        time.sleep(0.2)
    left = members(pgid)
    signal_group(pgid, signal.SIGKILL)
    t.update(kill=time.strftime("%F %T"), killed=left)
    log("KILL after %.0f s: %s" % (GRACE, left))
    end = time.time() + 5
    while members(pgid) and time.time() < end:
        time.sleep(0.2)
    t["left_after_kill"] = members(pgid)
    return t


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record", required=True)
    ap.add_argument("--watch", action="append", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=float, required=True)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd

    def log(msg):
        print("== %s guard: %s" % (time.strftime("%F %T"), msg), flush=True)
    end_at = time.time() + a.limit
    with open(a.out, "ab") as out:
        child = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    pgid = child.pid
    log("started pid / pgid %d: %s" % (pgid, " ".join(cmd)))
    watch = Watch([a.out], a.watch)
    rec: dict = {"pgid": pgid, "cmd": cmd, "started": time.strftime("%F %T")}
    code = None
    while code is None:
        hit = watch.scan()
        if hit:
            rec.update(gpu_error=dict(hit, seen=time.strftime("%F %T")))
            log("GPU error in %s: %s" % (hit["file"], hit["line"]))
            rec["end"] = end_group(pgid, "gpu_error", log)
            child.wait()
            code = 70
            break
        if time.time() >= end_at:
            log("limit reached")
            rec["end"] = end_group(pgid, "limit", log)
            child.wait()
            code = 124
            break
        try:
            code = child.wait(timeout=POLL)
        except subprocess.TimeoutExpired:
            code = None
    if "gpu_error" not in rec:                        # the last scan after the command ended
        hit = watch.scan()
        if hit:
            rec.update(gpu_error=dict(hit, seen=time.strftime("%F %T"), after_exit=True))
            log("GPU error found after exit in %s: %s" % (hit["file"], hit["line"]))
            code = 70
    if members(pgid):                                   # leftovers of the group after the command itself ended
        rec["leftovers"] = end_group(pgid, "leftovers", log)
    rec.update(exit=code, child_returncode=child.returncode, ended=time.strftime("%F %T"))
    tmp = a.record + ".tmp"
    json.dump(rec, open(tmp, "w"), indent=1)
    os.replace(tmp, a.record)
    log("exit %s" % code)
    return code


if __name__ == "__main__":
    sys.exit(main())
