"""scripts/v04full/finalize.sh (critic V04-K-FULL-MID-01-A) with a stub uv and a fake chain process: it waits for the
chain PID; after a non-zero exit it runs only seal / check (v0.3's only when v0.3 was frozen by the chain) and lists
the result; after exit 0 it runs nothing; it never starts run-extras, the pipeline or a model; it stays inside the
total deadline."""
import os
import shutil
import stat
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STUB = '#!/bin/bash\necho "uv $*" >> "${MARK_CALLS}"\necho "stub output for $*"\nexit 0\n'


def setup(tmp_path, exit_line, v03_frozen):
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    for rel in ("scripts/v04full/finalize.sh", "scripts/exp35/guard.sh"):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join(ROOT, rel), repo / rel)
    (repo / "runs").mkdir()
    (repo / "runs/v04-k-full.started").write_text("x")
    time.sleep(0.01)
    if v03_frozen:
        (repo / "runs/r15-vision/eval").mkdir(parents=True)
        os.utime(repo / "runs/v04-k-full.started", (time.time() - 100, time.time() - 100))
        (repo / "runs/r15-vision/eval/IDENTITY.json").write_text("{}")
    bdir = tmp_path / "bin"
    bdir.mkdir()
    (bdir / "uv").write_text(STUB)
    (bdir / "uv").chmod((bdir / "uv").stat().st_mode | stat.S_IEXEC)
    pid = subprocess.run(["bash", "-c", "sleep 2 >/dev/null 2>&1 & echo $!"], capture_output=True, text=True).stdout.strip()   # orphaned: reaped by init
    (repo / "runs/v04-k-full.out").write_text("== 2026-10-19 23:48:21 steps: ...\n" + exit_line)
    env = {**os.environ, "PATH": "%s:%s" % (bdir, os.environ["PATH"]), "MARK_CALLS": str(tmp_path / "calls"),
           "FINAL_PID": pid, "FINAL_DEADLINE": str(int(time.time()) + 60), "FINAL_REPO": str(repo), "FINAL_POLL": "1"}
    p = subprocess.run(["bash", str(repo / "scripts/v04full/finalize.sh")], env=env, capture_output=True, text=True, timeout=90)
    calls = (tmp_path / "calls").read_text().splitlines() if (tmp_path / "calls").exists() else []
    return p, calls, repo


def test_a_failed_chain_is_sealed_and_checked_without_models(tmp_path):
    p, calls, repo = setup(tmp_path / "a", "== 2026-10-19 23:48:22 exit 124\n", v03_frozen=False)
    assert p.returncode == 0, p.stdout + p.stderr
    assert [c.split()[4] for c in calls] == ["seal", "check"] and all("v04-full-eval" in c for c in calls)
    assert not any(w in " ".join(calls) for w in ("run-extras", "devision-pipeline", "freeze", "devision-eval"))
    out = (repo / "runs/v04-k-finalize.txt").read_text()
    assert "exit 124" in out and "all 6 missing" in out and "rc=0" in out


def test_v03_items_only_when_frozen_by_the_chain(tmp_path):
    p, calls, _ = setup(tmp_path / "b", "== 2026-10-19 23:48:22 exit 1\n", v03_frozen=True)
    assert p.returncode == 0 and [c.split()[4] for c in calls] == ["seal", "check", "seal", "check"]
    assert sum("v04-v03-confirm" in c for c in calls) == 2


def test_an_exit_0_chain_needs_nothing(tmp_path):
    p, calls, repo = setup(tmp_path / "c", "== 2026-10-19 23:00:00 exit 0\n", v03_frozen=False)
    assert p.returncode == 0 and not calls and "no close-out" in (repo / "runs/v04-k-finalize.txt").read_text()


def test_waits_for_the_chains_evaluation_processes(tmp_path):
    """An evaluation process of the chain (its command line holds the repo path) still alive after the chain PID ended:
    seal waits until it is gone."""
    repo = tmp_path / "w" / "repo"
    repo.mkdir(parents=True)
    t0 = time.time()
    subprocess.run(["bash", "-c", "(exec -a 'python eval_main %s/x' sleep 6) >/dev/null 2>&1 & echo $!" % repo], capture_output=True, text=True)
    p, calls, _ = setup(tmp_path / "w", "== 2026-10-19 23:48:22 exit 124\n", v03_frozen=False)
    assert p.returncode == 0 and "waiting for the chain's evaluation processes" in p.stdout and time.time() - t0 >= 6 and calls
