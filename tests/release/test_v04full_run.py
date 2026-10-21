"""scripts/v04full/run.sh (critic V04-K-FULL-RUN-01) with a stub uv: the order of the existing commands, the first
formal output verified (seal / preflight / exist sealed) before the pipeline, the first failure stops the chain, one
start only, limit and reserve."""
import os
import shutil
import stat
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STUB = """#!/bin/bash
echo "uv $*" >> "${MARK_CALLS}"
case "$*" in
  *"python -c"*) [ -n "${FAIL_FIRST_OK:-}" ] && exit 3; exit 0;;
  "run python - "*) echo "echo first-exist"; exit 0;;
  *"seal configs/v04-full-eval.yaml"*) [ -n "${FAIL_SEAL:-}" ] && exit 5; exit 0;;
esac
exit 0
"""


def chain(tmp_path, extra):
    root = tmp_path / "repo"
    for rel in ("scripts/v04full/run.sh", "scripts/exp35/guard.sh", "scripts/exp38/steps.sh", "configs/v04-full-eval.yaml",
                "configs/v04-extra-evidence.yaml", "configs/v04-v03-confirm.yaml", "configs/v04-confirm-extras.yaml"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join(ROOT, rel), root / rel)
    (root / "runs/v04-k/eval").mkdir(parents=True)
    (root / "runs/v04-k/eval/IDENTITY.json").write_text("{}")
    bdir = tmp_path / "bin"
    bdir.mkdir()
    for name, body in {"uv": STUB, "pgrep": "#!/bin/bash\nexit 1\n", "caffeinate": "#!/bin/bash\nexit 0\n"}.items():
        (bdir / name).write_text(body)
        (bdir / name).chmod((bdir / name).stat().st_mode | stat.S_IEXEC)
    env = {**os.environ, "PATH": "%s:%s" % (bdir, os.environ["PATH"]), "MARK_CALLS": str(tmp_path / "calls"),
           "FULL_LIMIT": "120", "FULL_RESERVE": "4", **extra}
    p = subprocess.run(["bash", str(root / "scripts/v04full/run.sh")], env=env, capture_output=True, text=True, timeout=120, cwd=str(root))
    calls = (tmp_path / "calls").read_text().splitlines() if (tmp_path / "calls").exists() else []
    return p, calls, root


def test_order_and_first_output_gate(tmp_path):
    p, calls, root = chain(tmp_path / "a", {})
    assert p.returncode == 0, p.stdout + p.stderr
    keys = ["--dry-run", "preflight configs/v04-full-eval", "run python - ", "seal configs/v04-full-eval", "preflight configs/v04-full-eval",
            "python -c", "devision-pipeline configs/v04-full-eval.yaml", "run-extras configs/v04-full-eval", "seal configs/v04-full-eval",
            "check configs/v04-full-eval", "freeze configs/v04-v03", "preflight configs/v04-v03", "run-extras configs/v04-v03",
            "seal configs/v04-v03", "check configs/v04-v03"]
    i = 0
    for c in calls:
        if i < len(keys) and keys[i] in c and not (keys[i] == "devision-pipeline configs/v04-full-eval.yaml" and "--dry-run" in c):
            i += 1
    assert i == len(keys), calls
    assert not any("--force" in c for c in calls)
    mark = (root / "runs/v04-k-full.started").read_text()
    assert "sha256" in mark and "IDENTITY.json" in mark
    p, calls2, _ = chain(tmp_path / "a2", {})
    p2 = subprocess.run(["bash", str(root / "scripts/v04full/run.sh")], env={**os.environ, "PATH": "%s:%s" % (tmp_path / "a" / "bin", os.environ["PATH"]),
                        "MARK_CALLS": str(tmp_path / "x")}, capture_output=True, text=True, cwd=str(root))
    assert p2.returncode == 1 and "already started once" in p2.stdout


def test_a_failed_first_output_stops_before_the_pipeline(tmp_path):
    for env in ({"FAIL_SEAL": "1"}, {"FAIL_FIRST_OK": "1"}):
        p, calls, _ = chain(tmp_path / list(env)[0], env)
        assert p.returncode != 0 and not any(c.endswith("devision-pipeline configs/v04-full-eval.yaml") for c in calls), calls


def test_limit_and_reserve():
    s = open(os.path.join(ROOT, "scripts/v04full/run.sh")).read()
    assert "${FULL_LIMIT:-37800}" in s and "${FULL_RESERVE:-900}" in s
