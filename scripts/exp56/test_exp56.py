"""exp56 runner without the real model: start refuses a selector / head that is not the registered one; the read
fidelity rule reproduces on EXP53's own strings and fails when 2% differ; the run finalizes inside its limit through the
real watch."""
import importlib.util
import json
import os
import sys

import pytest

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.dirname(HERE))
spec = importlib.util.spec_from_file_location("exp56", os.path.join(HERE, "exp56.py"))
assert spec is not None and spec.loader is not None
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
e54 = m.e54


def test_start_refuses_other_components(tmp_path, monkeypatch):
    fake = tmp_path / "selector.pt"
    fake.write_bytes(b"not the selector")
    monkeypatch.setattr(m, "SELECTOR", str(fake))
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        with pytest.raises(SystemExit, match="selector"):
            m.stage_start(0, out=str(tmp_path / "out"))
    finally:
        os.chdir(cwd)
    assert not (tmp_path / "out" / "START.json").exists()


def test_read_fidelity_rule(tmp_path):
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        rows = {r["id"]: r for r in map(json.loads, open(e54.TUNE))}
        qs = [json.loads(x) for x in open(e54.QUESTIONS)]
        pred, _, _ = e54.p49.read_details(os.path.join(m.E53, "details_auto.jsonl"), qs)
        per = {}
        for q, p in zip(qs, pred):
            if q["set"] == "rl_tune":
                d = e54.p49.decode(p)
                t = rows[q["id"]]["questions"][q["qid"]]["instructions"]
                per.setdefault(q["id"], []).append({"question": t, "decoded": d["text"], "valid": d["valid"]})
        path = tmp_path / "reads.jsonl"
        path.write_text("".join(json.dumps({"id": k, "reads": v}) + "\n" for k, v in per.items()))
        ok = m.read_fidelity(rows, str(path))
        assert ok["questions"] == 1800 and ok["same_strings"] == 1.0 and ok["reproduced"]
        n = 0
        for v in per.values():                                         # 2% of the strings changed
            for r in v:
                if n < 36:
                    r["decoded"], r["valid"] = "9999", True
                    n += 1
        path.write_text("".join(json.dumps({"id": k, "reads": v}) + "\n" for k, v in per.items()))
        assert not m.read_fidelity(rows, str(path))["reproduced"]
    finally:
        os.chdir(cwd)


STAGE = """import signal, subprocess, sys, time
child = subprocess.Popen([sys.executable, '-c', 'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(300)'])
open('runs/exp56/child.pid', 'w').write(str(child.pid))
time.sleep(300)
"""
STUB = """#!/bin/bash
echo "uv $*" >> "${MARK}"
case "$*" in
  *"watch.py"*) shift 2; exec python3 "$@";;
  *"exp56.py start"*) exit 0;;
  *"exp56.py live"*) exec python3 "${STAGE}";;
  *"exp56.py report --finalize"*) echo final > runs/exp56/report.txt; exit 0;;
esac
exit 9
"""


def test_run_stops_at_the_watch_deadline_and_finalizes(tmp_path):
    import shutil
    import stat
    import subprocess
    import time
    root = tmp_path / "repo"
    for rel in ("scripts/exp56/run.sh", "scripts/exp53/watch.py", "scripts/exp35/guard.sh", "scripts/exp38/steps.sh"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join(ROOT, rel), root / rel)
    (tmp_path / "stage.py").write_text(STAGE)
    b = tmp_path / "bin"
    b.mkdir()
    for name, body in {"uv": STUB, "pgrep": '#!/bin/bash\n[ "$1" = "-fl" ] && exit 1\nexec /usr/bin/pgrep "$@"\n'}.items():
        (b / name).write_text(body)
        (b / name).chmod((b / name).stat().st_mode | stat.S_IEXEC)
    env = {**os.environ, "PATH": "%s:%s" % (b, os.environ["PATH"]), "MARK": str(tmp_path / "calls"), "STAGE": str(tmp_path / "stage.py"),
           "EXP56_LIMIT": "60", "EXP56_RESERVE": "4"}
    t = time.time()
    p = subprocess.run(["bash", str(root / "scripts/exp56/run.sh")], env=env, capture_output=True, text=True, timeout=120)
    assert p.returncode == 124 and time.time() - t < 60, p.stdout + p.stderr
    assert "report --finalize" in (tmp_path / "calls").read_text()
    with pytest.raises(ProcessLookupError):
        os.kill(int((root / "runs/exp56/child.pid").read_text()), 0)
