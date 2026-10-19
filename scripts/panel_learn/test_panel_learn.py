"""EXP74 report rules and runner (PANEL-LEARN-PREP-01), CPU, no model: D1 gate exit codes; D2 outcomes A / B / C /
invalid from fabricated monitor records; the runner's order, one start per phase, and D2 refused without D1's gate."""
import importlib.util
import json
import os
import shutil
import stat
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def load():
    spec = importlib.util.spec_from_file_location("panel_report", os.path.join(HERE, "report.py"))
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def syn(prefix, fams, p_read, p_cmp, height=16):
    """readA / readB / cmp records, both value views; a family's questions right with the given shares (by family)."""
    out = []
    for i in range(fams):
        for v in (0, 1):
            sid = "exp21:%s%04d:v%d:h%d" % (prefix, i, v, height)
            for q, p in (("readA", p_read), ("readB", p_read), ("cmp", p_cmp)):
                out.append({"sample_id": sid, "qid": q, "correct": (i % 100) < p * 100})
    return out


def plain(n, p):
    return [{"sample_id": "s%d" % i, "qid": "q", "correct": i < p * n} for i in range(n)]


def write(d2, fit, erased, rs_temp=1.0, exist=0.95):
    step = os.path.join(d2, "d2", "monitor", "step1050")
    os.makedirs(step)
    sets = {"val": syn("t", 200, fit, fit), "dev16": syn("d", 200, 0.5, 0.5), "erased": syn("e", 100, erased, erased, height=0),
            "k_text": plain(240, 1.0), "rs_temp": plain(41, rs_temp), "rs_force": plain(120, 0.5), "rs_temp_mism": plain(41, 0.4),
            "rs_force_mism": plain(120, 0.4)}
    for n, rs in sets.items():
        open(os.path.join(step, n + ".details.jsonl"), "w").write("".join(json.dumps(r) + "\n" for r in rs))
    os.makedirs(os.path.join(d2, "eval"))
    for n, a in (("exist", exist), ("science", 0.645)):
        json.dump({"accuracy_all": a}, open(os.path.join(d2, "eval", n + ".json"), "w"))


@pytest.fixture
def sealed(tmp_path):
    s = tmp_path / "sealed"
    s.mkdir()
    for n, a in (("exist", 0.9457), ("science", 0.6447)):
        json.dump({"accuracy_all": a}, open(s / (n + ".json"), "w"))
    return str(s)


@pytest.mark.parametrize("fit,erased,rs_temp,exist,want", [
    (0.95, 0.2, 1.0, 0.95, "A:"), (0.95, 0.2, 0.8, 0.95, "B:"), (0.95, 0.2, 1.0, 0.90, "B:"), (0.3, 0.2, 1.0, 0.95, "C:"),
    (0.95, 0.7, 1.0, 0.95, "invalid")])
def test_d2_outcomes(tmp_path, sealed, fit, erased, rs_temp, exist, want):
    m = load()
    write(str(tmp_path / "d2"), fit, erased, rs_temp, exist)
    out = m.d2_report(str(tmp_path / "d2"), sealed)
    assert out["outcome"].startswith(want), out


def test_d1_gate(tmp_path):
    m = load()
    for gap, code in ((0.31, 0), (0.29, 10)):
        p = tmp_path / ("r%s.json" % gap)
        json.dump({"val": [{"val/mlm_acc_real": 0.2, "val/mlm_acc_shuffled": 0.1, "val/mlm_acc_gap": 0.1, "step": 0},
                           {"val/mlm_acc_real": 0.9, "val/mlm_acc_shuffled": 0.9 - gap, "val/mlm_acc_gap": gap, "step": 1250}]}, open(p, "w"))
        assert m.main(["d1", str(p)]) == code


STUB = """#!/bin/bash
echo "uv $*" >> "${MARK_CALLS}"
case "$*" in
  *"report.py d1"*) [ -n "${GATE_FAIL:-}" ] && exit 10; exit 0;;
esac
exit 0
"""


def chain(tmp_path, env_extra):
    root = tmp_path / "repo"
    for rel in ("scripts/panel_learn/run.sh", "scripts/exp35/guard.sh", "scripts/exp38/steps.sh", "configs/exp74-panel-d1.yaml",
                "configs/exp74-panel-d2.yaml", "scripts/panel_learn/report.py"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join(ROOT, rel), root / rel)
    (root / "data/panel_learn").mkdir(parents=True)
    (root / "data/panel_learn/MANIFEST.json").write_text("{}")
    b = tmp_path / "bin"
    b.mkdir(exist_ok=True)
    for name, body in {"uv": STUB, "pgrep": "#!/bin/bash\nexit 1\n"}.items():
        (b / name).write_text(body)
        (b / name).chmod((b / name).stat().st_mode | stat.S_IEXEC)
    env = {**os.environ, "PATH": "%s:%s" % (b, os.environ["PATH"]), "MARK_CALLS": str(tmp_path / "calls"), "EXP74_LIMIT": "60", "EXP74_RESERVE": "4", **env_extra}
    p = subprocess.run(["bash", str(root / "scripts/panel_learn/run.sh")], env=env, capture_output=True, text=True, timeout=120)
    calls = (tmp_path / "calls").read_text().splitlines() if (tmp_path / "calls").exists() else []
    return p, calls, root


def test_runner(tmp_path):
    p, calls, root = chain(tmp_path / "a", {"EXP74_PHASE": "d1"})
    assert p.returncode == 0, p.stdout + p.stderr
    order = ["consistency.py", "exp74-panel-d1.yaml", "report.py d1"]
    pos = [next(i for i, c in enumerate(calls) if k in c) for k in order]
    assert pos == sorted(pos) and not any("exp74-panel-d2" in c for c in calls)
    p2 = subprocess.run(["bash", str(root / "scripts/panel_learn/run.sh")], env={**os.environ, "EXP74_PHASE": "d1"}, capture_output=True, text=True)
    assert p2.returncode == 1 and "already started once" in p2.stdout
    p, calls, _ = chain(tmp_path / "b", {"EXP74_PHASE": "d2", "GATE_FAIL": "1"})
    assert p.returncode == 1 and "no D2" in p.stdout and not any("exp74-panel-d2.yaml" in c for c in calls)
    p, calls, _ = chain(tmp_path / "c", {"EXP74_PHASE": "d2"})
    assert p.returncode == 0 and any("exp74-panel-d2.yaml" in c for c in calls) and any("report.py d2" in c for c in calls)
    p, calls, _ = chain(tmp_path / "d", {"EXP74_PHASE": "x"})
    assert p.returncode == 2 and not calls


def test_limits():
    s = open(os.path.join(HERE, "run.sh")).read()
    assert "EXP74_LIMIT:-4500" in s and "EXP74_LIMIT:-5700" in s and "EXP74_RESERVE:-120" in s
