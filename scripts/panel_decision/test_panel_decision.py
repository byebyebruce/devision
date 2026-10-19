"""EXP76 preparation (PANEL-DECISION-PREP-01), CPU only: the config is EXP74 D2's with only name / start changed; the
preflight refuses another R, EXP75 evidence that does not name R worth a decision diagnostic, and an EXP76 directory
that already exists, and passes on the real files; the runner's order, one start, refusal and limits."""
import hashlib
import importlib.util
import json
import os
import shutil
import stat
import subprocess

import pytest
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def load():
    spec = importlib.util.spec_from_file_location("pd_preflight", os.path.join(HERE, "preflight.py"))
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_config_is_d2_with_another_start():
    a, b = (yaml.safe_load(open(os.path.join(ROOT, c))) for c in ("configs/exp74-panel-d2.yaml", "configs/exp76-panel-decision.yaml"))
    assert b["stages"][0].pop("init") == "runs/exp75-panel-r/r" and a["stages"][0].pop("init") == "runs/exp74-panel-d1/d1"
    a.pop("name"), b.pop("name")
    assert a == b


@pytest.fixture
def pf(tmp_path, monkeypatch):
    m = load()
    monkeypatch.chdir(ROOT)
    r = tmp_path / "r"
    r.mkdir()
    for k in ("model.safetensors", "config.json", "align_report.json"):
        (r / k).write_text(k)
    rr = {"path": str(r), **{k: hashlib.sha256(k.encode()).hexdigest() for k in ("model.safetensors", "config.json", "align_report.json")}}
    cmp = tmp_path / "compare.json"
    cmp.write_text(json.dumps({"arms": {"r": {"status": "completed", "worth_decision_diagnostic": True}}}))
    monkeypatch.setattr(m, "R", rr)
    monkeypatch.setattr(m, "EXP75", {str(cmp): m.sha(str(cmp))})
    monkeypatch.setattr(m, "FRESH", (str(tmp_path / "exp76"),))
    real_load = json.load

    def load_json(f):
        if getattr(f, "name", "") == "runs/panel-number/compare.json":
            return real_load(open(cmp))
        return real_load(f)
    monkeypatch.setattr(m.json, "load", load_json)
    cfg = tmp_path / "cfg.yaml"
    c = yaml.safe_load(open("configs/exp76-panel-decision.yaml"))
    c["stages"][0]["init"] = str(r)
    cfg.write_text(yaml.safe_dump(c))
    monkeypatch.setattr(m, "CONFIG", str(cfg))
    return m, tmp_path


def test_preflight_good(pf):
    m, _ = pf
    assert m.problems() == []


@pytest.mark.parametrize("case", ["other_r", "not_worth", "fresh"])
def test_preflight_refusals(pf, case):
    m, tmp = pf
    if case == "other_r":
        (tmp / "r" / "config.json").write_text("another config")
    elif case == "not_worth":
        (tmp / "compare.json").write_text(json.dumps({"arms": {"r": {"status": "completed", "worth_decision_diagnostic": False}}}))
        m.EXP75 = {str(tmp / "compare.json"): m.sha(str(tmp / "compare.json"))}
    else:
        (tmp / "exp76").mkdir()
    assert m.problems()


def test_preflight_on_the_real_files(tmp_path, monkeypatch):
    m = load()
    monkeypatch.chdir(ROOT)
    assert m.problems() == []


STUB = """#!/bin/bash
echo "uv $*" >> "${MARK_CALLS}"
case "$*" in
  *"preflight.py"*) [ -n "${PF_FAIL:-}" ] && exit 12; echo '{"ok": true}' > runs/exp76-preflight.json; exit 0;;
esac
exit 0
"""


def test_runner(tmp_path):
    root = tmp_path / "repo"
    for rel in ("scripts/panel_decision/run.sh", "scripts/panel_decision/preflight.py", "scripts/panel_learn/report.py", "configs/exp76-panel-decision.yaml",
                "scripts/exp35/guard.sh", "scripts/exp38/steps.sh"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join(ROOT, rel), root / rel)
    (root / "data/panel_learn").mkdir(parents=True)
    (root / "data/panel_learn/MANIFEST.json").write_text("{}")
    (root / "runs").mkdir()
    b = tmp_path / "bin"
    b.mkdir()
    # the process guards see no model process (other processes on this machine may name these paths; the guard's own
    # detection is tested where it was introduced)
    for name, body in {"uv": STUB, "pgrep": "#!/bin/bash\nexit 1\n", "ps": "#!/bin/bash\nexit 0\n"}.items():
        (b / name).write_text(body)
        (b / name).chmod((b / name).stat().st_mode | stat.S_IEXEC)
    env = {**os.environ, "PATH": "%s:%s" % (b, os.environ["PATH"]), "MARK_CALLS": str(tmp_path / "calls"), "EXP76_LIMIT": "60", "EXP76_RESERVE": "4"}
    p = subprocess.run(["bash", str(root / "scripts/panel_decision/run.sh")], env={**env, "PF_FAIL": "1"}, capture_output=True, text=True, timeout=60)
    calls = (tmp_path / "calls").read_text().splitlines()
    assert p.returncode == 12 and "nothing trained" in p.stdout and not any("devision-pipeline" in c for c in calls)
    assert not (root / "runs/exp76-panel-decision.started").exists()
    (tmp_path / "calls").unlink()
    p = subprocess.run(["bash", str(root / "scripts/panel_decision/run.sh")], env=env, capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stdout + p.stderr
    calls = (tmp_path / "calls").read_text().splitlines()
    keys = ["preflight.py", "devision-pipeline configs/exp76-panel-decision.yaml",
            "report.py d2 runs/exp76-panel-decision runs/v04-k/eval configs/exp76-panel-decision.yaml"]
    pos = [next(i for i, c in enumerate(calls) if k in c) for k in keys]
    assert pos == sorted(pos) and not any("exp74-panel-d2" in c for c in calls)
    p2 = subprocess.run(["bash", str(root / "scripts/panel_decision/run.sh")], env=env, capture_output=True, text=True)
    assert p2.returncode == 1 and "already started once" in p2.stdout
    s = open(os.path.join(HERE, "run.sh")).read()
    assert "EXP76_LIMIT:-5700" in s and "EXP76_RESERVE:-120" in s
