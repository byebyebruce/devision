"""scripts/calib/calib.py (critic V04-K-PREP-02-ACCEPT-01: the calibration runner) on tiny files in a scratch workspace:
a LIVE-K-like source verified, source / clone, the real ordinary calibration on a tiny val set (progress per forward chunk,
report moved out of the candidate, release name set, candidate checked), a tiny rl_calib by the real builder, the temporary
bundle, collect with the identity and runtime registered, analyse, and a report whose branch comes from the recomputed
analysis; a missing stage is incomplete, a changed record invalid, a wrong source refused at start; run.sh's order, limit
and reserve."""
import importlib.util
import json
import os
import shutil
import subprocess
import sys

import pytest
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture
def ws(tmp_path, monkeypatch):
    """A scratch workspace with a LIVE-K-like run, a tiny val set and the runner pointed at them."""
    ks = load("ks_helpers", "tests/release/test_k_source.py")
    monkeypatch.delitem(sys.modules, "build", raising=False)
    m = load("calib_runner", "scripts/calib/calib.py")
    pins = ks.livek_run(tmp_path)
    monkeypatch.chdir(tmp_path)
    os.makedirs("data/img")
    rows = []
    for i in range(8):
        Image.new("RGB", (40, 30), [(200, 30, 30), (30, 30, 200)][i % 2]).save("data/img/%d.png" % i)
        gold = ["red", "blue"][i % 2]
        rows.append({"id": "v%d" % i, "image": "img/%d.png" % i, "image_id": "v%d" % i,
                     "questions": {"q": {"type": "choice", "instructions": "what color is it ?", "criteria": {"red": None, "blue": None, "green": None}}},
                     "gold": {"q": {"probabilities": {"red": float(gold == "red"), "blue": float(gold == "blue"), "green": 0.0}}}})
    open("data/val.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(m, "VAL", "data/val.jsonl")
    monkeypatch.setattr(m, "VAL_QUESTIONS", 8)
    monkeypatch.setattr(m, "PINS", pins)
    monkeypatch.setattr(m, "LIVEK", {"run": str(tmp_path / "livek"), "decision": str(tmp_path / "end"), "bundle": str(tmp_path / "bundle_k")})
    monkeypatch.setattr(m, "RL_WANT", {"temp": 6, "force": 6})

    def tiny_rl_calib():                     # the real builder, then short options / hint for the tiny model
        rl = load("calib_rl_t", "scripts/release/rl_calib.py")
        monkeypatch.setattr(rl, "used_cases", lambda root: set())
        man = rl.build("data", want={"temp": 6, "force": 6})
        p = "data/rl_calib/questions.jsonl"
        rs = [json.loads(x) for x in open(p)]
        short = ["red", "blue", "green"]
        for r in rs:
            opts = list(r["questions"]["q0"]["criteria"])
            r["state_text"] = ""
            r["questions"]["q0"]["criteria"] = {short[i]: None for i in range(len(opts))}
            r["gold"]["q0"]["probabilities"] = {short[i]: r["gold"]["q0"]["probabilities"][o] for i, o in enumerate(opts)}
        open(p, "w").write("".join(json.dumps(r) + "\n" for r in rs))
        return man
    monkeypatch.setattr(m, "build_rl_calib", tiny_rl_calib)
    return m, pins


def run_chain(m, skip=()):
    import time
    m.stage_start(int(time.time()) + 2700)
    for s, fn in (("calibrate", lambda: m.stage_calibrate(device="cpu")), ("rl_calib", m.stage_rl_calib), ("temp_bundle", m.stage_temp_bundle),
                  ("collect", lambda: m.stage_collect(device="cpu")), ("analyse", m.stage_analyse)):
        if s in skip:
            break
        fn()
    return m.report()


def test_the_whole_chain_on_tiny_files(ws):
    m, _ = ws
    res = run_chain(m)
    assert all(v == "completed" for v in res["states"].values()), res["states"]
    assert res["branch"][0] in "1234" and "not a candidate" in res["branch"] and res["candidate_eligible"] is False
    cand = os.path.join(m.ROUND, "cand")
    assert not os.path.exists(os.path.join(cand, "calibrate_report.json")) and os.path.exists(os.path.join(m.OUT, "calibrate_report.json"))
    assert json.load(open(os.path.join(cand, "config.json")))["model_name"] == m.RELEASE_NAME
    assert not os.path.exists(os.path.join(cand, "reading"))                         # nothing bound here
    lines = open(os.path.join(m.OUT, "progress-calibrate.txt")).read().splitlines()
    assert lines and lines[-1].split()[1] == "8"                                      # progress reached every question
    assert open(os.path.join(m.OUT, "progress-collect.txt")).read().splitlines()[-1].split()[1] == "24"


def test_missing_stage_is_incomplete(ws):
    m, _ = ws
    res = run_chain(m, skip=("collect",))
    assert res["branch"].startswith("incomplete") and res["states"]["collect"] == "missing"


def test_a_changed_record_is_invalid(ws):
    m, _ = ws
    run_chain(m)
    with open(os.path.join(m.OUT, "calib", "records.jsonl"), "a") as fh:
        fh.write("\n")
    res = m.report()
    assert res["branch"].startswith("invalid")


def test_start_refuses_another_source(ws, monkeypatch):
    m, pins = ws
    monkeypatch.setattr(m, "PINS", dict(pins, **{"report.json": "0" * 64}))
    with pytest.raises(SystemExit, match="does not verify"):
        m.stage_start(0)


STUB = """#!/bin/bash
echo "uv $*" >> "${MARK_CALLS}"
case "$*" in
  *"watch.py"*) shift 2; exec python3 "$@";;
  *"calib.py start"*) mkdir -p runs/v04-k-calib; exit 0;;
  *"calib.py collect"*) [ -n "${FAIL_COLLECT:-}" ] && exit 4; exit 0;;
  *"calib.py report"*) echo "report $*" > runs/v04-k-calib/report.txt; exit 0;;
  *"calib.py "*) exit 0;;
esac
exit 9
"""


def chain(tmp_path, extra):
    import stat as st_
    root = tmp_path / "repo"
    for rel in ("scripts/calib/run.sh", "scripts/exp53/watch.py", "scripts/exp35/guard.sh", "scripts/exp38/steps.sh"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join(ROOT, rel), root / rel)
    bdir = tmp_path / "bin"
    bdir.mkdir()
    for name, body in {"uv": STUB, "pgrep": '#!/bin/bash\n[ "$1" = "-fl" ] && exit 1\nexec /usr/bin/pgrep "$@"\n'}.items():
        (bdir / name).write_text(body)
        (bdir / name).chmod((bdir / name).stat().st_mode | st_.S_IEXEC)
    env = {**os.environ, "PATH": "%s:%s" % (bdir, os.environ["PATH"]), "MARK_CALLS": str(tmp_path / "calls"), "CALIB_LIMIT": "60",
           "CALIB_RESERVE": "4", **extra}
    p = subprocess.run(["bash", str(root / "scripts/calib/run.sh")], env=env, capture_output=True, text=True, timeout=120)
    return p, (tmp_path / "calls").read_text().splitlines()


def test_run_order_failure_and_limit(tmp_path):
    p, calls = chain(tmp_path / "a", {})
    assert p.returncode == 0, p.stdout + p.stderr
    order = ["start", "calibrate", "rl_calib", "temp_bundle", "collect", "analyse", "report"]
    pos = [next(i for i, c in enumerate(calls) if "calib.py %s" % s in c) for s in order]
    assert pos == sorted(pos)
    p, calls = chain(tmp_path / "b", {"FAIL_COLLECT": "1"})
    assert p.returncode == 4 and "report --finalize" in "\n".join(calls) and not any("calib.py analyse" in c for c in calls)
    s = open(os.path.join(ROOT, "scripts/calib/run.sh")).read()
    assert "${CALIB_LIMIT:-2700}" in s and "${CALIB_RESERVE:-180}" in s
