"""exp54 without the real model: P never reads gold (changing gold changes only G); an invalid reading stays
"unreadable"; option order / gold do not change any condition's table; D is the twin picture's P table and a twin with
other rows or twins on two pictures are refused; the ordinary decide path (tiny model, image=None, raw temperature)
gives finite normalised probabilities per original option; details with a missing / repeated question, forged
probabilities, another condition's identity or an unregistered edit are refused; the gates / branch need P and G; the
run finalizes within its limit through the real watch."""
import copy
import importlib.util
import json
import os
import sys

import pytest

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tests"))
spec = importlib.util.spec_from_file_location("exp54", os.path.join(HERE, "exp54.py"))
assert spec is not None and spec.loader is not None
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)


@pytest.fixture(scope="module")
def data():
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        qs = [json.loads(x) for x in open(e.QUESTIONS)]
        pred, _, _ = e.p49.read_details(os.path.join(e.E53, "details_auto.jsonl"), qs)
        rows = {r["id"]: r for r in map(json.loads, open(e.TUNE))}
        return qs, {q["key"]: p for q, p in zip(qs, pred)}, rows
    finally:
        os.chdir(cwd)


def test_p_never_reads_gold_and_invalid_stays_unreadable(data):
    qs, preds, rows = data
    base = e.build_inputs(rows, qs, preds)
    qs2 = copy.deepcopy(qs)
    for q in qs2:
        q["target"] = "999"                                                  # other gold numbers
    rows2 = copy.deepcopy(rows)
    for r in rows2.values():                                                 # other gold options, reordered options
        for qid, q in r["questions"].items():
            q["criteria"] = dict(reversed(list(q["criteria"].items())))
            r["gold"][qid]["probabilities"] = {o: 1.0 / len(q["criteria"]) for o in q["criteria"]}
    other = e.build_inputs(rows2, qs2, preds)
    for c in ("P", "D", "E"):
        assert other[c] == base[c]
    assert other["G"] != base["G"]
    k = next(q["key"] for q in qs if q["set"] == "rl_tune")
    bad = dict(preds)
    bad[k] = [1, 2, 3, 4, 5]                                                 # no EOS: invalid
    t = e.build_inputs(rows, qs, bad)["P"][k.split("|")[1]]["table"]
    assert e.UNREADABLE in [v for _, _, v in t]


def test_d_is_the_twin_table_and_bad_twins_are_refused(data):
    qs, preds, rows = data
    inp = e.build_inputs(rows, qs, preds)
    for rid, d in list(inp["D"].items())[:50]:
        assert d["table"] == inp["P"][d["twin_row"]]["table"] and d["twin_row"] != rid
    qs2 = copy.deepcopy(qs)
    first = next(q for q in qs2 if q["set"] == "rl_tune")
    other = next(q for q in qs2 if q["set"] == "rl_tune" and q["family"] != first["family"])
    first["twin"] = other["key"]                                            # a twin on another family's picture
    with pytest.raises(SystemExit):
        e.build_inputs(rows, qs2, preds)


def test_templates_only(data):
    with pytest.raises(SystemExit):
        e.rows_of(["What is the colour of sample A?"])
    assert e.rows_of(list(reversed(e.ORDER))) == e.ORDER


def tiny():
    from conftest import tiny_decider  # type: ignore
    d = tiny_decider(seed=2)
    d.cfg.temperature, d.cfg.temperature_by_options = [1.0, 1.0, 1.0], {}
    return d


ROWS = {"r%d" % i: {"id": "r%d" % i, "image_id": "i%d" % i, "kind": "temp", "axis": "x",
                    "questions": {"a": {"type": "choice", "instructions": "what color is it ?", "criteria": {"red": None, "blue": None}}},
                    "gold": {"a": {"probabilities": {"red": 1.0, "blue": 0.0}}}} for i in range(3)}


def test_ordinary_decide_text_only(tmp_path):
    d = tiny()
    inputs = {rid: {"state": "Values read from the picture:\nSample A: mass = 9"} for rid in ROWS}
    recs = e.decide_condition(d, ROWS, inputs)
    assert len(recs) == 3 and all(abs(sum(r["probabilities"].values()) - 1) < 1e-5 for r in recs)
    assert all(r["options"] == ["red", "blue"] for r in recs)


def fake_out(tmp_path, conds=("P", "G", "E", "D")):
    out = str(tmp_path / "exp54")
    os.makedirs(out, exist_ok=True)
    json.dump({"x": 1}, open(os.path.join(out, "START.json"), "w"))
    files = {}
    for c in e.CONDS:
        f = "inputs_%s.jsonl" % c
        open(os.path.join(out, f), "w").write(json.dumps({"id": "r0", "state": c}) + "\n")
        files[f] = e.sha(os.path.join(out, f))
    e.register(out, "inputs", files)
    d = tiny()
    for c in conds:
        recs = e.decide_condition(d, ROWS, {rid: {"state": c} for rid in ROWS})
        ident = {"condition": c, "start_sha256": e.started(out), "inputs_sha256": files["inputs_%s.jsonl" % c], "model": e.T_SHA,
                 "temperature": "none (raw)", "image": None}
        f = "details_%s.jsonl" % c
        with open(os.path.join(out, f), "w") as fh:
            fh.write(json.dumps({"identity": ident}) + "\n")
            fh.write("".join(json.dumps(r) + "\n" for r in recs))
        e.register(out, "decide_" + c, {f: e.sha(os.path.join(out, f))})
    return out


def rewrite(out, c, change):
    p = os.path.join(out, "details_%s.jsonl" % c)
    lines = [json.loads(x) for x in open(p)]
    change(lines)
    open(p, "w").write("".join(json.dumps(x) + "\n" for x in lines))
    e.register(out, "decide_" + c, {"details_%s.jsonl" % c: e.sha(p)})       # re-registered: the content checks act


def test_details_checks(tmp_path):
    out = fake_out(tmp_path)
    assert len(e.read_condition(out, "P", ROWS)) == 3
    cases = {
        "missing": lambda L: L.pop(),
        "repeated": lambda L: L.append(L[-1]),
        "forged": lambda L: L[1]["probabilities"].update({"red": 0.9, "blue": 0.9}),
        "other_condition": lambda L: L[0]["identity"].update(condition="G"),
        "options": lambda L: L[1].update(options=["blue", "red"]),
    }
    for name, change in cases.items():
        o = fake_out(tmp_path / name)
        rewrite(o, "P", change)
        with pytest.raises(SystemExit):
            e.read_condition(o, "P", ROWS)
    o = fake_out(tmp_path / "edited")
    open(os.path.join(o, "details_P.jsonl"), "a").write("\n")                 # edited after registration
    with pytest.raises(SystemExit, match="changed"):
        e.read_condition(o, "P", ROWS)


def test_gates_and_branch():
    full = {c: {"state": "completed"} for c in e.CONDS}
    good = {"all": 0.95, "both": 0.9, "twin": 0.9, **{"all:" + f: 0.9 for f in e.r42.FIELDS}, "read_right": {"accuracy": 0.95}}
    diff = {"P-E": {"both": {"value": 0.3, "ci95": [0.2, 0.4]}}, "P-D": {"changed": {"value": 0.5, "ci95": [0.4, 0.6]}}}
    res = {"states": full, "conditions": {"G": good, "P": good}, "differences": diff}
    res["gates"] = e.gates(res)
    assert e.branch(res).startswith("G and P pass")
    res["conditions"]["G"] = dict(good, all=0.5)
    res["gates"] = e.gates(res)
    assert e.branch(res).startswith("G below")
    res["states"] = dict(full, P={"state": "missing"})
    assert e.branch(res).startswith("incomplete")
    res = {"states": {c: {"state": "completed"} for c in e.CONDS}, "conditions": {"G": good, "P": dict(good, read_right={"accuracy": 0.5})}, "differences": diff}
    res["gates"] = e.gates(res)
    assert res["gates"]["P_use"] is False and e.branch(res).startswith("G passes, P does not")


def test_finalize_without_details(tmp_path, monkeypatch):
    out = str(tmp_path / "exp54")
    os.makedirs(out)
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        res = e.report(out, finalize=" decide=124")
    finally:
        os.chdir(cwd)
    assert all(v["state"] == "missing" for v in res["states"].values()) and res["branch"].startswith("incomplete")


STAGE = """import signal, subprocess, sys, time
child = subprocess.Popen([sys.executable, '-c', 'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(300)'])
open('runs/exp54/child.pid', 'w').write(str(child.pid))
time.sleep(300)
"""
STUB = """#!/bin/bash
echo "uv $*" >> "${MARK}"
case "$*" in
  *"watch.py"*) shift 2; exec python3 "$@";;
  *"exp54.py start"*) exit 0;;
  *"exp54.py decide"*) exec python3 "${STAGE}";;
  *"exp54.py report --finalize"*) echo final > runs/exp54/report.txt; exit 0;;
esac
exit 9
"""


def test_run_stops_at_the_watch_deadline_and_finalizes(tmp_path):
    import shutil
    import stat
    import subprocess
    import time
    root = tmp_path / "repo"
    for rel in ("scripts/exp54/run.sh", "scripts/exp53/watch.py", "scripts/exp35/guard.sh", "scripts/exp38/steps.sh"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join(ROOT, rel), root / rel)
    (tmp_path / "stage.py").write_text(STAGE)
    b = tmp_path / "bin"
    b.mkdir()
    for name, body in {"uv": STUB, "pgrep": '#!/bin/bash\n[ "$1" = "-fl" ] && exit 1\nexec /usr/bin/pgrep "$@"\n'}.items():
        (b / name).write_text(body)
        (b / name).chmod((b / name).stat().st_mode | stat.S_IEXEC)
    env = {**os.environ, "PATH": "%s:%s" % (b, os.environ["PATH"]), "MARK": str(tmp_path / "calls"), "STAGE": str(tmp_path / "stage.py"),
           "EXP54_LIMIT": "60", "EXP54_RESERVE": "4"}
    t = time.time()
    p = subprocess.run(["bash", str(root / "scripts/exp54/run.sh")], env=env, capture_output=True, text=True, timeout=120)
    assert p.returncode == 124 and time.time() - t < 60, p.stdout + p.stderr
    assert "report --finalize" in (tmp_path / "calls").read_text()
    assert "stopping: deadline" in (root / "runs/exp54/watch-decide.txt").read_text()
    child = int((root / "runs/exp54/child.pid").read_text())
    try:
        os.kill(child, 0)
        assert False, "child left"
    except ProcessLookupError:
        pass
