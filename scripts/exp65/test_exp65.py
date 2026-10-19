# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false
"""exp65 without a model: the six rule classes and the W weights (train counts only, totals 1,200 / 9,600, every label
the generator's rule); the diagnostic (480 requests, 40 new canonical cases per class, A / B swapped, labels by rule,
no existing case); the arms differ only in the weight file; W - C on the conflict classes; the practical condition and
the decision; the run's order and bounded finalize."""
import importlib.util
import json
import os
from collections import Counter

import pytest

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.dirname(HERE))
spec = importlib.util.spec_from_file_location("exp65", os.path.join(HERE, "exp65.py"))
assert spec is not None and spec.loader is not None
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
b = m.b65
has = os.path.exists(os.path.join(ROOT, m.D65, "MANIFEST.json"))
data = pytest.mark.skipif(not has, reason="data/exp65 not built")


@pytest.fixture(autouse=True)
def in_root(monkeypatch):
    monkeypatch.chdir(ROOT)


def test_rule_classes():
    assert b.class_of((5, 100), (5, 100)) == "equal" and b.class_of((5, 100), (5, 200)) == "same_mass"
    assert b.class_of((5, 100), (9, 100)) == "same_speed" and b.class_of((9, 200), (5, 100)) == "same_direction"
    assert b.class_of((45, 1480), (5, 1890)) == "opposite_mass_wins"           # 45 * 1480^2 > 5 * 1890^2
    assert b.class_of((10, 100), (5, 1000)) == "opposite_speed_wins"
    assert b.winner((45, 1480), (5, 1890)) == "sample A" and b.winner((5, 100), (5, 100)) == "neither"


@data
def test_weights():
    w = m.jsonl(os.path.join(m.D65, "weights.jsonl"))
    n = Counter(x["class"] for x in w if "class" in x)
    assert dict(n) == {"equal": 139, "same_mass": 469, "same_speed": 304, "same_direction": 140, "opposite_mass_wins": 52, "opposite_speed_wins": 96}
    assert sum(x["weight"] for x in w) == pytest.approx(9600) and sum(x["weight"] for x in w if "class" in x) == pytest.approx(1200)
    assert all(x["weight"] == pytest.approx(200 / n[x["class"]]) for x in w if "class" in x) and all(x["weight"] == 1.0 for x in w if "class" not in x)
    train = m.jsonl(os.path.join(m.e63.FACTS, "train.jsonl"))
    assert {(x["id"], x["qid"]) for x in w} == {(r["id"], q) for r in train for q in r["questions"]} and len(w) == 9600


@data
def test_diagnostic():
    d = m.jsonl(os.path.join(m.D65, "diag.jsonl"))
    assert len(d) == 480 and Counter(r["axis"] for r in d) == {c: 80 for c in b.CLASSES}
    fam = Counter(r["family"] for r in d)
    assert len(fam) == 240 and set(fam.values()) == {2}
    seen = b.existing_cases("data")
    for r in d:
        a, bb = b.state_values(r["state_text"])
        assert b.class_of(a, bb) == r["axis"] and b.canonical(a, bb) not in seen
        g = next(o for o, p in r["gold"]["q0"]["probabilities"].items() if p == 1.0)
        want = b.winner(a, bb)
        assert (g.startswith("neither") if want == "neither" else g == want)
    by = {}
    for r in d:
        by.setdefault(r["family"], []).append(r)
    for rs in by.values():                                                       # A / B swapped; equal: options reordered
        x, y = rs
        if x["axis"] == "equal":
            assert x["state_text"] == y["state_text"]
        else:
            assert b.state_values(x["state_text"]) == tuple(reversed(b.state_values(y["state_text"])))


def test_conflict_difference_and_decision():
    rows = {}
    w, c = [], []
    for k in range(10):
        for cl in m.CONFLICT:
            rid = "%s:%d" % (cl, k)
            rows[rid] = {"axis": cl, "family": "%s:%d" % (cl, k)}
            w.append({"sample_id": rid, "qid": "q0", "correct": True})
            c.append({"sample_id": rid, "qid": "q0", "correct": k < 5})
    d = m.conflict_difference(w, c, rows)
    assert d["value"] == pytest.approx(0.5) and d["ci95"][0] > 0
    assert m.conflict_difference(w, w, rows)["ci95"] == [0.0, 0.0]
    assert m.decision(True, True, True).endswith("W (W - C conflict lower bound > 0)")
    assert "C (no reliable W advantage" in m.decision(True, True, False)
    assert m.decision(True, False, False).startswith("C meets")
    assert m.decision(False, True, False).startswith("only W meets") and m.decision(False, True, True).startswith("W meets")
    assert m.decision(False, False, True).startswith("neither") and m.decision("incomplete", True, True).startswith("incomplete")


def test_practical_condition():
    g = {k: True for k in ("fit_temp", "fit_force", "G_temp", "G_force", "P_temp", "reading", "screens")}
    diag = {c: {"accuracy": 0.85} for c in b.CLASSES}
    assert m.practical(g, diag, {"ok": True}) is True
    assert m.practical(dict(g, fit_temp=False), diag, {"ok": True}) is False
    assert m.practical(g, dict(diag, opposite_mass_wins={"accuracy": 0.79}), {"ok": True}) is False
    assert m.practical(g, diag, {"ok": False}) is False
    assert m.practical(g, None, {"ok": True}) == "incomplete" and m.practical(dict(g, P_temp="incomplete"), diag, {"ok": True}) == "incomplete"


def test_arms_differ_only_in_the_weight_file(tmp_path, monkeypatch):
    m.check_yamls()
    y = open(m.YAML["w"]).read().replace("lr_head: 1.0e-05", "lr_head: 2.0e-05")
    (tmp_path / "w.yaml").write_text(y)
    monkeypatch.setattr(m, "YAML", dict(m.YAML, w=str(tmp_path / "w.yaml")))
    with pytest.raises(SystemExit):
        m.check_yamls()
    c = open(m.YAML["c"]).read().replace("    seed: 1\n", "    seed: 1\n    question_weight_file: data/exp65/weights.jsonl\n")
    (tmp_path / "c.yaml").write_text(c)
    monkeypatch.setattr(m, "YAML", {"c": str(tmp_path / "c.yaml"), "w": "configs/exp65-w.yaml"})
    with pytest.raises(SystemExit):
        m.check_yamls()


STUB = """#!/bin/bash
echo "uv $*" >> "${MARK_CALLS}"
case "$*" in
  *"watch.py"*) shift 2; exec python3 "$@";;
  *"exp65.py start"*) mkdir -p runs/exp65; exit 0;;
  *"exp65.py train_w"*) [ -n "${FAIL_W:-}" ] && exit 7; exit 0;;
  *"exp65.py train_c"*|*"exp65.py freeze_"*|*"exp65.py decide"*|*"devision-eval"*|*"freeze.py record"*) exit 0;;
  *"exp65.py report"*) echo report > runs/exp65/report.txt; exit 0;;
esac
exit 9
"""


def chain(tmp_path, extra):
    import shutil
    import stat
    import subprocess
    root = tmp_path / "repo"
    for rel in ("scripts/exp65/run.sh", "scripts/exp53/watch.py", "scripts/exp35/guard.sh", "scripts/exp38/steps.sh"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join(ROOT, rel), root / rel)
    bdir = tmp_path / "bin"
    bdir.mkdir()
    for name, body in {"uv": STUB, "pgrep": '#!/bin/bash\n[ "$1" = "-fl" ] && exit 1\nexec /usr/bin/pgrep "$@"\n'}.items():
        (bdir / name).write_text(body)
        (bdir / name).chmod((bdir / name).stat().st_mode | stat.S_IEXEC)
    env = {**os.environ, "PATH": "%s:%s" % (bdir, os.environ["PATH"]), "MARK_CALLS": str(tmp_path / "calls"), "EXP65_LIMIT": "60", "EXP65_RESERVE": "4", **extra}
    p = subprocess.run(["bash", str(root / "scripts/exp65/run.sh")], env=env, capture_output=True, text=True, timeout=120)
    return p, (tmp_path / "calls").read_text().splitlines()


def test_run_order_and_failed_stage(tmp_path):
    p, calls = chain(tmp_path / "a", {})
    assert p.returncode == 0, p.stdout + p.stderr
    first = lambda s: next(i for i, c in enumerate(calls) if s in c)  # noqa: E731
    assert first("exp65.py train_c") < first("exp65.py freeze_c") < first("exp65.py train_w") < first("exp65.py freeze_w") \
        < first("exp65.py decide") < first("exp65-c/eval/exist") < first("exp65-w/eval/science") < first("exp65.py report")
    p, calls = chain(tmp_path / "b", {"FAIL_W": "1"})
    joined = "\n".join(calls)
    assert p.returncode == 7 and "exp65.py decide" not in joined and "report --finalize" in joined


def fake_run(tmp_path, monkeypatch, start: str):
    """report() with both arms faked to pass every condition and EXP63's start-point diagnostic written for real
    (write_set, registry, identity) and then broken per `start` (critic EXP65-R1-01)."""
    out = tmp_path / "exp65"
    out.mkdir()
    (out / "START.json").write_text("{}")
    rows = {}
    for cl in b.CLASSES:
        for k in range(2):
            rid = "%s:%d" % (cl, k)
            rows[rid] = {"id": rid, "axis": cl, "family": rid, "image_id": rid, "state_text": "s",
                         "questions": {"q0": {"criteria": {"sample A": None, "sample B": None}}},
                         "gold": {"q0": {"probabilities": {"sample A": 1.0, "sample B": 0.0}}}}
    inputs = {rid: {"state": "s"} for rid in rows}
    recs = [{"sample_id": rid, "qid": "q0", "image_id": rid, "gold": r["gold"]["q0"]["probabilities"], "options": ["sample A", "sample B"],
             "probabilities": {"sample A": 0.9, "sample B": 0.1}, "prediction": "sample A", "gold_answer": "sample A", "correct": True,
             "nll": 0.1} for rid, r in rows.items()]
    ident = {"set": "exp63_diag", "start_sha256": m.sha(str(out / "START.json")), "inputs_sha256": m.e57.hash_inputs(inputs),
             "model": dict(m.E63_SHA), "temperature": "none (raw)", "image": None}
    if start == "identity":
        ident["model"] = {"model.safetensors": "0" * 64}
    if start != "missing":
        m.e57.write_set(str(out), "exp63_diag", recs, ident)
    if start == "unregistered":
        reg = json.load(open(out / "registry.json"))
        reg.pop("decide_exp63_diag")
        json.dump(reg, open(out / "registry.json", "w"))
    if start == "changed":
        with open(out / "details_exp63_diag.jsonl", "a") as fh:
            fh.write(json.dumps(recs[0]) + "\n")
    real = m.e57.read_set
    monkeypatch.setattr(m, "check_start", lambda o, code=False: {"exp57_screens": {}})
    monkeypatch.setattr(m.e63, "set_inputs", lambda c: ({}, {}))
    monkeypatch.setattr(m, "inputs_of", lambda c: (rows, inputs))
    monkeypatch.setattr(m.e57, "read_set", lambda o, name, *a: real(o, name, *a) if name == "exp63_diag" else recs if name.endswith("_diag") else [])
    monkeypatch.setattr(m.fz, "check", lambda run, evals: {"weights": {"model.safetensors": run}})
    monkeypatch.setattr(m.e57, "SCREENS", {})
    monkeypatch.setattr(m.e54, "QUESTIONS", str(tmp_path / "none.jsonl"))
    monkeypatch.setattr(m.e54, "TUNE", str(tmp_path / "none.jsonl"))
    (tmp_path / "none.jsonl").write_text("")
    monkeypatch.setattr(m.e54, "load_inputs", lambda *a: {})
    monkeypatch.setattr(m.e57, "reading_summary", lambda *a: {})
    monkeypatch.setattr(m, "arm_gates", lambda a: {"fit_temp": True})
    monkeypatch.setattr(m, "practical", lambda g, diag, vs: True)
    return m.report(str(out))


@pytest.mark.parametrize("start", ["missing", "unregistered", "changed", "identity"])
def test_no_route_without_a_valid_start_diagnostic(tmp_path, monkeypatch, start):
    res = fake_run(tmp_path, monkeypatch, start)
    assert res["arms"]["c"]["practical"] is True and res["arms"]["w"]["practical"] is True     # the arms stay reported
    assert "diag" in res["arms"]["c"] and "w_minus_c_conflict" in res
    want = "missing" if start == "missing" else "invalid"
    assert res["states"]["exp63_diag"].startswith(want) and "exp63_diag" not in res
    assert res["decision"].startswith("incomplete" if start == "missing" else "invalid") and "no route decision" in res["decision"]


def test_a_complete_start_diagnostic_restores_the_decision(tmp_path, monkeypatch):
    res = fake_run(tmp_path, monkeypatch, "complete")
    assert res["states"]["exp63_diag"] == "completed" and res["exp63_diag"]["conflict_macro"] == 1.0
    assert res["decision"] == m.decision(True, True, res["attribution"]) and res["decision"].startswith("both arms meet")


def test_run_limit_default_is_100_minutes():
    s = open("scripts/exp65/run.sh").read()
    assert "${EXP65_LIMIT:-6000}" in s and "start + 100" in s and "120 s for the report" in s
