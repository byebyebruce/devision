# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false
"""exp63 without a model: comparisons counted by semantic class (magnets "weaker in Pair 1" = P2, same apart), per row
axis and both views of a family; a condition's records bind to their own arm (another arm's or a changed file is
refused); the gates / branch (fit first, missing never a pass or a fail); START binds EXP57's actual end; the run's order
(main conditions -> exist / science -> H) and its bounded finalize."""
import importlib.util
import json
import os

import pytest

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.dirname(HERE))
spec = importlib.util.spec_from_file_location("exp63", os.path.join(HERE, "exp63.py"))
assert spec is not None and spec.loader is not None
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
has = os.path.exists(os.path.join(ROOT, "data", "facts", "MANIFEST.json")) and os.path.exists(os.path.join(ROOT, "runs", "exp57", "FROZEN.json"))
real = pytest.mark.skipif(not has, reason="data / runs not here")


@pytest.fixture(autouse=True)
def in_root(monkeypatch):
    monkeypatch.chdir(ROOT)


INS = "Think about the magnetic force between the magnets in each pair. Which of the following statements is true?"


def frow(fam, view, gold_opt, axis):
    opts = ["The magnetic force is weaker in Pair 1", "The magnetic force is weaker in Pair 2", "The strength of the magnetic force is the same in both pairs"]
    return {"id": "rl:%s:%s" % (fam, view), "kind": "force", "family": fam, "case": "c" + fam, "axis": axis,
            "questions": {"q0": {"type": "choice", "instructions": INS, "criteria": {o: None for o in opts}}},
            "gold": {"q0": {"probabilities": {o: float(o == gold_opt) for o in opts}}}}


def rec(row, pred):
    g = next(o for o, p in row["gold"]["q0"]["probabilities"].items() if p == 1.0)
    return {"sample_id": row["id"], "qid": "q0", "gold_answer": g, "prediction": pred, "correct": pred == g}


def test_semantic_classes_axis_and_two_views():
    w1, w2, same = ("The magnetic force is weaker in Pair 1", "The magnetic force is weaker in Pair 2",
                    "The strength of the magnetic force is the same in both pairs")
    rows = {r["id"]: r for r in [frow("f1", "v0", w1, "size"), frow("f1", "v1", w2, "size"), frow("f2", "v0", same, "equal"),
                                 frow("f2", "v1", w1, "distance")]}
    recs = [rec(rows["rl:f1:v0"], w1), rec(rows["rl:f1:v1"], w1), rec(rows["rl:f2:v0"], same), rec(rows["rl:f2:v1"], w1)]
    cs = m.comparisons(recs, rows, "force")
    assert [c["gold"] for c in cs] == ["P2", "P1", "same", "P2"]                       # weaker in Pair 1 = Pair 2 stronger
    s = m.summary(cs, "force")
    assert s["recall"] == {"P1": 0.0, "P2": 1.0, "same": 1.0} and s["accuracy"] == 0.75
    assert s["by_axis"]["size"] == {"n": 2, "accuracy": 0.5} and s["both_views_right"] == {"families": 2, "rate": 0.5}
    assert m.comparisons(recs, rows, "temp") == [] and m.summary([], "temp") == {"n": 0}


def stat(acc=0.97, rec=0.95, kind="temp"):
    return {"n": 10, "accuracy": acc, "recall": {c: rec for c in m.CLASSES[kind]}}


def good():
    c = {}
    for kind in ("temp", "force"):
        c["fit:" + kind] = stat(kind=kind)
        c["G:" + kind] = stat(0.85, 0.7, kind)
    c["P:temp"] = stat(0.7, 0.5)
    rd = {"all": 0.99, "both": 0.9, "twin": 0.9, "read_right": {"accuracy": 0.99}, **{"all:" + f: 0.9 for f in m.r42.FIELDS}}
    return {"comparison": c, "paired": {"G:temp": {"ci95": [0.1, 0.2]}, "G:force": {"ci95": [0.1, 0.2]}, "P:temp": {"ci95": [0.1, 0.2]}},
            "reading": {"G": rd, "P": dict(rd, all=0.88)}, "screens": {"exist": {"difference": 0.0}, "science": {"difference": -0.01}}}


@pytest.mark.parametrize("edit,text", [
    (lambda r: None, "pass"),
    (lambda r: r["comparison"]["fit:force"]["recall"].update(same=0.85), "fit not learned (['fit_force'])"),
    (lambda r: r["comparison"]["G:temp"].update(accuracy=0.79), "G-tune below its line (['G_temp'])"),
    (lambda r: r["paired"]["G:force"].update(ci95=[0.0, 0.1]), "G-tune below its line (['G_force'])"),
    (lambda r: r["comparison"]["P:temp"].update(accuracy=0.6), "reading / comparison coupling"),
    (lambda r: r["reading"]["P"].update(all=0.86), "protection repair only"),
    (lambda r: r["screens"]["science"].update(difference=-0.03), "protection repair only"),
    (lambda r: r["comparison"].pop("fit:temp"), "incomplete: fit missing"),
    (lambda r: r["paired"].pop("P:temp"), "incomplete: ['P_temp']"),
    (lambda r: r["comparison"]["G:force"]["recall"].update(same=None), "incomplete: ['G_force']"),
])
def test_gates_and_branch(edit, text):
    r = good()
    edit(r)
    assert text in m.branch(m.gates(r))


@real
def test_conditions_bind_to_their_arm(tmp_path):
    out = str(tmp_path)
    open(os.path.join(out, "START.json"), "w").write("{}")
    rows, inputs = m.set_inputs("G")
    some = dict(list(rows.items())[:2])
    recs = []
    for rid, r in some.items():
        for qid, q in r["questions"].items():
            opts = list(q["criteria"]) if q["type"] == "choice" else ["false", "true"]
            g = r["gold"][qid]["probabilities"]
            recs.append({"sample_id": rid, "qid": qid, "image_id": r["image_id"], "options": opts, "gold": g,
                         "gold_answer": max(g, key=g.get), "probabilities": {o: 1 / len(opts) for o in opts}, "prediction": opts[0]})
    from devision.train.evaluate import score_records
    recs = score_records(recs)
    sub = {k: inputs[k] for k in some}
    ident = {"set": "G", "model_of": "new", "start_sha256": m.sha(os.path.join(out, "START.json")), "inputs_sha256": m.e57.hash_inputs(sub),
             "model": {"x": "1"}, "temperature": "none (raw)", "image": None}
    m.e57.write_set(out, "G", recs, ident)
    assert len(m.e57.read_set(out, "G", some, {"x": "1"}, sub)) == len(recs)
    with pytest.raises(SystemExit, match="identity"):
        m.e57.read_set(out, "G", some, m.E57_SHA, sub)                                # another arm's model
    _, inp_p = m.set_inputs("P")
    with pytest.raises(SystemExit, match="identity"):
        m.e57.read_set(out, "G", some, {"x": "1"}, {k: inp_p[k] for k in some})        # another condition's inputs
    f = os.path.join(out, "details_G.jsonl")
    open(f, "a").write("\n")
    with pytest.raises(SystemExit, match="registered"):
        m.e57.read_set(out, "G", some, {"x": "1"}, sub)


@real
def test_start_binds_exp57_actual_end(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "E57_SHA", dict(m.E57_SHA, **{"config.json": "0" * 64}))
    with pytest.raises(SystemExit, match="EXP57"):
        m.stage_start(0, out=str(tmp_path / "out"))
    assert not (tmp_path / "out").exists()


@real
def test_changed_facts_after_start_are_refused(tmp_path, monkeypatch):
    out = str(tmp_path / "out")
    m.stage_start(0, out=out)
    monkeypatch.setattr(m, "DATA_SHA", dict(m.DATA_SHA, **{"tune_P.jsonl": "0" * 64}))
    with pytest.raises(SystemExit, match="data/facts"):
        m.check_start(out, code=False)
    assert m.report(out)["branch"].startswith("invalid")


STUB = """#!/bin/bash
echo "uv $*" >> "${MARK_CALLS}"
case "$*" in
  *"watch.py"*) shift 2; exec python3 "$@";;
  *"exp63.py start"*) mkdir -p runs/exp63; exit 0;;
  *"exp63.py train"*) exit 0;;
  *"exp63.py freeze"*) exit 0;;
  *"exp63.py decide_h"*) exit 0;;
  *"exp63.py decide"*) [ -n "${FAIL_DECIDE:-}" ] && exit 5; exit 0;;
  *"devision-eval"*) exit 0;;
  *"freeze.py record"*) exit 0;;
  *"exp63.py report"*) echo report > runs/exp63/report.txt; exit 0;;
esac
exit 9
"""


def chain(tmp_path, extra):
    import shutil
    import stat
    import subprocess
    root = tmp_path / "repo"
    for rel in ("scripts/exp63/run.sh", "scripts/exp53/watch.py", "scripts/exp35/guard.sh", "scripts/exp38/steps.sh"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join(ROOT, rel), root / rel)
    b = tmp_path / "bin"
    b.mkdir()
    for name, body in {"uv": STUB, "pgrep": '#!/bin/bash\n[ "$1" = "-fl" ] && exit 1\nexec /usr/bin/pgrep "$@"\n'}.items():
        (b / name).write_text(body)
        (b / name).chmod((b / name).stat().st_mode | stat.S_IEXEC)
    env = {**os.environ, "PATH": "%s:%s" % (b, os.environ["PATH"]), "MARK_CALLS": str(tmp_path / "calls"), "EXP63_LIMIT": "60", "EXP63_RESERVE": "4", **extra}
    p = subprocess.run(["bash", str(root / "scripts/exp63/run.sh")], env=env, capture_output=True, text=True, timeout=120)
    return root, p, (tmp_path / "calls").read_text().splitlines()


def test_run_order_main_then_screens_then_h(tmp_path):
    root, p, calls = chain(tmp_path, {})
    assert p.returncode == 0, p.stdout + p.stderr

    def first(pred):
        return next(i for i, c in enumerate(calls) if pred(c))
    i_main = first(lambda c: "exp63.py decide" in c and "decide_h" not in c)
    i_exist, i_science = first(lambda c: "panel_exist" in c), first(lambda c: "panel_science" in c)
    i_h, i_rep = first(lambda c: "decide_h" in c), first(lambda c: "exp63.py report" in c)
    assert first(lambda c: "exp63.py freeze" in c) < i_main < i_exist < i_science < i_h < i_rep
    assert "--finalize" not in calls[i_rep]


def test_failed_main_decide_finalizes_without_screens_or_h(tmp_path):
    root, p, calls = chain(tmp_path, {"FAIL_DECIDE": "1"})
    joined = "\n".join(calls)
    assert p.returncode == 5 and "devision-eval" not in joined and "decide_h" not in joined and "report --finalize" in joined
    assert (root / "runs/exp63.started").exists()


# ---------------------------------------------------------------- EXP63-R1-01 / 02

class FakeDecider:
    class cfg:
        temperature, temperature_by_options = [1.0, 1.0, 1.0], {}

    def decide(self, state, questions, image=None):
        return {"answers": {k: ({"type": "choice", "probabilities": {o: 1 / len(q["criteria"]) for o in q["criteria"]}}
                                if q["type"] == "choice" else {"type": "noul", "noul": 0.5}) for k, q in questions.items()}}


@pytest.fixture
def started(tmp_path, monkeypatch):
    if not has:
        pytest.skip("data / runs not here")
    out = str(tmp_path / "out")
    st = m.stage_start(0, out=out)
    return out, st


def test_h_writes_its_own_progress_and_leaves_the_main_file(started, monkeypatch):
    import time
    out, st = started
    stale = os.path.join(out, "progress-decide.txt")
    open(stale, "w").write("%.3f 2000 2000\n" % (time.time() - 400))                 # the main decide, long finished
    before = open(stale).read()
    import devision.infer as inf
    import devision.reading as R
    monkeypatch.setattr(m.fz, "check", lambda run, names: {"weights": {"w": "new"}, "evals": {}})
    monkeypatch.setattr(m.fz, "weights", lambda run: m.E57_SHA if run == m.E57 else {"w": "new"})
    monkeypatch.setattr(R, "tree_sha", lambda p: st["start_model"]["tokenizer"])
    monkeypatch.setattr(inf.Decider, "load", classmethod(lambda cls, *a, **k: FakeDecider()))
    t0 = time.time()
    m.stage_decide_h(out)
    rows = [x.split() for x in open(os.path.join(out, "progress-decide_h.txt")).read().splitlines()]
    assert rows[-1][1:] == ["600", "600"] and float(rows[0][0]) >= t0                  # H's own total and times
    assert open(stale).read() == before


def test_real_watch_does_not_stop_h_on_the_stale_main_file(tmp_path):
    import subprocess
    import sys
    import time
    stale, own = tmp_path / "progress-decide.txt", tmp_path / "progress-decide_h.txt"
    stale.write_text("%.3f 2000 2000\n" % (time.time() - 400))
    script = ("import time\nf = open(%r, 'a')\nfor n in (1, 300, 600):\n"
              "    f.write('%%.3f %%d 600\\n' %% (time.time(), n)); f.flush(); time.sleep(0.3)\n") % str(own)
    watch = os.path.join(ROOT, "scripts", "exp53", "watch.py")
    base = [sys.executable, watch, "--deadline", str(time.time() + 120), "--grace", "2"]
    ok = subprocess.run(base + ["--progress", str(own), "--log", str(tmp_path / "w1.txt"), "--", sys.executable, "-c", script], timeout=60)
    assert ok.returncode == 0                                                           # H's own file: completes
    old = subprocess.run(base + ["--progress", str(stale), "--log", str(tmp_path / "w2.txt"), "--", sys.executable, "-c", "import time; time.sleep(30)"],
                         timeout=60)
    assert old.returncode == 124 and "no progress for" in (tmp_path / "w2.txt").read_text()   # the old wiring would stop H


def test_changed_exp49_questions_after_start_give_no_verdict(started, tmp_path, monkeypatch):
    out, st = started
    q = tmp_path / "q.jsonl"
    lines = open(m.e54.QUESTIONS).read().splitlines()
    r = json.loads(lines[0])
    r["twin_changed"] = not r["twin_changed"]
    lines[0] = json.dumps(r)
    q.write_text("\n".join(lines) + "\n")
    monkeypatch.setattr(m.e54, "QUESTIONS", str(q))
    with pytest.raises(SystemExit, match="exp49"):
        m.check_start(out, code=False)
    assert m.report(out)["branch"].startswith("invalid")


def test_changed_exp54_p_table_is_refused_even_when_reregistered(started, tmp_path, monkeypatch):
    import shutil
    out, st = started
    d = tmp_path / "exp54"
    shutil.copytree(os.path.join(ROOT, m.e54.OUT), d, ignore=shutil.ignore_patterns("details_*", "*.npz"))
    f = d / "inputs_P.jsonl"
    lines = f.read_text().splitlines()
    r = json.loads(lines[0])
    r["table"][0][2] = "999"
    lines[0] = json.dumps(r)
    f.write_text("\n".join(lines) + "\n")
    m.e54.register(str(d), "inputs", {**json.load(open(d / "registry.json"))["inputs"], "inputs_P.jsonl": m.sha(str(f))})
    monkeypatch.setattr(m.e54, "OUT", str(d))
    m.e54.load_inputs(str(d), "P")                                                      # its own registry accepts it ...
    with pytest.raises(SystemExit, match="EXP54"):
        m.check_start(out, code=False)                                                  # ... START does not


def test_changed_tokenizer_is_refused_before_any_load(started, monkeypatch):
    out, st = started
    import devision.infer as inf
    import devision.reading as R
    monkeypatch.setattr(R, "tree_sha", lambda p: "0" * 64)
    with pytest.raises(SystemExit, match="tokenizer"):
        m.check_start(out, code=False)
    monkeypatch.setattr(m, "check_start", lambda out, code=True: st)
    monkeypatch.setattr(m.fz, "check", lambda run, names: {"weights": {"w": "new"}, "evals": {}})
    monkeypatch.setattr(m.fz, "weights", lambda run: {"w": "new"})
    monkeypatch.setattr(inf.Decider, "load", classmethod(lambda cls, *a, **k: (_ for _ in ()).throw(AssertionError("loaded"))))
    with pytest.raises(SystemExit, match="tokenizer"):
        m.stage_decide_h(out)
