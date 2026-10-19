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


def q3():
    return {k: {"type": "choice", "instructions": k, "criteria": {"x": None, "y": None}} for k in ("readA", "readB", "cmp")}


def syn_rows(prefix, fams, height=16):
    return [{"id": "exp21:%s%04d:v%d:h%d" % (prefix, i, v, height), "questions": q3(),
             "gold": {k: {"probabilities": {"x": 1.0, "y": 0.0}} for k in ("readA", "readB", "cmp")}} for i in range(fams) for v in (0, 1)]


def plain_rows(prefix, n):
    return [{"id": "%s%d" % (prefix, i), "questions": {"q": {"type": "choice", "instructions": "q", "criteria": {"x": None, "y": None}}},
             "gold": {"q": {"probabilities": {"x": 1.0, "y": 0.0}}}} for i in range(n)]


def recs(rows, p_right):
    """Records of rows: every question of family / row i right when (i % 10) < p * 10."""
    out = []
    for n, r in enumerate(rows):
        i = int(r["id"].split(":")[1][1:]) if r["id"].startswith("exp21:") else n
        for qid, q in r["questions"].items():
            right = (i % 10) < round(p_right * 10)
            pred = "x" if right else "y"
            out.append({"sample_id": r["id"], "qid": qid, "options": list(q["criteria"]), "gold": r["gold"][qid]["probabilities"],
                        "gold_answer": "x", "prediction": pred, "correct": right})
    return out


def wj(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write("".join(json.dumps(r) + "\n" for r in rows))


@pytest.fixture
def ws(tmp_path, monkeypatch):
    """A tiny world: data files, the sealed candidate's probe summaries, the D2 config, and a builder of D2 runs."""
    m = load()
    monkeypatch.chdir(tmp_path)
    data = {"val": syn_rows("t", 40), "dev16": syn_rows("d", 40), "erased": syn_rows("e", 20, height=0), "k_text": plain_rows("k", 50),
            "rs_temp": plain_rows("rt", 41), "rs_force": plain_rows("rf", 120), "rs_temp_mism": plain_rows("rtm", 41),
            "rs_force_mism": plain_rows("rfm", 120)}
    paths = {n: "data/%s.jsonl" % n for n in data}
    for n, rows in data.items():
        wj(paths[n], rows)
    probes = {"exist": plain_rows("x", 30), "science": plain_rows("s", 30)}
    for n, rows in probes.items():
        wj("data/%s.jsonl" % n, rows)
    wj("data/train.jsonl", [{"id": "a", "input": "picture", "group": "g", "questions": {"q": {}}}])
    os.makedirs("data/panel_learn")
    json.dump({"train_sha256": m.sha("data/train.jsonl")}, open("data/panel_learn/MANIFEST.json", "w"))
    os.makedirs("sealed")
    for n, a in (("exist", 0.95), ("science", 0.62)):
        json.dump({"accuracy_all": a}, open("sealed/%s.json" % n, "w"))
    os.makedirs("configs")
    cfg = {"name": "x", "stages": [{"name": "d2", "kind": "train", "data": "data/train.jsonl", "params": {"epochs": 1, "micro_batch": 8, "seed": 1}}]}
    json.dump(cfg, open("configs/d2.yaml", "w"))
    monkeypatch.setattr(m, "MONITOR", paths)
    monkeypatch.setattr(m, "PROBES", {n: "data/%s.jsonl" % n for n in probes})
    monkeypatch.setattr(m, "TRAIN", "data/train.jsonl")

    def make(fit=0.95, erased=0.2, rs_temp=1.0, exist=0.95, updates=100, stop_before_report=False, drop_probe=None,
             drop_question=None, duplicate=None, probe_ckpt=None, empty=None, truncate=None, final_empty=False,
             final_drop=None, final_metric_off=None, probe_summary_off=None):
        d2 = "runs/d2"
        if os.path.exists(d2):
            shutil.rmtree(d2)
        os.makedirs(d2 + "/d2")
        json.dump({"config": cfg}, open(d2 + "/run.json", "w"))
        step = d2 + "/d2/monitor/step%d" % updates
        shares = {"val": fit, "dev16": 0.5, "erased": erased, "k_text": 1.0, "rs_temp": rs_temp, "rs_force": 0.5, "rs_temp_mism": 0.4,
                  "rs_force_mism": 0.4}
        final_metrics = {}
        for n, rows in data.items():
            rs = recs(rows, shares[n])
            final_metrics["%s/accuracy" % n] = sum(r["correct"] for r in rs) / len(rs)
            if not final_empty and n != final_drop:
                wj("%s/d2/monitor/final/%s.details.jsonl" % (d2, n), rs)
            if n == drop_question:
                rs = rs[1:]
            if n == duplicate:
                rs = rs + rs[:1]
            wj("%s/%s.details.jsonl" % (step, n), rs)
            if n == empty:
                open("%s/%s.details.jsonl" % (step, n), "w").close()
            if n == truncate:
                text = open("%s/%s.details.jsonl" % (step, n)).read()
                open("%s/%s.details.jsonl" % (step, n), "w").write(text[: len(text) // 2])
        if final_metric_off:
            final_metrics[final_metric_off + "/accuracy"] -= 0.1
        os.makedirs(d2 + "/d2/monitor/final", exist_ok=True)
        if stop_before_report:
            return d2
        json.dump({"losses": [0.1] * updates, "evals": [{"step": updates}], "final": final_metrics}, open(d2 + "/d2/train_report.json", "w"))
        open(d2 + "/d2/model.safetensors", "w").write("w")
        for n, rows in probes.items():
            if n == drop_probe:
                continue
            acc = exist if n == "exist" else 0.64
            rs = recs(rows, acc)
            wj("%s/eval/%s.details.jsonl" % (d2, n), rs)
            json.dump({"n": len(rs), "accuracy_all": sum(r["correct"] for r in rs) / len(rs) + (0.05 if n == probe_summary_off else 0),
                       "run": {"checkpoint": probe_ckpt or d2 + "/d2", "data_sha256": m.sha("data/%s.jsonl" % n)}},
                      open("%s/eval/%s.json" % (d2, n), "w"))
        return d2
    return m, make


def outcome(m, d2):
    return m.d2_report(d2, "sealed", "configs/d2.yaml", planned=100)["outcome"]


@pytest.mark.parametrize("kw,want", [({}, "A:"), ({"rs_temp": 0.8}, "B:"), ({"exist": 0.80}, "B:"), ({"fit": 0.3}, "C:"),
                                     ({"erased": 0.7}, "invalid")])
def test_complete_runs_get_their_outcome(ws, kw, want):
    m, make = ws
    assert outcome(m, make(**kw)).startswith(want)


@pytest.mark.parametrize("kw,want", [({"stop_before_report": True}, "incomplete"), ({"drop_probe": "science"}, "incomplete"),
                                     ({"drop_question": "dev16"}, "invalid"), ({"duplicate": "rs_force"}, "invalid"),
                                     ({"probe_ckpt": "runs/other/d2"}, "invalid"), ({"updates": 60}, "invalid")])
def test_missing_or_wrong_evidence_never_gets_an_outcome(ws, kw, want):
    """Stopped before the end (monitors exist, no train report), a missing probe, a missing / repeated question, a probe
    of another endpoint, a run with another number of updates: incomplete or invalid, never A / B / C / D."""
    m, make = ws
    got = outcome(m, make(**kw))
    assert got.startswith(want) and not got[:2] in ("A:", "B:", "C:", "D:"), got


@pytest.mark.parametrize("kw,want", [({"empty": "erased"}, "invalid"), ({"truncate": "dev16"}, "invalid"),
                                     ({"final_empty": True}, "incomplete"), ({"final_drop": "rs_temp"}, "incomplete"),
                                     ({"final_metric_off": "val"}, "invalid"), ({"probe_summary_off": "exist"}, "invalid")])
def test_r2_failures_end_in_a_report_without_an_outcome(ws, kw, want):
    """PANEL-D2-R2: an empty erased file / a truncated file (no crash), an empty or partial final, final records that do
    not give train_report.final, a probe summary contradicting its records: a readable report, no A / B / C / D, and the
    other valid sets still reported."""
    m, make = ws
    out = m.d2_report(make(**kw), "sealed", "configs/d2.yaml", planned=100)
    assert out["outcome"].startswith(want) and out["outcome"][:2] not in ("A:", "B:", "C:", "D:"), out["outcome"]
    json.dumps(out)
    if kw.get("empty") == "erased":
        assert out["sets"]["end/erased"] == "invalid" and out["parts"]["fit"]["read"]["value"] > 0.8      # fit still reported
        assert out["parts"]["fit"]["read_class"].startswith("void")


def test_no_run_directory_still_writes_a_report(ws):
    m, _ = ws
    m.CONFIG = "configs/d2.yaml"
    assert m.main(["d2", "runs/never", "sealed"]) == 0
    rep = json.load(open("runs/never/report.json"))
    assert rep["outcome"].startswith("incomplete") and "no run.json" in rep["outcome"]


def test_the_real_plan_is_one_pass_of_1054_updates():
    m = load()
    os.chdir(ROOT)
    assert m.planned_updates("configs/exp74-panel-d2.yaml") == 1054


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


def test_versus_start_and_sealed(ws):
    """PANEL-DECISION-PREP-01: the end against the run's own step 0 and against the sealed candidate (descriptive)."""
    m, make = ws
    d2 = make()
    end = os.path.join(d2, "d2", "monitor", "step100")
    start = os.path.join(d2, "d2", "monitor", "step0")
    shutil.copytree(end, start)
    rows = [json.loads(x) for x in open(os.path.join(start, "rs_temp.details.jsonl"))]
    for r in rows[:10]:
        r["correct"], r["prediction"] = False, "y"
    open(os.path.join(start, "rs_temp.details.jsonl"), "w").write("".join(json.dumps(r) + "\n" for r in rows))
    out = m.d2_report(d2, "sealed", "configs/d2.yaml", planned=100)
    v = out["versus"]
    assert v["end_minus_start"]["rs_temp"] == pytest.approx(10 / 41) and "fit_read" in v["end_minus_start"]
    assert v["sealed"]["exist"] == 0.95 and "exist" in v["end_minus_sealed"] and out["outcome"].startswith("A:")
