"""exp57 runner without the real model: start refuses changed data / a wrong plan / another YAML and writes nothing; the
fit rows are the fit slice's; the pipeline log becomes progress lines; the export check accepts only LoRA-merged encoder
layers and head changes; comparison statistics, the case-cluster pairing and the pre-written gates / branches; the run
stops at the watch deadline (group killed) and finalizes, and a failed stage starts nothing after it."""
import importlib.util
import json
import os

import pytest
import torch

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.dirname(HERE))
spec = importlib.util.spec_from_file_location("exp57", os.path.join(HERE, "exp57.py"))
assert spec is not None and spec.loader is not None
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def in_root(fn, *a, **k):
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        return fn(*a, **k)
    finally:
        os.chdir(cwd)


def test_start_refuses_changed_data_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "DATA_SHA", {**m.DATA_SHA, "train.jsonl": "0" * 64})
    monkeypatch.setattr(m.fz, "check", lambda run, names: {"weights": m.e54.T_SHA, "evals": {}})
    with pytest.raises(SystemExit, match="data/cmp"):
        in_root(m.stage_start, 0, out=str(tmp_path / "out"))
    assert not (tmp_path / "out" / "START.json").exists()


def test_plan_and_yaml_checks(tmp_path):
    plan = json.load(open(os.path.join(ROOT, "data/cmp/PLAN.json"))) if os.path.exists(os.path.join(ROOT, "data/cmp/PLAN.json")) else None
    good = {"updates": 1000, "units": 4000, "unit_sizes": {"2": 4000}, "split_units": 0, "exposure_per_question": {"1": 8000},
            "every_item_once": True, "seed": 1, "micro_batch": 8, "batch_sizes": {"8": 1000}}
    m.check_plan(good)
    if plan is not None:
        m.check_plan(plan)
    with pytest.raises(SystemExit, match="PLAN"):
        m.check_plan({**good, "split_units": 3})
    in_root(m.check_yaml)
    y = open(os.path.join(ROOT, m.YAML)).read().replace("text_only: true", "text_only: false")
    (tmp_path / "x.yaml").write_text(y)
    with pytest.raises(SystemExit, match="fixed EXP57 arm"):
        m.check_yaml(str(tmp_path / "x.yaml"))


def test_fit_rows_are_the_slice(tmp_path):
    rows = [{"id": "r%d" % i, "family": "f%d" % (i // 2)} for i in range(8)]
    (tmp_path / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "fit_slice.json").write_text(json.dumps({"families": ["f1", "f3"]}))
    assert [r["id"] for r in m.fit_rows(str(tmp_path))] == ["r2", "r3", "r6", "r7"]
    assert m.write_fit(str(tmp_path))["rows"] == 4


def test_progress_from_the_pipeline_log():
    assert m.progress_of(["== x", "step 0 {'val/accuracy': 0.5}"]) == 0
    assert m.progress_of(["step 0 {}", "epoch 1 step 25/1000 loss 1.0 nll 1.0 acc 0.5 step_s 0.4", "noise step 7"]) == 25
    assert m.progress_of(["temperatures (choice, score, noul): [1.0]"]) is None


def test_train_stage_writes_progress(tmp_path, monkeypatch):
    out = tmp_path / "out"
    (out / "logs").mkdir(parents=True)
    monkeypatch.setattr(m, "check_start", lambda out, code=True: {})
    script = ("import time\nf = open(%r, 'a')\nfor s in ['step 0 {}', 'epoch 1 step 25/1000 loss 1 nll 1 acc 1 step_s 1']:\n"
              "    f.write(s + '\\n'); f.flush(); time.sleep(0.3)\n") % str(out / "logs" / "stage2.log")
    import sys
    assert m.stage_train(str(out), poll=0.05, cmd=[sys.executable, "-c", script]) == 0
    lines = [x.split() for x in open(out / "progress-train.txt").read().splitlines()]
    assert [x[1:] for x in lines] == [["0", "1000"], ["25", "1000"]]


def tensors():
    return {"vision.a": torch.ones(2), "projector.w": torch.ones(2), "condition.q": torch.ones(2),
            "decision.encoder.embeddings.tok.weight": torch.ones(2), "decision.encoder.layers.0.attn.Wqkv.weight": torch.ones(2),
            "decision.scorer.weight": torch.ones(2), "decision.type_emb.weight": torch.ones(2)}


def test_export_check():
    a = tensors()
    b = {k: v.clone() for k, v in a.items()}
    b["decision.encoder.layers.0.attn.Wqkv.weight"] += 1
    b["decision.scorer.weight"] += 1
    r = m.export_check(a, b)
    assert r["ok"] and r["changed_groups"] == {"decision.encoder.layers": 1, "decision.scorer": 1}
    for k in ("vision.a", "projector.w", "condition.q", "decision.encoder.embeddings.tok.weight"):
        c = {x: v.clone() for x, v in b.items()}
        c[k] += 1
        assert not m.export_check(a, c)["ok"], k
    assert not m.export_check(a, {k: v.clone() for k, v in a.items()})["ok"]           # nothing trained
    with pytest.raises(SystemExit):
        m.export_check(a, {**b, "extra": torch.ones(1)})


def cmp_recs(right, n=40):
    """Comparison records over n families x 2 views; right(i) decides correctness; gold cycles A / B / neither."""
    out = []
    golds = ["sample A", "sample B", "neither"]
    for i in range(n * 2):
        out.append({"key": "k%d" % i, "family": "f%d" % (i // 2), "case": "c%d" % i, "axis": ["same mass", "equal"][i % 2],
                    "gold": golds[i % 3], "prediction": golds[i % 3] if right(i) else "sample A", "right": bool(right(i))})
    return out


def test_comparison_statistics_and_pairing():
    s = m.comparison_summary(cmp_recs(lambda i: i % 3 != 2))
    assert s["recall"] == {"sample A": 1.0, "sample B": 1.0, "neither": 0.0} and s["both_views_right"]["families"] == 40
    same = m.paired_difference(cmp_recs(lambda i: i % 2 == 0), cmp_recs(lambda i: i % 2 == 0))
    assert same["value"] == 0 and same["ci95"] == [0, 0]
    up = m.paired_difference(cmp_recs(lambda i: True), cmp_recs(lambda i: i % 2 == 0))
    assert up["value"] == 0.5 and up["ci95"][0] > 0 and up["wrong_to_right"] == 40
    with pytest.raises(SystemExit):
        m.paired_difference(cmp_recs(lambda i: True, 10), cmp_recs(lambda i: True, 11))


def result(fit=0.97, fit_rec=0.95, g=0.85, g_rec=0.7, g_lo=0.1, p=0.7, p_lo=0.1, g_all=0.99, p_all=0.88, screen=-0.01):
    rd = {"all": p_all, "both": 0.8, "twin": 0.8, "read_right": {"accuracy": 0.99}, **{"all:" + f: 0.8 for f in m.r42.FIELDS}}
    rec = lambda v: {"sample A": v, "sample B": v, "neither": v}  # noqa: E731
    return {"comparison": {"fit": {"accuracy": fit, "recall": rec(fit_rec)}, "G": {"accuracy": g, "recall": rec(g_rec)},
                           "P": {"accuracy": p, "recall": rec(0.5)}},
            "paired": {"G": {"ci95": [g_lo, 0.2]}, "P": {"ci95": [p_lo, 0.2]}},
            "reading": {"G": {**rd, "all": g_all}, "P": rd},
            "screens": {"exist": {"difference": screen}, "science": {"difference": 0.0}}}


@pytest.mark.parametrize("over,gate,text", [
    ({}, None, "all pass"),
    ({"fit_rec": 0.85}, "fit", "fit not learned"),
    ({"g": 0.79}, "G_tune", "generalisation"),
    ({"g_lo": -0.01}, "G_tune", "generalisation"),
    ({"p": 0.6}, "P_tune", "reading errors couple"),
    ({"p_lo": 0.0}, "P_tune", "reading errors couple"),
    ({"g_all": 0.97}, "reading", "protection repair only"),
    ({"p_all": 0.85}, "reading", "protection repair only"),
    ({"screen": -0.03}, "screens", "protection repair only"),
])
def test_gates_and_branches(over, gate, text):
    g = m.gates(result(**over))
    if gate:
        assert g[gate] is False and all(v is True for k, v in g.items() if k not in (gate, "reading_detail"))
    else:
        assert all(v is True for k, v in g.items() if k != "reading_detail")
    assert text in m.branch(g)


def test_missing_items_are_incomplete_not_low():
    res = result()
    del res["comparison"]["P"], res["screens"]["science"]
    g = m.gates(res)
    assert g["P_tune"] == "incomplete" and g["screens"] == "incomplete" and m.branch(g).startswith("incomplete")
    res = result(fit_rec=0.5)
    del res["comparison"]["G"]
    assert "fit not learned" in m.branch(m.gates(res))                # fit's verdict stands without the rest


STAGE = """import signal, subprocess, sys, time
child = subprocess.Popen([sys.executable, '-c', 'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(300)'])
open('runs/exp57/child.pid', 'w').write(str(child.pid))
time.sleep(300)
"""
STUB = """#!/bin/bash
echo "uv $*" >> "${MARK}"
case "$*" in
  *"watch.py"*) shift 2; exec python3 "$@";;
  *"exp57.py start"*) mkdir -p runs/exp57; exit 0;;
  *"exp57.py train"*) [ -n "${FAIL_TRAIN:-}" ] && exit 0; exec python3 "${STAGE}";;
  *"exp57.py freeze"*) [ -n "${FAIL_TRAIN:-}" ] && exit 3; exit 0;;
  *"exp57.py report --finalize"*) echo final > runs/exp57/report.txt; exit 0;;
esac
exit 9
"""


def run_stub(tmp_path, extra):
    import shutil
    import stat
    import subprocess
    root = tmp_path / "repo"
    for rel in ("scripts/exp57/run.sh", "scripts/exp53/watch.py", "scripts/exp35/guard.sh", "scripts/exp38/steps.sh"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join(ROOT, rel), root / rel)
    (tmp_path / "stage.py").write_text(STAGE)
    b = tmp_path / "bin"
    b.mkdir()
    for name, body in {"uv": STUB, "pgrep": '#!/bin/bash\n[ "$1" = "-fl" ] && exit 1\nexec /usr/bin/pgrep "$@"\n'}.items():
        (b / name).write_text(body)
        (b / name).chmod((b / name).stat().st_mode | stat.S_IEXEC)
    env = {**os.environ, "PATH": "%s:%s" % (b, os.environ["PATH"]), "MARK": str(tmp_path / "calls"), "STAGE": str(tmp_path / "stage.py"),
           "EXP57_LIMIT": "60", "EXP57_RESERVE": "4", **extra}
    return root, subprocess.run(["bash", str(root / "scripts/exp57/run.sh")], env=env, capture_output=True, text=True, timeout=120)


def test_run_stops_at_the_watch_deadline_and_finalizes(tmp_path):
    import time
    t = time.time()
    root, p = run_stub(tmp_path, {})
    assert p.returncode == 124 and time.time() - t < 60, p.stdout + p.stderr
    calls = (tmp_path / "calls").read_text()
    assert "report --finalize" in calls and "exp57.py freeze" not in calls and "devision-eval" not in calls
    with pytest.raises(ProcessLookupError):
        os.kill(int((root / "runs/exp57/child.pid").read_text()), 0)


def test_failed_stage_starts_nothing_after_it(tmp_path):
    root, p = run_stub(tmp_path, {"FAIL_TRAIN": "1"})
    calls = (tmp_path / "calls").read_text()
    assert p.returncode == 3, p.stdout + p.stderr
    assert "exp57.py decide" not in calls and "devision-eval" not in calls and "report --finalize  start=0 train=0 freeze=3" in calls
