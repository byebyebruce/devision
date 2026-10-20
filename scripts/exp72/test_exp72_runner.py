# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false
"""exp72's runner without a model: the arms' YAMLs (EXP68-X's arm, differing only by name / data / val) and readout pairs;
G's reading gate is EXP63's 'all' over exactly 1,800 unique reading questions (reproduces EXP65-C's registered value);
the pre-written route at its boundaries (G 0.979 stops, 0.98 passes the reading item, protection failure stops, missing /
invalid evidence gives no route); a report-level run with fake records; the run's order and limit."""
import importlib.util
import json
import os
import shutil

import pytest

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.dirname(HERE))
spec = importlib.util.spec_from_file_location("exp72_runner", os.path.join(HERE, "exp72.py"))
assert spec is not None and spec.loader is not None
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
has = os.path.exists(os.path.join(ROOT, "data", "exp72", "MANIFEST.json")) and os.path.exists(os.path.join(ROOT, "runs", "exp65", "details_c_G.jsonl"))
data = pytest.mark.skipif(not has, reason="data/exp72 / EXP65's run not present")
FAKE = {"raw": {"model.safetensors": "a" * 64, "config.json": "b" * 64}, "k": {"model.safetensors": "c" * 64, "config.json": "d" * 64}}


@pytest.fixture(autouse=True)
def in_root(monkeypatch):
    monkeypatch.chdir(ROOT)


@data
def test_yamls_and_pairs(tmp_path, monkeypatch):
    m.check_yamls()
    m.check_pairs()
    y = open(m.YAML["k"]).read().replace("lr_head: 1.0e-05", "lr_head: 2.0e-05")
    (tmp_path / "k.yaml").write_text(y)
    monkeypatch.setattr(m, "YAML", dict(m.YAML, k=str(tmp_path / "k.yaml")))
    with pytest.raises(SystemExit, match="not EXP68-X's arm"):
        m.check_yamls()


def g_context():
    qs53 = {q["key"]: q for q in m.jsonl(m.e54.QUESTIONS) if q["set"] == "rl_tune"}
    pin = m.e54.load_inputs(m.e54.OUT, "P")
    tune = {r["id"]: r for r in m.jsonl(m.e54.TUNE)}
    p_right = {k: {(o, qn): v for o, qn, v in pin[q["id"]]["table"]}[m.e54.TEMPLATES[tune[q["id"]]["questions"][q["qid"]]["instructions"]]] == q["target"]
               for k, q in qs53.items()}
    return qs53, p_right


@data
def test_g_reading_is_exp63s_metric_on_1800_questions():
    qs53, p_right = g_context()
    rows = m.inputs_of("raw", "G")[0]
    c = [json.loads(x) for x in open("runs/exp65/details_c_G.jsonl")][1:]
    g = m.g_reading(c, rows, qs53, p_right)
    assert g["n"] == 1800 and g["all"] == pytest.approx(json.load(open("runs/exp65/report.json"))["arms"]["c"]["reading"]["G"]["all"])
    assert len(c) == 2400                                                             # 600 comparisons are not in the denominator
    from devision.reading import TEMPLATES
    reading = [i for i, r in enumerate(c) if rows[r["sample_id"]]["questions"][r["qid"]]["instructions"] in TEMPLATES]
    for bad in ([r for i, r in enumerate(c) if i != reading[0]], c + [c[reading[0]]]):              # one missing / one twice
        with pytest.raises(SystemExit, match="G reading"):
            m.g_reading(bad, rows, qs53, p_right)


def arms(g_k=0.99, screens_k=True, fit_k=0.95, conf=0.85, g_raw=0.99):
    full = lambda v: {c: v for c in m.CLASSES}  # noqa: E731
    return {"k": {"fit": full(fit_k), "diag": dict(full(1.0), opposite_mass_wins=conf, opposite_speed_wins=conf), "g_all": g_k, "screens_ok": screens_k},
            "raw": {"fit": full(0.95), "diag": full(0.6), "g_all": g_raw, "screens_ok": True}}


def test_route_boundaries():
    up, flat = {"ci95": [0.05, 0.3]}, {"ci95": [-0.02, 0.1]}
    assert m.route("completed", arms(g_k=0.979), up).startswith("stop: the K arm's numeric reading")              # comparisons all pass
    assert m.route("completed", arms(g_k=0.98), up).startswith("the K arm learns to use the given K")              # the boundary passes
    assert m.route("completed", arms(screens_k=False), up).startswith("stop:")                                      # protection fails
    assert m.route("completed", arms(fit_k=0.89), up).startswith("the K arm did not fit")
    assert m.route("completed", arms(), flat).startswith("no K gain")
    assert m.route("completed", arms(conf=0.79), up).startswith("partial")
    assert "P (live tables)" in m.route("completed", arms(), up) and "not a candidate" in m.route("completed", arms(), up)
    for ev in ("incomplete", "invalid"):
        assert m.route(ev, arms(), up).startswith(ev) and "no route" in m.route(ev, arms(), up)
    assert m.route("completed", arms(g_raw=0.5), up).startswith("the K arm learns")                                 # raw's G only reported


def right(rows, wrong_axes=()):
    out = []
    for rid, r in rows.items():
        for qid, q in r["questions"].items():
            gold = r["gold"][qid]["probabilities"]
            g = max(gold, key=gold.get)
            opts = list(q["criteria"])
            pred = g if r.get("axis") not in wrong_axes else next(o for o in opts if o != g)
            out.append({"sample_id": rid, "qid": qid, "image_id": r.get("image_id", rid), "kind": r.get("kind", ""), "axis": r.get("axis", ""),
                        "options": opts, "gold": gold, "gold_answer": g, "probabilities": {o: (0.9 if o == pred else 0.1 / (len(opts) - 1)) for o in opts},
                        "prediction": pred, "correct": pred == g, "nll": 0.1, "brier": 0.0})
    return out


@pytest.fixture
def started(tmp_path):
    import time
    out = str(tmp_path / "exp72")
    m.stage_start(int(time.time()) + 2700, out=out)
    return out


def fake_run(out, monkeypatch, skip=(), screens_low=False):
    """Records written / registered by the runner's own writer under fake frozen identities: fid = EXP68's registered
    records, slice / diag / fit right, G = EXP65-C's G records (same questions); screens = EXP63's (or 0.05 lower)."""
    s_sha = m.sha(os.path.join(out, "START.json"))
    rows, inputs = m.e70.inputs_of("fid_raw")
    ref = {(r["sample_id"], r["qid"]): r for r in m.e70.raw_registered()}
    m.e57.write_set(out, "fid", [ref[(rid, q)] for rid in sorted(rows) for q in rows[rid]["questions"]],
                    {"set": "fid", "start_sha256": s_sha, "inputs_sha256": m.e57.hash_inputs(inputs), "model": m.X_SHA, "temperature": "none (raw)", "image": None})
    fake_runs = {"runs/exp72-raw": FAKE["raw"], "runs/exp72-k": FAKE["k"]}
    real_state = m.frozen_state
    monkeypatch.setattr(m, "frozen_state", lambda run: ("completed", fake_runs[run]) if run in fake_runs else real_state(run))
    e63s = {n: m.e57.screen_side(m.e65.E63, n, m.e57.SCREENS[n])[0] for n in m.e57.SCREENS}
    monkeypatch.setattr(m.e57, "screen", lambda run, n, *a, **k: {"new": e63s[n] - (0.05 if screens_low and run.endswith("-k") else 0.0)})
    cg = [json.loads(x) for x in open("runs/exp65/details_c_G.jsonl")][1:]
    for arm in m.ARMS:
        for n in m.REQUIRED + ("fit",):
            if (arm, n) in skip:
                continue
            rows, inputs = m.inputs_of(arm, n)
            recs = cg if n == "G" else right(rows)
            m.e57.write_set(out, "%s_%s" % (arm, n), recs, {"set": "%s_%s" % (arm, n), "start_sha256": s_sha, "inputs_sha256": m.e57.hash_inputs(inputs),
                                                            "model": FAKE[arm], "temperature": "none (raw)", "image": None})


@data
def test_report_routes_and_evidence(started, monkeypatch):
    fake_run(started, monkeypatch)
    res = m.report(started)
    assert res["evidence"] == "completed" and res["arms"]["k"]["G_reading"]["n"] == 1800
    assert res["arms"]["k"]["G_reading"]["all"] >= 0.98 and res["k_minus_raw"]["conflict_macro"]["value"] == 0
    assert res["route"].startswith("no K gain") and "95 %" in res["k_minus_raw"]["interval"]                    # equal arms: no gain


@data
@pytest.mark.parametrize("case,want", [("k_G_missing", "incomplete"), ("k_G_changed", "invalid"), ("screens_low", "stop:")])
def test_report_missing_invalid_or_protection(started, monkeypatch, case, want):
    fake_run(started, monkeypatch, skip={("k", "G")} if case == "k_G_missing" else (), screens_low=case == "screens_low")
    if case == "k_G_changed":
        with open(os.path.join(started, "details_k_G.jsonl"), "a") as fh:
            fh.write("\n")
    res = m.report(started)
    assert res["route"].startswith(want)
    if want != "stop:":
        assert "no route" in res["route"]


STUB = """#!/bin/bash
echo "uv $*" >> "${MARK_CALLS}"
case "$*" in
  *"watch.py"*) shift 2; exec python3 "$@";;
  *"exp72.py start"*) mkdir -p runs/exp72; exit 0;;
  *"exp72.py fid"*) [ -n "${FAIL_FID:-}" ] && exit 4; exit 0;;
  *"exp72.py train_"*|*"exp72.py freeze_"*|*"exp72.py decide"*|*"devision-eval"*|*"freeze.py record"*) exit 0;;
  *"exp72.py report"*) echo "report $*" > runs/exp72/report.txt; exit 0;;
esac
exit 9
"""


def chain(tmp_path, extra):
    import stat as st_
    import subprocess
    root = tmp_path / "repo"
    for rel in ("scripts/exp72/run.sh", "scripts/exp53/watch.py", "scripts/exp35/guard.sh", "scripts/exp38/steps.sh"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join(ROOT, rel), root / rel)
    bdir = tmp_path / "bin"
    bdir.mkdir()
    for name, body in {"uv": STUB, "pgrep": '#!/bin/bash\n[ "$1" = "-fl" ] && exit 1\nexec /usr/bin/pgrep "$@"\n'}.items():
        (bdir / name).write_text(body)
        (bdir / name).chmod((bdir / name).stat().st_mode | st_.S_IEXEC)
    env = {**os.environ, "PATH": "%s:%s" % (bdir, os.environ["PATH"]), "MARK_CALLS": str(tmp_path / "calls"), "EXP72_LIMIT": "60",
           "EXP72_RESERVE": "4", **extra}
    p = subprocess.run(["bash", str(root / "scripts/exp72/run.sh")], env=env, capture_output=True, text=True, timeout=120)
    return root, p, (tmp_path / "calls").read_text().splitlines()


def test_run_order_failed_fid_and_limit(tmp_path):
    root, p, calls = chain(tmp_path / "a", {})
    assert p.returncode == 0, p.stdout + p.stderr
    first = lambda s: next(i for i, c in enumerate(calls) if s in c)  # noqa: E731
    assert first("exp72.py start") < first("exp72.py fid") < first("exp72.py train_raw") < first("exp72.py freeze_raw") \
        < first("exp72.py train_k") < first("exp72.py freeze_k") < next(i for i, c in enumerate(calls) if c.endswith("exp72.py decide")) < first("exp72-raw/eval/exist") \
        < first("exp72-k/eval/science") < first("exp72.py decide_fit") < first("exp72.py report")
    root, p, calls = chain(tmp_path / "b", {"FAIL_FID": "1"})
    joined = "\n".join(calls)
    assert p.returncode == 4 and "train_raw" not in joined and "report --finalize" in joined
    s = open(os.path.join(ROOT, "scripts/exp72/run.sh")).read()
    assert "${EXP72_LIMIT:-2700}" in s and "${EXP72_RESERVE:-120}" in s


# ------------------------------------------------------------------ round 2 (critic EXP72-R2-01 / R2-02)

def exp63_copy(tmp_path, refreeze=False):
    """runs/exp63 linked file by file, with eval/exist.{json,details.jsonl} replaced by another internally consistent
    result (one answer flipped, the summary's accuracy recomputed); with refreeze, FROZEN.json re-records the new hashes."""
    src = os.path.join(ROOT, "runs", "exp63")
    dst = tmp_path / "exp63"
    dst.mkdir()
    for n in os.listdir(src):
        if n not in ("eval", "FROZEN.json"):
            os.symlink(os.path.join(src, n), dst / n)
    (dst / "eval").mkdir()
    for n in os.listdir(os.path.join(src, "eval")):
        if not n.startswith("exist."):
            os.symlink(os.path.join(src, "eval", n), dst / "eval" / n)
    det = [json.loads(x) for x in open(os.path.join(src, "eval", "exist.details.jsonl"))]
    r = det[0]
    flip = next(o for o in r["probabilities"] if o != r["prediction"])
    r["probabilities"] = {o: (0.9 if o == flip else 0.1) for o in r["probabilities"]}
    r["prediction"], r["correct"] = flip, flip == r.get("gold_answer", flip)
    (dst / "eval" / "exist.details.jsonl").write_text("".join(json.dumps(x) + "\n" for x in det))
    summ = json.load(open(os.path.join(src, "eval", "exist.json")))
    acc = sum(bool(x["correct"]) for x in det) / len(det)
    summ["accuracy_all"] = acc
    summ["accuracy"] = {k: (acc if v is not None else None) for k, v in summ["accuracy"].items()}
    (dst / "eval" / "exist.json").write_text(json.dumps(summ, indent=2))
    fr = json.load(open(os.path.join(src, "FROZEN.json")))
    if refreeze:
        fr["evals"]["exist"] = {f: m.sha(str(dst / "eval" / f)) for f in ("exist.json", "exist.details.jsonl")}
    (dst / "FROZEN.json").write_text(json.dumps(fr, indent=1))
    return str(dst), acc


@data
@pytest.mark.parametrize("refreeze,match", [(False, "changed after it was recorded"), (True, "not the one EXP68's START bound")])
def test_exp63_baseline_bound(started, tmp_path, monkeypatch, refreeze, match):
    st = m.check_start(started, code=False)
    assert st["exp63"] == m.bound_exp63() and set(st["exp63"]["screens"]) == {"exist", "science"}
    path, acc = exp63_copy(tmp_path, refreeze)
    assert acc != pytest.approx(0.94701087, abs=1e-9)                                      # a different, consistent result
    monkeypatch.setattr(m.e65, "E63", path)
    with pytest.raises(SystemExit, match=match):
        m.bound_exp63()
    with pytest.raises(SystemExit):
        m.check_start(started, code=False)
    assert m.report(started)["route"].startswith("invalid")
    with pytest.raises(SystemExit):
        m.stage_start(0, out=str(tmp_path / "fresh"))


def broken_arm(tmp_path, case):
    run = tmp_path / "exp72-k"
    (run / "stage2").mkdir(parents=True)
    (run / "stage2" / "model.safetensors").write_bytes(b"weights")
    (run / "stage2" / "config.json").write_text("{}")
    good = {"weights": {"model.safetensors": m.sha(str(run / "stage2" / "model.safetensors")), "config.json": m.sha(str(run / "stage2" / "config.json"))},
            "evals": {}}
    if case == "frozen_truncated":
        (run / "FROZEN.json").write_text(json.dumps(good)[:25])
    elif case == "weights_missing":
        (run / "FROZEN.json").write_text(json.dumps(good))
        os.remove(run / "stage2" / "model.safetensors")
    elif case == "wrong_sha":
        (run / "FROZEN.json").write_text(json.dumps(dict(good, weights=dict(good["weights"], **{"model.safetensors": "0" * 64}))))
    return str(run)


@data
@pytest.mark.parametrize("case,state", [("frozen_missing", "missing"), ("frozen_truncated", "invalid"), ("weights_missing", "missing"),
                                        ("wrong_sha", "invalid")])
def test_broken_frozen_arm(started, tmp_path, monkeypatch, case, state):
    fake_run(started, monkeypatch)
    k_run = broken_arm(tmp_path, case) if case != "frozen_missing" else broken_arm(tmp_path, "none")
    monkeypatch.setattr(m, "ARMS", dict(m.ARMS, k=k_run))
    res = m.report(started)
    assert res["states"]["k"].startswith(state)
    assert "fit_slice" in res["arms"]["raw"] and "G_reading" in res["arms"]["raw"]               # the other arm kept
    assert os.path.exists(os.path.join(started, "report.json")) and json.load(open(os.path.join(started, "report.json")))["route"] == res["route"]
    assert res["route"].startswith("incomplete" if state == "missing" else "invalid") and "no route" in res["route"]
