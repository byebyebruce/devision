# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false
"""exp68's runner without a model: the arm is EXP65-C's on data/exp68 (anything else refused); START binds EXP63, EXP65
(run, C's end and registered records), data/exp68 and refuses changes; the report reads C's reused EXP65 records and the
arm's through the same checks and gives the pre-written decision (fixtures: the arm = C's own records -> does not
meet; every diagnostic right -> meets, also with low but valid new diagnostics; a missing new diagnostic of X or C ->
incomplete, a changed / inconsistent one -> invalid, both without a route); the run's order and limit."""
import importlib.util
import json
import os
import shutil

import pytest

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.dirname(HERE))
spec = importlib.util.spec_from_file_location("exp68_runner", os.path.join(HERE, "exp68.py"))
assert spec is not None and spec.loader is not None
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
has = os.path.exists(os.path.join(ROOT, "runs", "exp65", "details_c_diag.jsonl")) and os.path.exists(os.path.join(ROOT, "data", "exp68", "MANIFEST.json"))
data = pytest.mark.skipif(not has, reason="EXP65's run / data/exp68 not present")
FAKE_X = {"model.safetensors": "f" * 64, "config.json": "e" * 64}


@pytest.fixture(autouse=True)
def in_root(monkeypatch):
    monkeypatch.chdir(ROOT)


@data
def test_the_arm_is_exp65_c_on_the_new_data(tmp_path, monkeypatch):
    m.check_yaml()
    y = open(m.YAML).read().replace("lr_head: 1.0e-05", "lr_head: 2.0e-05")
    (tmp_path / "x.yaml").write_text(y)
    monkeypatch.setattr(m, "YAML", str(tmp_path / "x.yaml"))
    with pytest.raises(SystemExit, match="not EXP65-C's arm"):
        m.check_yaml()


@pytest.fixture
def started(tmp_path):
    import time
    out = str(tmp_path / "exp68")
    m.stage_start(int(time.time()) + 3600, out=out)
    return out


@data
def test_start_binds_and_refuses(started, tmp_path, monkeypatch):
    st = m.check_start(started, code=False)
    assert st["requests"]["total"] == 3040 and st["exp65"]["c_end"]["model.safetensors"].startswith("382e0a44")
    with pytest.raises(SystemExit, match="exists"):
        m.stage_start(0, out=started)
    d = tmp_path / "d68"
    shutil.copytree(m.D68, d)
    with open(d / "diag.jsonl", "a") as fh:
        fh.write("\n")
    monkeypatch.setattr(m, "D68", str(d))
    with pytest.raises(SystemExit, match="data/exp68"):
        m.check_start(started, code=False)


def records_of(out_src: str, name: str):
    return [json.loads(x) for x in open(os.path.join(out_src, "details_%s.jsonl" % name))][1:]


def right(rows, wrong_axes=()):
    """Records answering every question right (a diag row of an axis in wrong_axes: always 'sample B')."""
    out = []
    for rid, r in rows.items():
        for qid, q in r["questions"].items():
            gold = r["gold"][qid]["probabilities"]
            g = max(gold, key=gold.get)
            pred = "sample B" if r.get("axis") in wrong_axes else g
            probs = {o: (0.98 if o == pred else 0.01) for o in q["criteria"]}
            out.append({"sample_id": rid, "qid": qid, "image_id": r.get("image_id", rid), "kind": r.get("kind", ""), "axis": r.get("axis", ""),
                        "options": list(q["criteria"]), "gold": gold, "gold_answer": g, "probabilities": probs, "prediction": pred,
                        "correct": pred == g, "nll": 0.02 if pred == g else 4.6, "brier": 0.0})
    return out


def fake_run(out, monkeypatch, x_recs, c68=True):
    """The arm's records written and registered by the runner's own writer under a fake frozen identity; C's new-diag
    records likewise; screens = C's recorded ones."""
    s_sha = m.sha(os.path.join(out, "START.json"))
    c_scr = {n: m.e57.screen(m.C_RUN, n, json.load(open(os.path.join(out, "START.json")))["exp57_screens"][n], start_run=m.e63.E57) for n in m.e57.SCREENS}
    real = m.fz.check
    monkeypatch.setattr(m.fz, "check", lambda run, evals: {"weights": FAKE_X, "evals": {}} if run == m.RUN else real(run, evals))
    monkeypatch.setattr(m.e57, "screen", lambda run, n, *a, **k: c_scr[n])
    for c, recs in x_recs.items():
        _, inputs = m.inputs_of(c)
        m.e57.write_set(out, "x_" + c, recs, {"set": "x_" + c, "model_of": "x", "start_sha256": s_sha, "inputs_sha256": m.e57.hash_inputs(inputs),
                                              "model": FAKE_X, "temperature": "none (raw)", "image": None, "device": "cpu"})
    if c68:
        rows, inputs = m.inputs_of("diag68")
        st = json.load(open(os.path.join(out, "START.json")))
        m.e57.write_set(out, "c_diag68", right(rows, m.CONFLICT), {"set": "c_diag68", "model_of": "c", "start_sha256": s_sha,
                                                                    "inputs_sha256": m.e57.hash_inputs(inputs), "model": st["exp65"]["c_end"],
                                                                    "temperature": "none (raw)", "image": None, "device": "cpu"})


@data
def test_report_arm_equal_to_c_does_not_meet(started, monkeypatch):
    x = {c: records_of(m.E65, "c_" + c) for c in ("fit", "G", "P", "diag")}
    x["diag68"] = right(m.inputs_of("diag68")[0])
    fake_run(started, monkeypatch, x)
    res = m.report(started)
    a = res["x"]
    assert all(v is True for v in a["gates"].values())                                      # C passed EXP63's gates
    assert a["diag"]["opposite_mass_wins"]["accuracy"] == pytest.approx(38 / 80)
    assert a["conflict_gain_over_c"]["value"] == 0 and a["practical"] is False and res["decision"].startswith("the new data recipe does not")
    assert a["diag68"]["accuracy"]["opposite_mass_wins"] == 1.0 and res["c_diag68"]["accuracy"]["opposite_mass_wins"] == 0.5   # always "sample B": right in the orders where B wins
    assert a["diag68_conflict_gain_over_c"]["ci95"][0] > 0 and res["candidate_eligible"] is False


def all_right_x():
    x = {c: records_of(m.E65, "c_" + c) for c in ("fit", "G", "P")}
    x["diag"] = right(m.inputs_of("diag")[0])
    return x


@data
def test_report_meets_with_valid_new_diags_even_when_they_score_low(started, monkeypatch):
    x = all_right_x()
    x["diag68"] = right(m.inputs_of("diag68")[0], m.CLASSES)                 # every new-diag answer "sample B": low, but valid
    fake_run(started, monkeypatch, x)
    res = m.report(started)
    assert res["new_diag_evidence"] == "completed" and res["x"]["diag68"]["accuracy"]["equal"] == 0.0
    assert res["x"]["practical"] is True and res["decision"].startswith("the new data recipe meets")      # report-only in score


@data
@pytest.mark.parametrize("case", ["x_missing", "c_missing"])
def test_missing_new_diag_gives_no_route(started, monkeypatch, case):
    x = all_right_x()
    if case != "x_missing":
        x["diag68"] = right(m.inputs_of("diag68")[0])
    fake_run(started, monkeypatch, x, c68=case != "c_missing")
    res = m.report(started)
    assert res["x"]["practical"] is True                                     # the metrics stay computed ...
    assert res["new_diag_evidence"] == "incomplete" and res["decision"].startswith("incomplete") and "no route" in res["decision"]


@data
@pytest.mark.parametrize("case", ["changed_after_registration", "prediction_not_argmax"])
def test_invalid_new_diag_gives_no_route(started, monkeypatch, case):
    x = all_right_x()
    x["diag68"] = right(m.inputs_of("diag68")[0])
    if case == "prediction_not_argmax":
        r = x["diag68"][0]
        r["prediction"] = next(o for o in r["options"] if o != r["prediction"])                  # a legal option, not the argmax
        r["correct"] = r["prediction"] == r["gold_answer"]
    fake_run(started, monkeypatch, x)
    if case == "changed_after_registration":
        with open(os.path.join(started, "details_x_diag68.jsonl"), "a") as fh:
            fh.write("\n")
    res = m.report(started)
    assert res["states"]["x_diag68"].startswith("invalid") and res["new_diag_evidence"] == "invalid"
    assert res["decision"].startswith("invalid") and "no route" in res["decision"]


@data
def test_report_without_the_arm_is_incomplete(started):
    res = m.report(started)
    assert res["decision"].startswith("incomplete") and res["x"]["practical"] == "incomplete"


STUB = """#!/bin/bash
echo "uv $*" >> "${MARK_CALLS}"
case "$*" in
  *"watch.py"*) shift 2; exec python3 "$@";;
  *"exp68.py start"*) mkdir -p runs/exp68; exit 0;;
  *"exp68.py train"*) [ -n "${FAIL_TRAIN:-}" ] && exit 6; exit 0;;
  *"exp68.py freeze"*|*"exp68.py decide"*|*"devision-eval"*|*"freeze.py record"*) exit 0;;
  *"exp68.py report"*) echo report > runs/exp68/report.txt; exit 0;;
esac
exit 9
"""


def chain(tmp_path, extra):
    import stat as st_
    import subprocess
    root = tmp_path / "repo"
    for rel in ("scripts/exp68/run.sh", "scripts/exp53/watch.py", "scripts/exp35/guard.sh", "scripts/exp38/steps.sh"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join(ROOT, rel), root / rel)
    bdir = tmp_path / "bin"
    bdir.mkdir()
    for name, body in {"uv": STUB, "pgrep": '#!/bin/bash\n[ "$1" = "-fl" ] && exit 1\nexec /usr/bin/pgrep "$@"\n'}.items():
        (bdir / name).write_text(body)
        (bdir / name).chmod((bdir / name).stat().st_mode | st_.S_IEXEC)
    env = {**os.environ, "PATH": "%s:%s" % (bdir, os.environ["PATH"]), "MARK_CALLS": str(tmp_path / "calls"), "EXP68_LIMIT": "60",
           "EXP68_RESERVE": "4", **extra}
    p = subprocess.run(["bash", str(root / "scripts/exp68/run.sh")], env=env, capture_output=True, text=True, timeout=120)
    return root, p, (tmp_path / "calls").read_text().splitlines()


def test_run_order_failed_stage_and_limit(tmp_path):
    root, p, calls = chain(tmp_path / "a", {})
    assert p.returncode == 0, p.stdout + p.stderr
    first = lambda s: next(i for i, c in enumerate(calls) if s in c)  # noqa: E731
    assert first("exp68.py start") < first("exp68.py train") < first("exp68.py freeze") < first("exp68.py decide") \
        < first("exp68-x/eval/exist") < first("exp68-x/eval/science") < first("exp68.py report")
    root, p, calls = chain(tmp_path / "b", {"FAIL_TRAIN": "1"})
    joined = "\n".join(calls)
    assert p.returncode == 6 and "exp68.py decide" not in joined and "report --finalize" in joined
    s = open(os.path.join(ROOT, "scripts/exp68/run.sh")).read()
    assert "${EXP68_LIMIT:-3600}" in s and "${EXP68_RESERVE:-120}" in s
