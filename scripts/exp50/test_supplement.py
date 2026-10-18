"""exp50 supplement without a model (critic EXP50-SUP-PREP-01), in a fake run tree: the missing list comes from C's status
alone, in the chain's order; nothing is planned before the chain's report and deadline, for an untrained or unfrozen
C, or when T is missing a set; a record is refused for changed weights, another checkpoint / temperature / input, or
details that do not answer every question; the report refuses a changed original product or a changed supplement
file, and never writes the original files."""
import hashlib
import importlib.util
import json
import os

import pytest

HERE = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("exp50_supplement", os.path.join(HERE, "supplement.py"))
assert _spec is not None and _spec.loader is not None
sup = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sup)
RUNS = {"X": "runs/exp48-x", "T": "runs/exp50-t", "C": "runs/exp50-c"}
ROWS = [{"id": "r%d" % i, "image_id": "i%d" % i, "image": "p.png", "questions": {"q": {"type": "choice", "instructions": "x",
         "criteria": {"a": None, "b": None}}}, "gold": {"q": {"probabilities": {"a": 1.0, "b": 0.0}}}} for i in range(3)]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def details(rows, drop=0):
    return [{"sample_id": r["id"], "qid": "q", "gold": r["gold"]["q"]["probabilities"], "image_id": r["image_id"],
             "probabilities": {"a": 0.7, "b": 0.3}, "prediction": "a", "gold_answer": "a", "correct": True} for r in rows[drop:]]


def write_eval(d, name, ckpt, data, rows, drop=0, **run):
    os.makedirs(os.path.join(d, "eval"), exist_ok=True)
    info = {"checkpoint": ckpt, "data": data, "data_sha256": sha(data), "temperature": "none (raw)", "control": "none",
            "code": "52ff548+dirty", **run}
    json.dump({"run": info}, open(os.path.join(d, "eval", name + ".json"), "w"))
    with open(os.path.join(d, "eval", name + ".details.jsonl"), "w") as fh:
        fh.write("".join(json.dumps(x) + "\n" for x in details(rows, drop)))


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = {}
    for n in sup.ORDER:
        p = "data/%s.jsonl" % n
        os.makedirs("data", exist_ok=True)
        open(p, "w").write("".join(json.dumps(r) + "\n" for r in ROWS))
        data[n] = p
    monkeypatch.setattr(sup, "DATA", data)
    monkeypatch.setattr(sup.rep, "arm_identity", lambda *a: "")
    monkeypatch.setattr(sup.r41, "sha", sha)
    os.makedirs("data/exp42", exist_ok=True)
    json.dump({"files": {}}, open("data/exp42/MANIFEST.json", "w"))
    os.makedirs("data/exp48", exist_ok=True)
    open("data/exp48/quads.jsonl", "w").write("{}\n")
    for who, run in RUNS.items():
        os.makedirs(run + "/stage2")
        open(run + "/stage2/model.safetensors", "wb").write(who.encode())
        open(run + "/stage2/config.json", "w").write("{}")
        fr = sup.frozen.freeze(run)
        done = sup.ORDER if who != "C" else sup.ORDER[:3]
        for n in done:
            write_eval(run, n, run + "/stage2", data[n], ROWS)
            sup.frozen.record(run, n)
        if who == "X":
            monkeypatch.setattr(sup.rep, "X_SHA", fr["weights"])
        else:
            st = ["train completed"] + ["eval:%s completed" % n for n in done]
            if who == "C":                                     # rl_tune_L half-written and interrupted, then none
                write_eval(run, "rl_tune_L", run + "/stage2", data["rl_tune_L"], ROWS, drop=2)
                st.append("eval:rl_tune_L interrupted")
            open(run + "-status.txt", "w").write("\n".join(st) + "\n")
    open(sup.ORIG_REPORT, "w").write(json.dumps({"conditions": {"progress_C": "incomplete"}, "runs": {}}))
    open(sup.ORIG_TXT, "w").write("x\n")
    open(sup.ORIG_OUT, "w").write("== final report (exit 0)\n")
    open(sup.DEADLINE_FILE, "w").write("100")
    return data


def originals():
    fs = [sup.ORIG_REPORT, sup.ORIG_TXT, sup.ORIG_OUT] + [os.path.join(dp, f) for r in RUNS.values()
                                                         for dp, _, fn in os.walk(r) for f in fn] + [r + "-status.txt" for r in ("runs/exp50-t", "runs/exp50-c")]
    return {f: sha(f) for f in fs}


def test_missing_by_status_only_in_chain_order():
    st = {"train": "completed", "eval:fit_O": "completed", "eval:rl_tune": "completed", "eval:fit_L": "failed",
          "eval:rl_tune_L": "interrupted", "eval:exist": "missing"}
    assert sup.missing_of(st) == ["fit_L", "rl_tune_L", "exist", "science"]


def test_plan_preconditions(world):
    s = sup.plan(RUNS, now=200)
    assert s["missing"] == ["rl_tune_L", "exist", "science"] and s["reused_C"] == ["fit_O", "fit_L", "rl_tune"]
    assert s["inputs"]["rl_tune_L"] == sha(world["rl_tune_L"]) and s["original"]["report.json"] == sha(sup.ORIG_REPORT)
    with pytest.raises(SystemExit, match="one supplement"):
        sup.plan(RUNS, now=200)
    os.remove(sup.OUT + "/START.json")
    with pytest.raises(SystemExit, match="deadline"):
        sup.plan(RUNS, now=50)
    open(sup.ORIG_OUT, "w").write("still running\n")
    with pytest.raises(SystemExit, match="final report"):
        sup.plan(RUNS, now=200)
    open(sup.ORIG_OUT, "w").write("== final report (exit 0)\n")
    open("runs/exp50-c-status.txt", "a").write("train interrupted\n")
    with pytest.raises(SystemExit, match="no resume"):
        sup.plan(RUNS, now=200)


def test_t_missing_is_refused(world):
    open("runs/exp50-t-status.txt", "a").write("eval:science interrupted\n")
    with pytest.raises(SystemExit, match="only C"):
        sup.plan(RUNS, now=200)


def test_record_checks_and_report_keeps_originals(world, monkeypatch):
    before = originals()
    sup.plan(RUNS, now=200)
    ev = sup.OUT + "/c"
    write_eval(ev, "rl_tune_L", "runs/exp50-c/stage2", world["rl_tune_L"], ROWS, drop=1)        # incomplete details
    with pytest.raises(SystemExit):
        sup.record("rl_tune_L", RUNS)
    write_eval(ev, "rl_tune_L", "runs/exp50-t/stage2", world["rl_tune_L"], ROWS)                 # another checkpoint
    with pytest.raises(SystemExit, match="not C's end"):
        sup.record("rl_tune_L", RUNS)
    write_eval(ev, "rl_tune_L", "runs/exp50-c/stage2", world["rl_tune_L"], ROWS, temperature="fitted")
    with pytest.raises(SystemExit, match="not C's end"):
        sup.record("rl_tune_L", RUNS)
    with pytest.raises(SystemExit, match="not a missing set"):
        sup.record("fit_O", RUNS)
    write_eval(ev, "rl_tune_L", "runs/exp50-c/stage2", world["rl_tune_L"], ROWS)
    m = sup.record("rl_tune_L", RUNS)
    assert set(m["evals"]) == {"rl_tune_L"}
    src = sup.sources(RUNS)
    assert src[("C", "rl_tune_L")][2] == "supplement" and src[("C", "fit_O")][2] == "original"
    assert ("C", "exist") not in src                                                        # not read: stays missing
    monkeypatch.setattr(sup, "readout", lambda s: {"conditions": {"progress_C": "x"}, "readout": {}, "differences": {},
                                                   "cut_slice": {}, "risk": {}, "quads": {}})
    res = sup.report(RUNS)
    assert res["still_missing_C"] == ["exist", "science"] and res["original_90min"]["conditions"] == {"progress_C": "incomplete"}
    assert originals() == before                                                            # nothing original written
    open(ev + "/eval/rl_tune_L.details.jsonl", "a").write("\n")
    with pytest.raises(SystemExit, match="changed after it was recorded"):
        sup.sources(RUNS)
    write_eval(ev, "rl_tune_L", "runs/exp50-c/stage2", world["rl_tune_L"], ROWS)
    open("runs/exp50-c/stage2/model.safetensors", "wb").write(b"other")                     # weights changed
    with pytest.raises(SystemExit):
        sup.record("rl_tune_L", RUNS)
    with pytest.raises(SystemExit):
        sup.sources(RUNS)


def test_changed_original_product_is_refused(world):
    sup.plan(RUNS, now=200)
    open(sup.ORIG_TXT, "a").write("edited\n")
    with pytest.raises(SystemExit, match="original report.txt changed"):
        sup.sources(RUNS)
