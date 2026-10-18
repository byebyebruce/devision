"""exp48 scripts without a model: the sidecar covers every training row once in strict quads; malformed quads (missing
corner, options differing, golds not exchanged) are refused; the report refuses an arm with another weight or sidecar,
reproduces G's EXP45 readout, and gives "incomplete" conditions unless both arms' four sets are read."""
import importlib.util
import json
import os

import pytest

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(ROOT, "data")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


b = load("exp48_build", os.path.join(HERE, "build.py"))
rep = load("exp48_report", os.path.join(HERE, "report.py"))


def test_sidecar_covers_every_row_once():
    side = [json.loads(x) for x in open(os.path.join(DATA, "exp48", "quads.jsonl"))]
    train = [json.loads(x) for x in open(os.path.join(DATA, "exp42", "train.jsonl"))]
    assert len(side) == len(train) == 4608 and len({(s["id"], s["qid"]) for s in side}) == 4608
    by = {}
    for s in side:
        by.setdefault(s["quad"], set()).add((s["view"], s["object"]))
    assert len(by) == 1152 and all(v == {(0, 0), (0, 1), (1, 0), (1, 1)} for v in by.values())


def quad_rows(gold_v1_first="5 u", options2=None):
    def row(view, obj, gold, opts=None):
        opts = opts or ["4 u", "5 u", "6 u"]
        ins = "What is the mass of each particle in sample %s?" % ("A" if obj == 0 else "B")
        return {"id": "real-layout:f1:v%d:%d" % (view, obj), "family": "f1", "view": "v%d" % view, "image": "p%d.png" % view,
                "state_text": "s", "questions": {"q": {"type": "choice", "instructions": ins, "criteria": {o: None for o in opts}}},
                "gold": {"q": {"probabilities": {o: float(o == gold) for o in opts}}}}
    return [row(0, 0, "4 u"), row(0, 1, "5 u"), row(1, 0, gold_v1_first), row(1, 1, "4 u", options2)]


def test_quads_of_refuses_malformed_quads():
    key = lambda r: (r["family"],)  # noqa: E731
    assert len(b.quads_of(quad_rows(), key)) == 1
    with pytest.raises(SystemExit):
        b.quads_of(quad_rows()[:3], key)                                         # a corner missing
    with pytest.raises(SystemExit):
        b.quads_of(quad_rows(gold_v1_first="6 u"), key)                          # not exchanged
    with pytest.raises(SystemExit):
        b.quads_of(quad_rows(options2=["5 u", "4 u", "6 u"]), key)               # options in another order
    loose = b.quads_of(quad_rows(gold_v1_first="6 u"), key, strict=False)
    assert loose["quads"] == {} and loose["excluded"] == {"no exchange": 1}


def test_report_identity_conditions_and_g_readout(tmp_path):
    man = json.load(open(os.path.join(DATA, "exp42", "MANIFEST.json")))
    qsha = json.load(open(os.path.join(DATA, "exp48", "MANIFEST.json")))["sha256"]

    def make(name, arm, weight, sha=qsha):
        run = tmp_path / name
        (run / "stage2").mkdir(parents=True)
        json.dump({"init": rep.G_CKPT, "init_sha256": rep.G_SHA, "data_sha256": man["files"]["train"], "interaction_weight": weight,
                   "quads_sha256": sha}, open(run / "START.json", "w"))
        json.dump({"config": {"stages": [{"data": "data/exp42/train.jsonl", "init": rep.G_CKPT, "params": {
            "objective": "ce", "interaction_file": rep.QUADS, "interaction_weight": weight}}]}}, open(run / "run.json", "w"))
        json.dump({"objective": "ce", "interaction": {"weight": weight, "file_sha256": sha}}, open(run / "stage2" / "train_report.json", "w"))
        return str(run)
    assert rep.arm_identity("X", make("x", "X", 1.0), man, qsha) == ""
    assert rep.arm_identity("X", make("x0", "X", 0.0), man, qsha).startswith("wrong")
    assert rep.arm_identity("C", make("c1", "C", 0.0, sha="0" * 64), man, qsha).startswith("wrong")
    os.chdir(ROOT)
    out = rep.build({"G": rep.G_RUN, "C": "/nonexistent/c", "X": "/nonexistent/x"})
    assert set(out["conditions"].values()) == {"incomplete"}
    assert out["readout"]["rl_tune"]["G"]["both"] == pytest.approx(0.0035, abs=1e-3)
    assert out["quads"]["rl_tune"]["G"]["quads"] == 176 and out["quads"]["rl_tune"]["G"]["excluded"] == {"no exchange": 274}
