"""exp44 scripts without a real model: the number readout runs end to end on a tiny checkpoint (one record per reading
question, same input for both views, invalid when no end token / number, never corrected to an option); picture
dependence scores the twin's prediction against this question's gold only where the number inputs are identical; J's
identity refuses another objective, weight or head; a missing J is "incomplete"."""
import importlib.util
import json
import os
import sys

import pytest

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tests"))
sys.path.insert(0, os.path.join(ROOT, "tests", "train"))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rep = load("exp44_report", os.path.join(HERE, "report.py"))
readout = load("exp44_aux_readout", os.path.join(HERE, "aux_readout.py"))


def test_number_readout_end_to_end_on_a_tiny_checkpoint(tmp_path):
    from conftest import tiny_decider  # type: ignore
    from test_number_aux import number_samples  # type: ignore
    rows = number_samples(tmp_path / "data", 6)
    rows = [dict(r, questions={"q": dict(r["questions"]["q"])}) for r in rows]
    path = tmp_path / "set.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    tiny_decider(seed=1).save(tmp_path / "ckpt")
    readout.main([str(tmp_path / "ckpt"), str(path), str(tmp_path / "out"), "aux_x", "--head", "", "--device", "cpu",
                  "--data-root", str(tmp_path / "data")])
    recs = [json.loads(x) for x in open(tmp_path / "out" / "aux_x.details.jsonl")]
    info = json.load(open(tmp_path / "out" / "aux_x.json"))["run"]
    assert len(recs) == 6 and info["slots"] == 8 and info["head"] == {"repo": ""}
    assert all(r["correct"] == (r["valid"] and r["prediction"] == r["gold_number"]) for r in recs)
    assert len({r["aux_input_sha256"] for r in recs}) == 1          # same state text and question -> same input


def rec(fam, view, obj, gold, pred, inp="x"):
    return {"family": fam, "case": fam, "field": "mass", "object": obj, "view": view, "picture": fam + view,
            "gold": gold, "prediction": pred, "right": pred == gold, "aux_input": inp}


def test_picture_dependence_uses_the_twin_only_with_identical_inputs():
    recs = [rec("f1", "v0", "A", "10", "10"), rec("f1", "v1", "A", "20", "20"),     # reads each picture: +1, +1
            rec("f2", "v0", "A", "10", "10"), rec("f2", "v1", "A", "20", "10"),     # always 10: v0 1-0, v1 0-1
            rec("f3", "v0", "A", "5", "5"), rec("f3", "v1", "A", "5", "5"),         # value unchanged: not counted
            rec("f4", "v0", "A", "1", "1", "a"), rec("f4", "v1", "A", "2", "2", "b")]   # different inputs: skipped
    dep = rep.dependence(recs)
    assert dep["_skipped"]["n"][0] == 1
    assert dep["f1"]["dep:mass"] == [2.0, 2] and dep["f2"]["dep:mass"] == [0.0, 2] and "f3" not in dep


def test_j_identity_refuses_another_recipe(tmp_path):
    check = json.load(open(os.path.join(ROOT, "data", "exp44", "CHECK.json")))
    man = json.load(open(os.path.join(ROOT, "data", "exp42", "MANIFEST.json")))
    head = {"repo": "answerdotai/ModernBERT-large", **{k: v for k, v in check["head"].items() if k != "file_sha256"}}

    def make(name, weight=1.0, objective="ce", head=head):
        run = tmp_path / name
        (run / "stage2").mkdir(parents=True)
        json.dump({"init": rep.INIT, "init_sha256": rep.INIT_SHA, "data_sha256": man["files"]["train"]}, open(run / "START.json", "w"))
        json.dump({"config": {"stages": [{"data": "data/exp42/train.jsonl", "init": rep.INIT,
                                          "params": {"objective": objective, "aux_weight": weight, "aux_slots": 8}}]}}, open(run / "run.json", "w"))
        json.dump({"objective": objective, "aux": {"weight": weight, "slots": 8, "head": head}}, open(run / "stage2" / "train_report.json", "w"))
        return str(run)
    assert rep.j_identity(make("ok"), check, man) == ""
    assert rep.j_identity(make("w", weight=0.5), check, man).startswith("wrong")
    assert rep.j_identity(make("o", objective="rlcd"), check, man).startswith("wrong")
    assert rep.j_identity(make("h", head=dict(head, **{"decoder.bias": "0"})), check, man).startswith("wrong")
    assert rep.j_identity(str(tmp_path / "none"), check, man).startswith("incomplete")


def test_missing_j_is_incomplete():
    os.chdir(ROOT)
    out = rep.build({"M": "runs/exp42-start", "C": "runs/exp42-c", "J": "/nonexistent/exp44-j"}, m_aux="/nonexistent/m-aux")
    assert out["runs"]["J"].startswith("incomplete") and out["conditions"]["progress"] == "incomplete"
    assert out["aux"]["aux_rl_tune"]["M"] == "missing"
    assert out["readout"]["rl_tune"]["C"]["both"] == pytest.approx(0.0128, abs=1e-3)
