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


def aux_case(tmp_path, preds, mutate=None):
    """A number readout of the first 6 fit_O questions under exp40-M's checkpoint name (its tokenizer is local);
    preds: per question the prediction (None = no end token); mutate(record) for the refusal cases."""
    from transformers import AutoTokenizer
    from devision.train.rlcd import aux_example, aux_number
    tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, rep.INIT, "tokenizer"))
    rows = [json.loads(x) for x in open(os.path.join(ROOT, "data", "exp42", "fit_O.jsonl"))][:6]
    data = tmp_path / "set.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in rows))
    run = tmp_path / "run"
    (run / "eval").mkdir(parents=True)
    check = json.load(open(os.path.join(ROOT, "data", "exp44", "CHECK.json")))
    head = {k: v for k, v in check["head"].items() if k != "file_sha256"}
    recs = []
    for r, pred in zip(rows, preds):
        (q, v), = r["questions"].items()
        gold = next(o for o, p in r["gold"][q]["probabilities"].items() if p == 1.0)
        ids, _ = aux_example(tok, r["state_text"], v["instructions"], gold, 8)
        valid = pred is not None and bool(rep.NUMBER.match(pred))
        g = aux_number(gold)
        x = {"sample_id": r["id"], "qid": q, "image_id": r["image_id"], "image_used": r["image"],
             "aux_input_sha256": __import__("hashlib").sha256(json.dumps(ids).encode()).hexdigest(),
             "gold_number": g, "prediction": g if pred == "GOLD" else pred, "valid": valid or pred == "GOLD",
             "correct": pred == "GOLD"}
        recs.append(mutate(x) if mutate and len(recs) == 0 else x)
    (run / "eval" / "aux_x.details.jsonl").write_text("".join(json.dumps(x) + "\n" for x in recs))
    json.dump({"run": {"checkpoint": rep.INIT, "data_sha256": rep.r41.sha(str(data)), "slots": 8, "head": head}},
              open(run / "eval" / "aux_x.json", "w"))
    os.chdir(ROOT)
    return rep.aux_records(str(run), "aux_x", str(data), rep.INIT, head)


def test_validity_follows_the_parse_rule(tmp_path):
    recs = aux_case(tmp_path, ["GOLD", "abc", None, "12", "GOLD", "1,2x"])
    s = rep.aux_summary(recs)
    assert s["valid"] == {"valid": 3, "of": 6, "rate": 0.5}               # GOLD, "12", GOLD; "abc" / None / "1,2x" invalid
    assert sum(r["right"] for r in recs) == 2


@pytest.mark.parametrize("mutate", [
    lambda x: dict(x, gold_number="999"),                                  # another gold
    lambda x: dict(x, image_used="exp37/images/L/" + x["image_used"].split("/")[-1]),   # another picture
    lambda x: dict(x, aux_input_sha256="0" * 64),                          # another input
    lambda x: dict(x, prediction="abc", valid=True, correct=False),       # claims a non-number valid
])
def test_number_records_must_match_their_source(tmp_path, mutate):
    with pytest.raises(SystemExit):
        aux_case(tmp_path, ["GOLD"] * 6, mutate)


def test_conditions_need_all_four_main_sets():
    good = {"different": 0.9, "both": 0.5, "twin": 0.5}
    ci = {"value": 0.3, "ci95": [0.1, 0.5]}
    out = {"readout": {n: {"J": good} for n in rep.SETS},
           "differences": {"%s|J-%s" % (n, b): {"both": ci, "twin": ci, "different": ci} for n in rep.SETS for b in "MC"}}
    assert rep.conditions(out) == {"progress": True, "number_loss_attribution": True}
    del out["readout"]["rl_tune_L"]                                        # rl_tune passes, rl_tune_L missing
    assert rep.conditions(out) == {"progress": "incomplete", "number_loss_attribution": "incomplete"}
