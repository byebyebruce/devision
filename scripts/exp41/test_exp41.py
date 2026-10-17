"""exp41 without a model: the pictures are cut from the original PNGs (keep: same canvas, original pixels inside the
box, white outside; zoom / other: the asked / other object's box), each row asks that object's original reading
questions with the original options and gold and the original state_text; the readout refuses missing or repeated
answers and counts same / different value pairs, both objects right and the other object's gold."""
import importlib.util
import json
import os

import numpy as np
import pytest
from PIL import Image

HERE = os.path.dirname(__file__)
DATA = os.path.join(os.path.dirname(os.path.dirname(HERE)), "data")


def load(name):
    spec = importlib.util.spec_from_file_location("exp41_" + name, os.path.join(HERE, name + ".py"))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rep = load("report")


def test_pictures_and_questions_map_the_originals():
    rows = [json.loads(x) for x in open(os.path.join(DATA, "exp41", "questions.jsonl"))]
    pics = json.load(open(os.path.join(DATA, "exp41", "pictures.json")))
    tune = {json.loads(x)["id"]: json.loads(x) for x in open(os.path.join(DATA, "real_layout", "questions_tune.jsonl"))}
    assert len(rows) == 720 and len({r["family"] for r in rows}) == 60
    asked = {}
    for r in rows:
        o = tune[r["orig_id"]]
        assert r["state_text"] == o["state_text"] and r["image"] != o["image"]
        for q, v in r["questions"].items():
            assert v == o["questions"][q] and r["gold"][q] == o["gold"][q]
            assert r["object"].lower() in v["instructions"].lower()
            asked.setdefault(r["condition"], set()).add((r["orig_id"], q))
    reading = {(i, q) for i, o in tune.items() if o["family"] in {r["family"] for r in rows}
               for q, v in o["questions"].items() if any(w in v["instructions"] for w in ("mass", "speed", "distance"))}
    assert all(s == reading for s in asked.values())                      # every reading question once per condition
    for r in rows:
        orig = np.asarray(Image.open(os.path.join(DATA, tune[r["orig_id"]]["image"])).convert("RGB"))
        img = np.asarray(Image.open(os.path.join(DATA, r["image"])).convert("RGB"))
        box = pics[r["image"].split("/")[-1][:-4]]["box"]
        x0, y0, x1, y1 = box
        assert 0 <= x0 < x1 <= orig.shape[1] and 0 <= y0 < y1 <= orig.shape[0]
        if r["condition"] == "keep":
            assert img.shape == orig.shape and (img[y0:y1, x0:x1] == orig[y0:y1, x0:x1]).all()
            outside = np.ones(orig.shape[:2], bool)
            outside[y0:y1, x0:x1] = False
            assert (img[outside] == 255).all()
        else:
            assert (img == orig[y0:y1, x0:x1]).all()


def rec(cond, oid, obj, field, gold, pred, fam="rl1", kind="temp"):
    return {"condition": cond, "orig_id": oid, "object": obj, "field": field, "gold": gold, "prediction": pred,
            "right": gold == pred, "family": fam, "kind": kind}


def test_units_count_pairs():
    recs = [rec("keep", "a:v0", "sample A", "mass", "10 u", "10 u"), rec("keep", "a:v0", "sample B", "mass", "20 u", "20 u"),
            rec("other", "a:v0", "sample A", "mass", "10 u", "20 u"), rec("other", "a:v0", "sample B", "mass", "20 u", "10 u"),
            rec("keep", "a:v0", "sample A", "speed", "5 m/s", "5 m/s"), rec("keep", "a:v0", "sample B", "speed", "5 m/s", "6 m/s")]
    fam = rep.units(recs)
    fs = sorted(fam)
    assert rep.value(fam, fs, "keep:different") == 1.0 and rep.value(fam, fs, "keep:both") == 1.0
    assert rep.value(fam, fs, "keep:same") == 0.5
    assert rep.value(fam, fs, "other:shown_other") == 1.0 and rep.value(fam, fs, "other:wrong_is_other") == 1.0
    with pytest.raises(SystemExit):
        rep.units(recs[:-1])                                              # one object of a pair missing


def test_records_refuse_missing_and_repeated():
    data = [{"id": "x", "family": "rl1", "kind": "temp", "orig_id": "o:v0", "object": "sample A", "condition": "zoom",
             "questions": {"q1": {"instructions": "What is the mass of each particle in sample A?"}},
             "gold": {"q1": {"probabilities": {"10 u": 1.0, "20 u": 0.0}}}}]
    g = {"sample_id": "x", "qid": "q1", "gold_answer": "10 u", "prediction": "10 u", "correct": True}
    assert len(rep.records(data, [g], [])) == 1
    with pytest.raises(SystemExit):
        rep.records(data, [g, g], [])
    with pytest.raises(SystemExit):
        rep.records(data, [], [])
