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
    assert len(rows) == 840 and len({r["family"] for r in rows}) == 60
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
    temp_reading = {k for k in reading if tune[k[0]]["kind"] == "temp"}
    assert all(s == reading for c, s in asked.items() if c != "keep_object")   # every reading question once per condition
    assert asked["keep_object"] == temp_reading                                  # keep_object: temperature only
    halves = {}
    for r in rows:
        orig = np.asarray(Image.open(os.path.join(DATA, tune[r["orig_id"]]["image"])).convert("RGB"))
        img = np.asarray(Image.open(os.path.join(DATA, r["image"])).convert("RGB"))
        box = pics[r["image"].split("/")[-1][:-4]]["box"]
        x0, y0, x1, y1 = box
        assert 0 <= x0 < x1 <= orig.shape[1] and 0 <= y0 < y1 <= orig.shape[0]
        if r["condition"] == "keep_object":        # the asked object's whole half (by its drawn side), the other half white
            w = orig.shape[1]
            assert (x0, y0, y1) == (0, 0, orig.shape[0]) and x1 == w // 2 or (x0, y0, x1, y1) == (w // 2, 0, w, orig.shape[0])
            halves.setdefault(r["orig_id"], set()).add((x0, r["object"]))
        if r["condition"] in ("keep", "keep_object"):
            assert img.shape == orig.shape and (img[y0:y1, x0:x1] == orig[y0:y1, x0:x1]).all()
            outside = np.ones(orig.shape[:2], bool)
            outside[y0:y1, x0:x1] = False
            assert (img[outside] == 255).all()
        else:
            assert (img == orig[y0:y1, x0:x1]).all()
    assert len(halves) == 60 and all(len({x for x, _ in v}) == 2 for v in halves.values())   # A / B on different halves


def fake_details(rows, wrong=lambda r, q: False):
    """devision-eval-like records: the most probable option is the gold unless wrong(r, q), then the first other one."""
    out = []
    for r in rows:
        for q, g in r["gold"].items():
            probs = g["probabilities"]
            gold = next(o for o, p in probs.items() if p == 1.0)
            pred = next(o for o in probs if o != gold) if wrong(r, q) else gold
            p = {o: (0.8 if o == pred else 0.2 / (len(probs) - 1)) for o in probs}
            out.append({"sample_id": r["id"], "qid": q, "image_id": r["image_id"], "gold": probs, "gold_answer": gold,
                        "options": list(probs), "probabilities": p, "prediction": pred, "correct": pred == gold})
    return out


def fake_run(tmp_path, tamper=None):
    """A frozen M with rl_tune recorded and an exp41 evaluation of the real exp41 questions."""
    fz = rep.frozen
    m = tmp_path / "exp40-m"
    (m / "stage2").mkdir(parents=True)
    (m / "eval").mkdir()
    (m / "stage2" / "model.safetensors").write_bytes(b"w")
    (m / "stage2" / "config.json").write_text("{}")
    tune = [json.loads(x) for x in open(os.path.join(DATA, "real_layout", "questions_tune.jsonl"))]
    write = lambda path, recs: open(path, "w").writelines(json.dumps(x) + "\n" for x in recs)  # noqa: E731
    write(m / "eval" / "rl_tune.details.jsonl", fake_details(tune, lambda r, q: q == "q2"))
    json.dump({"run": {}}, open(m / "eval" / "rl_tune.json", "w"))
    fz.freeze(str(m))
    fz.record(str(m), "rl_tune")
    out = tmp_path / "exp41"
    out.mkdir()
    data = [json.loads(x) for x in open(os.path.join(DATA, "exp41", "questions.jsonl"))]
    recs = fake_details(data, lambda r, q: r["condition"] == "keep_object")
    info = {"checkpoint": str(m / "stage2"), "temperature": "fitted", "control": "none",
            "data_sha256": json.load(open(os.path.join(DATA, "exp41", "MANIFEST.json")))["sha256"]}
    if tamper:
        recs, info = tamper(recs, info, m)
    write(out / "m.details.jsonl", recs)
    json.dump({"run": info}, open(out / "m.json", "w"))
    return str(m), str(out)


def build(m, out):
    return rep.build(m, out, os.path.join(DATA, "exp41", "questions.jsonl"), os.path.join(DATA, "exp41", "MANIFEST.json"))


def test_readout_end_to_end_on_a_fake_frozen_m(tmp_path):
    res = build(*fake_run(tmp_path))
    assert res["families"] == 60 and res["clusters"] == 42
    c = res["counts"]
    assert set(c["mass"]) == {"original", "keep", "keep_object", "zoom", "other"}
    assert set(c["distance"]) == {"original", "keep", "keep_object", "zoom", "other"}       # force keep doubles as keep_object
    assert c["distance"]["keep_object"] == c["distance"]["keep"]
    for f, pairs in (("mass", 26), ("speed", 39), ("distance", 16)):
        for cond in ("original", "keep", "zoom", "other"):
            assert c[f][cond]["both"][1] == pairs and c[f][cond]["different"][1] == 2 * pairs
    assert c["mass"]["keep_object"]["different"] == [0, 52] and c["mass"]["keep"]["different"] == [52, 52]
    d = res["differences"]["mass|keep-keep_object|different"]
    assert d["value"] == 1.0 and d["ci95"] == [1.0, 1.0]
    assert any("NOT excluded" in n or "fits" in n for n in res["observations"]["distance"])


def test_clusters_keep_shared_case_families_together():
    data = [json.loads(x) for x in open(os.path.join(DATA, "exp41", "questions.jsonl"))]
    cl = rep.clusters([{"family": r["family"], "case": r["case"]} for r in data])
    assert len(cl) == 42 and sum(map(len, cl.values())) == 60
    by_case = {}
    for r in data:
        by_case.setdefault(r["case"], set()).add(r["family"])
    for fams in by_case.values():                      # families sharing a case are always in one cluster
        assert any(fams <= v for v in cl.values())


def test_readout_refuses_changed_or_inconsistent_inputs(tmp_path):
    def swap_image(recs, info, m):
        recs[0] = dict(recs[0], image_id="real-layout:" + recs[0]["image_id"])
        return recs, info

    def bad_correct(recs, info, m):
        recs[1] = dict(recs[1], correct=not recs[1]["correct"])
        return recs, info

    def other_data(recs, info, m):
        return recs, dict(info, data_sha256="0" * 64)

    def replaced_original(recs, info, m):
        p = m / "eval" / "rl_tune.details.jsonl"
        lines = open(p).readlines()
        open(p, "w").writelines(lines[1:] + lines[:1])          # same answers, different file: not the frozen record
        return recs, info

    def control(recs, info, m):
        return recs, dict(info, control="mismatched")
    for i, t in enumerate((swap_image, bad_correct, other_data, replaced_original, control)):
        with pytest.raises(SystemExit):
            build(*fake_run(tmp_path / str(i), t))
