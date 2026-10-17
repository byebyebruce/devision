"""exp42 without a model: the data are exp40-M's rows (only id / group / repeat added), 144 eligible units of 8 by
the fixed hash order, each unit original + L and its two objects' v0 golds different, repeats in their own group
namespace; the report refuses a wrong arm, gives "incomplete" for a missing arm, and computes the readout,
case-cluster differences and pre-written conditions from temperature-1 records."""
import importlib.util
import json
import os
from collections import Counter

import pytest

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(ROOT, "data")


def load(name):
    spec = importlib.util.spec_from_file_location("exp42_" + name, os.path.join(HERE, name + ".py"))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


b, rep = load("build"), load("report")
t41 = importlib.util.spec_from_file_location("t41", os.path.join(HERE, "..", "exp41", "test_exp41.py"))
assert t41 is not None and t41.loader is not None
_t41 = importlib.util.module_from_spec(t41)
t41.loader.exec_module(_t41)
fake_details = _t41.fake_details


def rows(path):
    return [json.loads(x) for x in open(path)]


def test_data_are_exp40_rows_in_whole_units():
    src = {r["id"]: r for r in rows(os.path.join(DATA, "exp40", "train_M.jsonl"))}
    train = rows(os.path.join(DATA, "exp42", "train.jsonl"))
    m = json.load(open(os.path.join(DATA, "exp42", "MANIFEST.json")))
    assert m["available_units"] == {"mass": 160, "speed": 153, "distance": 113} and len(train) == 4608
    for r in train:
        o = src[r["id"].rsplit(":", 1)[0]]
        assert {k: v for k, v in r.items() if k not in ("id", "group", "repeat")} == {k: v for k, v in o.items() if k not in ("id", "group")}
        assert r["group"] == "r%d:%s" % (r["repeat"], o["group"])
    units = Counter((r["group"], next(iter(r["questions"]))) for r in train)
    assert set(units.values()) == {8} and len(units) == 576
    us = b.units(rows(os.path.join(DATA, "exp40", "train_M.jsonl")))
    chosen = [tuple(k) for k in m["units"]]
    assert Counter(k[1] for k in chosen) == {"mass": 48, "speed": 48, "distance": 48}
    for k in chosen:
        u = us[k]
        assert b.eligible(u) and Counter(r["exposure"] for r in u) == {0: 4, 1: 4}
        assert {r["image"].split("/")[0] for r in u if r["exposure"] == 1} <= {"exp37", "exp40"}
    for f in b.FIELDS:                                   # the first 48 eligible by the fixed hash order
        order = sorted((k for k, u in us.items() if k[1] == f and b.eligible(u)),
                       key=lambda k: __import__("hashlib").sha256(("%s:%s" % k).encode()).hexdigest())
        assert [k for k in chosen if k[1] == f] == order[:48]
    fit_o, fit_l = rows(os.path.join(DATA, "exp42", "fit_O.jsonl")), rows(os.path.join(DATA, "exp42", "fit_L.jsonl"))
    assert len(fit_o) == len(fit_l) == 576 and {r["exposure"] for r in fit_o} == {0} and {r["exposure"] for r in fit_l} == {1}


def fake_run(base, who, right, objective=None, sets=None, missing=()):
    """A run directory with temperature-1 evaluations; arms also START / run.json / train_report / frozen records."""
    run = base / ("exp42-" + ("start" if who == "start" else who.lower()))
    (run / "eval").mkdir(parents=True)
    ck = rep.INIT if who == "start" else str(run / "stage2")
    status = []
    if who != "start":
        (run / "stage2").mkdir()
        (run / "stage2" / "model.safetensors").write_bytes(who.encode())
        (run / "stage2" / "config.json").write_text("{}")
        obj = objective or rep.OBJECTIVE[who]
        m = json.load(open(os.path.join(DATA, "exp42", "MANIFEST.json")))
        json.dump({"objective": obj, "init": rep.INIT, "init_sha256": rep.INIT_SHA, "data": "data/exp42/train.jsonl",
                   "data_sha256": m["files"]["train"]}, open(run / "START.json", "w"))
        json.dump({"config": {"stages": [{"data": "data/exp42/train.jsonl", "init": rep.INIT, "params": {"objective": obj}}]}},
                  open(run / "run.json", "w"))
        json.dump({"objective": obj}, open(run / "stage2" / "train_report.json", "w"))
        rep.frozen.freeze(str(run))
        status.append("train completed")
    for n, path in {**rep.SETS, **rep.RISK}.items():
        if n in missing:
            continue
        data = rows(os.path.join(ROOT, path))
        recs = fake_details(data, lambda r, q: not right(n, r, q))
        open(run / "eval" / (n + ".details.jsonl"), "w").writelines(json.dumps(x) + "\n" for x in recs)
        json.dump({"run": {"checkpoint": ck, "temperature": "none (raw)", "control": "none",
                           "data_sha256": rep.sha(os.path.join(ROOT, path))}}, open(run / "eval" / (n + ".json"), "w"))
        if who != "start":
            rep.frozen.record(str(run), n)
        status.append("eval:%s completed" % n)
    (base / ("exp42-%s-status.txt" % ("start" if who == "start" else who.lower()))).write_text("\n".join(status) + "\n")
    return str(run)


def test_report_readout_and_conditions(tmp_path):
    os.chdir(ROOT)
    start = fake_run(tmp_path, "start", lambda n, r, q: False)
    r = fake_run(tmp_path, "R", lambda n, r, q: any(w in r["questions"][q]["instructions"] for w in ("sample A", "Pair 1")))
    c = fake_run(tmp_path, "C", lambda n, r, q: True)
    out = rep.build({"start": start, "R": r, "C": c})
    rt = out["readout"]["rl_tune"]
    assert rt["C"]["different"] == 1.0 and rt["C"]["both"] == 1.0 and rt["start"]["both"] == 0.0
    assert rt["C"]["different:mass"]["of"] == 276 and rt["C"]["both:speed"]["of"] == 191 and rt["C"]["both:distance"]["of"] == 85
    assert out["conditions"]["progress_C"] is True and out["conditions"]["ce_attribution"] is True
    assert out["conditions"]["stop_route"] is False
    assert out["differences"]["rl_tune|C-R"]["both"]["value"] == pytest.approx(1.0)


def test_report_refuses_wrong_arm_and_reports_missing(tmp_path):
    os.chdir(ROOT)
    start = fake_run(tmp_path, "start", lambda n, r, q: True)
    r = fake_run(tmp_path, "R", lambda n, r, q: True, objective="ce")             # R's identity says ce
    with pytest.raises(SystemExit):
        rep.build({"start": start, "R": r, "C": str(tmp_path / "none")})
    base = tmp_path / "b"
    start = fake_run(base, "start", lambda n, r, q: True)
    r = fake_run(base, "R", lambda n, r, q: True, missing=("rl_tune",))
    out = rep.build({"start": start, "R": r, "C": str(base / "none")})
    assert out["runs"]["C"].startswith("incomplete") and "rl_tune" in out["runs"]["R"]["missing"]
    assert out["conditions"]["progress_R"] == "incomplete" and out["conditions"]["ce_attribution"] == "incomplete"
