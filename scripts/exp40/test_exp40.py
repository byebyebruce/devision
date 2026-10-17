"""exp40 without a model (critic EXP40-READOUT-01): the two arms' data map B's rows as written (other blocks
untouched; real-layout questions, options and golds unchanged; M's second exposure on the L picture, O's on the
original); an L picture whose numbers do not match the golds is refused; the report refuses a wrong arm, reports a
missing run as incomplete and only calls a bridge when the paired lower bound is > 0; the 256 watch limits."""
import importlib.util
import json
import os
from collections import Counter

import pytest

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(ROOT, "data")


def load(name):
    spec = importlib.util.spec_from_file_location("exp40_" + name, os.path.join(HERE, name + ".py"))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


b, rep, w = load("build"), load("report"), load("watch")


def rows(path):
    return [json.loads(x) for x in open(path)]


def test_arms_map_b_rows():
    src = rows(os.path.join(DATA, "v04b", "train_mix.jsonl"))
    by_id = {r["id"]: r for r in src}
    arms = {a: rows(os.path.join(DATA, "exp40", "train_%s.jsonl" % a)) for a in ("O", "M")}
    other = lambda rs: Counter(json.dumps(r, sort_keys=True) for r in rs if not r.get("source", "").startswith("real-layout"))  # noqa: E731
    for a, rs in arms.items():
        assert other(rs) == other(src)                                   # B's other blocks, as they are
        rl = [r for r in rs if r.get("source", "").startswith("real-layout")]
        assert sum(len(r["questions"]) for r in rl) == 9600
        per_fam = Counter(r["family"] for r in rl)
        assert len(per_fam) == 600 and Counter(by_id[r["orig_id"]]["kind"] for r in rl if r["exposure"] == 0 and r["view"] == "v0" and r["orig_qid"] == "q0") == {"temp": 300, "force": 300}
        for r in rl:
            o = by_id[r["orig_id"]]
            (q, g), = zip(r["questions"].values(), r["gold"].values())
            assert q == o["questions"][r["orig_qid"]] and g == o["gold"][r["orig_qid"]] and r["state_text"] == o["state_text"]
            assert len(r["questions"]) == 1 and r["group"] == o["family"]
            if r["exposure"] == 0 or a == "O":
                assert r["image"] == o["image"] and r["image_id"] == o["image_id"]
            else:
                assert r["image"].split("/")[0] in ("exp37", "exp40") and "/L/" in r["image"] and r["image_id"].startswith("exp40-L:")
        for f, n in per_fam.items():                                     # both views, every question, twice
            o0 = by_id["real-layout:%s:v0" % f]
            assert n == 2 * 2 * len(o0["questions"])
    strip = lambda r: {k: v for k, v in r.items() if k not in ("image", "image_id")}  # noqa: E731
    assert [strip(r) for r in arms["O"]] == [strip(r) for r in arms["M"]]


def test_l_numbers_must_match_golds():
    r = next(x for x in rows(os.path.join(DATA, "real_layout", "questions_tune.jsonl")) if x["kind"] == "temp")
    meta = json.load(open(os.path.join(DATA, "exp37", "pictures.json")))["%s_%s" % (r["family"], r["id"].split(":")[2])]
    assert b.numbers_match(r, meta["numbers"])
    bad = [dict(n, text="99") if n["field"] == "mass" else n for n in meta["numbers"]]
    assert not b.numbers_match(r, bad)


def fake_run(base, arm, data_arm=None, run_data=None):
    run = base / ("exp40-" + arm.lower())
    run.mkdir(parents=True)
    man = json.load(open(os.path.join(DATA, "exp40", "MANIFEST.json")))
    d = data_arm or arm
    (run / "START.json").write_text(json.dumps({"arm": arm, "init": rep.INIT, "init_sha256": rep.INIT_SHA,
                                                "data": "data/exp40/train_%s.jsonl" % d, "data_sha256": man["arms"][d]["sha256"]}))
    (run / "run.json").write_text(json.dumps({"config": {"stages": [{"data": run_data or "data/exp40/train_%s.jsonl" % arm, "init": rep.INIT}]}}))
    return str(run), man


def test_report_refuses_wrong_arm_and_reports_missing(tmp_path):
    run, man = fake_run(tmp_path / "a", "M")
    assert rep.identity("M", run, man) == ""
    run, man = fake_run(tmp_path / "b", "M", data_arm="O")                      # M's START names O's data
    assert rep.identity("M", run, man).startswith("wrong arm")
    with pytest.raises(SystemExit):
        rep.arm_screen("M", run, man)
    run, man = fake_run(tmp_path / "c", "O", run_data="data/exp40/train_M.jsonl")   # O's stage trained on M's file
    assert rep.identity("O", run, man).startswith("wrong arm")
    assert rep.arm_screen("M", str(tmp_path / "none"), man)["outcome"].startswith("incomplete")   # never started
    run, man = fake_run(tmp_path / "d", "M")
    (tmp_path / "d" / "exp40-m-status.txt").write_text("train completed\n")      # trained, nothing screened
    out = rep.arm_screen("M", run, man)
    assert out["outcome"].startswith("incomplete: sqa_temp") and "model_sha256" not in out


def recs(right_m):
    """Two families in one case cluster, temperature mass A / B on both views, values differ."""
    out = []
    for fam, case in (("rl1", "c1"), ("rl2", "c2")):
        for view in ("v0", "v1"):
            for obj, text in (("sample A", "10 u"), ("sample B", "20 u")):
                out.append({"sample_id": "%s:%s" % (fam, view), "family": fam, "case": case, "field": "mass", "object": obj,
                            "text": text if view == "v0" else ("20 u" if text == "10 u" else "10 u"),
                            "prediction": text if right_m else "10 u", "right": right_m or obj == "sample A",
                            "mode": "real", "wrong_picture_kind": "none"})
    return out


def test_bridge_needs_a_paired_lower_bound_above_zero():
    c = rep.compare(recs(True), recs(False))
    assert c["M_minus_O"]["both"]["value"] > 0 and c["bridge"]
    assert c["O"]["same_prediction"] == 1.0
    c = rep.compare(recs(False), recs(False))
    assert not c["bridge"]
    with pytest.raises(SystemExit):
        rep.compare(recs(True), [r for r in recs(False) if r["family"] == "rl1"])


def test_watch_limits_at_256():
    wins = lambda **kw: {int(k[1:]): v for k, v in kw.items()}  # noqa: E731
    assert w.decide(wins(s100=3.2), {}, {}, 0).startswith("first window")
    assert w.decide(wins(s100=1.5, s200=3.1, s300=3.2), {}, {}, 0).startswith("two windows")
    assert w.decide(wins(s100=1.5, s1600=1.5, s1700=8.0, s1800=3.1), {}, {}, 0) == ""
    assert w.decide(wins(s100=1.5), {100: 16.5}, {100: 1.5}, 0).startswith("window 100")
    assert w.decide({}, {}, {}, 1700) == "" and w.decide({}, {}, {}, 1900).startswith("no new window")
    assert w.decide(wins(s100=1.5, s3200=1.5), {}, {}, 2300) == ""
    assert w.decide(wins(s100=1.5, s200=1.5), {}, {}, 700).startswith("no new window")


def test_missing_run_json_is_never_a_pass(tmp_path):
    """START, frozen weights and every required set present, but no run.json: no actual training identity."""
    run, man = fake_run(tmp_path, "M")
    os.remove(os.path.join(run, "run.json"))
    st = os.path.join(run, "stage2")
    os.makedirs(os.path.join(run, "eval"))
    os.makedirs(st)
    open(os.path.join(st, "model.safetensors"), "wb").write(b"w")
    open(os.path.join(st, "config.json"), "w").write("{}")
    rep.v04b.frozen.freeze(run)
    for n in rep.v04b.REQUIRED:
        json.dump({"run": {"checkpoint": st, "temperature": "fitted"}}, open(os.path.join(run, "eval", n + ".json"), "w"))
        open(os.path.join(run, "eval", n + ".details.jsonl"), "w").write("{}\n")
        rep.v04b.frozen.record(run, n)
    (tmp_path / "exp40-m-status.txt").write_text("train completed\n" + "".join("eval:%s completed\n" % n for n in rep.v04b.REQUIRED))
    out = rep.arm_screen("M", run, man)
    assert out["outcome"] == "incomplete: no run.json" and "model_sha256" not in out
    with pytest.raises(ValueError):
        rep.tune_records("M", run, man, {"train": "completed", "eval:rl_tune": "completed"})


def tune_arm(base, arm, man, mutate=None):
    """An arm with only train and rl_tune done (frozen, recorded); its rl_tune details from the real tune file."""
    run, _ = fake_run(base, arm)
    st = os.path.join(run, "stage2")
    os.makedirs(st)
    os.makedirs(os.path.join(run, "eval"))
    open(os.path.join(st, "model.safetensors"), "wb").write(arm.encode())
    open(os.path.join(st, "config.json"), "w").write("{}")
    recs = []
    for d in rows(os.path.join(DATA, "real_layout", "questions_tune.jsonl")):
        for q, g in d["gold"].items():
            gold = next(o for o, p in g["probabilities"].items() if p == 1.0)
            right = arm == "M" or q == "q0"
            recs.append({"sample_id": d["id"], "qid": q, "image_id": d["image_id"], "gold": g["probabilities"],
                         "gold_answer": gold, "prediction": gold if right else "x", "correct": right})
    recs = mutate(recs) if mutate else recs
    with open(os.path.join(run, "eval", "rl_tune.details.jsonl"), "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in recs)
    json.dump({"run": {"checkpoint": st, "temperature": "fitted"}}, open(os.path.join(run, "eval", "rl_tune.json"), "w"))
    rep.v04b.frozen.freeze(run)
    rep.v04b.frozen.record(run, "rl_tune")
    (base / ("exp40-%s-status.txt" % arm.lower())).write_text("train completed\neval:rl_tune completed\n")
    return run


def test_comparison_needs_only_rl_tune_and_every_question(tmp_path):
    man = json.load(open(os.path.join(DATA, "exp40", "MANIFEST.json")))
    runs = {a: tune_arm(tmp_path / "ok", a, man) for a in ("M", "O")}
    out = rep.build(runs)
    assert all(a["outcome"].startswith("incomplete") for a in out["arms"].values()) and out["candidate"] == "none"
    c = out["comparison"]
    assert c["n_clusters"] == 165 and c["n_families"] == 300 and c["bridge"]
    assert out["strata"]["M"]["different"]["pairs"] == 414 and out["strata"]["M"]["same"]["questions"] == 972
    assert c["M"]["denominators"]["both:mass"] == 138 and c["M"]["denominators"]["both:speed"] == 191 and c["M"]["denominators"]["both:distance"] == 85
    drop = lambda rs: [r for r in rs if not (r["sample_id"].endswith(":v1") and r["qid"] == "q2" and r["sample_id"] == rs[-1]["sample_id"])]  # noqa: E731
    dup = lambda rs: rs + [rs[5]]  # noqa: E731
    for name, mut in (("drop", drop), ("dup", dup)):
        base = tmp_path / name
        runs = {"M": tune_arm(base, "M", man), "O": tune_arm(base, "O", man, mut)}
        c = rep.build(runs)["comparison"]
        assert isinstance(c, str) and c.startswith("incomplete") and ("differ in questions" in c or "duplicate" in c), c
