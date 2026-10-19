"""scripts/release/route_calib.py (prep03 5.3 / R4; critic V04-K-PREP-02) on synthetic records, no model: the two
counter-examples written in 5.3 land in branches 3 and 4; "ECE down but NLL up" cannot select a routed temperature;
an over-confident routed path (true temperature 2, config temperature 1) selects branch 1 with an all-data temperature near
2, a calibrated one keeps the current temperatures; zero / non-finite probabilities, unread records and missing families
are refused; both views of a family share a fold."""
import importlib.util
import math
import os
import random

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
spec = importlib.util.spec_from_file_location("route_calib", os.path.join(ROOT, "scripts", "release", "route_calib.py"))
assert spec and spec.loader
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)


def d(value, lo, hi):
    return {"value": value, "ci95": [lo, hi]}


def test_the_written_counter_examples():
    # 5.3 / R4: ordinary - raw = +0.10 [+0.09, +0.11], routed - ordinary = -0.03 [-0.04, -0.02]: routed - raw ~ +0.07
    one = {"R": d(-0.03, -0.04, -0.02), "O": d(0.10, 0.09, 0.11), "B": d(0.07, 0.06, 0.08)}
    assert rc.branch(one, {"temperature": -0.03, "magnets": -0.03}).startswith("3 clear degradation")
    # Delta_O [-0.01, +0.05], Delta_R [-0.06, +0.01]: insufficient evidence
    two = {"R": d(-0.025, -0.06, 0.01), "O": d(0.02, -0.01, 0.05), "B": d(-0.005, -0.05, 0.04)}
    assert rc.branch(two, {"temperature": -0.025, "magnets": -0.025}).startswith("4 insufficient")
    # ECE may fall while NLL rises by 0.05: a routed temperature is never chosen on that
    three = {"R": d(0.05, 0.03, 0.07), "O": d(0.0, -0.01, 0.01), "B": d(0.05, 0.03, 0.07)}
    assert rc.branch(three, {"temperature": 0.05, "magnets": 0.05}).startswith("2 keep")


def synthetic(true_t, n_fam=300, seed=1):
    """Families of two views x one 3-option question per domain; raw probabilities softmax(z), gold drawn from softmax(z / true_t)."""
    rng = random.Random(seed)
    recs = []
    for f in range(n_fam):
        dom = "temperature" if f % 2 else "magnets"
        for v in range(2):
            z = [rng.gauss(0, 2.5) for _ in range(3)]
            opts = ["a", "b", "c"]
            raw = rc.scaled({o: math.exp(x) for o, x in zip(opts, z)}, 1.0)
            true = rc.scaled(raw, true_t)
            u, acc, gold = rng.random(), 0.0, opts[-1]
            for o in opts:
                acc += true[o]
                if u <= acc:
                    gold = o
                    break
            recs.append({"sample_id": "f%d:v%d" % (f, v), "qid": "q0", "family": "fam%d" % f, "route": "read", "reading_domain": dom,
                         "probabilities": raw, "gold_answer": gold})
    return recs


def test_an_over_confident_routed_path_selects_a_routed_temperature():
    res = rc.analyse(synthetic(2.0, n_fam=400), {"temperature": [1.0, 1.0, 1.0], "temperature_by_options": {}})
    assert res["branch"].startswith("1 separate routed temperature"), (res["branch"], res["pooled"])
    assert all(1.6 < t < 2.5 for t in res["routed_temperature_all_data"].values())
    assert res["pooled"]["R"]["value"] < -0.02 and res["pooled"]["O"]["value"] == pytest.approx(0.0)


def test_a_calibrated_routed_path_keeps_the_current_temperatures():
    res = rc.analyse(synthetic(1.0, n_fam=400), {"temperature": [1.0, 1.0, 1.0], "temperature_by_options": {}})
    assert res["branch"].startswith("2 keep"), (res["branch"], res["pooled"])


def test_the_config_bucket_temperature_is_the_ordinary_one():
    r = {"probabilities": {"a": 0.5, "b": 0.3, "c": 0.2}}
    assert rc.ordinary_t(r, {"temperature": [1.4, 1.0, 1.0], "temperature_by_options": {"choice:3-5": 0.8}}) == 0.8
    assert rc.ordinary_t(r, {"temperature": [1.4, 1.0, 1.0], "temperature_by_options": {"choice:2": 0.8}}) == 1.4


@pytest.mark.parametrize("bad,why", [("zero", "zero or not finite"), ("nan", "zero or not finite"), ("unread", "not a routed"),
                                     ("no_family", "not a routed"), ("gold", "not an option")])
def test_uncalibratable_records_are_refused(bad, why):
    recs = synthetic(1.0, n_fam=10)
    r = recs[3]
    if bad == "zero":
        r["probabilities"] = {"a": 0.0, "b": 0.5, "c": 0.5}
    elif bad == "nan":
        r["probabilities"] = {"a": float("nan"), "b": 0.5, "c": 0.5}
    elif bad == "unread":
        r["route"] = "ordinary"
    elif bad == "no_family":
        r.pop("family")
    else:
        r["gold_answer"] = "z"
    with pytest.raises(SystemExit, match=why):
        rc.analyse(recs, {"temperature": [1.0, 1.0, 1.0]})


def test_both_views_share_a_fold():
    assert rc.fold_of("fam7") == rc.fold_of("fam7") and {rc.fold_of("fam%d" % i) for i in range(20)} == {0, 1}


def test_collect_reads_every_rl_calib_question_through_the_k_combination(tmp_path, monkeypatch):
    """A tiny rl_calib (two families, real renderer, fresh seed) through a tiny K combination and the formal evaluation
    path: raw temperatures, every comparison question read, families joined; a combination without K is refused."""
    import torch
    from conftest import tiny_decider

    from devision import reading as R
    spec2 = importlib.util.spec_from_file_location("rl_calib", os.path.join(ROOT, "scripts", "release", "rl_calib.py"))
    assert spec2 and spec2.loader
    rl = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(rl)
    monkeypatch.setattr(rl, "used_cases", lambda root: set())
    import sys
    monkeypatch.delitem(sys.modules, "build", raising=False)      # real_layout imports exp28's build; other tests load another "build"
    man = rl.build(str(tmp_path), want={"temp": 1, "force": 1})
    rows = [__import__("json").loads(x) for x in open(tmp_path / "rl_calib" / "questions.jsonl")]
    assert man["pictures"] == 4 and all(list(r["questions"]) == ["q0"] for r in rows)
    short = ["red", "blue", "green"]                       # the tiny model's input is short: same questions, short options / hint
    for r in rows:
        opts = list(r["questions"]["q0"]["criteria"])
        r["state_text"] = ""
        r["questions"]["q0"]["criteria"] = {short[i]: None for i in range(len(opts))}
        r["gold"]["q0"]["probabilities"] = {short[i]: r["gold"]["q0"]["probabilities"][o] for i, o in enumerate(opts)}
    d = tiny_decider(seed=3)
    d.model.add_question_conditioning()
    d.cfg.question_conditioning = True
    d.model.enable_local_views(12)
    d.cfg.local_views, d.cfg.local_overlap, d.cfg.local_source = 12, 0.25, "T"
    d.cfg.temperature = [1.7, 1.0, 1.2]
    d.save(tmp_path / "cand")
    torch.manual_seed(7)
    torch.save(R.Selector(32, 64).state_dict(), tmp_path / "sel.pt")
    torch.save(R.CharHead(32).state_dict(), tmp_path / "head.pt")
    for name, k in (("tmp_k", True), ("tmp_plain", False)):
        R.write_bundle(str(tmp_path / name), str(tmp_path / "cand"), str(tmp_path / "sel.pt"), str(tmp_path / "head.pt"), {},
                       dims={"selector": [32, 64], "head": 32}, k_rule=k)
    rd = R.Reading.load(str(tmp_path / "cand"), str(tmp_path / "tmp_k"), device="cpu", full_rules=True)
    recs = rc.collect(rd, rows, str(tmp_path))
    assert len(recs) == 4 and all(r["route"] == "read" and r["family"].startswith("rc") for r in recs)
    assert rd.cfg.temperature == [1.0, 1.0, 1.0]                                        # raw
    with pytest.raises(SystemExit, match="no K rule"):
        rc.collect(R.Reading.load(str(tmp_path / "cand"), str(tmp_path / "tmp_plain"), device="cpu", full_rules=True), rows, str(tmp_path))
