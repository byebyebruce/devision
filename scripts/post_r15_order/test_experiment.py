"""Behavioural fixtures for option semantics, grouping, image isolation, and frozen exits."""
import copy
import json
import random

import pytest

import build
import report
import run
from devision.train.rlcd import plan_batches


def sample(kind="choice"):
    q = {"type": kind, "instructions": "How many red objects?"}
    if kind == "choice":
        q["criteria"] = {"two": "Two objects", "three": "Three objects", "four": None}
        p = {"two": .2, "three": .8, "four": 0.}
    else:
        p = {"false": .2, "true": .8}
    return {"id": "x", "group": "family", "image_id": "coco:1", "image": "1.png",
            "questions": {"q": q}, "gold": {"q": {"probabilities": p}}, "source": "fixture"}


def test_choice_views_preserve_soft_gold_and_descriptions_and_pairing():
    s = sample()
    a = build.views(s, "a", random.Random(1))
    b = build.views(s, "b", random.Random(1))
    assert a[0] == b[0]
    assert list(a[1]["questions"]["q"]["criteria"]) != list(b[1]["questions"]["q"]["criteria"])
    assert a[1]["questions"] == b[1]["questions"]
    assert a[1]["gold"] == b[1]["gold"]
    assert [s["id"] for s in a] == [s["id"] for s in b]
    assert s == sample()


def test_binary_choice_changes_order_but_noul_does_not():
    s = sample()
    s["questions"]["q"]["criteria"].pop("four")
    s["gold"]["q"]["probabilities"].pop("four")
    assert list(build.views(s, "b", random.Random(4))[1]["questions"]["q"]["criteria"]) == ["three", "two"]
    assert build.views(sample("noul"), "a", random.Random(2)) == build.views(sample("noul"), "b", random.Random(2))


def test_mirror_family_and_both_views_stay_in_the_same_update():
    orig, mirror = sample(), dict(sample(), id="flip:x", image_id="coco:1:flip", image="1-flip.png")
    ss = build.views(orig, "b", random.Random(1)) + build.views(mirror, "b", random.Random(2))
    ss += [dict(sample(), id="single%d" % i, group="other%d" % i) for i in range(4)]
    its = [dict(group=s["group"], qid="q") for s in ss]
    batches = plan_batches(its, 8, random.Random(1))
    assert len(batches) == 1 and set(batches[0]) == set(range(8))


def test_identity_alias_and_mirror_are_excluded_and_shortfall_fails():
    s = dict(sample(), image_id="vg:9:flip")
    selector = build.Selector({"vg:9": "coco:1"}, {"coco:1"}, set(), set())
    assert not selector.allowed(s)
    with pytest.raises(ValueError, match="eligible questions"):
        selector.take([s], 1, "test")


def record(sid="x", probabilities=None, gold=None):
    p = probabilities or {"cat": .8, "dog": .2}
    g = gold or {"cat": 1., "dog": 0.}
    return dict(sample_id=sid, qid="q", image_id="coco:1", type="choice", source="fixture", options=list(p),
                probabilities=p, gold=g, gold_answer=max(g, key=g.get), nll=.2)


def test_order_metrics_compare_semantics_not_positions_and_keep_ties_stable():
    original = report.normalise([record()])
    reverse = report.normalise([record(probabilities={"dog": .2, "cat": .8})])
    assert report.order_rows(original, reverse, "flip")[0]["correct"] == 0
    assert report.order_rows(original, reverse, "joint")[0]["correct"] == 1
    a = report.normalise([record(probabilities={"dog": .5, "cat": .5})])
    b = report.normalise([record(probabilities={"cat": .5, "dog": .5})])
    assert report.order_rows(a, b, "flip")[0]["correct"] == 0


def test_order_missing_questions_or_changed_gold_are_rejected():
    a = report.normalise([record()])
    with pytest.raises(ValueError, match="questions"):
        report.order_rows(a, [], "flip")
    b = report.normalise([record(gold={"cat": 0., "dog": 1.})])
    with pytest.raises(ValueError, match="gold"):
        report.order_rows(a, b, "flip")


def test_duplicate_occurrences_and_shared_images_survive_cluster_comparison():
    a = report.normalise([record(), record()])
    b = report.normalise([record(probabilities={"cat": .2, "dog": .8})] * 2)
    result = report.compare(a, b, resamples=20)
    assert result["questions"] == 2 and result["pictures"] == 1
    assert result["difference"] == -1 and result["ci95"] == [-1, -1]


def test_macro_nll_uses_equal_source_weight_not_number_of_questions(monkeypatch):
    monkeypatch.setattr(report, "EXT", ("large", "small"))
    a = [dict(record("x%d" % i), source="large", image_id="img:%d" % i) for i in range(9)]
    a += [dict(record("small"), source="small", image_id="img:small")]
    b = [dict(r, nll=r["nll"] + (1 if r["source"] == "small" else 0)) for r in a]
    result = report.compare(report.normalise(a), report.normalise(b), "macro_nll", resamples=20)
    assert result["difference"] == pytest.approx(.5)


def comparison():
    def value(diff=0., lo=0., hi=0.):
        return {"difference": diff, "ci95": [lo, hi]}
    out = {name: {"accuracy": value(), "nll": value()} for name in report.GUARDS +
           ("count_wide", "count_narrow", "count_wide_reverse", "count_narrow_reverse")}
    out["count_wide"].update(flip=value(-.06, -.1, -.02), joint=value())
    return out


def test_success_requires_flip_improvement_and_count_protection():
    a, b = comparison(), comparison()
    assert report.decide(a, b)["screen_pass"]
    b["count_wide"]["joint"]["ci95"][0] = -.02
    result = report.decide(a, b)
    assert not result["screen_pass"] and result["count_not_held"] == ["wide_joint"]


def test_guard_nll_and_early_stop_boundary():
    a, b = comparison(), comparison()
    b["vsr"]["accuracy"] = {"difference": -.05, "ci95": [-.09, -.001]}
    assert report.clear_losses(b, early=True) == ["vsr"]
    assert not report.decide(a, b)["screen_pass"]
    b = comparison()
    b["aokvqa"]["nll"] = {"difference": .021, "ci95": [.001, .04]}
    assert report.decide(a, b)["nll_warnings"] == ["aokvqa"]
    a["count_wide"]["flip"]["ci95"][1] = 0
    assert not report.decide(a, comparison())["primary"]


@pytest.mark.parametrize("screen_pass,replication_pass,expected", [
    (False, False, ["a", "b"]),
    (True, False, ["a", "b", "b-seed2"]),
    (True, True, ["a", "b", "b-seed2", "evaluate"]),
])
def test_controller_enforces_conditional_budget(tmp_path, monkeypatch, screen_pass, replication_pass, expected):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run, "ROOT", tmp_path)
    monkeypatch.setattr(run, "RUN", tmp_path / "run")
    monkeypatch.setattr(run, "STATE", tmp_path / "run/state.json")
    monkeypatch.setattr(run.signal, "signal", lambda *args: None)
    calls = []
    monkeypatch.setattr(run, "run_arm", lambda name: calls.append(name) or True)
    monkeypatch.setattr(run, "final_report", lambda *args: {"screen_pass": screen_pass})
    monkeypatch.setattr(run, "replication", lambda: replication_pass)
    monkeypatch.setattr(run, "final_evaluation", lambda: calls.append("evaluate"))
    run.main()
    assert calls == expected
    if not screen_pass or not replication_pass:
        assert json.loads(run.STATE.read_text())["finished"] is True


def test_controller_refuses_changed_frozen_inputs(tmp_path, monkeypatch):
    p = tmp_path / "config.yaml"
    p.write_text("original")
    frozen = {"file_sha256": {str(p): build.sha(p)}}
    (tmp_path / "frozen.json").write_text(json.dumps(frozen))
    monkeypatch.setattr(run, "RUN", tmp_path)
    p.write_text("changed after approval")
    with pytest.raises(ValueError, match="frozen input changed"):
        run.check_frozen()
