"""exp28 without a model: the transcriptions parse, field options are well formed, the wording set keeps the gold
with its option, the control rebuilds temperature-1 probabilities, and each branch of the next-fix rule."""
import importlib.util
import json
import math
import os
import random

import pytest

HERE = os.path.dirname(__file__)


def _load(name):
    spec = importlib.util.spec_from_file_location("exp28_" + name, os.path.join(HERE, name + ".py"))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


build, report = _load("build"), _load("report")
NOTES = json.load(open(os.path.join(HERE, "..", "exp21", "sqa_transcribed.json")))


def test_every_transcription_parses():
    temp = force = 0
    for k, v in NOTES.items():
        if k == "_note":
            continue
        if v.startswith("Sample"):
            vals = build.parse_temp(v)
            assert all(vals[s]["mass"] > 0 and vals[s]["speed"] > 0 for s in "AB")
            temp += 1
        else:
            vals = build.parse_force(v)
            assert all(vals[p]["distance"] > 0 for p in "12")
            force += 1
    assert (temp, force) == (12, 12)
    assert build.parse_force(NOTES["v3-sqa-test:test-3898"])["1"]["distance"] == 42.8


@pytest.mark.parametrize("field,right,other", [("mass", 44, 28), ("mass", 46, 46), ("speed", 1300, 1300),
                                               ("distance", 0.25, 0.1), ("distance", 16.9, 16.9)])
def test_field_options(field, right, other):
    seen_slots = set()
    for slot in range(3):
        probs = build.options(right, other, field, slot, "u")
        assert len(probs) == 3 and sum(probs.values()) == 1.0
        keys = list(probs)
        seen_slots.add(keys.index(next(k for k, v in probs.items() if v == 1.0)))
        if other != right:
            assert "%s u" % build.fmt_num(other) in probs
    assert seen_slots == {0, 1, 2}


def test_wording_keeps_the_gold_with_its_option(tmp_path):
    q = {"type": "choice", "instructions": "Which?", "criteria": {"sample A": None, "sample B": None,
                                                                  "neither; the samples have the same temperature": None}}
    rows = [{"id": "t%d" % i, "source": "s", "image_id": "p%d" % i, "image": "x.png", "questions": {"q": q},
             "gold": {"q": {"probabilities": {"sample A": 0.0, "sample B": 0.0,
                                              "neither; the samples have the same temperature": 1.0}}}} for i in range(2)]
    (tmp_path / "exp21").mkdir()
    (tmp_path / "v10").mkdir()
    (tmp_path / "exp21" / "dev_sqa_temp.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "v10" / "dev_sqa_conc.jsonl").write_text("")
    out = build.equal_wording(str(tmp_path))
    assert len(out) == 2 * 2 * 3
    for r in out:
        gold = next(k for k, v in r["gold"]["q"]["probabilities"].items() if v == 1.0)
        assert gold == ("the two samples have equal temperatures" if ":w1:" in r["id"]
                        else "neither; the samples have the same temperature")
        assert list(r["gold"]["q"]["probabilities"]) == list(r["questions"]["q"]["criteria"])
    orders = {tuple(r["questions"]["q"]["criteria"]) for r in out if r["id"].startswith("t0:w0")}
    assert len(orders) == 3


def test_control_rebuilds_temperature_one(tmp_path, monkeypatch):
    t = 1.7
    logits = {"A": 1.0, "B": 0.2, "neither; they are equal": -0.5}
    soft = lambda z, s: {k: math.exp(v / s) / sum(math.exp(x / s) for x in z.values()) for k, v in z.items()}  # noqa: E731
    mon = {"sample_id": "exp21:d0000:v0:h16", "qid": "cmp", "type": "choice", "probabilities": soft(logits, 1.0),
           "prediction": "A"}
    dec = dict(mon, probabilities=soft(logits, t))
    (tmp_path / "m.jsonl").write_text(json.dumps(mon) + "\n")
    (tmp_path / "d.jsonl").write_text(json.dumps(dec) + "\n")
    monkeypatch.setattr(report, "CONTROL_MONITOR", str(tmp_path / "m.jsonl"))
    monkeypatch.setattr(report, "path", lambda stage, name: str(tmp_path / "d.jsonl"))

    class Cfg:
        def temperature_for(self, qtype, k):
            return t
    monkeypatch.setattr(report, "config_path", lambda p: str(tmp_path / "m.jsonl"))
    monkeypatch.setattr(report.ModelConfig, "from_dict", staticmethod(lambda d: Cfg()))
    c = report.control()
    assert c["pass"] and c["disagreements"] == 0 and c["max_prob_diff_t1"] < 1e-9


def outcome(control=True, temp=0.9, rule=(3, 3), wording=False, cost_stage="exp26B"):
    return {"control": {"pass": control}, "fields": {"exp26B": {"temp_all": {"accuracy": temp}}},
            "rule": {"exp26B": {"plain": rule[0], "rule": rule[1], "n": 12}},
            "wording": {"exp26B": {"template_dependence": wording}}, "stages": {"first_cost_stage": cost_stage}}


@pytest.mark.parametrize("kw,start", [
    (dict(control=False, temp=0.1), "fix the export"),
    (dict(temp=0.5), "real-layout reading"),
    (dict(temp=0.7, rule=(3, 10)), "relation-rule"),
    (dict(temp=0.7, wording=True), "answer-wording"),
    (dict(temp=0.9), "field-to-decision"),
    (dict(temp=0.7), "uncertain"),
])
def test_next_fix_priority(kw, start):
    assert report.decide(outcome(**kw))["next_fix"].startswith(start)


@pytest.mark.parametrize("stage,start", [("exp24A", "replay real tasks in the alignment"),
                                         ("exp26B", "replay real tasks in the decision"), (None, "no stage")])
def test_replay_placement(stage, start):
    assert report.decide(outcome(cost_stage=stage))["replay"].startswith(start)
