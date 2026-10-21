# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportOptionalMemberAccess=false
"""exp60 without a model (fake Reading replaying EXP59's real P records; real EXP58 / EXP59 products read only): the
oracle boxes are the fixed rule's output on the original pixels; an auto condition that does not reproduce EXP59 (or
whose mirrored read differs from the product read) stops before the oracle condition; the oracle condition gets exactly
the fixed boxes and builds its state from its own reads (gold only scores); a details file with another box is refused;
the pre-written direction."""
import importlib.util
import json
import os

import pytest

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.dirname(HERE))
spec = importlib.util.spec_from_file_location("exp60", os.path.join(HERE, "exp60.py"))
assert spec is not None and spec.loader is not None
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, m.E59, "START.json")), reason="runs/exp59 not here")


@pytest.fixture(autouse=True)
def in_root(monkeypatch):
    monkeypatch.chdir(ROOT)


def test_oracle_boxes_are_the_rule_s_output():
    bx = importlib.util.spec_from_file_location("boxes", os.path.join(HERE, "boxes.py"))
    b = importlib.util.module_from_spec(bx)
    bx.loader.exec_module(b)
    its = m.e59.items()
    fixed = {k: v for k, v in json.load(open(m.BOXES)).items() if not k.startswith("_")}
    assert b.build(its) == fixed and sum(len(v["boxes"]) for v in fixed.values()) == 72
    for v in fixed.values():
        w, h = v["size"]
        for t, (x, y, bw, bh) in v["boxes"].items():
            left = t.endswith(("sample A?", "Pair 1?"))
            assert y >= h / 2 - 2 and ((x + bw <= w / 2 + 2) if left else (x >= w / 2 - 2))
    m.oracle_boxes(its)


class FakeReading:
    """Replays EXP59's P (answers and diagnostics) on the product path; ordinary decide answers from the state."""
    def __init__(self, p, change=None):
        self.p, self.change, self.seen = p, change, []
        self.by_key = {it["key"]: it["row"]["id"] for it in m.e59.items()}
        outer = self

        class D:
            def decide(self, state, questions, image=None):
                outer.seen.append((state, image))
                return {"answers": {k: {"type": "choice", "probabilities": {o: 1 / len(q["criteria"]) for o in q["criteria"]}}
                                    for k, q in questions.items()}}
        self.decider = D()

    def decide(self, text, questions, image=None, diagnostics=None):
        (k, q), = questions.items()
        r = json.loads(json.dumps(self.p[self.by_key[k]]))
        if self.change:
            self.change(r)
        diagnostics.update(r["diagnostics"])
        return {"answers": {k: {"type": "choice", "probabilities": r["probabilities"]}}}


def fake_read_with(p, seen, mirror_change=None):
    """boxes None: EXP59's reads (the mirror); boxes given: those boxes, the gold numbers read (a success to score)."""
    gold = m.e59.gold_numbers(m.e59.items())
    by_q = {}
    for rid, r in p.items():
        by_q[tuple(x["question"] for x in r["diagnostics"]["reads"]), r["diagnostics"]["reads"][0]["box"][1]] = rid

    def rw(reading, image, templates, boxes=None):
        from devision.reading import TEMPLATES, fmt
        if boxes is None:
            rid = seen["mirror_order"].pop(0)
            reads = json.loads(json.dumps(p[rid]["diagnostics"]["reads"]))
            if mirror_change:
                mirror_change(reads)
        else:
            rid = seen["oracle_order"].pop(0)                       # the run asks in item order (layouts repeat)
            seen["oracle_boxes"].append((rid, boxes))
            reads = [{"question": t, "box": boxes[t], "degenerate": False, "cells": 4, "views": 2,
                      "decoded": m.e54.number_of(gold[rid][t]), "valid": True, "value": fmt(m.e54.number_of(gold[rid][t]))} for t in templates]
        return {"reads": reads, "table": [[*TEMPLATES[x["question"]], x["value"]] for x in reads]}
    return rw


def started(tmp_path, monkeypatch, change=None, mirror_change=None):
    p = m.exp59_p()
    out = str(tmp_path / "out")
    m.stage_start(0, out=out, data_root="data")
    fr = FakeReading(p, change)
    pics = json.load(open(os.path.join(m.e59.E58, "pictures.json")))
    monkeypatch.setattr(m.e58, "load_reading", lambda *a, **k: (fr, m.e58.check_run(m.e59.E58, data_root="", code=False), pics))
    monkeypatch.setattr(m, "check_start", lambda out, data_root="data", code=True: json.load(open(os.path.join(out, "START.json"))))
    order = [it["row"]["id"] for it in m.e59.items()]
    seen = {"mirror_order": list(order), "oracle_order": list(order), "oracle_boxes": []}
    monkeypatch.setattr(m, "read_with", fake_read_with(p, seen, mirror_change))
    return out, fr, seen


def test_reproduced_auto_then_oracle_with_the_fixed_boxes(tmp_path, monkeypatch):
    out, fr, seen = started(tmp_path, monkeypatch)
    m.stage_run(out)
    fixed = {k: v["boxes"] for k, v in json.load(open(m.BOXES)).items() if not k.startswith("_")}
    assert len(seen["oracle_boxes"]) == 24 and all(fixed[rid] == b for rid, b in seen["oracle_boxes"])
    oracle = {r["sample_id"]: r for r in m.jsonl(os.path.join(out, "details_oracle.jsonl"))[1:]}
    assert [s for s, _ in fr.seen] == [oracle[it["row"]["id"]]["diagnostics"]["state"] for it in m.e59.items()]   # own reads -> state
    assert all(img is None for _, img in fr.seen)
    rep = m.report(out)
    assert rep["states"] == {"auto": "completed", "oracle": "completed"} and rep["reproduction"]["reproduced"]
    assert rep["fields"]["auto"]["speed"]["exact"] == 0 and rep["fields"]["oracle"]["speed"]["exact"] == 24
    assert rep["branch"].startswith("speed read from the right box") and rep["candidate_eligible"] is False


def _prob(r):
    k = next(iter(r["probabilities"]))
    r["probabilities"][k] = r["probabilities"][k] + 0.01


def _mirror(reads):
    reads[0]["decoded"] = reads[0]["decoded"] + "1"


@pytest.mark.parametrize("change,mirror", [(_prob, None), (None, _mirror)])
def test_auto_not_reproduced_stops_before_oracle(tmp_path, monkeypatch, change, mirror):
    out, fr, seen = started(tmp_path, monkeypatch, change, mirror)
    with pytest.raises(SystemExit, match="does not reproduce EXP59"):
        m.stage_run(out)
    assert not seen["oracle_boxes"] and not os.path.exists(os.path.join(out, "details_oracle.jsonl"))
    assert m.report(out)["branch"].startswith("invalid / incomplete")


def test_oracle_details_with_another_box_are_refused(tmp_path, monkeypatch):
    out, fr, seen = started(tmp_path, monkeypatch)
    m.stage_run(out)
    f = os.path.join(out, "details_oracle.jsonl")
    lines = [json.loads(x) for x in open(f)]
    lines[1]["diagnostics"]["reads"][0]["box"][1] -= 5
    open(f, "w").write("".join(json.dumps(x) + "\n" for x in lines))
    m.register(out, "decide_oracle", {"details_oracle.jsonl": m.sha(f)})
    rep = m.report(out)
    assert rep["states"]["oracle"].startswith("missing / invalid") and rep["branch"].startswith("incomplete")


@pytest.mark.parametrize("exact,lo,text", [(18, 0.1, "selector supervision first"), (24, 0.0, "partial rescue"),
                                           (12, 0.5, "cross-layout supervision"), (15, 0.2, "partial rescue")])
def test_direction(exact, lo, text):
    res = {"fields": {"oracle": {"speed": {"exact": exact}}}, "speed_oracle_minus_auto": {"ci95": [lo, 1.0]}}
    assert text in m.branch(res)
