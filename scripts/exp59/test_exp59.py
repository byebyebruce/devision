# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportOptionalMemberAccess=false
"""exp59 without a model (fake Reading / Decider; the real EXP58 products and data, read only): the 24 items, the 72 gold
numbers and the 12 facts; diagnostics do not change Reading.decide's answer; P never sees a gold number (the actual
picture and the row's own text only); a P that does not reproduce EXP58 stops before G_num / G_facts; a P that does
reproduces, then G_num / G_facts get exactly their registered inputs; a changed or incomplete product is refused."""
import importlib.util
import json
import os

import pytest

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.dirname(HERE))
spec = importlib.util.spec_from_file_location("exp59", os.path.join(HERE, "exp59.py"))
assert spec is not None and spec.loader is not None
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, m.E58, "START.json")), reason="runs/exp58 not here")


@pytest.fixture(autouse=True)
def in_root(monkeypatch):
    monkeypatch.chdir(ROOT)


def test_items_golds_and_facts():
    its = m.items()
    gold = m.gold_numbers(its)
    facts = m.load_facts(its)
    assert len(its) == 24 and sum(len(v) for v in gold.values()) == 72 and len(facts) == 12
    g = m.g_inputs(its, gold, facts)
    assert len(g["G_num"]) == 24 and len(g["G_facts"]) == 12
    for rid, v in g["G_facts"].items():
        assert v["state"].startswith(g["G_num"][rid]["state"]) and "force" not in v["facts"].lower()


def test_diagnostics_do_not_change_the_answer():
    from devision.reading import Reading

    class Dec:
        def decide(self, state, questions, image=None):
            return {"answers": {k: {"type": "choice", "probabilities": {o: (len(state) % 7 + i) / 1.0 for i, o in enumerate(q["criteria"])}}
                                for k, q in questions.items()}, "usage": {"input_tokens": 1}}
    r = object.__new__(Reading)
    r.decider = Dec()
    r.read = lambda image, ts: {"table": [["Pair 1", "distance between the magnets", "3"]], "reads": [], "seconds": 0.0}
    it = m.items()[-1]
    q = {it["key"]: it["row"]["questions"][it["qid"]]}
    from PIL import Image
    img = Image.new("RGB", (8, 8))
    diag: dict = {}
    assert r.decide("s", q, image=img) == r.decide("s", q, image=img, diagnostics=diag) and diag["table"]


class FakeReading:
    """Records what the P path receives; answers from a fixed table (EXP58's probabilities, or not)."""
    def __init__(self, answers, unreadable=()):
        self.answers, self.seen, self.unreadable = answers, [], set(unreadable)      # (row id, template) read as unreadable
        self.gold = m.gold_numbers(m.items())
        self.by_key = {it["key"]: it["row"]["id"] for it in m.items()}
        outer = self

        class D:
            cfg = None

            def decide(self, state, questions, image=None):
                outer.seen.append(("text", state, image))
                return {"answers": {k: {"type": "choice", "probabilities": {o: 1 / len(q["criteria"]) for o in q["criteria"]}}
                                    for k, q in questions.items()}}
        self.decider = D()

    def decide(self, text, questions, image=None, diagnostics=None):
        self.seen.append(("P", text, image is not None))
        (k, q), = questions.items()
        if diagnostics is not None:            # the structure Reading.read returns, numbers from the transcribed gold
            from devision.reading import TEMPLATES, UNREADABLE, fmt, reads_for, serialise
            rid, reads = self.by_key[k], []
            for t in reads_for(questions):
                if (rid, t) in self.unreadable:
                    reads.append({"question": t, "box": [0.1, 0.1, 0.2, 0.2], "degenerate": False, "decoded": "12", "valid": False, "value": UNREADABLE})
                else:
                    d = m.e54.number_of(self.gold[rid][t])
                    reads.append({"question": t, "box": [0.1, 0.1, 0.2, 0.2], "degenerate": False, "decoded": d, "valid": True, "value": fmt(d)})
            table = [[*TEMPLATES[x["question"]], x["value"]] for x in reads]
            diagnostics.update(reads=reads, table=table, state=serialise(text, table))
        return {"answers": {k: {"type": "choice", "probabilities": self.answers(k, q)}}}


def exp58_answers():
    st58 = m.e58.check_run(m.E58, data_root="", code=False)
    old = {}
    for name in ("sqa_temp", "sqa_force"):
        for r in m.e58.set_records(m.E58, name, st58):
            old[r["sample_id"]] = r["probabilities"]
    keyed = {it["key"]: old[it["row"]["id"]] for it in m.items()}
    return lambda k, q: dict(keyed[k])


def started(tmp_path, monkeypatch, reading):
    out = str(tmp_path / "out")
    m.stage_start(0, out=out, data_root="data")
    monkeypatch.setattr(m.e58, "load_reading", lambda *a, **k: (reading, m.e58.check_run(m.E58, data_root="", code=False),
                                                              json.load(open(os.path.join(m.E58, "pictures.json")))))
    monkeypatch.setattr(m, "check_start", lambda out, data_root="data", code=True: json.load(open(os.path.join(out, "START.json"))))
    return out


def test_p_that_does_not_reproduce_stops_before_g(tmp_path, monkeypatch):
    out = started(tmp_path, monkeypatch, FakeReading(lambda k, q: {o: 1 / len(q["criteria"]) for o in q["criteria"]}))
    with pytest.raises(SystemExit, match="does not reproduce"):
        m.stage_run(out)
    assert os.path.exists(os.path.join(out, "details_P.jsonl")) and not os.path.exists(os.path.join(out, "details_G_num.jsonl"))
    rep = m.report(out)
    assert rep["branch"].startswith("invalid / incomplete: P does not reproduce")


def test_reproduced_p_then_g_inputs_only_offline(tmp_path, monkeypatch):
    fr = FakeReading(exp58_answers())
    out = started(tmp_path, monkeypatch, fr)
    m.stage_run(out)
    its = m.items()
    p_calls = [s for s in fr.seen if s[0] == "P"]
    assert [s[1] for s in p_calls] == [it["row"].get("state_text") or "" for it in its] and all(s[2] for s in p_calls)   # picture, own text
    g = [s for s in fr.seen if s[0] == "text"]
    inp = {v["state"] for v in m.e54.load_inputs(out, "G_num").values()}
    assert len(g) == 36 and all(s[2] is None for s in g) and {s[1] for s in g[:24]} == inp            # gold only in G, image=None
    assert json.load(open(os.path.join(out, "reproduction.json")))["reproduced"]
    rep = m.report(out)
    assert rep["states"] == {c: "completed" for c in m.CONDS} and rep["candidate_eligible"] is False
    assert {k: (v["fields"], v["exact"], v["invalid"]) for k, v in rep["reading"].items()} == {"temp": (48, 48, 0), "force": (24, 24, 0)}


def test_a_real_unreadable_is_a_reading_failure_not_invalid_evidence(tmp_path, monkeypatch):
    it = m.items()[0]
    from devision.reading import reads_for
    t = reads_for(it["row"]["questions"])[0]
    out = started(tmp_path, monkeypatch, FakeReading(exp58_answers(), unreadable=[(it["row"]["id"], t)]))
    m.stage_run(out)
    rep = m.report(out)
    assert rep["states"]["P"] == "completed" and rep["reading"]["temp"]["invalid"] == 1 and rep["reading"]["temp"]["exact"] == 47
    assert [q for q in rep["per_question"] if q["id"] == it["row"]["id"]][0]["numbers_right"] is False


def edit_p(out, fn):
    f = os.path.join(out, "details_P.jsonl")
    lines = [json.loads(x) for x in open(f) if x.strip()]
    lines[1] = fn(json.loads(json.dumps(lines[1])))
    open(f, "w").write("".join(json.dumps(x) + "\n" for x in lines))
    m.register(out, "decide_P", {"details_P.jsonl": m.sha(f)})           # written and registered as if legitimately


def _missing(r):
    r["diagnostics"]["reads"] = r["diagnostics"]["reads"][1:]
    return r


def _duplicate(r):
    r["diagnostics"]["reads"] = r["diagnostics"]["reads"] + r["diagnostics"]["reads"][:1]
    return r


def _table(r):
    r["diagnostics"]["table"][0][2] = "999"
    return r


def _state(r):
    r["diagnostics"]["state"] = r["diagnostics"]["state"] + " "
    return r


def _repaired(r):                       # an invalid decode turned into a number
    r["diagnostics"]["reads"][0].update(valid=False, value="14")
    return r


@pytest.mark.parametrize("change", [_missing, _duplicate, _table, _state, _repaired])
def test_inconsistent_p_evidence_is_invalid_not_a_low_reading(tmp_path, monkeypatch, change):
    out = started(tmp_path, monkeypatch, FakeReading(exp58_answers()))
    m.stage_run(out)
    edit_p(out, change)
    st = json.load(open(os.path.join(out, "START.json")))
    with pytest.raises(SystemExit) as e:
        m.cond_records(out, "P", st, m.items())
    print("REFUSED", change.__name__, e.value)
    rep = m.report(out)
    assert rep["states"]["P"].startswith("missing / invalid") and "reading" not in rep
    assert rep["branch"].startswith("invalid / incomplete")


@pytest.mark.parametrize("change", ["edited", "missing_question"])
def test_changed_products_are_refused(tmp_path, monkeypatch, change):
    out = started(tmp_path, monkeypatch, FakeReading(exp58_answers()))
    m.stage_run(out)
    f = os.path.join(out, "details_G_num.jsonl")
    lines = open(f).read().splitlines()
    if change == "edited":
        lines[1] = lines[1].replace('"correct": true', '"correct": false').replace('"correct": false', '"correct": true', 0)
        lines[1] = json.dumps(dict(json.loads(lines[1]), state="other"))
        open(f, "w").write("\n".join(lines) + "\n")
    else:
        open(f, "w").write("\n".join(lines[:-1]) + "\n")
        m.register(out, "decide_G_num", {"details_G_num.jsonl": m.sha(f)})
    st = json.load(open(os.path.join(out, "START.json")))
    with pytest.raises(SystemExit):
        m.cond_records(out, "G_num", st, m.items())
    rep = m.report(out)
    assert rep["states"]["G_num"].startswith("missing / invalid") and rep["states"]["G_facts"] == "completed"
    assert not any("picture's facts" in b for b in rep["branch"])        # no G_num, no "facts fixed it" (EXP59-R1-02)


def test_facts_fix_counts_only_against_a_valid_g_num():
    base = {"kind": "force", "numbers_right": True, "P": False}
    assert not any("picture's facts" in b for b in m.branches({"per_question": [dict(base, G_facts=True)]}))           # G_num missing
    assert any(b.startswith("magnets: 1 wrong with gold numbers became right")
               for b in m.branches({"per_question": [dict(base, G_num=False, G_facts=True)]}))
    assert not any("still wrong" in b for b in m.branches({"per_question": [dict(base, G_num=False)]}))           # G_facts missing
