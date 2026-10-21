"""Complete-fact CE data without a model: the facts line carries the caption's sizes only (no relation); canonical classes
map "weaker in Pair 1" to P2; reordered options keep the gold on its option; a training state is the row's text + its own
gold table + (magnets) the facts line; G / H / P differ exactly as defined (H and P carry EXP56's live table, P without
facts); isolation and the plan's counts."""
import importlib.util
import json
import os

import pytest

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.dirname(HERE))
spec = importlib.util.spec_from_file_location("facts_build", os.path.join(HERE, "build.py"))
assert spec is not None and spec.loader is not None
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
has = os.path.exists(os.path.join(ROOT, "data", "facts", "MANIFEST.json"))
data = pytest.mark.skipif(not has, reason="data/facts not built")


@pytest.fixture(autouse=True)
def in_root(monkeypatch):
    monkeypatch.chdir(ROOT)


def test_facts_line_and_classes():
    line = b.facts_line("Pair 1: two large magnets, 4.5 in apart. Pair 2: one large magnet and one small magnet, 3.75 in apart.")
    assert line == "Also shown in the picture: Pair 1: two large magnets. Pair 2: one large magnet and one small magnet."
    assert not any(w in line for w in ("stronger", "weaker", "greater", "smaller", "force", "in apart"))
    with pytest.raises(SystemExit):
        b.facts_line("Pair 1: two large magnets, 4.5 in apart.")
    cases = {"The magnetic force is weaker in Pair 1.": "P2", "The magnitude of the magnetic force is greater in Pair 1": "P1",
             "The magnitude of the magnetic force is smaller in Pair 2": "P1", "The strength of the magnetic force is the same in both pairs": "same"}
    assert all(b.gold_class("force", a) == c for a, c in cases.items())
    assert b.gold_class("temp", "neither; the samples have the same temperature") == "same" and b.gold_class("temp", "sample B") == "sample B"


def test_reordered_options_keep_the_gold():
    r = {"id": "x:1", "source": "s", "kind": "temp", "axis": "a", "image_id": "i", "image": "p", "family": "f", "case": "c",
         "questions": {"q": {"type": "choice", "instructions": "?", "criteria": {"a": None, "b": None, "c": None}}},
         "gold": {"q": {"probabilities": {"a": 0.0, "b": 1.0, "c": 0.0}}}}
    out = b.row_of(r, "state")
    opts = list(out["questions"]["q"]["criteria"])
    assert sorted(opts) == ["a", "b", "c"] and out["gold"]["q"]["probabilities"]["b"] == 1.0 and list(out["gold"]["q"]["probabilities"]) == opts


@data
def test_states_and_conditions():
    rows = {r["id"]: r for r in b.jsonl("data/real_layout/questions_train.jsonl")}
    caps = {c["id"]: c["captions"][0] for c in b.jsonl("data/real_layout/captions_train.jsonl")}
    train = b.jsonl("data/facts/train.jsonl")
    for x in train[::97]:
        r = rows[x["id"]]
        want = b.serialise(r.get("state_text", ""), b.gold_table(r))
        if x["kind"] == "force":
            want += "\n" + b.facts_line(caps[x["id"]])
        assert x["state_text"] == want
    live = {x["id"]: x for x in b.jsonl("runs/exp56/inputs_P.jsonl")}
    G, H, P = (b.jsonl("data/facts/tune_%s.jsonl" % c) for c in "GHP")
    assert len(G) == len(H) == len(P) == 600
    for g, hy, p in list(zip(G, H, P))[::53]:
        assert p["state_text"] == live[p["id"]]["state"]                           # P: the live table only
        assert hy["state_text"].startswith(live[hy["id"]]["state"])                # H: the live table + gold sizes
        assert (b.FACTS_HEADER in hy["state_text"]) == (hy["kind"] == "force") and b.FACTS_HEADER not in p["state_text"]
        assert g["questions"] == hy["questions"] == p["questions"]


@data
def test_isolation_and_plan():
    m = json.load(open("data/facts/MANIFEST.json"))
    assert m["isolation"] == {"train_tune_families": 0, "train_tune_cases": 0, "train_exp57_families": 0}
    assert (m["rows"], m["questions"], m["families"]["temp"], m["families"]["force"], m["families"]["fit"]) == (2400, 9600, 600, 600, 200)
    p = json.load(open("data/facts/PLAN.json"))
    assert (p["updates"], p["units"], p["split_units"], p["every_item_once"], p["batch_sizes"]) == (1200, 4800, 0, True, {"8": 1200})
    assert all(v["state_dropped"] == 0 and v["options_not_distinct"] == 0 for v in p["truncation"].values())
    fit = {r["family"] for r in b.jsonl("data/facts/fit.jsonl")}
    assert len(fit) == 200 and fit <= {r["family"] for r in b.jsonl("data/facts/train.jsonl")}
