# pyright: reportAttributeAccessIssue=false, reportArgumentType=false
"""exp72 data without a model: the two arms differ only by the K block on temperature states (rows, ids, groups,
questions, golds, order and magnets identical); the comparison-led mix (two passes of the 1,200 comparisons, the fixed
reading / magnet replay) and its counts; the readout sets; the audited plan equality and truncation."""
import importlib.util
import json
import os
from collections import Counter

import pytest

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.dirname(HERE))
spec = importlib.util.spec_from_file_location("exp72_build", os.path.join(HERE, "build.py"))
assert spec is not None and spec.loader is not None
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
has = os.path.exists(os.path.join(ROOT, "data", "exp72", "MANIFEST.json"))
data = pytest.mark.skipif(not has, reason="data/exp72 not built")


@pytest.fixture(autouse=True)
def in_root(monkeypatch):
    monkeypatch.chdir(ROOT)


def jl(p):
    return [json.loads(x) for x in open(p)]


@data
def test_arms_differ_only_by_the_k_block():
    raw, k = jl("data/exp72/train_raw.jsonl"), jl("data/exp72/train_K.jsonl")
    assert len(raw) == len(k)
    for a, c in zip(raw, k):
        assert {x: a[x] for x in a if x != "state_text"} == {x: c[x] for x in c if x != "state_text"}
        if a["kind"] == "temp":
            assert c["state_text"] == a["state_text"] + "\n" + b.b70.k_block(a["state_text"])
        else:
            assert c["state_text"] == a["state_text"]
    for n in ("slice", ):
        r_, k_ = jl("data/exp72/%s_raw.jsonl" % n), jl("data/exp72/%s_K.jsonl" % n)
        assert all(x["state_text"] + "\n" + b.b70.k_block(x["state_text"]) == y["state_text"] for x, y in zip(r_, k_))


@data
def test_comparison_led_mix():
    raw = jl("data/exp72/train_raw.jsonl")
    comps = [r for r in raw if r["kind"] == "temp" and all(q["instructions"] not in b.fb.TEMPLATES for q in r["questions"].values())]
    p2 = [r for r in comps if r["id"].endswith("#p2")]
    assert len(comps) == 2400 and len(p2) == 1200 and all(r["group"].endswith("#p2") for r in p2)
    assert Counter(r["axis"] for r in comps) == {c: 400 for c in b.CLASSES}
    reads = [r for r in raw if r["kind"] == "temp" and r not in comps]
    assert sum(len(r["questions"]) for r in reads) == 800 and all(q["instructions"] in b.fb.TEMPLATES for r in reads for q in r["questions"].values())
    mags = [r for r in raw if r["kind"] == "force"]
    src = {x for x in open("data/exp68/train.jsonl").read().splitlines()}
    assert sum(len(r["questions"]) for r in mags) == 600 and all(json.dumps(r) in src for r in mags)
    m = json.load(open("data/exp72/MANIFEST.json"))
    assert m["training"]["by_kind"]["temperature comparison"]["share"] == pytest.approx(2400 / 3800, abs=1e-4)
    sl = jl("data/exp72/slice_raw.jsonl")
    assert len(sl) == 240 and Counter(r["axis"] for r in sl) == {c: 40 for c in b.CLASSES}


@data
def test_plan_and_truncation_audit():
    p = json.load(open("data/exp72/MANIFEST.json"))["plan"]
    assert p["plans_equal"] and p["raw"] == p["K"] and p["raw"]["updates"] == 475 and p["raw"]["split_units"] == 0 and p["raw"]["every_item_once"]
    assert all(not v["state_dropped"] and not v["options_not_distinct"] for v in p["truncation"].values())
