# pyright: reportAttributeAccessIssue=false, reportArgumentType=false
"""exp70 data without a model: the K block is built from the state's own numbers only (changing the gold or the option
order does not change it), in A-then-B order, exact; every row's K order is its gold; questions and options are the
source's; the train slice and the fidelity subset are the hashed ones."""
import importlib.util
import json
import os
from collections import Counter

import pytest

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.dirname(HERE))
spec = importlib.util.spec_from_file_location("exp70_build", os.path.join(HERE, "build.py"))
assert spec is not None and spec.loader is not None
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
has = os.path.exists(os.path.join(ROOT, "data", "exp70", "MANIFEST.json"))
data = pytest.mark.skipif(not has, reason="data/exp70 not built")


@pytest.fixture(autouse=True)
def in_root(monkeypatch):
    monkeypatch.chdir(ROOT)


def jl(p):
    return [json.loads(x) for x in open(p)]


def test_k_block_reads_only_the_state():
    state = "hint\nValues read from the picture (numbers only; units as in the question):\nSample A: mass of each particle = 44\n" \
            "Sample B: mass of each particle = 10\nSample A: average particle speed = 1,230\nSample B: average particle speed = 2,000"
    blk = b.k_block(state)
    assert blk.split("\n") == [b.K_HEADER, "Sample A: mass times speed squared = 66,567,600", "Sample B: mass times speed squared = 40,000,000"]
    r = {"id": "t", "state_text": state, "questions": {"q0": {"instructions": "Compare", "criteria": {"sample B": None, "sample A": None}}},
         "gold": {"q0": {"probabilities": {"sample B": 0.0, "sample A": 1.0}}}}
    rk = b.with_k(r)
    b.check(rk)
    flipped = dict(r, gold={"q0": {"probabilities": {"sample B": 1.0, "sample A": 0.0}}},
                   questions={"q0": {"instructions": "Compare", "criteria": {"sample A": None, "sample B": None}}})
    assert b.with_k(flipped)["state_text"] == rk["state_text"]                      # gold / option order never read
    with pytest.raises(SystemExit, match="K's order"):
        b.check(b.with_k(flipped))


@data
def test_built_sets():
    m = json.load(open("data/exp70/MANIFEST.json"))
    diag = {r["id"]: r for r in jl("data/exp68/diag.jsonl")}
    dk = jl("data/exp70/diag68_K.jsonl")
    assert len(dk) == 480 and all(r["state_text"].startswith(diag[r["id"]]["state_text"] + "\n" + b.K_HEADER) for r in dk)
    assert all(r["questions"] == diag[r["id"]]["questions"] and r["gold"] == diag[r["id"]]["gold"] for r in dk)
    for r in dk:
        b.check(r)
    fid = jl("data/exp70/fid_raw.jsonl")
    assert Counter(r["axis"] for r in fid) == {c: 8 for c in b.CLASSES} and all(r == diag[r["id"]] for r in fid)
    tr, tk = jl("data/exp70/train_raw.jsonl"), jl("data/exp70/train_K.jsonl")
    assert len(tr) == len(tk) == 120 and len({r["family"] for r in tr}) == 60 and all(len(r["questions"]) == 1 for r in tr)
    assert all(k["state_text"].startswith(r["state_text"] + "\n" + b.K_HEADER) for r, k in zip(tr, tk))
    assert m["new_requests"] == 768 and all(not v["state_dropped"] and not v["options_not_distinct"] for v in m["truncation"].values())
