"""scripts/protect/build.py (PROTECT-PREP-01, critic PROTECT-R1-01): the plan check works on the real data -- the
first build (data/protect, the pool's own groups coco:<id> vs coco:<id>:flip) splits every mirrored pair and is
refused; the rebuilt copy (data/protect2, one "protect-mirror:<base>" group per pair) keeps all of them in one batch
under the plan training draws, with the counts its MANIFEST records; a unit larger than a batch is refused."""
import importlib.util
import json
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
spec = importlib.util.spec_from_file_location("protect_build", os.path.join(HERE, "build.py"))
assert spec and spec.loader
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


def rows(name):
    return [json.loads(x) for x in open(os.path.join(ROOT, "data", name, "train_mix.jsonl"))]


data = pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "data", "protect2", "MANIFEST.json")), reason="data/protect2 not built")


@pytest.fixture(autouse=True)
def in_root(monkeypatch):
    monkeypatch.chdir(ROOT)


@data
def test_the_first_build_splits_the_mirrored_pairs():
    with pytest.raises(SystemExit, match="split mirrored pairs 383"):
        b.batch_plan(rows("protect"))


@data
def test_the_rebuilt_copy_keeps_every_pair_in_one_batch():
    plan = b.batch_plan(rows("protect2"))
    man = json.load(open(os.path.join(ROOT, "data", "protect2", "MANIFEST.json")))
    assert plan == man["plan"]
    assert plan["mirrored_pairs"] == plan["mirrored_pairs_in_one_batch"] == 383 and plan["updates"] == 1398
    assert plan["largest_unit"] <= 8 and plan["mixed_batches"] == 0 and plan["questions"] == 11180
    mirror = [r for r in rows("protect2") if r.get("block") == "mirror"]
    assert len(mirror) == 780 and all(r["group"].startswith("protect-mirror:") and r["source_group"] for r in mirror)


def test_a_unit_larger_than_a_batch_is_refused():
    mix = [{"questions": {"q": {}}, "input": "picture", "group": "big", "block": "science"} for _ in range(9)]
    with pytest.raises(SystemExit, match="larger than a batch"):
        b.batch_plan(mix)


def test_mirror_group_joins_original_and_flip():
    a = b.mirror_group({"block": "mirror", "image_id": "coco:5", "group": "coco:5"})
    f = b.mirror_group({"block": "mirror", "image_id": "coco:5:flip", "group": "coco:5:flip"})
    assert a["group"] == f["group"] == "protect-mirror:coco:5" and (a["source_group"], f["source_group"]) == ("coco:5", "coco:5:flip")
    assert b.mirror_group({"block": "science", "image_id": "x", "group": "x"}) == {}
