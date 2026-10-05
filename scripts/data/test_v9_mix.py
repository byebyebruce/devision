"""Round 9's mix: replay shares scale exactly, left/right pairs prefer unused ones, nothing on a held-out picture."""
import random

import pytest
from v9_mix import REPLAY_QUOTAS, pick_pairs, pick_replay, scale


def test_the_replay_table_sums_to_16000_and_scales_exactly():
    assert sum(n for _, n in REPLAY_QUOTAS) == 16000
    for total in (3600, 4800, 16000):
        got = scale(REPLAY_QUOTAS, total)
        assert sum(n for _, n in got) == total and [k for k, _ in got] == [k for k, _ in REPLAY_QUOTAS]
    assert dict(scale(REPLAY_QUOTAS, 4800))["vsr"] == 720


def flip(i, kind, typ, pic):
    return {"id": "flip:o%d" % i, "kind": kind, "image_id": "coco:%d:flip" % pic,
            "questions": {"q": {"type": typ}}}


def test_pairs_take_unused_first_then_reused_and_skip_held_pictures():
    flips = [flip(1, "relation", "noul", 1), flip(2, "relation", "noul", 2), flip(3, "relation", "noul", 3),
             flip(4, "relation", "noul", 4)]
    originals = {"o%d" % i: {"id": "o%d" % i, "image_id": "coco:%d" % i} for i in range(1, 5)}
    rows, report = pick_pairs(flips, originals, {("relation", "noul"): 2}, used={"flip:o2", "flip:o3"},
                              blocked={"coco:4"}, v2c={}, rng=random.Random(0))
    ids = {r["id"] for r in rows}
    assert "flip:o1" in ids and "o1" in ids and "flip:o4" not in ids
    assert report["relation/noul"]["unused"] == 1 and report["relation/noul"]["reused"] == 1
    with pytest.raises(SystemExit):
        pick_pairs(flips, originals, {("relation", "noul"): 4}, set(), {"coco:4"}, {}, random.Random(0))


def test_a_replay_ability_below_its_share_stops_the_build():
    pools = {"count": [{"id": i} for i in range(3)], "color": [{"id": 9}]}
    replay, got = pick_replay(pools, [("count", 3), ("color", 1)], random.Random(0))
    assert len(replay) == 4 and got == {"count": 3, "color": 1}
    with pytest.raises(SystemExit):
        pick_replay(pools, [("count", 4)], random.Random(0))
