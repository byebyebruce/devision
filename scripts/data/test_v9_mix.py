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


def test_a_training_picture_near_a_held_out_hash_is_excluded():
    from v9_mix import near_held, original_image
    held = {"v3/images/aokvqa/x.jpg": 0b1111, "coco/val2014/a.jpg": 0xFF00}
    train = {"coco/train2014/same.jpg": 0b1111, "coco/train2014/close.jpg": 0b1100, "coco/train2014/far.jpg": 0xF0F0F0}
    assert near_held(train, held) == {"coco/train2014/same.jpg", "coco/train2014/close.jpg"}
    assert original_image("v6/images/flip/COCO_train2014_000000000009.jpg") == \
        "coco/train2014/COCO_train2014_000000000009.jpg"


def test_diagrams_are_not_photos():
    from v9_mix import is_photo
    assert is_photo("coco/train2014/a.jpg") and is_photo("v3/images/aokvqa/x.jpg")
    assert not is_photo("v3/images/scienceqa/x.jpg") and not is_photo("lv_bench/images/scienceqa/1.png")


def test_a_fixed_pair_list_drops_whole_pairs_on_held_or_near_duplicate_pictures():
    from v9_mix import fixed_pairs

    def row(i, pic, flip=False):
        return {"id": ("flip:" if flip else "") + "o%d" % i, "kind": "position", "image": "coco/p%d.jpg" % pic,
                "image_id": "coco:%d" % pic + (":flip" if flip else ""), "questions": {"q": {"type": "choice"}}}
    rows = [row(1, 1), row(1, 1, True), row(2, 2), row(2, 2, True), row(3, 3), row(3, 3, True)]
    keep, report = fixed_pairs(rows, held={"coco:2"}, v2c={}, off=lambda r: r["image"] == "coco/p3.jpg")
    assert [r["id"] for r in keep] == ["o1", "flip:o1"]
    assert report["fixed_list"]["dropped_ids"] == ["flip:o2", "flip:o3", "o2", "o3"]


def test_count_share_gives_counting_its_share_and_keeps_the_rest_proportional():
    from v9_mix import replay_quotas
    assert replay_quotas(4800) == scale(REPLAY_QUOTAS, 4800)          # unset: round 9's behaviour
    q = dict(replay_quotas(2488, 0.5))
    assert q["count"] == 1244 and sum(q.values()) == 2488
    assert (q["vsr"], q["scienceqa"], q["v7w"]) == (210, 184, 132)
