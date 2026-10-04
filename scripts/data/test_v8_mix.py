"""Round 8's mix: replay quotas grow counting and GQA only, never fall short, and replay never touches held-out pictures."""
import json
import os
import random

import pytest
from v7_mix import QUOTAS as V7
from v8_mix import QUOTAS, check_quotas, held_pictures, pick_replay, replay_pool


def test_quotas_are_round_7s_plus_counting_and_gqa():
    v7 = dict(V7)
    assert {k: n - v7[k] for k, n in QUOTAS if n != v7[k]} == {"count": 1000, "gqa": 500}
    assert [k for k, _ in QUOTAS] == [k for k, _ in V7]


def test_replay_skips_test_pictures_questions_in_the_mix_and_scienceqa():
    rows = [{"id": "a", "image_id": "coco:1", "kind": "count", "source": "vqav2-choice"},
            {"id": "b", "image_id": "vg:9", "kind": "count", "source": "vqav2-choice"},     # VG alias of coco:2
            {"id": "c", "image_id": "coco:3", "kind": "count", "source": "vqav2-choice"},
            {"id": "d", "image_id": "coco:4", "kind": "sqa", "source": "scienceqa"},
            {"id": "e", "image_id": "coco:5", "kind": "relation", "axis": "lr", "source": "coco-relation"}]
    pool = replay_pool(rows, have={"c"}, held={"coco:2"}, v2c={"vg:9": "coco:2"})
    assert [r["id"] for r in pool] == ["a"]
    assert [r["id"] for r in pick_replay(pool, [("count", 5)], random.Random(0))] == ["a"]


def test_old_held_out_pictures_are_held_out_too_under_either_id(tmp_path):
    """align_val (caption alignment's validation) names a picture by COCO id; replay names it by its VG id."""
    def put(rel, ids):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps({"id": i, "image_id": i}) + "\n" for i in ids))
    for d in ("v2", "v3", "v4", "v6", "lv_bench"):
        os.makedirs(tmp_path / d, exist_ok=True)
    put("align_val.jsonl", ["coco:257450"])
    put("align_full_train.jsonl", ["coco:1"])                 # training, not held out
    for n in ("test_vsr", "test_v7w", "dev_vsr", "dev_v7w"):
        put("v5/%s.jsonl" % n, ["coco:9"] if n == "test_vsr" else [])
    v2c = {"vg:2411086": "coco:257450"}
    held = held_pictures(str(tmp_path), v2c)
    assert {"coco:257450", "coco:9"} <= held and "coco:1" not in held
    rows = [{"id": "v2-vgrel:2411086:tb:pepper-plate", "image_id": "vg:2411086", "kind": "relation", "axis": "tb",
             "source": "vg-relation"}]
    assert replay_pool(rows, set(), held, v2c) == []


def test_a_replay_ability_below_its_minimum_stops_the_build():
    full = [{"kind": "count"}] * 5 + [{"kind": "color"}] * 2
    assert check_quotas(full, [("count", 5), ("color", 2)])["count"] == 5
    with pytest.raises(SystemExit, match="count"):
        check_quotas(full[:4] + full[5:], [("count", 5), ("color", 2)])
