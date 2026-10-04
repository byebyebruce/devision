"""Round 8's mix: replay quotas grow counting and GQA only, and replay never touches data-v5's test pictures."""
import random

from v7_mix import QUOTAS as V7
from v8_mix import QUOTAS, pick_replay, replay_pool


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
