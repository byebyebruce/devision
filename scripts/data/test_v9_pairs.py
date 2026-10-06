"""Round 9's R list: only unused mirrors with a known original off the dev pictures, original before mirror."""
from v9_pairs import select


def flip(i, pic):
    return {"id": "flip:o%d" % i, "image_id": "coco:%d:flip" % pic}


def test_select_keeps_only_eligible_pairs_and_writes_original_then_mirror():
    originals = {"o%d" % i: {"id": "o%d" % i, "image_id": "coco:%d" % i} for i in range(6)}
    flips = [flip(i, i) for i in range(7)]          # o6 has no original
    used = {"flip:o1"}                               # round 6 trained on it
    dev_pics = {"coco:2"}                            # a monitoring picture
    rows = select(flips, originals, used, dev_pics, pairs=10)
    ids = [r["id"] for r in rows]
    assert sorted(ids[0::2]) == ["o0", "o3", "o4", "o5"]
    assert all(m == "flip:" + o for o, m in zip(ids[0::2], ids[1::2]))


def test_select_is_seeded_and_takes_the_first_pairs():
    originals = {"o%d" % i: {"id": "o%d" % i, "image_id": "coco:%d" % i} for i in range(50)}
    flips = [flip(i, i) for i in range(50)]
    a = select(flips, originals, set(), set(), pairs=5)
    assert a == select(flips, originals, set(), set(), pairs=5) and len(a) == 10
    assert [r["id"] for r in a] == [r["id"] for r in select(flips, originals, set(), set(), pairs=50)][:10]
