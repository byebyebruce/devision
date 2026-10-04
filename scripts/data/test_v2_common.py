"""v2 final steps: balance holds in the written rows, dev split is by picture, acceptance catches gaps."""
from v2_common import balance_overall, check, equalize, flatten_top, in_dev, split_dev


def row(i, group, answer, image=None):
    return {"id": "r%d" % i, "group": group, "answer_key": answer, "image_id": image or "coco:%d" % i}


def test_equalize_balances_every_group_and_drops_one_sided_groups():
    rows = [row(i, "dog", "true") for i in range(5)] + [row(10 + i, "dog", "false") for i in range(2)] + \
           [row(20 + i, "cat", "true") for i in range(3)]
    out = equalize(rows)
    assert sorted(r["answer_key"] for r in out) == ["false", "false", "true", "true"]
    assert check(out, "x", "equal") == []
    assert check(rows, "x", "equal")


def test_flatten_top_caps_the_dominant_answer_at_the_second():
    rows = [row(i, "sky", "blue") for i in range(8)] + [row(10 + i, "sky", "gray") for i in range(3)]
    out = flatten_top(rows)
    assert sum(r["answer_key"] == "blue" for r in out) == 3 and check(out, "x", "flat") == []


def test_overall_balance():
    rows = [row(i, "_", "true") for i in range(6)] + [row(10 + i, "_", "false") for i in range(4)]
    assert len(balance_overall(rows)) == 8


def test_dev_split_keeps_a_picture_on_one_side_under_any_of_its_names():
    names = {"vg:1": "coco:7", "vg:2": "coco:7"}
    picture = lambda i: names.get(i, i)
    rows = [row(i, "g", "a", image=img) for i, img in enumerate(["coco:7", "vg:1", "vg:2"] * 50)]
    train, dev = split_dev(rows, picture, 0.5)
    assert {picture(r["image_id"]) for r in train}.isdisjoint({picture(r["image_id"]) for r in dev})
    assert in_dev("coco:7", 0.5) == in_dev("coco:7", 0.5)


def test_duplicate_ids_are_reported():
    assert check([row(1, "g", "a"), row(1, "g", "b")], "x", "none")
