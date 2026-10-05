"""Pair statistics over a set and its mirrored copy: both sides right is what counts."""
import importlib.util
import os

spec = importlib.util.spec_from_file_location(
    "pair_stats", os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "pair_stats.py"))
assert spec is not None and spec.loader is not None
pair_stats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pair_stats)


def rec(sid, correct, pic, axis="lr", kind="relation", typ="choice"):
    return {"sample_id": sid, "qid": "q", "correct": correct, "image_id": pic, "axis": axis, "kind": kind, "type": typ}


def test_only_questions_right_on_both_sides_count_as_both():
    original = [rec("a", True, "coco:1"), rec("b", True, "coco:2"), rec("c", False, "coco:3"),
                rec("d", True, "coco:4", axis="tb")]
    mirrored = [rec("flip:a", True, "coco:1:flip"), rec("flip:b", False, "coco:2:flip"),
                rec("flip:c", True, "coco:3:flip"), rec("flip:d", True, "coco:4:flip")]
    ps = pair_stats.pairs(original, mirrored)
    assert len(ps) == 3                      # the above/below question is not a left/right pair
    s = pair_stats.summary(ps, resamples=50)
    assert (s["original"], s["mirrored"], s["both"]) == (2 / 3, 2 / 3, 1 / 3)
    assert s["both_ci95"][0] <= s["both"] <= s["both_ci95"][1]


def test_a_mirrored_record_without_its_original_is_ignored():
    assert pair_stats.pairs([rec("a", True, "coco:1")], [rec("flip:z", True, "coco:9:flip")]) == []


def test_paired_difference_counts_only_pairs_both_models_answered():
    new = pair_stats.pairs([rec("a", True, "coco:1"), rec("b", True, "coco:2"), rec("c", True, "coco:3")],
                           [rec("flip:a", True, "coco:1:flip"), rec("flip:b", True, "coco:2:flip"),
                            rec("flip:c", True, "coco:3:flip")])
    base = pair_stats.pairs([rec("a", True, "coco:1"), rec("b", False, "coco:2")],
                            [rec("flip:a", True, "coco:1:flip"), rec("flip:b", True, "coco:2:flip")])
    d = pair_stats.paired_diff(new, base, resamples=50)
    assert (d["pairs"], d["new_both"], d["base_both"], d["diff"]) == (2, 1.0, 0.5, 0.5)
    assert d["diff_ci95"][0] <= d["diff"] <= d["diff_ci95"][1]


def test_picture_gain_change_is_a_difference_of_differences():
    qs = [rec("a", True, "coco:1"), rec("b", True, "coco:2")]
    new_real, new_mis = qs, [rec("a", False, "coco:1"), rec("b", False, "coco:2")]       # gain 1.0
    base_real = [rec("a", True, "coco:1"), rec("b", False, "coco:2")]
    base_mis = [rec("a", True, "coco:1"), rec("b", False, "coco:2")]                      # gain 0.0
    g = pair_stats.gain_change(qs, new_real, new_mis, base_real, base_mis, resamples=50)
    assert (g["questions"], g["new_gain"], g["base_gain"], g["gain_change"]) == (2, 1.0, 0.0, 1.0)
