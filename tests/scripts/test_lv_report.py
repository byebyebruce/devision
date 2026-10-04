"""laya-vision report: paired difference, head to head and POPE counts from per-question records."""
import importlib.util
import os

spec = importlib.util.spec_from_file_location(
    "lv_report", os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "lv_report.py"))
assert spec is not None and spec.loader is not None
lv_report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lv_report)


def rec(sid, correct, pic, pred="true", gold="true"):
    return {"sample_id": sid, "image_id": pic, "correct": correct, "prediction": pred, "gold_answer": gold}


def test_paired_difference_and_head_to_head():
    ours = {"a": rec("a", 1, "p1"), "b": rec("b", 1, "p1"), "c": rec("c", 0, "p2")}
    theirs = {"a": rec("a", 1, "p1"), "b": rec("b", 0, "p1"), "c": rec("c", 1, "p2")}
    d = lv_report.interval(ours, theirs, {"a", "b", "c"})
    assert d["questions"] == 3 and abs(d["difference"]) < 1e-9 and d["ours"] == d["theirs"]
    assert lv_report.head_to_head(ours, theirs, ["a", "b", "c"]) == {"both": 1, "only_ours": 1, "only_theirs": 1, "neither": 0}


def test_pope_counts():
    rows = [rec("1", 1, "p", "true", "true"), rec("2", 0, "p", "true", "false"),
            rec("3", 0, "p", "false", "true"), rec("4", 1, "p", "false", "false")]
    s = lv_report.pope_stats(rows)
    assert (s["accuracy"], s["precision"], s["recall"], s["yes_ratio"]) == (0.5, 0.5, 0.5, 0.5)
