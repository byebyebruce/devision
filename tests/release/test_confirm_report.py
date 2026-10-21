"""scripts/release/confirm_report.py: subsets reported separately, mismatched records grouped by the ORIGINAL
picture (their own image_id is the shown one), gain and reversed sensitivity per model."""
import importlib.util
import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
spec = importlib.util.spec_from_file_location("confirm_report", os.path.join(ROOT, "scripts", "release", "confirm_report.py"))
assert spec is not None and spec.loader is not None
cr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cr)


@pytest.fixture
def data(tmp_path):
    rows = [{"id": "s%d" % i, "image_id": "img%d" % (i // 2), "kind": "object-property" if i < 6 else "chart-reading"}
            for i in range(10)]
    p = tmp_path / "confirm.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(p), rows


def recs(rows, correct, shown=lambda r: r["image_id"], flip=False):
    return [{"sample_id": r["id"], "qid": "q", "image_id": shown(r), "correct": correct(r),
             "prediction": "b" if flip else "a"} for r in rows]


def test_groups_by_original_picture_and_splits_subsets(data):
    path, rows = data
    shown = lambda r: "img%d" % ((int(r["image_id"][3:]) + 1) % 5)        # the mismatched file's own image_id
    v03 = {"none": recs(rows, lambda r: False), "mism": recs(rows, lambda r: False, shown), "rev": recs(rows, lambda r: False)}
    cand = {"none": recs(rows, lambda r: True), "mism": recs(rows, lambda r: False, shown), "rev": recs(rows, lambda r: True, flip=True)}
    out = cr.build(path, v03, cand)
    assert set(out) == {"object-property", "chart-reading"}
    op, ch = out["object-property"], out["chart-reading"]
    assert op["acc_change"]["pictures"] == 3 and ch["acc_change"]["pictures"] == 2   # original pictures, not shown ones
    assert op["cand_gain"]["value"] == 1.0 and op["gain_change"]["value"] == 1.0 and op["cand_flip"]["value"] == 1.0
    assert op["v03_flip"]["value"] == 0.0 and op["acc_change"]["n"] == 6


def test_refuses_different_questions(data):
    path, rows = data
    base = {"none": recs(rows, lambda r: True), "mism": recs(rows, lambda r: True), "rev": recs(rows, lambda r: True)}
    short = dict(base, mism=recs(rows[:-1], lambda r: True))
    with pytest.raises(SystemExit):
        cr.build(path, base, short)
