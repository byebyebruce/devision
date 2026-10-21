"""exp30 without a model: the isolation removes eval pictures by id (VG mapped to COCO, mirrors to originals) and
by path, the mix keeps both halves, and the readout's four outcomes."""
import importlib.util
import json
import os

import pytest

HERE = os.path.dirname(__file__)


def _load(name):
    spec = importlib.util.spec_from_file_location("exp30_" + name, os.path.join(HERE, name + ".py"))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


iso, report = _load("isolate"), _load("report")


def w(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def test_isolation_and_mix(tmp_path):
    root = str(tmp_path)
    w(os.path.join(root, "v5", "image_identity.json").replace(".json", ".tmp"), [])
    json.dump({"vg:7": "coco:3"}, open(os.path.join(root, "v5", "image_identity.json"), "w"))
    pool = [{"id": "c%d" % i, "image_id": "coco:%d" % i, "image": "coco/%d.jpg" % i, "captions": ["x"]} for i in range(10)]
    w(os.path.join(root, "align_full_train.jsonl"), pool)
    w(os.path.join(root, "r13b", "panel_exist.jsonl"), [{"image_id": "coco:1", "image": "coco/1.jpg"}])
    w(os.path.join(root, "v6", "dev_lr_pairs.jsonl"), [{"image_id": "coco:2:flip", "image": "flip/2.jpg"}])
    w(os.path.join(root, "v2", "test_gqa.jsonl"), [{"image_id": "vg:7", "image": "gqa/7.jpg"}])
    w(os.path.join(root, "lv_bench", "pope.jsonl"), [{"image_id": "other:9", "image": "coco/4.jpg"}])
    r = iso.build(root, n=6)
    kept = {json.loads(x)["image_id"] for x in open(os.path.join(root, "exp30", "real_captions.jsonl"))}
    assert not kept & {"coco:1", "coco:2", "coco:3", "coco:4"} and r["kept"] == 6
    w(os.path.join(root, "exp24", "align_train.jsonl"), [{"id": "t%d" % i} for i in range(6)])
    m = iso.mix(root)
    assert m == dict(m, text=6, real=6, rows=12)
    with pytest.raises(SystemExit):
        iso.build(root, n=7)


def rows(n, right, source="x"):
    return [{"sample_id": "%s%d" % (source, i), "qid": "q", "image_id": "%s-p%d" % (source, i), "source": source,
             "gold": {"a": 1.0, "b": 0.0}, "correct": bool(right(i)), "nll": 0.5} for i in range(n)]


def fixture(tmp_path, gap, base_ok, new_ok, a24_ok):
    run = tmp_path / "run"
    (run / "stage1").mkdir(parents=True)
    json.dump({"val": [{"step": 0, "val/mlm_acc_gap": 0.0}, {"step": 2500, "val/mlm_acc_gap": gap}]},
              open(run / "stage1" / "align_report.json", "w"))
    dirs = {"v03": tmp_path / "v03", "exp24A": tmp_path / "a24", "new": run / "eval"}
    for key, ok in (("v03", base_ok), ("exp24A", a24_ok), ("new", new_ok)):
        for p in report.GUARD:
            rs = [r for s in report.EXT_SOURCES for r in rows(20, lambda i, s=s: ok(p, i), s)] if p == "ext" else rows(60, lambda i: ok(p, i), p)
            w(str(dirs[key] / (p + ".details.jsonl")), rs)
        for k in ("sqa_temp", "sqa_force", "sqa_conc"):
            w(str(dirs[key] / (k + ".details.jsonl")), rows(10, lambda i: i % 2, k))
    return run, {"v03": str(dirs["v03"]), "exp24A": str(dirs["exp24A"])}


@pytest.mark.parametrize("gap,new,a24,start", [
    (0.4, lambda p, i: True, lambda p, i: i % 2, "passes this protection screen"),
    (0.4, lambda p, i: i % 4 != 0 if p in ("exist", "science") else i % 2, lambda p, i: i % 2 == 0 and i % 4 != 0, "both panels improve"),
    (0.4, lambda p, i: i % 2, lambda p, i: i % 2, "the pre-written two-panel improvement"),
    (0.2, lambda p, i: True, lambda p, i: i % 2, "reading gate not passed"),
])
def test_outcomes(tmp_path, monkeypatch, gap, new, a24, start):
    run, base = fixture(tmp_path, gap, lambda p, i: True, new, a24)
    monkeypatch.setattr(report, "BASE", base)
    assert report.build(str(run))["outcome"].startswith(start)
