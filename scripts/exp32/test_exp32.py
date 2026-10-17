"""exp32 readout on fabricated records, no model: the delivery-target rule, protection costs and outcome labels."""
import importlib.util
import json
import os

import pytest

HERE = os.path.dirname(__file__)
spec = importlib.util.spec_from_file_location("exp32_report", os.path.join(HERE, "report.py"))
assert spec is not None and spec.loader is not None
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def rows(n, right, source="x", qid="q", ids=None):
    return [{"sample_id": (ids[i] if ids else "%s%d" % (source, i)), "qid": qid, "image_id": "%s-p%d" % (source, i),
             "source": source, "gold": {"a": 1.0, "b": 0.0}, "correct": bool(right(i)), "nll": 0.5} for i in range(n)]


def w(path, rs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        for r in rs:
            fh.write(json.dumps(r) + "\n")


@pytest.fixture
def setup(tmp_path, monkeypatch):
    def make(new_true, new_mism, guard_drop=False):
        v03, run = tmp_path / "v03", tmp_path / "run"
        for d, new in ((v03, False), (run / "eval", True)):
            for k, n in (("temp", 41), ("force", 120)):
                w(str(d / ("sqa_%s.details.jsonl" % k)), rows(n, (lambda i: new_true(i)) if new else (lambda i: i % 3 == 0), k))
                w(str(d / ("sqa_%s_mism.details.jsonl" % k)), rows(n, (lambda i: new_mism(i)) if new else (lambda i: i % 3 == 0), k))
            for p in report.GUARD:
                rs = [r for s in report.EXT_SOURCES for r in rows(20, lambda i: not (new and guard_drop) or i % 2, s)] if p == "ext" \
                    else rows(60, lambda i: not (new and guard_drop) or i % 2, p)
                w(str(d / (p + ".details.jsonl")), rs)
            for k in ("sqa_conc", "sqa_repel", "sqa_text", "sqa_blank"):
                w(str(d / (k + ".details.jsonl")), rows(10, lambda i: i % 2, k))
        monkeypatch.setattr(report, "V03", str(v03))
        monkeypatch.setattr(report, "by_kind", lambda rs, path: {"x": 1.0})
        for split in ("rl_tune", "rl_confirm"):
            for d in (run / "eval_v03", run / "eval"):
                w(str(d / (split + ".details.jsonl")), rows(20, lambda i: i % 2, split))
        import sys
        sys.path.insert(0, os.path.join(HERE, "..", "exp21"))
        from test_readout import syn_records  # type: ignore
        w(str(run / "eval" / "dev16.details.jsonl"), syn_records(lambda f, v, h, q: True, heights=(16,)))
        return str(run)
    return make


@pytest.mark.parametrize("true,mism,drop,start", [
    (lambda i: True, lambda i: i % 3 == 0, False, "delivery target met with no"),
    (lambda i: True, lambda i: i % 3 == 0, True, "delivery target met, protection costs remain"),
    (lambda i: i % 3 == 0 or i % 3 == 1, lambda i: i % 3 == 0 or i % 3 == 1, False, "improves on v0.3 but below"),
    (lambda i: i % 3 == 0, lambda i: i % 3 == 0, False, "no improvement"),
])
def test_outcomes(setup, true, mism, drop, start):
    out = report.build(setup(true, mism, drop))
    assert out["outcome"].startswith(start)


def test_real_layout_families():
    """Two views of a family share one resampling unit; 'follows the values' counts pairs whose gold moved."""
    rs = []
    for f in range(4):
        for v in range(2):
            rs.append({"sample_id": "real-layout:rl%05d:v%d" % (f, v), "qid": "q0", "image_id": "real-layout:rl%05d:v%d" % (f, v),
                       "gold_answer": "sample A" if v == 0 or f % 2 else "sample B", "correct": f != 0 or v == 0})
    assert {r["image_id"] for r in report.by_family(rs)} == {"real-layout:rl%05d" % f for f in range(4)}
    v = report.views(rs)
    assert v["pairs"] == 4 and v["both_views_right"] == 0.75 and v["moved_pairs"] == 2 and v["follows_values"] == 0.5
