"""exp27 readout on fabricated devision-eval records, no model: the three outcomes and the protection cost."""
import importlib.util
import json
import os

import pytest

HERE = os.path.dirname(__file__)
spec = importlib.util.spec_from_file_location("exp27_report", os.path.join(HERE, "report.py"))
assert spec is not None and spec.loader is not None
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def rows(n, right, source="x"):
    return [{"sample_id": "%s%d" % (source, i), "qid": "q", "image_id": "%s-p%d" % (source, i), "source": source,
             "gold": {"a": 1.0, "b": 0.0}, "correct": bool(right(i)), "nll": 0.5} for i in range(n)]


def write(d, name, rs):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, name + ".details.jsonl"), "w") as fh:
        for r in rs:
            fh.write(json.dumps(r) + "\n")


def fixture(tmp_path, exp_true, exp_mism, guard_drop=False):
    run = str(tmp_path / "run")
    for m in ("v03", "exp26"):
        d = os.path.join(run, "eval", m)
        new = m == "exp26"
        for k, n in (("temp", 41), ("force", 120)):
            write(d, "sqa_" + k, rows(n, (lambda i: exp_true) if new else (lambda i: i % 3 == 0), k))
            write(d, "sqa_%s_mism" % k, rows(n, (lambda i: exp_mism) if new else (lambda i: i % 3 == 0), k))
        for n in report.REPORT_ONLY:
            write(d, n, rows(24, lambda i: i % 2 == 0, n))
        for p in report.GUARD:
            if p == "ext":
                rs = [r for s in report.EXT_SOURCES for r in rows(20, lambda i: not (new and guard_drop) or i % 2, s)]
            else:
                rs = rows(60, lambda i: not (new and guard_drop and p == "exist") or i % 2, p)
            write(d, p, rs)
    return run


@pytest.mark.parametrize("exp_true,exp_mism,start", [
    (True, False, "transfer observed"),
    (True, True, "accuracy up against v0.3, but a larger picture gain is not shown"),
    (False, False, "no transfer shown"),
])
def test_outcomes(tmp_path, capsys, exp_true, exp_mism, start):
    run = fixture(tmp_path, exp_true, exp_mism)
    report.main([run])
    out = json.load(open(os.path.join(run, "report.json")))
    assert out["outcome"].startswith(start) and out["costs"] == []
    assert out["transfer"]["temp"]["questions"] == 41 and "outcome:" in capsys.readouterr().out


def test_a_protection_cost_is_listed(tmp_path):
    out = report.build(fixture(tmp_path, True, False, guard_drop=True))
    assert "exist" in out["costs"] and "ext" in out["costs"] and "science" not in out["costs"]
