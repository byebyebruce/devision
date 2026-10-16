"""exp26 readout without a model: the binding numbers on fabricated records, and the report command end to end
against a fabricated exp25 run."""
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "exp21"))
from test_readout import control, syn_records, write  # type: ignore  # noqa: E402

spec = importlib.util.spec_from_file_location("exp26_report", os.path.join(HERE, "report.py"))
assert spec is not None and spec.loader is not None
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def pictures(binds: bool, n=20):
    """Records of n pictures with A=10, B=20 (unequal); binds -> right answers, else both answered with 20."""
    rows = []
    for i in range(n):
        sid = "exp21:d%04d:v0:h16" % i
        for qid, gold in (("readA", "10 cm"), ("readB", "20 cm")):
            pred = gold if binds else "20 cm"
            rows.append({"sample_id": sid, "qid": qid, "image_id": sid, "gold_answer": gold, "prediction": pred,
                         "correct": pred == gold})
        rows.append({"sample_id": sid, "qid": "cmp", "image_id": sid, "gold_answer": "B", "prediction": "B", "correct": True})
    return rows


def test_binding(tmp_path):
    for binds, same, other in ((True, 0.0, None), (False, 1.0, 1.0)):
        write(str(tmp_path / str(binds)), "x", pictures(binds))
        b = report.binding(str(tmp_path / str(binds) / "x.details.jsonl"))
        assert b["pictures"] == 20 and b["same_answer"] == same and b["other_panel"] == other


def answers(rows):
    """syn_records rows plus gold / predicted options: A's value 10, B's 20; a wrong reading takes the other's."""
    gold = {"readA": "10 cm", "readB": "20 cm", "cmp": "B"}
    other = {"readA": "20 cm", "readB": "10 cm", "cmp": "A"}
    return [dict(r, gold_answer=gold[r["qid"]], prediction=gold[r["qid"]] if r["correct"] else other[r["qid"]])
            for r in rows]


def run_dir(tmp_path, name, good):
    run = str(tmp_path / name)
    for step in (0, 1000, 2000, 3000, 4000, 5000):
        d = os.path.join(run, "stage2", "monitor", "step%d" % step)
        right = lambda f, v, h, q: step == 5000 and good   # noqa: E731
        write(d, "val", answers(syn_records(right, heights=(16,))))
        write(d, "dev16", answers(syn_records(right, heights=(16,))))
        write(d, "dev", syn_records(right))
        write(d, "erased", control(lambda q: False, "erased"))
        write(d, "text", control(lambda q: True, "text"))
    return run


def test_report_end_to_end(tmp_path, capsys):
    exp25 = run_dir(tmp_path, "exp25", False)
    exp26 = run_dir(tmp_path, "exp26", True)
    report.main([exp26, exp25])
    text = capsys.readouterr().out
    out = json.load(open(os.path.join(exp26, "report.json")))
    assert out["outcome"].startswith("reading and comparison both pass")
    assert out["vs_exp25"]["dev16"]["value"] == 1.0
    assert "binding improved" in text and "vs exp25, 16 px read dev16" in text
    assert out["binding_improved"] and out["binding"]["dev16"]["exp26"]["same_answer"] == 0.0
