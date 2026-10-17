"""exp29 without a model: cyclic orders keep the gold with its option, and the readout averages each option's
probability over the orders, against order 0 as the single-order baseline."""
import importlib.util
import json
import os

import pytest

HERE = os.path.dirname(__file__)


def _load(name):
    spec = importlib.util.spec_from_file_location("exp29_" + name, os.path.join(HERE, name + ".py"))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


build, report = _load("build"), _load("report")


def test_cyclic_orders():
    s = {"id": "s1", "image_id": "p", "image": "x.png",
         "questions": {"q": {"type": "choice", "instructions": "?", "criteria": {"a": None, "b": None, "c": None}},
                       "n": {"type": "noul", "instructions": "?"}},
         "gold": {"q": {"probabilities": {"a": 0.0, "b": 1.0, "c": 0.0}}, "n": {"probabilities": {"true": 1.0}}}}
    rows = build.cyclic(s)
    assert [r["id"] for r in rows] == ["s1|q:o0", "s1|q:o1", "s1|q:o2"]
    assert [list(r["questions"]["q"]["criteria"]) for r in rows] == [["a", "b", "c"], ["b", "c", "a"], ["c", "a", "b"]]
    assert all(r["gold"]["q"]["probabilities"]["b"] == 1.0 and list(r["gold"]["q"]["probabilities"]) ==
               list(r["questions"]["q"]["criteria"]) for r in rows)


def rec(order, probs, gold="b", image="p"):
    opts = list(probs)
    pred = max(probs, key=probs.get)
    return {"sample_id": "s1|q:o%d" % order, "qid": "q", "image_id": image, "options": opts, "probabilities": probs,
            "gold": {k: float(k == gold) for k in probs}, "prediction": pred, "correct": pred == gold, "nll": 0.0}


def test_average_over_orders_fixes_a_first_option_bias():
    # always picks the first option, but b is second-best everywhere
    rs = [rec(0, {"a": 0.5, "b": 0.3, "c": 0.2}), rec(1, {"b": 0.5, "c": 0.2, "a": 0.3}), rec(2, {"c": 0.5, "a": 0.2, "b": 0.3})]
    out = report.combine(rs)
    q = out[("s1|q", "q")]
    assert q["single"]["correct"] is False and q["average"]["correct"] is True and q["orders"] == 3
    assert all(q["first_choice"])


def test_a_missing_order_is_refused():
    with pytest.raises(ValueError):
        report.combine([rec(0, {"a": 0.5, "b": 0.3, "c": 0.2}), rec(2, {"c": 0.5, "a": 0.2, "b": 0.3})])


def test_report_end_to_end(tmp_path, capsys):
    run = tmp_path / "run"
    (run / "eval").mkdir(parents=True)
    for p in report.PANELS:
        with open(run / "eval" / (p + ".details.jsonl"), "w") as fh:
            for i in range(30):
                for r in (rec(0, {"a": 0.5, "b": 0.3, "c": 0.2}), rec(1, {"b": 0.5, "c": 0.2, "a": 0.3}),
                          rec(2, {"c": 0.5, "a": 0.2, "b": 0.3})):
                    r = dict(r, sample_id=r["sample_id"].replace("s1", "s%d" % i), image_id="p%d" % i)
                    fh.write(json.dumps(r) + "\n")
    report.main([str(run)])
    out = json.load(open(run / "report.json"))
    assert out["helps"] and out["panels"]["science"]["average"] == 1.0 and out["panels"]["science"]["single"] == 0.0
    assert "macro change" in capsys.readouterr().out
