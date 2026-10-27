"""Check label semantics and deterministic candidates, without a model or downloads."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("v05_chart_data", Path(__file__).with_name("chart_data.py"))
chart = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chart)


def row(answer, question="What is the value?"):
    return {"label": [answer], "query": question, "human_or_machine": 0}


def test_numeric_equivalents_cannot_be_wrong_options():
    pools = {"number": {chart.canonical(x): x for x in ["10", "10.0", "20", "30", "40", "50"]}}
    r = chart.convert(row("10.00"), pools, "id", "i.png", "picture")
    options = r["questions"]["q"]["criteria"]
    assert len(options) == 4
    assert sum(chart.canonical(x) == chart.canonical("10") for x in options) == 1
    assert r["gold"]["q"]["probabilities"]["10.00"] == 1
    assert r == chart.convert(row("10.00"), pools, "id", "i.png", "picture")


def test_yes_no_is_not_inverted_and_ambiguous_labels_are_refused():
    for answer, correct in [("Yes", "true"), ("No", "false")]:
        r = chart.convert(row(answer), {}, "id", "i.png", "picture")
        assert r["questions"]["q"]["type"] == "noul"
        assert r["gold"]["q"]["probabilities"][correct] == 1
    r = row("one")
    r["label"].append("two")
    assert chart.convert(r, {}, "id", "i.png", "picture") is None


def test_gray_alias_and_percent_semantics():
    assert chart.canonical("grey") == chart.canonical("gray")
    assert chart.canonical("1,000.0") == chart.canonical("1000")
    assert chart.canonical("10%") != chart.canonical("10")
    assert chart.pool_key("Which year?", "2014") == "year"
    assert chart.convert(row("10"), {"number": {"number:2E+1": "20"}}, "id", "i", "p") is None


def test_frequent_gold_does_not_get_only_rare_distractors():
    from collections import Counter
    pools = {"number": {chart.canonical(x): x for x in ["10", "20", "30", "40", "50"]}}
    frequencies = {"number": Counter({chart.canonical(x): n for x, n in [("10", 50), ("20", 40), ("30", 30), ("40", 25), ("50", 1)]})}
    r = chart.convert(row("10"), pools, "id", "i", "p", frequencies)
    assert set(r["questions"]["q"]["criteria"]) == {"10", "20", "30", "40"}
