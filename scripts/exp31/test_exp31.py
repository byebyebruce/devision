"""exp31 data and readout without a model: the gold follows the stated numbers / sizes and the wording, part of
the gold is "equal / the same", a family's two pair orders differ only in the swapped values (same wording, same
option arrangement), every family is a distinct number case, and the report runs end to end."""
import importlib.util
import json
import os
import re

HERE = os.path.dirname(__file__)


def load(name):
    spec = importlib.util.spec_from_file_location("exp31_" + name, os.path.join(HERE, name + ".py"))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


build = load("build")


def rows_of(tmp_path):
    m = build.build(str(tmp_path))
    return m, [json.loads(x) for x in open(tmp_path / "exp31" / "text_force.jsonl")]


def gold_of(r):
    return next(k for k, v in r["gold"]["q"]["probabilities"].items() if v == 1.0)


def test_labels(tmp_path):
    m, rows = rows_of(tmp_path)
    assert m["rows"] == 600 and min(m["slots"]) > 150
    for lv, cats in m["gold_cat"].items():
        assert cats["same"] >= 0.2 * sum(cats.values()), lv
    for r in rows:
        text, gold = r["state_text"], gold_of(r)
        if r["kind"] == "d-rule-size":
            s = re.findall(r"Pair \d: (two large magnets|one large magnet and one small magnet)", text)
            stronger = "same" if s[0] == s[1] else "1" if s[0].startswith("two") else "2"
        else:
            d = [float(x) for x in re.findall(r"([\d.]+) (?:in|cm|mm)", text)][:2]
            stronger = "same" if d[0] == d[1] else "1" if d[0] < d[1] else "2"
            assert r["kind"] != "a-compare-numbers" or "Pair" not in text + " ".join(r["option_cats"])
        weak = ":" in r["id"].split(":", 3)[3] and r["id"].rsplit(":", 1)[1] in ("smaller", "weaker")
        expect = stronger if stronger == "same" or not weak else {"1": "2", "2": "1"}[stronger]
        assert r["gold_cat"] == expect and r["option_cats"][gold] == expect


def test_pair_orders_only_swap_values(tmp_path):
    _, rows = rows_of(tmp_path)
    views = {}
    for r in rows:
        key = r["id"].replace(":s0", ":s").replace(":s1", ":s")
        views.setdefault(key, {})[r["id"].split(":")[3][:2]] = r
    for v in views.values():
        a, b = v["s0"], v["s1"]
        ca = [a["option_cats"][o] for o in a["questions"]["q"]["criteria"]]
        cb = [b["option_cats"][o] for o in b["questions"]["q"]["criteria"]]
        assert ca == cb and a["questions"]["q"]["instructions"] == b["questions"]["q"]["instructions"]
        if a["gold_cat"] == "same":
            assert b["gold_cat"] == "same"
        else:
            assert {a["gold_cat"], b["gold_cat"]} == {"1", "2"}


def test_families_are_distinct_cases(tmp_path):
    _, rows = rows_of(tmp_path)
    case = {}
    for r in rows:
        if r["kind"] in ("a-compare-numbers", "d-rule-size"):
            nums = re.findall(r"[\d.]+ (?:in|cm|mm)", r["state_text"])
            key = (r["case"], tuple(sorted(nums)), r["kind"])
            case.setdefault(r["family"], set()).add(key)
    keys = [k for v in case.values() for k in v]
    assert all(len(v) == 1 for v in case.values()) and len(keys) == len(set(keys)) == 90


def test_report(tmp_path):
    _, rows = rows_of(tmp_path)
    report = load("report")
    for m in report.MODELS:
        d = tmp_path / "run" / "eval" / m
        d.mkdir(parents=True)
        with open(d / "text_force.details.jsonl", "w") as fh:
            for r in rows:
                opts = list(r["questions"]["q"]["criteria"])
                pred = opts[0]
                fh.write(json.dumps({"sample_id": r["id"], "qid": "q", "image_id": r["image_id"], "options": opts,
                                     "prediction": pred, "correct": pred == gold_of(r)}) + "\n")
    out = report.build(str(tmp_path / "run"), str(tmp_path / "exp31" / "text_force.jsonl"))
    c = out["v03"]["c-rule-distance"]
    assert c["slot_share"][0] == 1.0 and c["strong"]["families"] == 60 and c["equal"]["families"] == 15
    assert abs(sum(c["pred_share"].values()) - 1) < 1e-9
