"""exp31 data without a model: every family is asked in both pair orders, the gold follows the stated distances /
sizes and the wording (strong vs weak), and the right option's slot is spread."""
import importlib.util
import json
import os
import re

HERE = os.path.dirname(__file__)
spec = importlib.util.spec_from_file_location("exp31_build", os.path.join(HERE, "build.py"))
assert spec is not None and spec.loader is not None
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


def test_labels(tmp_path):
    m = build.build(str(tmp_path))
    rows = [json.loads(x) for x in open(tmp_path / "exp31" / "text_force.jsonl")]
    assert m["rows"] == 600 and min(m["slots"]) > 150
    for r in rows:
        text = r["state_text"]
        gold = next(k for k, v in r["gold"]["q"]["probabilities"].items() if v == 1.0)
        if r["kind"] == "d-rule-size":
            stronger = 1 if text.split("Pair 1: ")[1].startswith("two large") else 2
        else:
            d = [float(x) for x in re.findall(r"([\d.]+) (?:in|cm|mm) apart", text)]
            stronger = 1 if d[0] < d[1] else 2
        if r["kind"] in ("a-smaller-number", "b-closer-pair"):
            assert gold.endswith(str(stronger))
        else:
            weak = "smaller" in gold or "weaker" in gold
            assert ("Pair %d" % (3 - stronger if weak else stronger)) in gold
    fams = {}
    for r in rows:
        fams.setdefault(r["family"], set()).add(r["id"].split(":")[3][:2])
    assert all(v == {"s0", "s1"} for v in fams.values())
