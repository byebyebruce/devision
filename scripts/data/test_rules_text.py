"""Text rule questions: the gold follows the physics stated in the text (temperature: M * V^2; force: closer or
larger magnets -> stronger), cases are balanced, every gold is an option."""
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from rules_text import TEMP_SAME, build  # noqa: E402


def test_gold_follows_the_rule():
    rows = build(2000)
    for r in rows:
        q = r["questions"]["q"]
        gold = next(k for k, v in r["gold"]["q"]["probabilities"].items() if v == 1.0)
        assert gold in q["criteria"] and len(q["criteria"]) == 3
        text = r["state_text"].split("\n")[1]
        if r["kind"] == "temperature":
            (ma, va), (mb, vb) = [(int(m), int(v.replace(",", ""))) for m, v in re.findall(r"particle (\d+) u, average particle speed ([\d,]+)", text)]
            ka, kb = ma * va * va, mb * vb * vb
            assert gold == ("sample A" if ka > kb else "sample B" if kb > ka else TEMP_SAME)
        else:
            d = [float(x) for x in re.findall(r"([\d.]+) (?:in|cm|mm) apart", text)]
            small = ["one small" in p for p in text.split("Pair ")[1:]]
            if d[0] == d[1] and small[0] == small[1]:
                assert "same" in gold
            else:
                stronger = (1 if d[0] < d[1] else 2) if d[0] != d[1] else (2 if small[0] else 1)
                weak_word = "weaker" in gold or "smaller" in gold
                pair = int(re.search(r"Pair (\d)", gold).group(1))
                assert pair == (3 - stronger if weak_word else stronger)


def test_cases_are_balanced():
    c = Counter(r["axis"] for r in build(4000))
    assert len(c) == 7 and min(c.values()) > 400
