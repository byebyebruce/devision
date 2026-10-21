"""exp21 synthetic reading data (critic EXP21-R1-02): labels follow the drawn values, not option positions;
the candidates and option order stay fixed inside a family; only the number changes size; families never cross
train / dev; the drawn digits are the stated height in pixels."""
import json
import random
import sys
import os
from collections import Counter

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from exp21_reading import (EQUAL, HEIGHTS, answers, build, digit_ink, family, fmt, questions,  # noqa: E402
                           render)


def families(n=400, seed=5):
    rng = random.Random(seed)
    return [family(rng, "f%d" % i) for i in range(n)]


def test_view_one_changes_every_answer_that_can_change():
    for f in families():
        a0, a1 = answers(f, 0), answers(f, 1)
        assert a0["cmp"] != a1["cmp"] and a0["readB"] != a1["readB"]
        if a0["cmp"] != EQUAL:
            assert a0["readA"] != a1["readA"]            # swapped: both readings change
        assert questions(f) == questions(f)                # same candidates and option order in both views


def test_answers_are_options_and_the_reading_candidates_include_both_panels():
    for f in families():
        q = questions(f)
        for view in (0, 1):
            ans = answers(f, view)
            for qid in q:
                assert ans[qid] in q[qid]["criteria"]
        a, b = f["views"][0]
        assert {fmt(a) + " " + f["unit"], fmt(b) + " " + f["unit"]} <= set(q["readA"]["criteria"])
        assert len({len(str(v)) for v in f["candidates"]}) == 1      # no length cue


def test_no_text_only_rule_gets_far_above_chance():
    """Rules a model could use without the picture: a fixed option slot, the middle / largest candidate,
    always A / B / equal for compare."""
    rows = [(f, v) for f in families(1200) for v in (0, 1)]
    def acc(rule, qid):
        return sum(rule(f, questions(f)[qid]) == answers(f, v)[qid] for f, v in rows) / len(rows)
    num = lambda k: int(k.split()[0].replace(",", ""))   # noqa: E731
    for qid in ("readA", "readB"):
        for slot in range(3):
            assert acc(lambda f, q: list(q["criteria"])[slot], qid) < 0.42
        assert acc(lambda f, q: sorted(q["criteria"], key=num)[1], qid) < 0.45
        assert acc(lambda f, q: max(q["criteria"], key=num), qid) < 0.45
    for k in ("A", "B", EQUAL):
        assert acc(lambda f, q: k, "cmp") < 0.45
    for slot in range(3):
        assert acc(lambda f, q: list(q["criteria"])[slot], "cmp") < 0.42


@pytest.mark.parametrize("h", HEIGHTS)
def test_digits_are_the_stated_height_and_only_the_number_changes(h):
    for f in families(30):
        img, ink = render(f, 0, h)
        assert all(abs(digit_ink(img, b) - h) <= 1 for b in ink.values())   # a number of only 4s is 1 px shorter
        assert all(b[0] >= 0 and b[2] <= 256 for b in ink.values())       # inside the picture
        erased, _ = render(f, 0, h, erased=True)
        small_erased, _ = render(f, 0, HEIGHTS[0], erased=True)
        assert erased.tobytes() == small_erased.tobytes()               # everything but the number is the same


def test_build_keeps_families_apart_and_counts(tmp_path):
    replay = tmp_path / "replay.jsonl"
    with open(replay, "w") as fh:
        for i in range(400):
            fh.write(json.dumps({"id": "r%d" % i, "source": "r14b", "image_id": "x:%d" % i, "image": "x.png",
                                 "questions": {"q": {"type": "noul", "instructions": "?"}},
                                 "gold": {"q": {"probabilities": {"false": 0.0, "true": 1.0}}}}) + "\n")
    m = build(str(tmp_path), str(replay), n_train=60, n_dev=8, n_controls=4, total=300)
    rows = {k: [json.loads(x) for x in open(tmp_path / "exp21" / (k + ".jsonl"))]
            for k in ("train_syn", "dev_syn", "dev_syn_erased", "dev_syn_text", "train_mix")}
    train_fam = {s["family"] for s in rows["train_syn"]}
    dev_fam = {s["family"] for s in rows["dev_syn"]}
    assert not train_fam & dev_fam and len(dev_fam) == 8
    assert m["files"]["dev_syn"]["questions"] == 8 * 2 * len(HEIGHTS) * 3
    assert m["files"]["dev_syn_erased"]["questions"] == m["files"]["dev_syn_text"]["questions"] == 4 * 2 * 3
    assert all("state_text" in s for s in rows["dev_syn_text"])
    assert all("state_text" not in s for s in rows["dev_syn_erased"])
    assert sum(len(s["questions"]) for s in rows["train_mix"]) >= 300
    assert Counter(s["height"] for s in rows["dev_syn"]) == {h: 8 * 2 for h in HEIGHTS}
    assert all(sum(m["dev_ink_height_px"][str(h)].values()) == 2 * 8 * 2 for h in HEIGHTS)
    assert all(abs(int(k) - h) <= 1 for h in HEIGHTS for k in m["dev_ink_height_px"][str(h)])
    # every question has exactly one correct option
    assert all(sum(g["probabilities"].values()) == 1.0 for s in rows["dev_syn"] for g in s["gold"].values())
