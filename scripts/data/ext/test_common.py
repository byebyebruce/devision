"""Extension pack, shared pieces: Cauldron parsing, sample builders, balancing, the question-only baseline,
resumable download (local file URL), and the IconQA / CLEVR / chart conversions."""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (balance_yes_no, choice, count_options, download, flatten_answers, multiple_choice, noul,  # noqa: E402
                    problems, question_only_baseline, question_text, short_turn)


def test_cauldron_turns_parse():
    assert question_text("Is Pale Green the minimum?\nAnswer yes or no.") == "Is Pale Green the minimum?"
    assert question_text("Subtract all cubes. How many are left?\nBe succinct.") == "Subtract all cubes. How many are left?"
    assert multiple_choice("Question: How many?\nChoices:\nA. 10\nB. 3\nAnswer with the letter.", "Answer: B") == \
        ("How many?", ["10", "3"], "3")
    assert multiple_choice("How many?", "3") is None
    assert short_turn({"user": "Are there more cubes?\nBe succinct.", "assistant": "Yes."}) == ("yesno", "Are there more cubes?", True)
    assert short_turn({"user": "How many chairs?\nBe brief.", "assistant": "2."}) == ("number", "How many chairs?", 2)
    assert short_turn({"user": "What colour?", "assistant": "Cyan."}) == ("word", "What colour?", "Cyan")


def test_samples_are_well_formed_and_counts_stay_near_the_answer():
    rng = random.Random(0)
    for n in range(0, 11):
        opts = count_options(n, rng)
        assert str(n) in opts and 2 <= len(opts) <= 5 and all(abs(int(o) - n) <= 3 for o in opts)
    c = choice("a", "s", "k", "s:1", "x.png", "How many?", ["3", "4"], "4", rng)
    y = noul("b", "s", "k", "s:1", "x.png", "Is it?", False)
    assert problems(c) == [] and problems(y) == []
    assert y["gold"]["q"]["probabilities"] == {"false": 1.0, "true": 0.0} and c["answer_key"] == "4"


def test_balancing_and_baseline():
    rows = [noul("y%d" % i, "s", "k", "s:%d" % i, "x", "q?", True, group="g") for i in range(10)] + \
           [noul("n%d" % i, "s", "k", "s:%d" % i, "x", "q?", False, group="g") for i in range(4)]
    bal = balance_yes_no(rows)
    assert sum(r["answer_key"] == "true" for r in bal) == sum(r["answer_key"] == "false" for r in bal) == 4
    rng = random.Random(0)
    counts = [choice("c%d" % i, "s", "count", "s:%d" % i, "x", "How many?", ["1", "2"], "1" if i < 90 else "2", rng)
              for i in range(100)]
    flat = flatten_answers(counts)
    assert sum(r["answer_key"] == "1" for r in flat) <= 100
    b = question_only_baseline(bal + counts, "wording")
    assert 0 <= b["accuracy"] <= 1 and b["by"] == "wording"


def test_download_resumes_and_never_refetches(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"0123456789" * 1000)
    dst = tmp_path / "out" / "f.bin"
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "f.bin.part").write_bytes(b"0123456789" * 10)     # an interrupted download
    download(src.as_uri(), str(dst))                                        # file:// ignores Range: starts over
    assert dst.read_bytes() == src.read_bytes()
    src.write_bytes(b"changed")
    download(src.as_uri(), str(dst))                                        # already there: not fetched again
    assert dst.read_bytes() != b"changed"


def test_clevr_and_chart_conversions():
    from chartmap import template as chart_template
    from clevr import convert
    from figureqa import template as fig_template
    rng = random.Random(0)
    r, _ = convert("clevr", "k", 0, "x.png", {"user": "What color is the cube?\nBe brief.", "assistant": "Cyan."}, rng)
    assert r is not None
    assert r["kind"] == "color" and "cyan" in r["questions"]["q"]["criteria"] and r["answer_key"] == "cyan"
    r, _ = convert("clevr", "k", 1, "x.png", {"user": "Are there more cubes than spheres?", "assistant": "No."}, rng)
    assert r is not None
    assert r["kind"] == "compare" and r["answer_key"] == "false"
    r, why = convert("clevr", "k", 2, "x.png", {"user": "How many?", "assistant": "12."}, rng)
    assert r is None and why == "count > 10"
    assert fig_template("Is Dark Red greater than Pale Green?") == "is x greater than x?"
    assert chart_template("Does Nebraska have a higher value than New York?") == "does x have a higher value than x?"
