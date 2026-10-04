"""data-v3 conversion: Cauldron multiple-choice turns, option shuffling and the samples it makes."""
import random
from collections import Counter

from v3_build import make_sample, parse_cauldron, position_baseline

TURN = ("Question: Which is a producer?\nChoices:\nA. Coyote\nB. Desert Grass\nC. Kangaroo\nD. Dingo\n"
        "Answer with the letter.")


def test_a_cauldron_turn_gives_question_options_and_answer():
    assert parse_cauldron(TURN, "Answer: B") == ("Which is a producer?", ["Coyote", "Desert Grass", "Kangaroo", "Dingo"], 1)


def test_turns_that_are_not_multiple_choice_are_refused():
    assert parse_cauldron("Describe the picture.", "A food web.") is None
    assert parse_cauldron(TURN, "Answer: F") is None


def test_the_gold_option_survives_shuffling_and_its_position_is_uniform():
    rng = random.Random(0)
    rows = [make_sample("ai2d", str(i), "img:1", "x.jpg", "Which?", ["a", "b", "c", "d"], 0, rng) for i in range(400)]
    for s in rows:
        g = s["gold"]["q"]["probabilities"]
        assert max(g, key=g.get) == "a" and set(s["questions"]["q"]["criteria"]) == {"a", "b", "c", "d"}
        assert list(s["questions"]["q"]["criteria"]) == list(g)   # options asked in the order of the gold
    pos = Counter(s["answer_position"] for s in rows)
    assert min(pos.values()) > 70                                 # about 100 each
    assert position_baseline(rows)["chance"] == 0.25


def test_options_are_cleaned_and_unusable_ones_dropped():
    rng = random.Random(0)
    s = make_sample("tqa", "1", "img:1", "x.jpg", "How many?", ["6.", "4.", "8."], 2, rng)
    assert set(s["questions"]["q"]["criteria"]) == {"6", "4", "8"} and s["answer_key"] == "8"
    assert make_sample("tqa", "2", "img:1", "x.jpg", "?", ["a", "a."], 0, rng) is None   # duplicate after cleaning
    assert make_sample("tqa", "3", "img:1", "x.jpg", "?", ["a", ""], 0, rng) is None


def test_a_hint_becomes_state_text():
    rng = random.Random(0)
    s = make_sample("scienceqa", "1", "img:1", "x.jpg", "Which is north?", ["x", "y"], 0, rng, state_text=" Look at the map. ")
    assert s["state_text"] == "Look at the map."
    assert "state_text" not in make_sample("scienceqa", "2", "img:1", "x.jpg", "?", ["x", "y"], 0, rng, state_text="")
