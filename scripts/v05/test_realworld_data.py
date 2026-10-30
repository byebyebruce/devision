from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from realworld_data import convert


def test_native_choices_keep_gold_mapping_and_strip_only_response_instruction():
    q, p = convert("Which is closer? A. A bus B. A sedan C. Equal Please answer directly with only the letter.", "B")
    assert q["instructions"] == "Which is closer?"
    assert list(q["criteria"]) == ["A bus", "A sedan", "Equal"]
    assert p == {"A bus": 0.0, "A sedan": 1.0, "Equal": 0.0}


def test_free_answers_and_broken_choice_lists_are_not_fabricated():
    assert convert("What color?", "Green") is None
    assert convert("Count?", "2") is None
    assert convert("Which? A. red C. blue", "A") is None
    assert convert("Which? A. red B. RED", "A") is None
    q, p = convert("Is the light green? Please answer directly with a single word or number.", "Yes")
    assert q["type"] == "noul" and p == {"false": 0.0, "true": 1.0}
