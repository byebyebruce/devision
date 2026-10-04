"""v2 VQAv2 multiple choice: every option is a sensible answer to the question, the right one is reliable."""
import random
from collections import Counter

from v2_vqa_choice import category_of, convert, finish, listed_alternatives, problems, select, vocabularies

VOCABS = vocabularies()


def vqa(text, answers, answer_type="other", qid=1, image=7):
    return ({"question": text, "question_id": qid, "image_id": image},
            {"answer_type": answer_type, "answers": [{"answer": a} for a in answers]})


def build(items, seed=0):
    """`items`: (text, answers) pairs, each on its own image; returns finished samples."""
    rng = random.Random(seed)
    rows = [convert(*vqa(t, a, "number" if t.lower().startswith("how many") else "other", qid=i, image=i),
                    "train2014", VOCABS, rng) for i, (t, a) in enumerate(items)]
    return finish(select([r for r in rows if r], 10000, set(), rng), VOCABS, rng)


def options(s):
    return list(s["questions"]["q"]["criteria"])


def gold(s):
    g = s["gold"]["q"]["probabilities"]
    return max(g, key=g.get)


def test_the_question_not_the_answer_decides_the_category():
    assert category_of("What type of fruit is this?") == "fruit"     # answer "orange" is also a colour
    assert category_of("What type of meat is in the picture?") == "meat"  # "chicken" is also an animal
    assert category_of("What color is the fruit?") == "color"
    assert category_of("What is he holding?") is None


def test_a_fruit_question_offers_only_fruits():
    rows = build([("What type of fruit is this?", ["orange"] * 9 + ["tangerine"])] +
                 [("What fruit is on the plate?", [f] * 10) for f in ["banana", "apple", "pear", "lemon"] * 5])
    for s in rows:
        assert set(options(s)) <= set(VOCABS["fruit"]) and problems(s, VOCABS) == []


def test_a_meat_question_offers_only_meats():
    rows = build([("What type of meat is in the picture?", ["chicken"] * 10)] +
                 [("What meat is this?", [m] * 10) for m in ["beef", "ham", "fish", "turkey"] * 5])
    s = next(r for r in rows if gold(r) == "chicken")
    assert set(options(s)) <= set(VOCABS["meat"]) and "elephant" not in options(s)


def test_named_alternatives_are_exactly_the_options():
    rows = build([("Are the bananas yellow or green?", ["yellow"] * 10),
                  ("Is the man sitting or standing?", ["standing"] * 9 + ["walking"])])
    by_q = {s["questions"]["q"]["instructions"]: options(s) for s in rows}
    assert set(by_q["Are the bananas yellow or green?"]) == {"yellow", "green"}
    assert set(by_q["Is the man sitting or standing?"]) == {"sitting", "standing"}


def test_named_alternatives_without_the_answer_or_with_synonyms_are_skipped():
    assert not build([("Is the shirt red or blue?", ["white"] * 10)])
    assert not build([("Is it cloudy or overcast?", ["cloudy"] * 10)])


def test_distractors_are_never_given_by_an_annotator_nor_synonyms():
    rows = build([("What color is the bus?", ["red"] * 8 + ["orange", "maroon"])] +
                 [("What color is the car?", [c] * 10) for c in ["orange", "white", "blue", "gray", "green"] * 4] +
                 [("What color is the cat?", ["gray"] * 10)])
    for s in rows:
        if gold(s) == "red":
            assert "orange" not in options(s)
        if gold(s) == "gray":
            assert not {"grey", "silver"} & set(options(s))


def test_counts_get_nearby_numbers():
    rows = build([("How many dogs are there?", ["3"] * 9 + ["4"])] +
                 [("How many cats are there?", [str(n)] * 10) for n in range(7)] * 3)
    s = next(r for r in rows if r["questions"]["q"]["instructions"] == "How many dogs are there?")
    assert "4" not in options(s) and max(abs(int(o) - 3) for o in options(s)) <= 3


def test_unreliable_unknown_and_reading_questions_are_skipped():
    rng = random.Random(0)
    assert convert(*vqa("What color is the bus?", ["red"] * 5 + ["blue"] * 5), "train2014", VOCABS, rng) is None
    assert convert(*vqa("What is he holding?", ["umbrella"] * 10), "train2014", VOCABS, rng) is None
    assert convert(*vqa("What does the sign say?", ["stop"] * 10), "train2014", VOCABS, rng) is None
    assert convert(*vqa("What fruit is this?", ["pizza"] * 10), "train2014", VOCABS, rng) is None


def test_a_recurring_wording_does_not_point_at_one_answer():
    items = [("What color is the sky?", ["blue"] * 10)] * 30 + [("What color is the sky?", ["gray"] * 10)] * 5
    count = Counter(gold(s) for s in build(items) if s["questions"]["q"]["instructions"] == "What color is the sky?")
    assert count["blue"] <= count["gray"]


def test_no_answer_dominates_its_category():
    items = [("What color is it?", ["white"] * 10)] * 100 + \
            [("What color is it?", [c] * 10) for c in ["red", "blue", "green", "black"] for _ in range(10)]
    count = Counter(gold(s) for s in build(items))
    assert count["white"] <= 10


def test_problems_catches_the_reviewed_failures():
    fruit = {"kind": "color", "questions": {"q": {"type": "choice", "instructions": "What type of fruit is this?",
                                                  "criteria": {"yellow": None, "orange": None, "white": None}}},
             "gold": {"q": {"probabilities": {"yellow": 0.0, "orange": 1.0, "white": 0.0}}}}
    bananas = {"kind": "color", "questions": {"q": {"type": "choice", "instructions": "Are the bananas yellow or green?",
                                                    "criteria": {"yellow": None, "white": None, "brown": None}}},
               "gold": {"q": {"probabilities": {"yellow": 1.0, "white": 0.0, "brown": 0.0}}}}
    assert problems(fruit, VOCABS) and problems(bananas, VOCABS)


def test_listed_alternatives_must_all_be_answers_of_the_category():
    colors, sports, vehicles = VOCABS["color"], VOCABS["sport"], VOCABS["vehicle"]
    assert listed_alternatives("Are the bananas yellow or green?", colors) == ["yellow", "green"]
    assert listed_alternatives("Is this a red bus or blue bus?", colors) == ["red", "blue"]
    assert listed_alternatives("Is this a car, a bus or a truck?", vehicles) == ["car", "bus", "truck"]
    assert listed_alternatives("Are the weather conditions dry, snowy, windy, or rainy?", VOCABS["weather"]) is None
    assert listed_alternatives("Is the dog looking at a tennis ball or frisbee?", sports) is None


def test_a_question_about_what_an_animal_does_is_skipped():
    assert not build([("What is this animal eating?", ["horse"] * 10)])
    kept = build([("Which animal is eating?", [a] * 10) for a in ["horse", "cow", "sheep", "dog", "cat", "zebra"]])
    assert kept and all(set(options(s)) <= set(VOCABS["animal"]) for s in kept)
