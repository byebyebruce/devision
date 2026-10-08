"""Converter rules of the round-13 sources (pure functions, no downloads)."""
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from base import problems  # noqa: E402


# ---------------------------------------------------------------- TallyQA complex

def test_tallyqa_simple_rule_matches_the_authors_examples():
    from tallyqa_complex import is_simple
    simple = ["How many dogs are there?", "How many dogs are in the picture?", "How many dogs can be seen?",
              "How many dogs are visible?", "How many dining tables are there?", "How many men in the photo?"]
    complex_ = ["How many people are wearing a red shirt?", "How many people can cross the street?",
                "How many giraffes are standing up?", "How many red dogs are there?",
                "How many chairs are in front of the tv?", "How many dogs are red?", "How many baby lions are there?"]
    assert all(is_simple(q) for q in simple)
    assert not any(is_simple(q) for q in complex_)
    assert is_simple("What is this?") is None


def test_tallyqa_unqualified_count_from_a_simple_question_on_the_same_picture():
    from tallyqa_complex import options, unqualified
    assert unqualified("How many people are wearing hats?", {"people": 5}) == 5
    assert unqualified("How many red dogs are there?", {"dogs": 3}) == 3
    assert unqualified("How many cats are sleeping?", {"dogs": 3}) is None
    opts = options(1, 5, random.Random(0))
    assert opts[:2] == ["1", "5"] and len(set(opts)) == len(opts) >= 3


def test_tallyqa_coco_category():
    from tallyqa_complex import coco_category
    assert coco_category("How many men are wearing hats?") == "person"
    assert coco_category("How many black cats are in the picture?") == "cat"
    assert coco_category("How many orange slices are on the toast?") == "orange"


# ---------------------------------------------------------------- CoGenT

def _clevr(last, answer, filters=True):
    prog = [{"function": "scene"}] + ([{"function": "filter_color"}] if filters else []) + [{"function": last}]
    return {"program": prog, "answer": answer, "question": "What is it", "question_index": 7,
            "question_family_index": 1, "image_filename": "CLEVR_trainA_000001.png", "image_index": 1}


def test_cogent_families_and_samples():
    from cogent import convert, family
    scene = {"objects": [{"color": "red"}, {"color": "blue"}, {"color": "yellow"}]}
    assert family(_clevr("count", "3", filters=False)) is None
    assert family(_clevr("exist", "yes")) is None
    rng = random.Random(0)
    r = convert(_clevr("query_color", "red"), scene, "trainA", rng, "cogent_a")
    opts = list(r["gold"]["q"]["probabilities"])
    assert "red" in opts and {"blue", "yellow"} <= set(opts) and len(opts) == 4
    assert r["group"] == r["image_id"] == "cogent:CLEVR_trainA_000001" and r["questions"]["q"]["instructions"].endswith("?")
    n = convert(_clevr("equal_shape", "no"), scene, "trainA", rng, "cogent_a")
    assert n["gold"]["q"]["probabilities"] == {"false": 1.0, "true": 0.0}
    c = convert(_clevr("count", "0"), scene, "trainA", rng, "cogent_a")
    assert c["answer_key"] == "0" and all(int(o) >= 0 for o in c["gold"]["q"]["probabilities"])
    assert not any(problems(x) for x in (r, n, c))


# ---------------------------------------------------------------- VAW

def _inst(name="dog", pos=("brown",), neg=("white", "tan", "black"), box=(10, 10, 100, 100)):
    return {"object_name": name, "positive_attributes": list(pos), "negative_attributes": list(neg),
            "instance_bbox": list(box), "instance_id": "1"}


def test_vaw_uniqueness_needs_one_named_object_whose_box_agrees():
    from vaw import unique_instance
    g = {"objects": {"a": {"name": "dog", "x": 12, "y": 8, "w": 98, "h": 104}, "b": {"name": "cat", "x": 0, "y": 0, "w": 5, "h": 5}}}
    assert unique_instance(_inst(), g, None) is None
    g2 = {"objects": {**g["objects"], "c": {"name": "dog", "x": 300, "y": 300, "w": 50, "h": 50}}}
    assert "not unique" in unique_instance(_inst(), g2, None)
    far = {"objects": {"a": {"name": "dog", "x": 300, "y": 300, "w": 50, "h": 50}}}
    assert "IoU" in unique_instance(_inst(), far, None)
    assert "COCO" in unique_instance(_inst(), g, Counter({"dog": 2}))
    assert unique_instance(_inst(), None, None)


def test_vaw_distractors_are_explicit_negatives_never_unknown_or_confusable():
    from vaw import questions
    specs = questions(_inst(), {"color": {"black"}, "material": set()}, random.Random(0))
    ch = [s for s in specs if s["type"] == "choice"]
    assert len(ch) == 1 and ch[0]["answer"] == "brown"
    assert set(ch[0]["options"][1:]) <= {"white", "black"}       # tan is confusable with brown
    assert ch[0]["options"][1] == "black"                          # another object has it: first
    no = {s["value"] for s in specs if s["type"] == "noul" and not s["answer"]}
    assert no == {"white", "black"}
    assert {s["value"] for s in specs if s["type"] == "noul" and s["answer"]} == {"brown"}
    # no explicit negative -> no choice and no "no" question (unknown is not negative)
    specs = questions(_inst(neg=()), {"color": set(), "material": set()}, random.Random(0))
    assert all(s["type"] == "noul" and s["answer"] for s in specs)


def test_vaw_names():
    from vaw import text, usable_name
    assert usable_name("dog") and usable_name("pants") and not usable_name("trees") and not usable_name("sky")
    assert text({"type": "choice", "cat": "color"}, "pants") == "What color are the pants?"
    assert text({"type": "noul", "cat": "material", "value": "wooden"}, "table") == "Is the table wooden?"


# ---------------------------------------------------------------- PixMo two kinds

def test_pixmo_labels_skip_empty_and_conflicting_counts():
    from pixmo_two_kind import usable_labels
    pts = lambda n: [{"x": 1.0, "y": 1.0}] * n  # noqa: E731
    rows = [{"label": "red cards", "count": 5, "points": pts(5)}, {"label": "cards", "count": 9, "points": pts(9)},
            {"label": "dogs", "count": 0, "points": []}, {"label": "cups", "count": 4, "points": pts(3)},
            {"label": "hats", "count": 4, "points": pts(4)}, {"label": "hats", "count": 6, "points": pts(6)}]
    d = Counter()
    labels = usable_labels(rows, d)
    assert set(labels) == {"red cards", "cards"}
    assert d["label with an empty count / no points"] == 1 and d["phrase asked twice with different counts"] == 1


def test_pixmo_pairs_prefer_the_qualifier_contrast_and_offer_the_other_count():
    from pixmo_two_kind import options, pairs
    labels = {"red cards": ("red cards", 5), "cards": ("cards", 9), "dogs": ("dogs", 7)}
    pr = pairs(labels, random.Random(0))
    assert {pr[0][0][0], pr[0][1][0]} == {"red cards", "cards"} and len(pr) == 2
    opts = options(5, 9, random.Random(0))
    assert opts[:2] == ["5", "9"] and len(set(opts)) == len(opts)


# ---------------------------------------------------------------- VisMin

def test_vismin_caption_check_and_question():
    from vismin import caption_question, text_ok
    assert text_ok("object", "cat", "dog", "A cat touches a banana.", "A dog touches a banana")
    assert not text_ok("object", "cat", "dog", "A kitten touches a banana.", "A dog touches a banana")
    assert not text_ok("counting", "2 dogs", "3 dogs", "Two dogs.", "two dogs")
    assert text_ok("counting", "2 dogs", "3 dogs", "Two dogs run.", "Three dogs run.")
    assert caption_question("A man with a red jacket.  ") == "Is it true that a man with a red jacket?"


# ---------------------------------------------------------------- benches

def test_countbench_candidates_do_not_depend_on_the_gold():
    """Critic R13-I07: the options are the same for every count, so they cannot give the answer away."""
    from countbench import CANDIDATES, candidates, option_baselines
    assert {tuple(candidates(n)) for n in range(2, 11)} == {tuple(CANDIDATES)}
    rows = [{"answer_key": str(n), "questions": {"q": {"criteria": {o: None for o in candidates(n)}}}}
            for n in range(2, 11) for _ in range(3)]
    b = option_baselines(rows)
    assert b["median_option"] < 0.2 and b["uniform_guess"] == 0.1


def test_whatsup_prepositions():
    from whatsup import OPPOSITE, preposition
    assert preposition("mug_right_of_knife.jpeg") == "right_of"
    assert preposition("cap_in-front_of_spoon.jpeg") == "in-front_of"
    assert preposition("beer-bottle_on_armchair.jpeg") == "on"
    assert OPPOSITE["left_of"] == OPPOSITE["right_of"]


def test_sugarcrepe_item_shuffles_and_keeps_gold():
    from sugarcrepe import item
    golds = set()
    for k in range(20):
        r = item("swap_att", str(k), {"filename": "000000000139.jpg", "caption": "a red car", "negative_caption":
                                      "a car red"}, "val2014", 139, random.Random(k))
        assert r["answer_key"] == "a red car" and not problems(r)
        golds.add(list(r["gold"]["q"]["probabilities"]).index("a red car"))
    assert golds == {0, 1}
