import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
from pixmo_points import options, phrase, picture_questions, sample_pictures  # noqa: E402


def test_phrases_become_plural_noun_phrases():
    assert phrase("gold coins") == "gold coins"
    assert phrase("People wearing glasses") == "people wearing glasses"
    assert phrase("purple grape") == "purple grapes"
    assert phrase("Person") == "people"
    assert phrase("strawberry") == "strawberries"
    assert phrase("TV") == "TVs"
    assert phrase("Kid's Menu") == "kid's menus"
    assert phrase("the word code underlined") is None      # a singular clause
    assert phrase('".com"') is None
    assert phrase("food") is None                          # a mass noun
    assert phrase("Books, top row") is None
    assert phrase("14 features") is None                   # the number would give the count away


def test_count_options_stay_where_counts_occur():
    for n in range(4, 11):
        for seed in range(20):
            opts = options(n, random.Random(seed))
            assert opts[0] == str(n) and 2 <= len(opts) <= 5
            assert all(4 <= int(o) <= 10 and abs(int(o) - n) <= 3 for o in opts)


def test_picture_questions_one_per_phrase_with_answers_from_counts():
    rows = [{"label": "dogs", "count": 0}, {"label": "Dogs", "count": 5}, {"label": "cats", "count": 0},
            {"label": "people", "count": 30}, {"label": "chairs", "count": 6}]
    out = picture_questions("k", "ext/pixmo_points/images/k.jpg", rows, 5, random.Random(0))
    asked = {r["questions"]["q"]["instructions"]: r["answer_key"] for r in out}
    assert not any("dogs" in q for q in asked)        # contradictory counts for one phrase: dropped
    assert asked["Are there any cats in the image?"] == "false"
    assert asked["Are there any people in the image?"] == "true"
    chairs = [a for q, a in asked.items() if "chairs" in q]
    assert chairs in (["6"], ["true"])
    assert len({r["id"] for r in out}) == len(out) and all(r["image_id"] == "pixmo_points:k" for r in out)
    assert len(picture_questions("k", "x.jpg", rows, 2, random.Random(0))) == 2


def test_picture_sample_is_deterministic():
    urls = ["https://a/%d.jpg" % i for i in range(100)]
    assert sample_pictures(urls, 10) == sample_pictures(list(reversed(urls)), 10)
    assert len(set(sample_pictures(urls, 10))) == 10
