"""Infographic questions: distractors are other answers about the same image of the same kind, never contain the
gold, at most one numeric filler, and the gold letter rotates."""
from collections import Counter

from v05_board_infovqa import clean_a, clean_q, image_questions

PAIRS = [("How many apps?\nKeep it brief.", "12."), ("How many users?\nAnswer briefly.", "40."),
         ("How many stores?\nBe succinct.", "7."), ("What share uses mobile?\nKeep it brief.", "35%."),
         ("Which app is first?\nKeep it brief.", "Facebook."), ("Which app is second?\nKeep it brief.", "Twitter."),
         ("Which app is third?\nKeep it brief.", "Instagram."), ("Which network is newest?\nKeep it brief.", "TikTok.")]


def test_cleaning():
    assert clean_q("How many apps?\nWrite a very short answer.") == "How many apps?"
    assert clean_a("Road Rider Cruiser.") == "road rider cruiser" and clean_a("2.5") == "2.5"


def test_distractors_are_same_image_same_kind():
    qs = image_questions("img", PAIRS, Counter(), Counter())
    by_q = {q: (opts, gold) for _, q, opts, gold in qs}
    opts, gold = by_q["Which app is first?"]
    assert gold == "facebook" and set(opts) == {"facebook", "twitter", "instagram", "tiktok"}
    opts, gold = by_q["How many apps?"]
    assert gold == "12" and opts.count("12") == 1 and "35%" not in opts      # % answers only against % answers
    assert all(not any(c.isalpha() for c in o) for o in opts)


def test_at_most_one_filler_and_unanswerable_dropped():
    stats = Counter()
    qs = image_questions("img", PAIRS, Counter(), stats)
    assert "What share uses mobile?" not in {q for _, q, _, _ in qs}          # no other % answer: dropped
    assert stats["few_same_image_answers"] >= 1
    two = [("How many apps?", "12"), ("How many users?", "40"), ("How many stores?", "7")]
    got = image_questions("img2", two, Counter(), stats)
    assert len(got) == 3 and all(len(o) == 4 for _, _, o, _ in got)            # 2 same-image + 1 filler


def test_gold_letter_rotates():
    slots = Counter()
    for k in range(40):
        for _, _, opts, gold in image_questions("img%d" % k, PAIRS, slots, Counter()):
            pass
    assert max(slots.values()) - min(slots.values()) <= 1
