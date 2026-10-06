import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
import snli_ve  # noqa: E402


def test_negation_cues_found_in_contractions_and_words():
    assert snli_ve.has_negation("The man isn't running.")
    assert snli_ve.has_negation("Nobody is outside.")
    assert snli_ve.has_negation("A dog is sleeping on the couch.")
    assert not snli_ve.has_negation("Two people are in a store.")
    assert not snli_ve.has_negation("A woman knows the answer.")   # "no" only as a whole word


def test_question_quotes_the_hypothesis():
    assert snli_ve.question("Two humans  in a store.") == 'Is this true of the picture? "Two humans in a store."'


def test_text_baseline_finds_a_giveaway_word_and_debias_removes_it():
    rng = random.Random(0)
    texts, labels = [], []
    for i in range(4000):
        yes = rng.random() < 0.5
        cue = "outdoors" if (yes if rng.random() < 0.9 else not yes) else "inside"
        texts.append("a person %s %s" % (cue, rng.choice(["walks", "runs", "stands", "sits"])))
        labels.append(yes)
    assert snli_ve.text_baseline(texts, labels)["accuracy"] > 0.85
    keep = snli_ve.debias(texts, labels, seed=0)
    kept_t, kept_l = [texts[i] for i in keep], [labels[i] for i in keep]
    assert sum(kept_l) * 2 == len(kept_l)
    assert abs(snli_ve.text_baseline(kept_t, kept_l)["accuracy"] - 0.5) < 0.05


def test_balance_gives_equal_yes_and_no_within_the_limit():
    rows = [{"answer_key": "true"}] * 30 + [{"answer_key": "false"}] * 10
    out = snli_ve.balance(rows, 100, random.Random(0))
    assert len(out) == 20 and sum(r["answer_key"] == "true" for r in out) == 10
    assert len(snli_ve.balance(rows, 8, random.Random(0))) == 8
