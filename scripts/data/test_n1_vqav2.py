import random

from n1_vqav2 import clash, convert, finish, flatten, text_baseline


def qa(qid, text, answers, answer_type="other", iid=7):
    return ({"question_id": qid, "image_id": iid, "question": text},
            {"question_id": qid, "answer_type": answer_type, "answers": [{"answer": a} for a in answers]})


def build(pairs, seed=0):
    rng = random.Random(seed)
    return finish([s for s in (convert(q, a, "train2014", rng) for q, a in pairs) if s], rng)


def test_yes_no_gold_is_the_share_of_yes_and_too_few_yes_no_answers_drop_it():
    (s,) = build([qa(1, "Is the dog asleep?", ["yes"] * 7 + ["no"] * 3, "yes/no")])
    assert s["questions"]["q"]["type"] == "noul"
    assert s["gold"]["q"]["probabilities"] == {"false": 0.3, "true": 0.7}
    assert build([qa(2, "Is the dog asleep?", ["yes"] * 5 + ["maybe"] * 5, "yes/no")]) == []


def test_count_offers_every_number_from_0_to_10_with_the_answers_shares():
    (s,) = build([qa(3, "How many dogs are there?", ["2"] * 6 + ["two"] * 2 + ["3"] * 2, "number")])
    opts = s["questions"]["q"]["criteria"]
    assert sorted(opts, key=int) == [str(i) for i in range(11)]
    p = s["gold"]["q"]["probabilities"]
    assert p["2"] == 0.8 and p["3"] == 0.2 and sum(p.values()) == 1.0
    assert build([qa(4, "How many dogs are there?", ["12"] * 10, "number")]) == []


def test_distractors_are_never_an_annotators_answer_or_its_synonym():
    pairs = [qa(10, "What color is the bus?", ["red"] * 6 + ["maroon"] * 3 + ["gray"])]
    pairs += [qa(20 + i, "What color is the car?", [c] * 10, iid=100 + i)
              for i, c in enumerate(["blue", "green", "silver", "white", "black", "yellow"])]
    for seed in range(20):
        s = next(x for x in build(pairs, seed) if x["id"] == "n1-vqa:10")
        opts = list(s["questions"]["q"]["criteria"])
        assert "red" in opts and "maroon" in opts and 3 <= len(opts) <= 5
        assert "gray" not in opts and "silver" not in opts and "grey" not in opts
        p = s["gold"]["q"]["probabilities"]
        assert p["red"] == 6 / 9 and p["maroon"] == 3 / 9


def test_reading_questions_and_answers_without_consensus_are_dropped():
    assert build([qa(30, "What does the sign say?", ["stop"] * 10)]) == []
    assert build([qa(31, "What is the man holding?", ["bat", "stick", "club", "pole"] * 2 + ["a", "b"])]) == []


def test_a_recurring_wording_does_not_point_at_one_answer():
    pairs = [qa(40 + i, "Is it sunny?", ["yes"] * 10, "yes/no", iid=200 + i) for i in range(9)]
    pairs += [qa(60, "Is it sunny?", ["no"] * 10, "yes/no", iid=300)]
    rng = random.Random(0)
    kept = flatten([convert(q, a, "train2014", rng) for q, a in pairs], rng)
    tops = [max(s["gold"]["q"]["probabilities"], key=s["gold"]["q"]["probabilities"].get) for s in kept]
    assert tops.count("true") == tops.count("false") == 1


def test_text_baseline_answers_from_the_wording_seen_in_train():
    train = build([qa(70 + i, "Is it sunny?", ["yes"] * 10, "yes/no", iid=i) for i in range(3)])
    dev = build([qa(80, "Is it sunny?", ["yes"] * 10, "yes/no"), qa(81, "Is it sunny?", ["no"] * 10, "yes/no")])
    assert text_baseline(train, dev) == {"yesno": 0.5}


def test_singular_and_plural_of_a_word_clash():
    assert clash("kite", "kites") and clash("shoes", "shoe") and clash("boxes", "box") and clash("red bus", "bus")
    assert not clash("glass", "grass") and not clash("bus", "car")
