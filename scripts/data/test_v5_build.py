"""data-v5 conversion and splitting: VSR statements, Visual7W options, pictures kept whole, balance, baselines."""
import random
from collections import Counter

from v5_build import (balance_answers, cap_per_picture, convert_v7w, convert_vsr, equalize_by_group,
                      majority_baseline, meta_categories, option_baselines, take_by_picture, vsr_question)


def test_vsr_statements_become_yes_no_questions():
    assert vsr_question("The cat is inside the refrigerator.") == "Is the cat inside the refrigerator?"
    assert vsr_question("The dogs are next to the bench.") == "Are the dogs next to the bench?"
    assert vsr_question("The suitcase contains the cat.") == "Does the suitcase contain the cat?"
    assert vsr_question("The bed has as a part the couch.") == "Does the bed have as a part the couch?"
    assert vsr_question("The cake consists of the banana.") == "Does the cake consist of the banana?"
    assert vsr_question("A cat sleeps.") is None


def test_a_vsr_record_keeps_its_label_relation_and_picture():
    meta = meta_categories("Projective: on top of, left of\nTopological: on, contains")
    rec = {"image": "000000296471.jpg", "caption": "The cat is left of the dog.", "label": 0, "relation": "left of"}
    s = convert_vsr(rec, "train-0", meta, train2014={296471})
    assert s is not None
    assert s["questions"]["q"] == {"type": "noul", "instructions": "Is the cat left of the dog?"}
    assert s["gold"]["q"]["probabilities"] == {"true": 0.0, "false": 1.0}
    assert (s["image_id"], s["image"]) == ("coco:296471", "coco/train2014/COCO_train2014_000000296471.jpg")
    assert (s["meta_category"], s["axis"]) == ("Projective", "lr")
    other = convert_vsr(dict(rec, image="000000000042.jpg", relation="on", label=1), "dev-1", meta, train2014=set())
    assert other is not None and other["image"].startswith("coco/val2014/") and other["axis"] == ""


def test_visual7w_answer_is_one_of_four_shuffled_options():
    qa = {"qa_id": 7, "question": "Where is he sitting?", "answer": "On a bench.",
          "multiple_choices": ["At a park.", "On the grass.", "At a dining table."], "type": "where"}
    rng = random.Random(0)
    rows = [convert_v7w(dict(qa, qa_id=i), "vg:1", "gqa/images/1.jpg", rng) for i in range(200)]
    for s in rows:
        assert s is not None
        g = s["gold"]["q"]["probabilities"]
        assert max(g, key=g.get) == "On a bench" and len(g) == 4 and s["category"] == "where"
    assert min(Counter(s["answer_position"] for s in rows if s).values()) > 30   # about 50 each
    assert convert_v7w(dict(qa, multiple_choices=["on a bench", "x", "y"]), "vg:1", "i.jpg", rng) is None


def _rows(pictures, per=2):
    return [{"id": "%s-%d" % (p, i), "image_id": p, "answer_key": "true" if i % 2 else "false", "group": "on",
             "category": "on"} for p in pictures for i in range(per)]


def test_pictures_are_taken_whole_and_reproducibly():
    rows = _rows(["coco:%d" % i for i in range(50)], per=3)
    key = lambda r: r["image_id"]  # noqa: E731
    a = take_by_picture(rows, 10, key, "test")
    assert a == take_by_picture(rows, 10, key, "test") and len(a) == 12           # 4 whole pictures
    assert all(Counter(r["image_id"] for r in a)[p] == 3 for p in {r["image_id"] for r in a})
    assert take_by_picture(rows, 10, key, "dev") != a


def test_balancing_drops_never_copies_or_relabels():
    rng = random.Random(0)
    rows = [{"id": str(i), "answer_key": "true" if i < 30 else "false", "group": "on" if i % 3 else "near"}
            for i in range(40)]
    out = balance_answers(rows, rng)
    assert Counter(r["answer_key"] for r in out) == {"true": 10, "false": 10}
    assert len({r["id"] for r in out}) == len(out)
    eq = equalize_by_group(rows, rng)
    by = Counter((r["group"], r["answer_key"]) for r in eq)
    assert by[("on", "true")] == by[("on", "false")] and by[("near", "true")] == by[("near", "false")]
    capped = cap_per_picture(_rows(["a", "b"], per=5), 3, lambda r: r["image_id"], rng)
    assert Counter(r["image_id"] for r in capped) == {"a": 3, "b": 3}


def test_text_only_baselines_are_fitted_on_training_rows():
    train = [{"group": "on", "answer_key": "true"}] * 3 + [{"group": "under", "answer_key": "false"}] * 3
    dev = [{"group": "on", "answer_key": "true"}, {"group": "under", "answer_key": "true"}]
    assert majority_baseline(train, dev, "group") == {"n": 2, "accuracy": 0.5}

    def q(gold, opts):
        return {"category": "what", "answer_key": gold, "gold": {"q": {"probabilities": {o: float(o == gold) for o in opts}}}}
    b = option_baselines([q("red", ["red", "a", "b", "c"])] * 5, [q("red", ["red", "a long one", "bb", "c"])])
    assert b["train_answer_frequency"] == 1.0 and b["longest"] == 0.0
