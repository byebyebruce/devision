"""v05_ego: the gold is unique, distractors come from the same frame / video and never name the gold's object (or
one a hand is touching), questions a single frame cannot answer are dropped, and participants never cross splits."""
from collections import Counter

from v05_ego import (action_question, choice_sample, egothink_rows, frame_questions, object_name, split_participants,
                     text_baselines)

BOX = [[[100, 100], [400, 100], [400, 400], [100, 400]]]       # 300 px at 1920 wide -> 40 px at 256
TINY = [[[100, 100], [130, 100], [130, 130]]]                   # 30 px -> 4 px at 256


def ann(i, name, cls, contact=None, seg=BOX):
    a = {"id": i, "name": name, "class_id": cls, "segments": seg}
    if contact is not None:
        a["in_contact_object"] = contact
    return a


def frame(anns, name="P01_01_frame_0000000500.jpg"):
    return {"image": {"name": name, "video": "P01_01"}, "annotations": anns}


def seg(i, start, stop, narration, noun_class, noun):
    return {"id": i, "start": start, "stop": stop, "narration": narration, "verb_class": 0, "noun": noun,
            "noun_class": noun_class, "noun_classes": [noun_class], "nouns": [noun]}


KITCHEN = [ann("lh", "left hand", 300, "k"), ann("rh", "right hand", 301, "hand-not-in-contact"),
           ann("k", "knife", 4), ann("b", "chopping board", 18), ann("o", "onion", 60), ann("p", "pan", 5),
           ann("l", "lid:pan", 77), ann("k2", "knife", 4), ann("t", "tap", 0)]
SEGS = [seg("a0", 0, 200, "turn on tap", 0, "tap"), seg("a1", 450, 600, "cut onion", 60, "onion"),
        seg("a2", 700, 800, "wash knife", 4, "knife"), seg("a3", 900, 950, "put pan on hob", 5, "pan"),
        seg("a4", 960, 990, "open fridge", 12, "fridge"), seg("a5", 1000, 1100, "take plate", 2, "plate"),
        seg("a6", 1200, 1300, "cut onion", 60, "onion")]


def golds(row):
    return [o for o, p in row["gold"]["q"]["probabilities"].items() if p == 1.0]


def test_contact_gold_is_the_touched_object_and_distractors_are_other_objects_in_the_frame():
    stats = Counter()
    rows = frame_questions(frame(KITCHEN), None, [], "x.jpg", 1920, 1080, stats)
    contact = [r for r in rows if r["kind"] == "contact"]
    assert len(contact) == 1 and contact[0]["hand"] == "left"
    r = contact[0]
    opts = list(r["questions"]["q"]["criteria"])
    assert golds(r) == ["knife"] and r["answer"] == "knife"
    assert 4 <= len(opts) <= 6 and len(set(opts)) == len(opts)
    assert not {"left hand", "right hand"} & set(opts)              # never a hand
    assert opts.count("knife") == 1                                  # the second knife (same class) is not a distractor
    assert not ({"pan", "pan lid"} <= set(opts))                    # related names never both offered
    assert stats["contact:hand_not_in_contact"] == 1


def test_both_hands_on_one_object_give_one_question_and_tiny_or_unsegmented_contacts_are_dropped():
    both = [ann("lh", "left hand", 300, "p"), ann("rh", "right hand", 301, "p")] + KITCHEN[2:]
    rows = [r for r in frame_questions(frame(both), None, [], "x.jpg", 1920, 1080, Counter()) if r["kind"] == "contact"]
    assert [r["hand"] for r in rows] == ["both"] and rows[0]["answer"] == "pan"
    assert "pan lid" not in rows[0]["questions"]["q"]["criteria"]   # 'pan lid' contains the gold 'pan'
    tiny = [ann("lh", "left hand", 300, "s"), ann("s", "spoon", 1, seg=TINY)] + KITCHEN[2:]
    stats = Counter()
    assert not [r for r in frame_questions(frame(tiny), None, [], "x.jpg", 1920, 1080, stats) if r["kind"] == "contact"]
    assert stats["contact:gold_below_6px"] == 1
    stats = Counter()
    frame_questions(frame([ann("lh", "left hand", 300, "none-of-the-above")] + KITCHEN[2:]), None, [], "x", 1920,
                    1080, stats)
    assert stats["contact:contact_unsegmented"] == 1


def test_touching_noul_false_object_is_touched_by_no_hand():
    for n in range(40):
        name = "P01_01_frame_%010d.jpg" % n
        rows = frame_questions(frame(KITCHEN, name), None, [], "x.jpg", 1920, 1080, Counter())
        t = [r for r in rows if r["kind"] == "touching"]
        assert len(t) == 1 and t[0]["questions"]["q"]["type"] == "noul"
        obj = t[0]["questions"]["q"]["instructions"][len("Is the person touching the "):-1]
        assert (obj == "knife") == (t[0]["answer"] == "true")


def test_action_gold_is_the_segment_at_the_frame_and_distractors_avoid_its_and_the_touched_objects():
    stats = Counter()
    got = action_question(frame(KITCHEN), 500, SEGS, stats)
    assert got is not None
    gold, ds = got
    assert gold["narration"] == "cut onion"
    assert "cut onion" not in ds and len(ds) == len(set(ds)) and 3 <= len(ds) <= 5
    assert "wash knife" not in ds                                   # knife is in the left hand
    assert ds[:2] == sorted(ds[:2], key=lambda d: d not in ("turn on tap", "put pan on hob"))  # visible ones first
    assert set(ds[:2]) == {"turn on tap", "put pan on hob"}
    row = choice_sample("ego:action:x", "action", "What is the person doing right now?", gold["narration"], ds,
                        "x.jpg", "visor:x", {})
    assert golds(row) == ["cut onion"]


def test_action_drops_frames_one_picture_cannot_answer():
    stats = Counter()
    assert action_question(frame(KITCHEN), 300, SEGS, stats) is None              # between actions
    overlap = SEGS + [seg("a7", 480, 520, "open fridge", 12, "fridge")]
    assert action_question(frame(KITCHEN), 500, overlap, stats) is None           # two nouns at once
    assert action_question(frame(KITCHEN), 1050, SEGS, stats) is None             # plate not in the frame
    assert action_question(frame(KITCHEN), 500, SEGS[:3], stats) is None          # too few distractors
    assert stats == Counter({"action:no_segment_at_frame": 1, "action:overlapping_actions": 1,
                             "action:gold_object_not_visible": 1, "action:fewer_than_3_distractors": 1})


def test_participants_never_cross_splits_and_the_split_is_deterministic():
    parts = {"P%02d" % i: 100 + 7 * i for i in range(1, 32)}
    a = split_participants(parts)
    assert a == split_participants(dict(reversed(list(parts.items()))))
    total = sum(parts.values())
    for s in ("test", "dev"):
        share = sum(n for p, n in parts.items() if a[p] == s) / total
        assert 0.10 <= share < 0.25
    assert set(a.values()) == {"train", "dev", "test"}


def test_object_names_and_baselines():
    assert object_name("lid:pan") == "pan lid" and object_name("chopping board") == "chopping board"
    rows = [choice_sample("q%d" % i, "contact", "?", "knife", ["pan", "tap", "onion"], "x", "x", {}) for i in range(8)]
    b = text_baselines(rows, rows)["contact"]
    assert b["train_answer_prior"] == 1.0 and b["chance"] == 0.25    # a constant gold is caught by the prior


def test_egothink_distractors_share_no_content_word_with_the_gold():
    items = [{"id": str(i), "category": "Activity", "question": "What am I doing?", "answer": a, "image": "x.jpg"}
             for i, a in enumerate(["cutting onions", "washing dishes", "cutting bread", "reading a book",
                                    "riding a bike", "playing the guitar with friends at a party tonight"])]
    rows, stats = egothink_rows(items)
    assert stats["long_answer"] == 1
    for r in rows:
        assert len(golds(r)) == 1
        gold_words = set(r["answer"].split()) - {"a", "the"}
        for o in r["questions"]["q"]["criteria"]:
            if o != r["answer"]:
                assert not gold_words & set(o.split())


def test_graspable_filter_keeps_only_often_touched_classes_as_contact_distractors():
    from v05_ego import contact_distractors, graspable_classes
    frames = [frame(KITCHEN)] * 30 + [frame([ann("rh", "right hand", 301, "p"), ann("p", "pan", 5),
                                             ann("o", "onion", 60)])] * 30
    g = graspable_classes(frames)
    assert set(g) == {4, 5}                                          # knife, pan touched; board, onion, tap never
    assert contact_distractors(frame(KITCHEN), KITCHEN[2], g) == ["pan"]
