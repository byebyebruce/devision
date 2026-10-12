"""Converters of the public-data pack: what each keeps, and the gold it gives (no downloads)."""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
import abo  # noqa: E402
import cosyn  # noqa: E402
import onejev  # noqa: E402
import vizwiz  # noqa: E402


def test_cosyn_keeps_closed_answers_only():
    qa = {"question": ["Is the arrow red?", "How many nodes are there?", "Which node feeds the model?",
                       "What is shown?", "How many nodes are blue?", "Explain the process."],
          "answer": ["Yes", "Four", "Dataset", "Sawmill operations", "12", "It starts with raw data and then ..."]}
    out, steps = cosyn.figure_questions("k", "ext/cosyn/images/k.png", "diagram", qa, random.Random(0))
    by = {r["questions"]["q"]["instructions"]: r for r in out}
    assert by["Is the arrow red?"]["answer_key"] == "true"
    count = by["How many nodes are there?"]
    assert count["answer_key"] == "4" and len(count["questions"]["q"]["criteria"]) == 11
    label = by["Which node feeds the model?"]["questions"]["q"]["criteria"]
    assert "Dataset" in label and "Sawmill operations" in label        # the other label of the same figure
    assert "How many nodes are blue?" not in by and "Explain the process." not in by     # 12 > 10; long answer


def test_cosyn_short_answer_needs_another_label():
    out, steps = cosyn.figure_questions("k", "x.png", "math", {"question": ["What is the shape?"], "answer": ["Circle"]},
                                        random.Random(0))
    assert out == [] and steps["short answer without another label in the figure"] == 1


def test_abo_colour_must_be_one_basic_colour():
    assert abo.basic_colour(["Black"]) == "black"
    assert abo.basic_colour(["Gray"]) == "grey"
    assert abo.basic_colour(["Navy Blue"]) is None and abo.basic_colour(["Black/Red"]) is None
    assert abo.basic_colour(["Multicolor"]) is None
    assert abo.english([{"language_tag": "nl_NL", "value": "Zwart"}, {"language_tag": "en_US", "value": "Black"}]) == ["Black"]


def ann(question, answers, kind, answerable=1):
    return {"image": "VizWiz_train_1.jpg", "question": question, "answer_type": kind, "answerable": answerable,
            "answers": [{"answer": x, "answer_confidence": "yes"} for x in answers]}


def test_vizwiz_soft_gold_from_votes():
    r, _ = vizwiz.convert(ann("Is it on?", ["yes"] * 7 + ["no"] * 2 + ["unsure"], "yes/no"), "train", "x.jpg")
    assert abs(r["gold"]["q"]["probabilities"]["true"] - 7 / 9) < 1e-9
    r, _ = vizwiz.convert(ann("How many pills?", ["2", "two", "3", "2", "pills", "2"] + ["x"] * 4, "number"), "train", "x.jpg")
    g = r["gold"]["q"]["probabilities"]
    assert len(g) == 11 and abs(g["2"] - 0.8) < 1e-9 and abs(g["3"] - 0.2) < 1e-9
    assert vizwiz.convert(ann("What is it?", ["x"] * 10, "unanswerable", 0), "train", "x.jpg")[0] is None
    assert vizwiz.convert(ann("What brand?", ["basil"] * 10, "other"), "train", "x.jpg")[1] == "open answer"


def row(media, question, target, state='{"image": "<image:1>"}'):
    return {"id": "s/1", "source": "flowers", "task": "species", "state": state, "question": json.dumps(question),
            "target": json.dumps(target), "media": json.dumps(media), "meta": '{"group": "g1"}',
            "images": [{"bytes": b"\xff\xd8x"}]}


PIC = [{"type": "image", "path": "media/image/flowers/a.jpg"}]


def test_onejev_static_picture_choice_and_noul():
    r, _ = onejev.convert(row(PIC, {"type": "choice", "instructions": "Which flower?",
                                    "criteria": {"rose": "Rose", "sweet_pea": "Sweet pea"}}, {"sweet_pea": 1.0}))
    assert r["questions"]["q"]["criteria"] == {"Rose": None, "Sweet pea": None}
    assert r["gold"]["q"]["probabilities"] == {"Rose": 0.0, "Sweet pea": 1.0}
    r, _ = onejev.convert(row(PIC, {"type": "noul", "instructions": "Is it red?"}, {"yes": 0.75, "no": 0.25}))
    assert r["gold"]["q"]["probabilities"] == {"false": 0.25, "true": 0.75}


def test_onejev_drops_what_does_not_fit():
    q = {"type": "noul", "instructions": "x?"}
    assert onejev.convert(row([{"type": "image", "path": "media/gui/a.jpg"}], q, {"yes": 1}))[0] is None
    assert onejev.convert(row(PIC * 2, q, {"yes": 1}))[0] is None
    assert onejev.convert(row(PIC, q, {"yes": 1}, state='{"image": "<image:1>", "prompt": "a cat"}'))[0] is None
    assert onejev.convert(row(PIC, {"type": "score", "instructions": "x", "criteria": ["a", "b"]}, {"1": 1}))[0] is None


def test_yes_no_are_balanced():
    import held
    rows = [{"questions": {"q": {"type": "noul"}}, "answer_key": "true"}] * 8 + \
           [{"questions": {"q": {"type": "noul"}}, "answer_key": "false"}] * 3 + \
           [{"questions": {"q": {"type": "choice"}}, "answer_key": "x"}] * 5
    rows = [dict(r) for r in rows]
    out = held.balance_noul(rows, random.Random(0))
    keys = [r["answer_key"] for r in out if r["questions"]["q"]["type"] == "noul"]
    assert keys.count("true") == keys.count("false") == 3 and sum(r["answer_key"] == "x" for r in out) == 5


def test_cosyn_drops_empty_questions():
    out, steps = cosyn.figure_questions("k", "x.png", "diagram", {"question": ["  ", "Is it red?"], "answer": ["Yes", "No"]},
                                        random.Random(0))
    assert len(out) == 1 and steps["empty question"] == 1
