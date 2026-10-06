import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
import visonlyqa as v  # noqa: E402

FORMAT = "Your response should only include the final answer (True, False). Do not include any reasoning."


def row(prompt, answer, options, task):
    return {"image_path": "images/SyntheticGeometry/syntheticgeometry_15131noise_1.jpeg", "answer": answer,
            "prompt_no_reasoning": prompt, "response_options": options, "id": "x1", "task_category": task}


def test_negated_triangle_becomes_noul_with_definition_and_own_group():
    p = "There is no triangle ADP in the figure. True or False?\n\nA triangle is a polygon.\n\n" + FORMAT
    s, why = v.convert(row(p, "False", ["True", "False"], "triangle"), "img.png", random.Random(0))
    assert why is None and s is not None
    assert s["questions"]["q"] == {"type": "noul", "instructions": "Is there no triangle ADP in the figure? A triangle is a polygon."}
    assert s["answer_key"] == "false" and s["template"] == "triangle-no"
    assert s["group"] == "is there no triangle adp in the figure? a triangle is a polygon."   # balanced per wording
    assert s["image_id"] == "visonlyqa:syntheticgeometry_15131noise_1"


def test_lettered_options_become_choice_over_option_texts():
    p = ("Line AB is X times longer than AE. Which of the following options is a reasonable estimate of X? You only "
         "need to estimate from the visual information. (a) 2 (b) 0.5 (c) 1 (d) 0.25 (e) 4 \n\n" + FORMAT)
    s, _ = v.convert(row(p, "d", list("abcde"), "length"), "img.png", random.Random(0))
    assert s is not None
    q = s["questions"]["q"]
    assert q["instructions"] == "Line AB is X times longer than AE. Which is a reasonable estimate of X?"
    assert sorted(q["criteria"]) == ["0.25", "0.5", "1", "2", "4"]
    assert s["answer_key"] == "0.25" and s["gold"]["q"]["probabilities"]["0.25"] == 1.0


def test_area_question_names_the_unknown_and_degrees_kept():
    p = "CAK is X times larger in area than ABD. Which of the following options is a reasonable estimate? (a) 2 (b) 1\n"
    assert v.lettered(p, "b") == ("CAK is X times larger in area than ABD. Which is a reasonable estimate of X?",
                                  ["2", "1"], "1")
    p = "Which of the following options is a reasonable estimate of the angle DFC in the figure? (a) 45 degrees (b) 10 degrees"
    mc = v.lettered(p, "b")
    assert mc is not None and mc[1:] == (["45 degrees", "10 degrees"], "10 degrees")


def test_unparsable_rows_are_dropped_with_a_reason():
    s, why = v.convert(row("Something else. True or False?", "True", ["True", "False"], "triangle"), "i", random.Random(0))
    assert s is None and why == "unparsed true / false"


def sample(sid, group, answer):
    return {"id": sid, "group": group, "answer_key": answer, "questions": {"q": {"type": "noul"}}}


def test_balance_is_per_wording_and_limit_keeps_whole_wordings():
    rows = [sample("1", "a", "true"), sample("2", "a", "true"), sample("3", "a", "false"),
            sample("4", "b", "true")]                       # "b" has one answer only: dropped
    kept = v.balance(rows)
    assert sorted(r["id"] for r in kept) == ["1", "3"]
    many = [sample(str(i), "g%d" % (i // 2), "true" if i % 2 else "false") for i in range(10)]
    out = v.first_wordings(many, 5, 0)
    assert len(out) == 4 and all(sum(r["group"] == g for r in out) == 2 for g in {r["group"] for r in out})
