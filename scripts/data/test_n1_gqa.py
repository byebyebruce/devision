import random
from collections import Counter

from n1_gqa import answer_pools, convert

GRAPH = {"width": 100, "height": 100, "objects": {
    "1": {"name": "wall", "w": 50, "h": 50, "attributes": ["white"]},
    "2": {"name": "pipe", "w": 20, "h": 20, "attributes": ["white", "metal"]},
    "3": {"name": "cup", "w": 20, "h": 20, "attributes": ["red"]},
    "4": {"name": "speck", "w": 1, "h": 1, "attributes": []}}}


def rec(qid, question, answer, semantic, structural="query"):
    return {"id": qid, "imageId": "9", "question": question, "answer": answer,
            "types": {"structural": structural}, "semantic": semantic}


ON_WALL = [{"operation": "select", "argument": "wall (1)", "dependencies": []},
           {"operation": "relate", "argument": "_,on,s (2)", "dependencies": [0]},
           {"operation": "query", "argument": "name", "dependencies": [1]}]
PIPE_COLOR = [{"operation": "select", "argument": "pipe (2)", "dependencies": []},
              {"operation": "query", "argument": "color", "dependencies": [0]}]


def pools_with(names, colors):
    recs = [rec(str(i), "q", n, ON_WALL) for i, n in enumerate(names)]
    recs += [rec(str(100 + i), "q", c, PIPE_COLOR) for i, c in enumerate(colors)]
    return answer_pools(recs)


def test_a_name_distractor_is_never_an_object_of_the_picture():
    pools = pools_with(["pipe", "cup", "wall", "clock", "sign", "lamp", "shelf"], [])
    for seed in range(30):
        s = convert("q1", rec("q1", "What is on the wall?", "pipe", ON_WALL), GRAPH, pools, random.Random(seed))
        opts = list(s["questions"]["q"]["criteria"])
        assert "pipe" in opts and not {"cup", "wall"} & set(opts) and 3 <= len(opts) <= 5
        assert s["gold"]["q"]["probabilities"]["pipe"] == 1.0


def test_an_attribute_distractor_is_never_an_attribute_of_the_queried_object():
    pools = pools_with([], ["white", "metal", "red", "blue", "green", "black"])
    for seed in range(30):
        s = convert("q2", rec("q2", "What color is the pipe?", "white", PIPE_COLOR), GRAPH, pools, random.Random(seed))
        assert "metal" not in s["questions"]["q"]["criteria"]


def test_no_for_an_existence_question_is_dropped_but_yes_is_kept():
    exist = [{"operation": "select", "argument": "cup (3)", "dependencies": []},
             {"operation": "exist", "argument": "?", "dependencies": [0]}]
    pools = Counter()
    assert convert("q3", rec("q3", "Is there a cup?", "no", exist, "verify"), GRAPH, pools, random.Random(0)) is None
    s = convert("q4", rec("q4", "Is there a cup?", "yes", exist, "verify"), GRAPH, pools, random.Random(0))
    assert s["gold"]["q"]["probabilities"] == {"false": 0.0, "true": 1.0}


def test_a_question_about_a_tiny_object_is_dropped():
    tiny = [{"operation": "select", "argument": "speck (4)", "dependencies": []},
            {"operation": "query", "argument": "color", "dependencies": [0]}]
    pools = pools_with([], ["white", "red", "blue", "green"])
    assert convert("q5", rec("q5", "What color is the speck?", "red", tiny), GRAPH, pools, random.Random(0)) is None
