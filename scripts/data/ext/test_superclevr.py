"""Extension pack, Super-CLEVR: question conversion and the attribute vocabulary (no network)."""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import problems  # noqa: E402
from superclevr import convert, query_attribute, vocabulary  # noqa: E402


def q(question, answer, last, template="zero_hop.json", index=0):
    return {"image_filename": "superCLEVR_new_000007.png", "image_index": 7, "question_index": index,
            "template_filename": template, "question": question, "answer": answer,
            "program": [{"type": "scene"}, {"type": last}]}


QS = [q("What color is the bus?", "red", "query_color"), q("What color is the jet?", "cyan", "query_color"),
      q("What color is the car?", "gray", "query_color"), q("What shape is it?", "school bus", "query_shape"),
      q("What shape is that?", "sedan", "query_shape"), q("What size is it?", "small", "query_size"),
      q("What size is that?", "large", "query_size"), q("How many buses?", 3, "count"),
      q("Are there more buses than cars?", True, "greater_than", "compare_integer.json"),
      q("Is there a jet?", False, "exist", "same_relate.json")]


def one(converted) -> dict:
    r, why = converted
    assert r is not None, why
    return r


def test_vocabulary_is_per_queried_attribute():
    v = vocabulary(QS)
    assert v == {"color": ["cyan", "gray", "red"], "shape": ["school bus", "sedan"], "size": ["large", "small"]}
    assert query_attribute(QS[0]["program"]) == "color" and query_attribute(QS[7]["program"]) is None


def test_conversion_by_answer_type():
    rng, v = random.Random(0), vocabulary(QS)
    out = [one(convert(x, "ext/superclevr/images/superCLEVR_new_000007.png", v, rng)) for x in QS]
    assert all(problems(r) == [] for r in out)
    color = out[0]
    assert color["kind"] == "color" and set(color["gold"]["q"]["probabilities"]) <= {"cyan", "gray", "red"}
    assert color["answer_key"] == "red"
    shape = out[3]
    assert set(shape["gold"]["q"]["probabilities"]) == {"school bus", "sedan"}
    count = out[7]
    assert count["kind"] == "count" and count["answer_key"] == "3"
    assert out[8]["kind"] == "compare" and out[8]["answer_key"] == "true" and out[8]["group"] == "superclevr-compare_integer"
    assert out[9]["kind"] == "exist" and out[9]["answer_key"] == "false"
    assert out[0]["image_id"] == "superclevr:superCLEVR_new_000007"
    assert convert(q("How many?", 11, "count"), "x.png", v, rng) == (None, "count > 10")


def test_shape_options_include_the_same_superclass():
    qs = [q("What shape is the bus?", w, "query_shape", index=i)
          for i, w in enumerate(["school bus", "double bus", "regular bus", "sedan", "jet", "articulated bus"])]
    v, rng = vocabulary(qs), random.Random(1)
    for _ in range(20):
        r = one(convert(qs[0], "x.png", v, rng))
        opts = set(r["gold"]["q"]["probabilities"])
        assert "school bus" in opts and len(opts & {"double bus", "regular bus", "articulated bus"}) >= 1
