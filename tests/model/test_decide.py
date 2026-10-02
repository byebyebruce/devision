"""Contract tests for `decide`: Jev-format request in, Jev-format response out."""
import pytest
from conftest import image_b64

from devision.model import InvalidRequest


def test_noul_question_about_an_image_returns_probability_of_yes(decider):
    res = decider.decide(
        state=[{"type": "image", "base64": image_b64()}],
        questions={"dog": {"type": "noul", "instructions": "Is there a dog in the image?"}},
    )

    answer = res["answers"]["dog"]
    assert answer["type"] == "noul"
    assert 0.0 <= answer["noul"] <= 1.0
    assert "confidence" not in answer
    assert res["model"] == "devision-tiny"
    assert res["usage"]["input_tokens"] > 0


def test_choice_answer_covers_exactly_the_requested_options(decider):
    options = {"standing": None, "sitting": "on a chair or the ground", "lying": ""}
    res = decider.decide(
        state=[{"type": "image", "base64": image_b64()}],
        questions={"pose": {"type": "choice", "instructions": "Is the person standing or sitting?",
                            "criteria": options}},
    )

    answer = res["answers"]["pose"]
    probs = answer["probabilities"]
    assert answer["type"] == "choice"
    assert set(probs) == set(options)
    assert abs(sum(probs.values()) - 1.0) < 1e-5
    assert answer["choice"] == max(probs, key=probs.__getitem__)
    # Jev's definition: 0 for a uniform answer, 1 for a certain one.
    n, p_max = len(options), max(probs.values())
    assert abs(answer["confidence"] - (n * p_max - 1) / (n - 1)) < 1e-6


NOUL = {"type": "noul", "instructions": "Is there a dog in the image?"}
CHOICE = {"type": "choice", "instructions": "What color is it?",
          "criteria": {"red": None, "green": None, "blue": None}}


def test_text_only_request_is_answered_like_jev(decider):
    res = decider.decide(state="a red dog is sitting", questions={"a": NOUL, "b": CHOICE})

    assert res["answers"]["a"]["type"] == "noul"
    assert res["answers"]["b"]["type"] == "choice"


def test_text_parts_next_to_the_image_are_accepted(decider):
    res = decider.decide(state=["photo from the park", {"type": "image", "base64": image_b64()}],
                         questions={"a": NOUL})

    assert 0.0 <= res["answers"]["a"]["noul"] <= 1.0


def test_any_aspect_ratio_is_accepted(decider):
    for size in [(256, 256), (31, 700), (1600, 90), (1, 1)]:
        res = decider.decide(state=[{"type": "image", "base64": image_b64(size)}], questions={"a": NOUL})
        assert 0.0 <= res["answers"]["a"]["noul"] <= 1.0


def test_same_request_gives_identical_answers(decider):
    state = [{"type": "image", "base64": image_b64()}]
    first = decider.decide(state=state, questions={"a": NOUL, "b": CHOICE})
    second = decider.decide(state=state, questions={"a": NOUL, "b": CHOICE})

    assert first == second


def test_question_ids_do_not_change_the_answers(decider):
    state = [{"type": "image", "base64": image_b64()}]
    a = decider.decide(state=state, questions={"x": NOUL, "y": CHOICE})["answers"]
    b = decider.decide(state=state, questions={"first_question": NOUL, "second": CHOICE})["answers"]

    assert a["x"] == b["first_question"]
    assert a["y"] == b["second"]


def test_answers_to_one_question_do_not_depend_on_the_others(decider):
    state = [{"type": "image", "base64": image_b64()}]
    alone = decider.decide(state=state, questions={"a": NOUL})["answers"]["a"]
    together = decider.decide(state=state, questions={"a": NOUL, "b": CHOICE})["answers"]["a"]

    assert abs(alone["noul"] - together["noul"]) < 1e-5

IMG = {"type": "image", "base64": image_b64()}


@pytest.mark.parametrize("state,questions,why", [
    ([IMG], {}, "no questions"),
    ([IMG], {"a": {"type": "vibe", "instructions": "?"}}, "unknown type"),
    ([IMG], {"a": {"type": "score", "instructions": "How big?", "criteria": ["small", "big"]}},
     "score is not supported yet"),
    ([IMG], {"a": {"type": "noul"}}, "missing instructions"),
    ([IMG], {"a": {"type": "choice", "instructions": "?", "criteria": {"only": None}}}, "one option"),
    ([IMG], {"a": {"type": "choice", "instructions": "?", "criteria": ["red", "blue"]}},
     "criteria not a map"),
    ([IMG], {"a": {"type": "choice", "instructions": "?",
                   "criteria": {str(i): None for i in range(256)}}}, "more than 255 options"),
    ([IMG], {"a": {"type": "noul", "instructions": "?", "criteria": {"maybe": "x"}}},
     "noul criteria keys"),
    ([{"type": "image", "base64": "bm90IGFuIGltYWdl"}], {"a": NOUL}, "not an image"),
    ([{"type": "image", "base64": "%%%"}], {"a": NOUL}, "not base64"),
    ([{"type": "image"}], {"a": NOUL}, "no image data"),
    ([{"type": "image", "base64": image_b64(), "url": "http://x/y.png"}], {"a": NOUL}, "both sources"),
    ([IMG, IMG], {"a": NOUL}, "two images"),
    (None, {"a": NOUL}, "state missing"),
])
def test_invalid_requests_are_rejected(decider, state, questions, why):
    with pytest.raises(InvalidRequest):
        decider.decide(state=state, questions=questions)


def test_image_can_be_given_by_url(decider, tmp_path):
    import base64
    import functools
    import http.server
    import threading

    (tmp_path / "dog.png").write_bytes(base64.b64decode(image_b64()))
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        base = "http://127.0.0.1:%d/" % server.server_address[1]
        by_url = decider.decide(state=[{"type": "image", "url": base + "dog.png"}], questions={"a": NOUL})
        by_b64 = decider.decide(state=[IMG], questions={"a": NOUL})
        assert by_url["answers"] == by_b64["answers"]

        with pytest.raises(InvalidRequest):
            decider.decide(state=[{"type": "image", "url": base + "missing.png"}], questions={"a": NOUL})
    finally:
        server.shutdown()


def test_options_that_do_not_fit_the_model_are_rejected_not_dropped(decider):
    many = {"option number %d with a long description" % i: None for i in range(250)}

    with pytest.raises(InvalidRequest):
        decider.decide(state=[IMG], questions={"a": {"type": "choice", "instructions": "?",
                                                     "criteria": many}})


def test_decompression_bomb_is_rejected(decider):
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("1", (20000, 20000)).save(buf, format="PNG")  # tiny file, 400M pixels
    bomb = base64.b64encode(buf.getvalue()).decode()

    with pytest.raises(InvalidRequest):
        decider.decide(state=[{"type": "image", "base64": bomb}], questions={"a": NOUL})


def test_a_saved_checkpoint_loads_through_the_package_and_answers_the_same(decider, tmp_path):
    import devision

    request = dict(state=[{"type": "image", "base64": image_b64()}],
                   questions={"dog": {"type": "noul", "instructions": "Is there a dog in the image?"}})
    decider.save(tmp_path / "ckpt")

    loaded = devision.load(str(tmp_path / "ckpt"))

    assert loaded.predict(**request) == decider.decide(**request)
