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


QUESTIONS_BY_BUCKET = {
    "noul": NOUL,
    "two": {"type": "choice", "instructions": "What color is it?", "criteria": {"red": None, "blue": None}},
    "three": CHOICE,
    "seven": {"type": "choice", "instructions": "What color is it?",
              "criteria": {c: None for c in ["red", "green", "blue", "black", "white", "dog", "cat"]}},
}


def test_a_checkpoint_saved_before_bucket_temperatures_answers_with_its_per_type_temperatures(tmp_path):
    import json

    from conftest import tiny_decider

    from devision.model import Decider

    decider = tiny_decider()
    decider.cfg.temperature = [2.0, 1.0, 0.6]
    decider.save(tmp_path / "ckpt")
    config = json.loads((tmp_path / "ckpt" / "config.json").read_text())
    del config["temperature_by_options"]   # as written before the field existed
    (tmp_path / "ckpt" / "config.json").write_text(json.dumps(config))

    old = Decider.load(tmp_path / "ckpt", device="cpu")   # the in-memory model is on the CPU: exact equality
    untempered = tiny_decider()

    state = [{"type": "image", "base64": image_b64()}]
    assert old.decide(state, QUESTIONS_BY_BUCKET) == decider.decide(state, QUESTIONS_BY_BUCKET)
    assert old.decide(state, QUESTIONS_BY_BUCKET) != untempered.decide(state, QUESTIONS_BY_BUCKET)


def test_a_bucket_temperature_changes_only_the_questions_in_that_bucket():
    from conftest import tiny_decider

    decider = tiny_decider()
    state = [{"type": "image", "base64": image_b64()}]
    before = decider.decide(state, QUESTIONS_BY_BUCKET)["answers"]

    decider.cfg.temperature_by_options = {"choice:3-5": 3.0}
    after = decider.decide(state, QUESTIONS_BY_BUCKET)["answers"]

    assert after["three"] != before["three"]
    assert after["three"]["choice"] == before["three"]["choice"]   # a temperature never changes the answer
    assert {q: after[q] for q in ("noul", "two", "seven")} == {q: before[q] for q in ("noul", "two", "seven")}


def test_options_that_become_identical_when_cut_to_fit_are_refused(decider):
    # each option is longer than the head gives one option, and they differ only at the end
    prefix = "the dog is sitting on the left side of the white cat in the picture " * 6
    q = {"type": "choice", "instructions": "Which is it?",
         "criteria": {prefix + "left": None, prefix + "right": None}}
    with pytest.raises(InvalidRequest, match="identical"):
        decider.decide(state=[{"type": "image", "base64": image_b64()}], questions={"q": q})


def test_image_argument_accepts_every_form_and_answers_like_an_image_part(decider, tmp_path):
    """decide(image=...) is the Python convenience: URL / data URI / file / base64 / bytes / Path / PIL give the
    same answers as the Jev-compatible image part in state."""
    import base64
    import io

    from PIL import Image

    b64 = image_b64()
    raw = base64.b64decode(b64)
    path = tmp_path / "photo.png"
    path.write_bytes(raw)
    questions = {"dog": {"type": "noul", "instructions": "Is there a dog in the image?"}}
    want = decider.decide(state=[{"type": "image", "base64": b64}, "a note"], questions=questions)
    for image in (b64, "data:image/png;base64," + b64, str(path), path, raw, Image.open(io.BytesIO(raw))):
        assert decider.decide(state="a note", questions=questions, image=image) == want


def test_image_argument_errors_are_invalid_requests(decider, tmp_path):
    questions = {"dog": {"type": "noul", "instructions": "Is there a dog in the image?"}}
    with pytest.raises(InvalidRequest, match="not both"):
        decider.decide(state=[{"type": "image", "base64": image_b64()}], questions=questions, image=image_b64())
    for bad in ("not an image at all!", str(tmp_path / "missing.jpg"), "data:image/png,abc", 42):
        with pytest.raises(InvalidRequest):
            decider.decide(state="", questions=questions, image=bad)


def test_an_image_key_in_an_object_state_stays_text_as_in_jev(decider, tmp_path):
    questions = {"dog": {"type": "noul", "instructions": "Is there a dog in the image?"}}
    text = decider.decide(state={"image": "a red sofa", "note": "n"}, questions=questions)
    seen = decider.decide(state={"image": "a red sofa", "note": "n"}, questions=questions, image=image_b64())
    assert text["usage"]["input_tokens"] < seen["usage"]["input_tokens"]    # only image= adds the picture


def test_an_older_checkpoint_with_devision_config_json_loads_the_same(decider, tmp_path):
    """Checkpoints saved before 2026-10-07 name the model config devision_config.json; they still load."""
    import os

    import devision

    request = dict(state=[{"type": "image", "base64": image_b64()}],
                   questions={"dog": {"type": "noul", "instructions": "Is there a dog in the image?"}})
    decider.save(tmp_path / "ckpt")
    os.rename(tmp_path / "ckpt" / "config.json", tmp_path / "ckpt" / "devision_config.json")
    assert devision.load(str(tmp_path / "ckpt")).predict(**request) == decider.decide(**request)
    decider.save(tmp_path / "ckpt")                              # saving over it leaves one config, config.json
    assert sorted(os.listdir(tmp_path / "ckpt"))[0] == "config.json"
    assert not (tmp_path / "ckpt" / "devision_config.json").exists()
