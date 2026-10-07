"""The HTTP request's top-level `image` (the same as Python's image=): URL / data URI / base64 answer like an
image part in `state`; a server-side path is never read; image and an image part together are refused."""
from conftest import image_b64
from fastapi.testclient import TestClient

from devision.serve import create_app

Q = {"dog": {"type": "noul", "instructions": "Is there a dog in the image?"}}


def post(decider, body):
    return TestClient(create_app(decider)).post("/v1/systemone", json=body)


def test_top_level_image_answers_like_an_image_part(decider):
    b64 = image_b64()
    part = post(decider, {"state": [{"type": "image", "base64": b64}, "a note"], "questions": Q}).json()
    for image in (b64, "data:image/png;base64," + b64):
        top = post(decider, {"image": image, "state": "a note", "questions": Q})
        assert top.status_code == 200
        assert abs(top.json()["answers"]["dog"]["noul"] - part["answers"]["dog"]["noul"]) < 1e-6


def test_top_level_image_never_reads_a_server_file(decider, tmp_path):
    path = tmp_path / "pic.png"
    path.write_bytes(__import__("base64").b64decode(image_b64()))
    res = post(decider, {"image": str(path), "state": "", "questions": Q})
    assert res.status_code == 422 and "URL, a data URI or base64" in res.json()["error"]


def test_top_level_image_and_an_image_part_together_are_refused(decider):
    b64 = image_b64()
    res = post(decider, {"image": b64, "state": [{"type": "image", "base64": b64}], "questions": Q})
    assert res.status_code == 422 and "not both" in res.json()["error"]
    assert post(decider, {"image": 42, "state": "", "questions": Q}).status_code == 422
