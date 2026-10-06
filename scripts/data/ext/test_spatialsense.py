import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import spatialsense as s  # noqa: E402


def rel(pred, sub, obj, label, i="r1"):
    return {"_id": i, "predicate": pred, "label": label,
            "subject": {"name": sub, "x": 0, "y": 0, "bbox": [0, 1, 0, 1]},
            "object": {"name": obj, "x": 0, "y": 0, "bbox": [0, 1, 0, 1]}}


def test_questions_read_naturally_with_is_or_are():
    assert s.question("Cat", "on", "ground") == "Is the cat on the ground?"
    assert s.question("shoes", "under", "bed") == "Are the shoes under the bed?"
    assert s.question("people", "in front of", "bus") == "Are the people in front of the bus?"
    assert s.question("glass", "next to", "plate") == "Is the glass next to the plate?"


def test_triples_become_noul_grouped_by_predicate():
    img = {"url": "https://farm4.staticflickr.com/3543/5704634119_8b8ccf3229.jpg", "split": "train",
           "annotations": [rel("on", "cat", "ground", True), rel("to the left of", "cat", "mirror", False, "r2"),
                           rel("on", "cat", "cat", True, "r3")]}
    rows = s.convert(img)
    assert [r["id"] for r in rows] == ["ext-spatialsense:r1", "ext-spatialsense:r2"]   # same subject / object dropped
    assert rows[0]["answer_key"] == "true" and rows[0]["group"] == "on" and rows[0]["kind"] == "relation"
    assert rows[1]["answer_key"] == "false" and rows[1]["kind"] == "relation-leftright"
    assert rows[0]["image"] == "ext/spatialsense/images/flickr/5704634119_8b8ccf3229.jpg"
    assert rows[0]["image_id"] == "spatialsense:flickr/5704634119_8b8ccf3229"


def test_nyu_pictures_map_to_their_folder():
    assert s.picture_rel("/images/nyu/nyu_bedroom_0118_r-1.2-3.png") == "ext/spatialsense/images/nyu/nyu_bedroom_0118_r-1.2-3.png"


def test_flickr_placeholder_is_not_usable(tmp_path):
    good, bad = tmp_path / "a.jpg", tmp_path / "b.jpg"
    good.write_bytes(b"\xff\xd8\xff\xe0rest")
    bad.write_bytes(b"\x89PNG\r\n\x1a\nrest")
    assert s.usable(str(good)) and not s.usable(str(bad)) and not s.usable(str(tmp_path / "c.jpg"))
