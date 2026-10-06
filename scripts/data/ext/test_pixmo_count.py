import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
from pixmo_count import convert, picture_rel, plural, points_inside  # noqa: E402


def _row(label, count):
    return {"label": label, "count": count, "image_sha256": "ab" * 32, "image_url": "https://x/y.jpg"}


def test_labels_read_as_plural_nouns():
    assert plural("butterflys") == "butterflies"
    assert plural("bowl/basins") == "bowls"
    assert plural("table tennis ") == "table tennis balls"
    assert plural("dogs") == "dogs"
    assert plural("other balls") is None


def test_count_row_gives_count_choice_and_exist_question():
    samples, why = convert(_row("dogs", 3), "ext/pixmo_count/images/a.jpg", random.Random(0))
    assert why == ""
    count, exist = samples
    assert count["questions"]["q"]["instructions"] == "How many dogs are there?"
    assert count["answer_key"] == "3" and "3" in count["questions"]["q"]["criteria"]
    assert exist["questions"]["q"]["instructions"] == "Are there any dogs in the image?"
    assert exist["answer_key"] == "true" and exist["group"] == "exist:dogs"
    assert count["image_id"] == exist["image_id"] == "pixmo_count:" + "ab" * 8


def test_zero_count_is_absent_and_large_counts_drop():
    samples, _ = convert(_row("ties", 0), "a.jpg", random.Random(0))
    assert samples[0]["answer_key"] == "0" and samples[1]["answer_key"] == "false"
    assert convert(_row("people", 11), "a.jpg", random.Random(0)) == ([], "count > 10")
    assert convert(_row("frenches", 2), "a.jpg", random.Random(0)) == ([], "unclear label")


def test_points_must_fit_the_downloaded_picture():
    assert points_inside({"x": [10, 600], "y": [5, 400]}, 640, 480)
    assert not points_inside({"x": [10, 900], "y": [5, 400]}, 640, 480)
    assert points_inside({"x": [], "y": []}, 32, 32)


def test_picture_names_keep_a_known_extension():
    assert picture_rel("s", "https://a/b.PNG?x=1", "f" * 64).endswith("f" * 64 + ".png")
    assert picture_rel("s", "https://a/b", "f" * 64).endswith(".jpg")
