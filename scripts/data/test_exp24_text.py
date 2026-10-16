"""exp24 reading-alignment pictures: the caption is the drawn text in reading order, every line inside the picture
at its stated digit height, no picture twice."""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
from exp21_reading import digit_ink  # noqa: E402
from exp24_text import HEIGHTS, build, caption, render, scene  # noqa: E402


def test_lines_go_top_to_bottom_inside_the_picture_and_the_caption_follows_them():
    rng = random.Random(3)
    for _ in range(300):
        s = scene(rng)
        img, boxes = render(s)
        assert len(boxes) == len(s["lines"])
        assert all(0 <= b[0] and b[2] <= 256 and 0 <= b[1] and b[3] <= 256 for b in boxes)
        assert all(boxes[i][3] <= boxes[i + 1][1] for i in range(len(boxes) - 1))   # no overlap, in order
        assert caption(s) == ". ".join(t for _, t, _ in s["lines"])


def test_numbers_are_drawn_at_their_height():
    rng = random.Random(4)
    checked = 0
    while checked < 100:
        s = scene(rng)
        img, boxes = render(s)
        for (kind, text, h), b in zip(s["lines"], boxes):
            if kind == "number" and "," not in text and "4" not in text:   # nothing below the baseline
                assert abs(digit_ink(img, b) - h) <= 1                         # (a 4 alone can be 1 px short)
                checked += 1


def test_build_counts_and_no_picture_twice(tmp_path):
    m = build(str(tmp_path), n_train=60, n_val=20)
    rows = [json.loads(x) for f in ("align_train", "align_val") for x in open(tmp_path / "exp24" / (f + ".jsonl"))]
    assert len(rows) == 80 and len({r["image_id"] for r in rows}) == 80
    assert all(r["captions"][0] and os.path.exists(tmp_path / r["image"]) for r in rows)
    assert set(m["files"]["align_train"]["heights"]) <= {str(h) for h in HEIGHTS}
