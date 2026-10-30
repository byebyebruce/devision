"""Set-of-marks layout: the gold box sits at the requested letter, the 3 others never overlap it or each other and lie
inside the window; the question is the board's template."""
import random

from v05_board_som import layout, overlaps, question


def test_layout_places_gold_and_non_overlapping_negatives():
    g = [100.0, 1500.0, 300.0, 1540.0]
    negs: list = [[100, 1500, 200, 1520],                                  # overlaps the gold
            [400, 1400, 500, 1440], [400, 1600, 500, 1640], [600, 1700, 700, 1740], [610, 1705, 690, 1735],
            [50, 100, 150, 140]]                                      # far above the window
    for slot in range(4):
        got = layout(g, negs, 1280, 4000, random.Random(slot), slot)
        assert got is not None
        top, vh, boxes = got
        assert boxes[slot] == [g[0], g[1] - top, g[2], g[3] - top]
        others = [b for k, b in enumerate(boxes) if k != slot]
        assert len(others) == 3 and all(0 <= b[1] and b[3] <= vh for b in boxes)
        assert not any(overlaps(a, b) for i, a in enumerate(boxes) for b in boxes[i + 1:])
    assert layout(g, negs[:2], 1280, 4000, random.Random(0), 0) is None      # too few negatives


def test_question_template():
    q = question("Book a table", ["[link] Restaurants -> CLICK", "[button] Search -> CLICK"], 1, "CLICK")
    assert q == ("Task: Book a table\nPrevious actions:\n- [link] Restaurants -> CLICK\n"
                 "Which marked element should the next action (CLICK) be performed on?")
    assert "- (none)" in question("t", ["a"], 0, "TYPE")
