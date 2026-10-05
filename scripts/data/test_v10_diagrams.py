"""Round 10 synthetic attract / repel: the label follows the facing poles; turning one magnet round changes it,
mirroring the whole scene does not."""
import random

from v10_diagrams import answer, repel_answer, scene, variants


def test_facing_poles_decide():
    assert repel_answer("N", "S") == "attract" and repel_answer("S", "S") == "repel"
    # first magnet S-N faces with N; second N-S faces with N -> repel
    assert answer({"poles": [("S", "N"), ("N", "S")]}) == "repel"
    assert answer({"poles": [("S", "N"), ("S", "N")]}) == "attract"


def test_controls_change_or_keep_the_answer():
    rng = random.Random(1)
    for _ in range(200):
        s = scene(rng)
        (_, o), (_, f), (_, m) = variants(s)
        assert answer(f) != answer(o) and answer(m) == answer(o)
