"""exp26 panel captions: A's value first whatever side panel A is drawn on, values match the drawn view, no family
shows a dev family's candidates or value pair, no picture twice."""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
from exp21_reading import build as exp21_build, family, fmt  # noqa: E402
from exp26_panels import allowed, build, caption, dev_keys, pair_key  # noqa: E402


def test_caption_names_a_first_with_the_view_values():
    rng = random.Random(2)
    for _ in range(200):
        f = family(rng, "x")
        for view in (0, 1):
            a, b = f["views"][view]
            c = caption(f, view)
            assert c.startswith("A: %s %s %s." % (f["name"], fmt(a), f["unit"]))
            assert c.endswith("B: %s %s %s." % (f["name"], fmt(b), f["unit"]))


def test_dev_families_are_excluded(tmp_path):
    replay = tmp_path / "r.jsonl"
    replay.write_text(json.dumps({"id": "r", "source": "r14b", "image_id": "x", "image": "x.png",
                                  "questions": {"q": {"type": "noul", "instructions": "?"}},
                                  "gold": {"q": {"probabilities": {"false": 0.0, "true": 1.0}}}}) + "\n")
    exp21_build(str(tmp_path), str(replay), n_train=10, n_dev=15, n_controls=2, total=10)
    cands, pairs = dev_keys(str(tmp_path))
    assert len(pairs) >= 15
    # a family built to repeat a dev pair (swapped, in view 1) is refused
    rng = random.Random(9)
    f = family(rng, "y")
    name, unit, lo, hi = next(iter(pairs))
    f = dict(f, name=name, unit=unit, views=[(lo + 1000, hi + 1000), (hi, lo)])
    assert not allowed(f, cands, pairs)
    m = build(str(tmp_path), n_train=40, n_val=10)
    rows = [json.loads(x) for s in ("align_train", "align_val") for x in open(tmp_path / "exp26" / (s + ".jsonl"))]
    assert len(rows) == 50 and len({r["image_id"] for r in rows}) == 50
    assert not {r["family"] for r in rows[:40]} & {r["family"] for r in rows[40:]}
    for r in rows:
        parts = [p.split() for p in r["captions"][0].rstrip(".").split(". ")]
        name, unit = parts[0][1], " ".join(parts[0][3:])
        a, b = (int(p[2].replace(",", "")) for p in parts)
        assert pair_key(name, unit, a, b) not in pairs
    assert m["files"]["align_train"]["pictures"] == 40
