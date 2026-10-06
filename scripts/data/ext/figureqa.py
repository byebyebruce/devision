"""Extension pack, FigureQA (synthetic bar / line / pie / dot charts; yes / no questions comparing two named series:
greater / less than, maximum / minimum, intersects, area under the curve...). Licence: Microsoft Research FigureQA
terms (not verified, see docs/research/data-pack-sources-2026-10-06.md).

    uv run python scripts/data/ext/figureqa.py --root data [--share 0.3] [--per-picture 3] [--force]

Source: HuggingFaceM4/the_cauldron, subset figureqa (100,000 charts, about 13 questions each). A deterministic
--share of the charts is used (by row key), at most --per-picture questions each. Every question is yes / no ->
noul. Balance: per template (the question with its series names replaced by X), as many yes as no.
Synthetic charts, so no held-out picture can occur.
"""
import os
import random
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (already_built, balance_yes_no, base_args, cap_per_picture, cauldron_rows, extract_picture,  # noqa: E402
                    finish, keep_row, noul, picture_bytes, short_turn)

SOURCE = "figureqa"


def template(question: str) -> str:
    """'Is Dark Red greater than Pale Green?' -> 'is X greater than X?' (FigureQA names series by colour)."""
    first, _, rest = question.partition(" ")
    return (first + " " + re.sub(r"(?:[A-Z][a-z]+)(?:\s+[A-Z][a-z]+)*", "X", rest)).lower()


def main(argv=None):
    p = base_args(__doc__)
    p.add_argument("--share", type=float, default=0.3)
    p.add_argument("--per-picture", type=int, default=3)
    a = p.parse_args(argv)
    if already_built(a.root, SOURCE, a.force):
        return
    rng = random.Random(a.seed)
    rows, steps = [], Counter()
    for key, texts, images in cauldron_rows(a.root, "figureqa"):
        steps["charts"] += 1
        if not keep_row(key, a.share, SOURCE):
            continue
        data = picture_bytes(images)
        if data is None:
            continue
        steps["charts_used"] += 1
        image = extract_picture(a.root, SOURCE, key, data)
        for k, turn in enumerate(texts):
            t = short_turn(turn)
            if not t or t[0] != "yesno":
                steps["not yes/no"] += 1
                continue
            _, q, ans = t
            rows.append(noul("ext-figureqa:%s:%d" % (key, k), SOURCE, "chart", "figureqa:%s" % key, image, q, ans,
                             group=template(q)))
    steps["questions"] = len(rows)
    rows = cap_per_picture(rows, a.per_picture, rng)
    steps["after_per_picture_cap"] = len(rows)
    rows = balance_yes_no(rows)
    steps["after_balance"] = len(rows)
    finish(a.root, SOURCE, rows, dict(steps), {"licence": "Microsoft Research FigureQA terms (unverified)",
                                                "templates": len({r["group"] for r in rows})})


if __name__ == "__main__":
    main()
