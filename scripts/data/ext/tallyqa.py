"""Extension pack, TallyQA (counting on real photos, simple and complex "how many" questions). Licence Apache-2.0.

    uv run python scripts/data/ext/tallyqa.py --root data [--share 0.5] [--per-picture 2] [--force]

Source: HuggingFaceM4/the_cauldron, subset tallyqa (train; about 98k pictures with several counting questions).
TallyQA's pictures are COCO and Visual Genome photos, and the Cauldron copy carries no picture ids, so held-out
pictures (COCO val2014 test sets, GQA val, POPE, ...) are removed by dHash: every extracted picture within
NEAR_BITS of a held-out photo is dropped (common.HeldOut.near_held; hashes cached in data/raw/ext/tallyqa.hashes.json).
Each "how many" question with an answer 0..10 -> choice with nearby numbers (common.count_options). Counts are
flattened (no answer above twice the mean); at most --per-picture questions per picture.
"""
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (HeldOut, already_built, base_args, cap_per_picture, cauldron_rows, choice, count_options,  # noqa: E402
                    extract_picture, finish, flatten_answers, keep_row, picture_bytes, short_turn)

SOURCE = "tallyqa"


def main(argv=None):
    p = base_args(__doc__)
    p.add_argument("--share", type=float, default=0.5)
    p.add_argument("--per-picture", type=int, default=2)
    a = p.parse_args(argv)
    if already_built(a.root, SOURCE, a.force):
        return
    rng = random.Random(a.seed)
    rows, steps, dropped = [], Counter(), Counter()
    for key, texts, images in cauldron_rows(a.root, "tallyqa"):
        steps["pictures"] += 1
        if not keep_row(key, a.share, SOURCE):
            continue
        data = picture_bytes(images)
        if data is None:
            continue
        steps["pictures_used"] += 1
        image = extract_picture(a.root, SOURCE, key, data)
        for k, turn in enumerate(texts):
            t = short_turn(turn)
            if not t or t[0] != "number" or not t[1].lower().startswith("how many"):
                dropped["not a count"] += 1
                continue
            _, q, n = t
            if n > 10:
                dropped["count > 10"] += 1
                continue
            rows.append(choice("ext-tallyqa:%s:%d" % (key, k), SOURCE, "count", "tallyqa:%s" % key, image, q,
                               count_options(n, rng), str(n), rng, group="count"))
    steps["questions"] = len(rows)
    near = HeldOut(a.root).near_held({r["image"] for r in rows}, SOURCE)
    rows = [r for r in rows if r["image"] not in near]
    steps["held_out_pictures_dropped"] = len(near)
    steps["after_held_out"] = len(rows)
    rows = cap_per_picture(rows, a.per_picture, rng)
    steps["after_per_picture_cap"] = len(rows)
    rows = flatten_answers(rows)
    steps["after_flatten"] = len(rows)
    finish(a.root, SOURCE, rows, dict(steps), {"dropped": dict(dropped), "licence": "Apache-2.0",
                                                "count_answers": dict(sorted(Counter(r["answer_key"] for r in rows).items(),
                                                                            key=lambda kv: int(kv[0])))})


if __name__ == "__main__":
    main()
