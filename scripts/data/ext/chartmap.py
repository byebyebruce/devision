"""Extension pack, DVQA (synthetic bar charts) and MapQA (US choropleth maps): only the yes / no comparison
questions ("Is the value of X larger than Y?", "Does Nebraska have a higher value than Pennsylvania?"), which need
the picture and no reading of numbers. Licences: DVQA CC BY-NC 4.0, MapQA CC BY-SA 4.0.

    uv run python scripts/data/ext/chartmap.py --root data --subset dvqa|mapqa [--share 0.2] [--per-picture 3]

Source: HuggingFaceM4/the_cauldron, subsets dvqa and mapqa. A deterministic --share of pictures, at most
--per-picture yes / no questions each; balance yes = no per template (the question with its capitalised names
replaced by X). Synthetic pictures, so no held-out picture can occur. Output data/ext/<subset>/.
"""
import os
import random
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (already_built, balance_yes_no, base_args, cap_per_picture, cauldron_rows, extract_picture,  # noqa: E402
                    finish, keep_row, noul, picture_bytes, short_turn)

LICENCE = {"dvqa": "CC BY-NC 4.0", "mapqa": "CC BY-SA 4.0"}


def template(question: str) -> str:
    """Names (capitalised words after the first word, or quoted labels) -> X."""
    first, _, rest = question.partition(" ")
    rest = re.sub(r"'[^']*'|\"[^\"]*\"", "X", rest)
    rest = re.sub(r"(?:[A-Z][\w.-]*)(?:\s+[A-Z][\w.-]*)*", "X", rest)
    return (first + " " + rest).lower()


def main(argv=None):
    p = base_args(__doc__)
    p.add_argument("--subset", required=True, choices=["dvqa", "mapqa"])
    p.add_argument("--share", type=float, default=0.2)
    p.add_argument("--per-picture", type=int, default=3)
    a = p.parse_args(argv)
    source = a.subset
    if already_built(a.root, source, a.force):
        return
    rng = random.Random(a.seed)
    rows, steps = [], Counter()
    for key, texts, images in cauldron_rows(a.root, a.subset):
        steps["pictures"] += 1
        if not keep_row(key, a.share, source):
            continue
        data = picture_bytes(images)
        if data is None:
            continue
        yes_no = [(k, t) for k, turn in enumerate(texts) for t in [short_turn(turn)] if t and t[0] == "yesno"]
        if not yes_no:
            continue
        steps["pictures_used"] += 1
        image = extract_picture(a.root, source, key, data)
        for k, (_, q, ans) in yes_no:
            rows.append(noul("ext-%s:%s:%d" % (source, key, k), source, "chart" if source == "dvqa" else "map",
                             "%s:%s" % (source, key), image, q, ans, group=template(q)))
    steps["questions"] = len(rows)
    rows = cap_per_picture(rows, a.per_picture, rng)
    steps["after_per_picture_cap"] = len(rows)
    rows = balance_yes_no(rows)
    steps["after_balance"] = len(rows)
    finish(a.root, source, rows, dict(steps), {"licence": LICENCE[source], "templates": len({r["group"] for r in rows})})


if __name__ == "__main__":
    main()
