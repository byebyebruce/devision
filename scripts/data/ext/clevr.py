"""Extension pack, CLEVR / CLEVR-Math (synthetic 3D scenes; labels computed from the scene, so exact). CLEVR:
compare counts ("more A than B?"), same attribute ("same colour as ...?"), exist, count, query an attribute.
CLEVR-Math: add / subtract objects, then count. Licence CC BY 4.0.

    uv run python scripts/data/ext/clevr.py --root data [--subset clevr|clevr_math] [--share 0.4] [--per-picture 3]

Source: HuggingFaceM4/the_cauldron, subsets clevr (70,000 scenes) and clevr_math (70,000 scenes), each scene with
about 10 short-answer questions. A deterministic --share of scenes, at most --per-picture questions each.
  - yes / no -> noul (group: "compare" for more / fewer / same / equal questions, else "exist");
  - a number -> choice with nearby numbers (count);
  - a colour / shape / material / size word -> choice among 2-5 words of the same attribute (CLEVR's vocabulary).
Balance: yes = no per group; counts flattened (no answer above twice the mean). Output data/ext/<subset>/.
"""
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (already_built, balance_yes_no, base_args, cap_per_picture, cauldron_rows, choice,  # noqa: E402
                    count_options, extract_picture, finish, flatten_answers, keep_row, noul, picture_bytes,
                    short_turn)

VOCAB = {"color": ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"],
         "shape": ["cube", "sphere", "cylinder"], "material": ["rubber", "metal"], "size": ["small", "large"]}
WORD = {w: k for k, ws in VOCAB.items() for w in ws}
COMPARE = ("more", "fewer", "less", "same", "equal", "greater")


def convert(subset: str, key: str, k: int, image: str, turn: dict, rng: random.Random):
    t = short_turn(turn)
    if not t:
        return None, "open answer"
    kind, q, ans = t
    sid, image_id = "ext-%s:%s:%d" % (subset, key, k), "%s:%s" % (subset, key)
    if kind == "yesno":
        group = "compare" if any(w in q.lower().split() for w in COMPARE) else "exist"
        return noul(sid, subset, group, image_id, image, q, bool(ans), group="%s-%s" % (subset, group)), None
    if kind == "number":
        n = int(ans)
        if n > 10:
            return None, "count > 10"
        return choice(sid, subset, "count", image_id, image, q, count_options(n, rng), str(n), rng,
                      group="%s-count" % subset), None
    word = str(ans).lower()
    if word in WORD:
        attr = WORD[word]
        others = [w for w in VOCAB[attr] if w != word]
        rng.shuffle(others)
        n = min(len(others), rng.choice([1, 2, 3, 4]))
        return choice(sid, subset, attr, image_id, image, q, [word] + others[:n], word, rng,
                      group="%s-%s" % (subset, attr)), None
    return None, "other word"


def main(argv=None):
    p = base_args(__doc__)
    p.add_argument("--subset", default="clevr", choices=["clevr", "clevr_math"])
    p.add_argument("--share", type=float, default=0.4)
    p.add_argument("--per-picture", type=int, default=3)
    a = p.parse_args(argv)
    source = a.subset
    if already_built(a.root, source, a.force):
        return
    rng = random.Random(a.seed)
    rows, steps, dropped = [], Counter(), Counter()
    for key, texts, images in cauldron_rows(a.root, a.subset):
        steps["scenes"] += 1
        if not keep_row(key, a.share, source):
            continue
        data = picture_bytes(images)
        if data is None:
            continue
        steps["scenes_used"] += 1
        image = extract_picture(a.root, source, key, data)
        for k, turn in enumerate(texts):
            r, why = convert(a.subset, key, k, image, turn, rng)
            if r:
                rows.append(r)
            else:
                dropped[why] += 1
    steps["questions"] = len(rows)
    rows = cap_per_picture(rows, a.per_picture, rng)
    steps["after_per_picture_cap"] = len(rows)
    rows = balance_yes_no(rows)
    counts = flatten_answers([r for r in rows if r["kind"] == "count"])
    rows = [r for r in rows if r["kind"] != "count"] + counts
    steps["after_balance"] = len(rows)
    finish(a.root, source, rows, dict(steps), {"dropped": dict(dropped), "licence": "CC BY 4.0",
                                                "count_answers": dict(sorted(Counter(r["answer_key"] for r in counts).items(),
                                                                            key=lambda kv: int(kv[0])))})


if __name__ == "__main__":
    main()
