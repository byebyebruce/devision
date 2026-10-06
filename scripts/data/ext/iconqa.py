"""Extension pack, IconQA (K-12 math pictures: counting, comparing, shapes, fractions, time...). Licence CC BY-NC-SA 4.0.

    uv run python scripts/data/ext/iconqa.py --root data [--force]

Source: HuggingFaceM4/the_cauldron, subset iconqa (IconQA's text-choice and fill-in-the-blank train questions;
its image-choice questions, whose options are pictures, are not in it). Each turn is converted:
  - "Question ... Choices: A. ... Answer with the letter." -> choice over the given options; a yes / no pair of
    options -> noul;
  - a short numeric answer (fill-in-the-blank, mostly counts) -> choice with nearby numbers (common.count_options);
  - anything else is dropped (counted in the manifest).
Pictures are extracted once to data/ext/iconqa/images/. IconQA comes from the same authors as ScienceQA, so a
picture with exactly the pixels of a held-out diagram (ScienceQA validation / test sets we evaluate on) is dropped.
"""
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (HeldOut, already_built, base_args, cauldron_rows, choice, count_options, extract_picture,  # noqa: E402
                    finish, multiple_choice, noul, picture_bytes, question_text, short_answer)

SOURCE = "iconqa"


def convert(key: str, k: int, image: str, turn: dict, rng: random.Random):
    """One Cauldron turn -> a sample or (None, reason)."""
    sid = "ext-iconqa:%s:%d" % (key, k)
    image_id = "iconqa:%s" % key
    mc = multiple_choice(turn["user"], turn["assistant"])
    if mc:
        q, opts, ans = mc
        low = {o.lower() for o in opts}
        if low == {"yes", "no"}:
            return noul(sid, SOURCE, "iconqa", image_id, image, q, ans.lower() == "yes", group="iconqa-yesno"), None
        if not 2 <= len(opts) <= 10:
            return None, "options"
        return choice(sid, SOURCE, "iconqa", image_id, image, q, opts, ans, rng, group="iconqa-choice"), None
    ans = short_answer(turn["assistant"])
    if ans.isdigit() and int(ans) <= 20:
        q = question_text(turn["user"])
        return choice(sid, SOURCE, "count", image_id, image, q, count_options(int(ans), rng, top=20), ans, rng,
                      group="iconqa-count"), None
    return None, "open answer"


def main(argv=None):
    a = base_args(__doc__).parse_args(argv)
    if already_built(a.root, SOURCE, a.force):
        return
    rng = random.Random(a.seed)
    held = HeldOut(a.root)
    rows, dropped, steps = [], Counter(), Counter()
    for key, texts, images in cauldron_rows(a.root, "iconqa"):
        steps["rows"] += 1
        data = picture_bytes(images)
        if data is None:
            dropped["no single picture"] += len(texts)
            continue
        image = extract_picture(a.root, SOURCE, key, data)
        if held.same_diagram(image):
            dropped["held-out diagram"] += len(texts)
            continue
        for k, turn in enumerate(texts):
            steps["turns"] += 1
            r, why = convert(key, k, image, turn, rng)
            if r:
                rows.append(r)
            else:
                dropped[why] += 1
    finish(a.root, SOURCE, rows, dict(steps, kept=len(rows)), {"dropped": dict(dropped), "licence": "CC BY-NC-SA 4.0"})


if __name__ == "__main__":
    main()
