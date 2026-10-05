"""Round 10 focused screens (critic/plan-round10.md, section 4): one diagram type's training questions, repeated to
a fixed number of question views, so every screen has the same update budget.

    uv run python scripts/data/v10_focus.py --root data --type repel --views 2400

Source "real": the type's questions in data/v4/scienceqa.jsonl (data-v4's training pool, already kept apart from
every held-out picture by identical pixels). They are checked again against every held file's picture ids
(v8_mix.held_pictures, which now includes data/v10's monitoring set) and the build stops on any overlap.
The pool is shuffled once per pass and cut at --views; repeated questions get the id suffix "#r<pass>".
Writes data/v10/focus_<type>.jsonl and data/v10/focus_<type>.MANIFEST.json (unique questions, passes, answers).
"""
import argparse
import json
import os
import random
from collections import Counter

from convert import picture
from v10_build import diagram_type
from v8_mix import held_pictures, read


def repeat(rows, views: int, rng: random.Random):
    """`views` rows: whole shuffled passes over `rows`, the last one cut; pass k > 1 gets the id suffix #r<k>."""
    out, k = [], 0
    while len(out) < views:
        k += 1
        order = list(rows)
        rng.shuffle(order)
        for r in order[:views - len(out)]:
            out.append(dict(r, id=r["id"] + ("#r%d" % k if k > 1 else "")))
    return out, k


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    p.add_argument("--type", required=True, choices=["repel", "force", "temp", "conc"])
    p.add_argument("--views", type=int, default=2400)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    from fetch import vg_to_coco
    v2c = vg_to_coco(a.root)
    held = held_pictures(a.root, v2c)
    pool = [r for r in read(os.path.join(a.root, "v4", "scienceqa.jsonl"))
            if diagram_type(r["questions"]["q"]["instructions"]) == a.type]
    overlap = sorted({r["image_id"] for r in pool if picture(r["image_id"], v2c) in held})
    if overlap:
        raise SystemExit("training pictures in held-out sets: %s" % overlap[:5])
    rows, passes = repeat(pool, a.views, random.Random(a.seed))
    out = os.path.join(a.root, "v10")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "focus_%s.jsonl" % a.type), "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    manifest = {"type": a.type, "source": "data/v4/scienceqa.jsonl", "unique_questions": len(pool),
                "pictures": len({r["image_id"] for r in pool}), "views": len(rows), "passes": passes,
                "passes_exact": round(len(rows) / len(pool), 2), "held_out_pictures": len(held),
                "held_out_overlap": 0, "answers": dict(Counter(r["answer_key"][:40] for r in pool)),
                "steps_at_micro_batch_8": -(-len(rows) // 8), "seed": a.seed}
    with open(os.path.join(out, "focus_%s.MANIFEST.json" % a.type), "w") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
    print(json.dumps(manifest, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
