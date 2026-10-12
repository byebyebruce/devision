"""Quick checks on the exp14 base (docs/experiments/2026-10-12-exp15-steps.md): the same 12,000-question replay subset
exp14-pixmo used (75% of every exp14-base block) plus 4,000 questions of ONE new block, so each arm compares with
exp14-base (pure replay, same seed / steps / learning rates) on one variable.

    uv run python scripts/exp15/build.py --root data --block bind     # -> data/exp15/train_bind.jsonl

Blocks:
  bind   real-photo binding: GQA binding 2,000 + VisMin 2,000 (whole pictures / original-edit pairs; round-13 train
         splits, pictures isolated from every dev / test set in round 13)
"""
import argparse
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "r13b"))
import build as r13b  # noqa: E402


def block_bind(root, rng):
    R = lambda rel: r13b.rows(os.path.join(root, rel))
    return (r13b.take(R("r13/gqa/train.jsonl"), lambda s: s["group"], 2000, rng)
            + r13b.take(R("r13/vismin/train.jsonl"), lambda s: s["group"], 2000, rng))


BLOCKS = {"bind": block_bind}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    p.add_argument("--block", required=True, choices=sorted(BLOCKS))
    a = p.parse_args()
    rng = random.Random(15)
    base = [s for s in r13b.rows(os.path.join(a.root, "exp14/train_pixmo.jsonl")) if s["block"] != "pixmo"]
    new = [dict(s, block=a.block) for s in BLOCKS[a.block](a.root, rng)]
    mix = base + new
    assert len({s["id"] for s in mix}) == len(mix), "duplicate ids"
    random.Random(1).shuffle(mix)
    os.makedirs(os.path.join(a.root, "exp15"), exist_ok=True)
    out = os.path.join(a.root, "exp15", "train_%s.jsonl" % a.block)
    r13b.write(out, mix)
    m = {"replay": r13b.nq(base), "new": r13b.nq(new), "total": r13b.nq(mix),
         "new_by_source": dict(Counter(s["source"] for s in new for _ in s["questions"])),
         "new_pictures": len({s["image_id"] for s in new}), "blocks": dict(Counter(s["block"] for s in mix))}
    json.dump(m, open(out.replace(".jsonl", ".MANIFEST.json"), "w"), indent=1)
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
