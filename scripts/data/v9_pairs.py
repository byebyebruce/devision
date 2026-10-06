"""Round 9's real left/right questions (R, docs/experiments/2026-10-05-v9-lr-focus-plan.md): the 2,500-question
list v9b trains on, written to data/v9/r_pairs.jsonl.

    uv run python scripts/data/v9_pairs.py --root data

The selection rule of the round 9 screen script (scripts/r9/real_lr.py on branch r9-screen, seed 9), without the
training: mirrored questions from data/v6/flip_lr.jsonl that round 6's mix (data/v6/train_flip_mix.jsonl) did not
use, whose original is in data-v2's COCO position / relation pools and whose picture is not a dev_lr_pairs picture;
shuffled with seed 9, the first 1,250 taken, each written as original then mirror. On this machine's data the
output is byte for byte the file v9b used (SHA256 6c6b596c...26fee). v9_mix.py --pairs-file then drops the pairs on
near-duplicate photos of held-out pictures (12 questions).
"""
import argparse
import json
import os
import random
from typing import Dict, List


def read(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def select(flips: List[dict], originals: Dict[str, dict], used: set, dev_pics: set, pairs: int, seed: int = 9):
    cand = [f for f in flips
            if f["id"] not in used and f["id"][5:] in originals and f["image_id"].split(":flip")[0] not in dev_pics]
    random.Random(seed).shuffle(cand)
    return [x for f in cand[:pairs] for x in (originals[f["id"][5:]], f)]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    p.add_argument("--pairs", type=int, default=1250)
    p.add_argument("--out", help="default <root>/v9/r_pairs.jsonl")
    a = p.parse_args(argv)
    root = a.root
    dev_pics = {r["image_id"].split(":flip")[0] for r in read(os.path.join(root, "v6", "dev_lr_pairs.jsonl"))}
    used = {r["id"] for r in read(os.path.join(root, "v6", "train_flip_mix.jsonl"))}
    originals = {}
    for f in ("coco_position.jsonl", "coco_relation.jsonl"):
        for r in read(os.path.join(root, "v2", f)):
            originals[r["id"]] = r
    rows = select(read(os.path.join(root, "v6", "flip_lr.jsonl")), originals, used, dev_pics, a.pairs)
    out = a.out or os.path.join(root, "v9", "r_pairs.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    print("%d questions (%d pairs) -> %s" % (len(rows), len(rows) // 2, out))


if __name__ == "__main__":
    main()
