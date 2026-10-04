"""Training data for round 4 (kept out of the repo): above/below relations plus replay, to continue v3.

    uv run python scripts/data/v4_relation_mix.py --root data

v3's mix held only 172 relative above/below questions and v3 fell from 0.86 to 0.57 on them. This file
takes every above/below question of the COCO relation pool (data/v2/coco_relation.jsonl, axis tb) and
of data/v2/vg_relation_tb.jsonl (scene graphs, scripts/data/v2_vg_relation.py), plus --replay times as
many questions drawn at random from data/v2/train_mix.jsonl so the rest is not forgotten.
Writes data/v2/train_relation_tb.jsonl.
"""
import argparse
import json
import os
import random
from collections import Counter


def read(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="data")
    p.add_argument("--replay", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    v2 = os.path.join(a.root, "v2")
    tb = [r for r in read(os.path.join(v2, "coco_relation.jsonl")) if r.get("axis") == "tb"]
    tb += read(os.path.join(v2, "vg_relation_tb.jsonl"))
    mix = read(os.path.join(v2, "train_mix.jsonl"))
    rng = random.Random(a.seed)
    ids = {r["id"] for r in tb}
    pool = [r for r in mix if r["id"] not in ids]
    replay = rng.sample(pool, min(len(pool), int(a.replay * len(tb))))
    out = tb + replay
    rng.shuffle(out)
    assert len({r["id"] for r in out}) == len(out), "duplicate ids"
    with open(os.path.join(v2, "train_relation_tb.jsonl"), "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in out)
    print("train_relation_tb.jsonl: %d questions = %d above/below (%s) + %d replayed" % (
        len(out), len(tb), dict(Counter(r["source"] for r in tb)), len(replay)))
    print("above/below answers:", dict(Counter(r.get("answer_key") for r in tb)))


if __name__ == "__main__":
    main()
