"""Training data for round 5, continuing round 3 (kept out of the repo):

    uv run python scripts/data/v5_mix.py --root data

Round 3 has already trained one pass over data-v2's mix, so round 5 does not repeat it: it takes every
question of data/v3/train_mix.jsonl that is not in data/v2/train_mix.jsonl (the reasoning and diagram
sets and the above/below relation questions) plus --replay questions drawn at random from data-v2's mix,
so the abilities round 3 has are kept. Round 4 showed that passing over the same data again makes the
model over-confident. Writes data/v3/train_cont.jsonl.
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
    p.add_argument("--replay", type=int, default=30000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    v2 = read(os.path.join(a.root, "v2", "train_mix.jsonl"))
    v3 = read(os.path.join(a.root, "v3", "train_mix.jsonl"))
    seen = {r["id"] for r in v2}
    new = [r for r in v3 if r["id"] not in seen]
    rng = random.Random(a.seed)
    out = new + rng.sample(v2, min(a.replay, len(v2)))
    rng.shuffle(out)
    assert len({r["id"] for r in out}) == len(out)
    with open(os.path.join(a.root, "v3", "train_cont.jsonl"), "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in out)
    print("train_cont.jsonl: %d questions = %d new + %d replayed; new by source %s" % (
        len(out), len(new), len(out) - len(new), dict(Counter(r.get("source") for r in new))))


if __name__ == "__main__":
    main()
