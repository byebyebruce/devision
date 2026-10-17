"""exp33 decision questions (docs/experiments/2026-10-17-exp33-decision-protect-plan.md): 20,000 questions of v0.3's
training mix (data/r14b/train.jsonl, 81,875; every block is an old task for v0.3), drawn per block in proportion
(largest remainder), fixed seed, no selection by any model output. r14b keeps its isolation from every monitoring /
evaluation picture.

    uv run python scripts/exp33/build.py --root data
"""
import argparse
import hashlib
import json
import os
import random
from collections import defaultdict


def build(root: str, n: int = 20000, seed: int = 33) -> dict:
    rows = [json.loads(x) for x in open(os.path.join(root, "r14b", "train.jsonl"))]
    by = defaultdict(list)
    for r in rows:
        by[r["block"]].append(r)
    exact = {b: n * len(v) / len(rows) for b, v in by.items()}
    take = {b: int(x) for b, x in exact.items()}
    for b in sorted(exact, key=lambda b: exact[b] - take[b], reverse=True)[: n - sum(take.values())]:
        take[b] += 1
    rng = random.Random(seed)
    chosen = [r for b in sorted(by) for r in rng.sample(by[b], take[b])]
    rng.shuffle(chosen)
    out = os.path.join(root, "exp33")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "decisions.jsonl")
    with open(path, "w") as f:
        for r in chosen:
            f.write(json.dumps(r) + "\n")
    m = {"source": "data/r14b/train.jsonl", "source_sha256": hashlib.sha256(open(os.path.join(root, "r14b", "train.jsonl"), "rb").read()).hexdigest(),
         "seed": seed, "rows": len(chosen), "questions": sum(len(r["questions"]) for r in chosen), "by_block": dict(sorted(take.items())),
         "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}
    json.dump(m, open(os.path.join(out, "MANIFEST.json"), "w"), indent=1)
    return m


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    print(json.dumps(build(p.parse_args().root), indent=1))
