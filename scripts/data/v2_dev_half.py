"""Half of data/v2/dev_mix.jsonl, for cheaper evaluation during training (kept out of the repo):

    uv run python scripts/data/v2_dev_half.py --root data

Per source, a seeded random half of the pictures with all their questions, so every source keeps its
share and no picture is split. Writes data/v2/dev_mix_half.jsonl and prints per-source counts and the
answer balance before and after.
"""
import argparse
import json
import os
import random
from collections import Counter, defaultdict


def gold_answer(s):
    q = next(iter(s["gold"].values()))["probabilities"]
    return max(q, key=q.get)


def half(rows, seed=0):
    by_source = defaultdict(lambda: defaultdict(list))
    for s in rows:
        by_source[s.get("source", "")][s["image_id"]].append(s)
    rng = random.Random(seed)
    out = []
    for src in sorted(by_source):
        pics = sorted(by_source[src])
        rng.shuffle(pics)
        target, n = sum(len(v) for v in by_source[src].values()) / 2, 0
        for p in pics:
            if n >= target:
                break
            out.extend(by_source[src][p])
            n += len(by_source[src][p])
    keep = {s["id"] for s in out}
    return [s for s in rows if s["id"] in keep]   # original order


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--root", default="data")
    a.add_argument("--seed", type=int, default=0)
    args = a.parse_args(argv)
    src = os.path.join(args.root, "v2", "dev_mix.jsonl")
    rows = [json.loads(l) for l in open(src)]
    sub = half(rows, args.seed)
    with open(os.path.join(args.root, "v2", "dev_mix_half.jsonl"), "w") as f:
        f.writelines(json.dumps(s) + "\n" for s in sub)
    for name, rs in (("dev_mix", rows), ("dev_mix_half", sub)):
        by = defaultdict(list)
        for s in rs:
            by[s.get("source", "")].append(s)
        print(name, len(rs), {k: (len(v), dict(Counter(gold_answer(s) for s in v).most_common(2)))
                              for k, v in sorted(by.items())})


if __name__ == "__main__":
    main()
