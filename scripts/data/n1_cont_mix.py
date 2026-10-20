"""A continuation mix: new questions plus a random sample of an earlier training mix as replay.

    uv run python scripts/data/n1_cont_mix.py --root data --new n1/exist_train.jsonl --replay n1/train_mix.jsonl \\
        --replay-count 120000 --out n1/r17_mix.jsonl

Replay rows whose id is also among the new ones are skipped. Paths are under --root. Writes the mix and
<out without .jsonl>.MIX.json (counts per part and per source).
"""
import argparse
import json
import os
import random
from collections import Counter


def rows(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def main(argv=None) -> None:
    from prepare import _write_jsonl

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    p.add_argument("--new", nargs="+", required=True)
    p.add_argument("--replay", required=True)
    p.add_argument("--replay-count", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    rng = random.Random(a.seed)
    new = [s for f in a.new for s in rows(os.path.join(a.root, f))]
    ids = {s["id"] for s in new}
    pool = [s for s in rows(os.path.join(a.root, a.replay)) if s["id"] not in ids]
    rng.shuffle(pool)
    replay = pool[:a.replay_count]
    mix = new + replay
    rng.shuffle(mix)
    out = os.path.join(a.root, a.out)
    info = {"train": _write_jsonl(out, mix), "new": len(new), "replay": len(replay), "new_files": a.new,
            "replay_file": a.replay, "by_source": dict(Counter(s["source"] for s in mix).most_common())}
    with open(out[:-len(".jsonl")] + ".MIX.json", "w") as f:
        json.dump(info, f, indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
