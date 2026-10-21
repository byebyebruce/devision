"""r21: a short continuation of r20 that softens the hard 0 / 1 labels of existence and size questions
(plan docs/experiments/2026-10-21-v04-plan.md, stage 4).

    uv run python scripts/data/n2/soft_mix.py --root data [--eps 0.1] [--exist 40000] [--replay 60000]

r20 (temperatures refitted on data/v5/dev_mix) passes the frozen rules 1-3 but not 4: POPE NLL 0.372 -> 0.454 and
size 0.350 -> 0.440 (runs/v04-interp/a1.0/compare/accept_v04.md). Both are judged from COCO existence / size questions
with one-hot golds (99,042 and 4,331 in the mix), which push those outputs to the extremes; one yes / no temperature
cannot fix existence without hurting VQAv2 (offline refit). This mix, drawn from data/n2/train_mix_fix.jsonl:
- --exist existence questions and every size question with the gold softened: the right answer gets 1 - eps, the
  others share eps (two options: 0.9 / 0.1);
- --replay questions drawn at random from the rest (unchanged), so the other abilities keep their training.
Writes <root>/n2/soft_mix.jsonl and soft_mix.MIX.json.
"""
import argparse
import json
import os
import random
from collections import Counter

SOFT = ("coco-exist", "coco-size")


def soften(r: dict, eps: float) -> dict:
    r = json.loads(json.dumps(r))
    g = r["gold"]["q"]["probabilities"]
    best = max(g, key=lambda o: g[o])
    others = [o for o in g if o != best]
    for o in g:
        g[o] = 1.0 - eps if o == best else eps / len(others)
    return r


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    p.add_argument("--eps", type=float, default=0.1)
    p.add_argument("--exist", type=int, default=40000)
    p.add_argument("--replay", type=int, default=60000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    rng = random.Random(a.seed)
    seen, exist, size, rest = set(), [], [], []
    for line in open(os.path.join(a.root, "n2", "train_mix_fix.jsonl")):
        r = json.loads(line)
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        hard = max(r["gold"]["q"]["probabilities"].values()) == 1.0
        if r["source"] == "coco-exist" and hard:
            exist.append(r)
        elif r["source"] == "coco-size" and hard:
            size.append(r)
        else:
            rest.append(r)
    rng.shuffle(exist)
    rng.shuffle(rest)
    rows = [soften(r, a.eps) for r in exist[:a.exist] + size] + rest[:a.replay]
    rng.shuffle(rows)
    out = os.path.join(a.root, "n2", "soft_mix.jsonl")
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    m = {"built_by": "scripts/data/n2/soft_mix.py", "args": vars(a), "rows": len(rows),
         "softened": {"coco-exist": min(a.exist, len(exist)), "coco-size": len(size)},
         "by_source": dict(Counter(r["source"] for r in rows).most_common())}
    with open(out.replace(".jsonl", ".MIX.json"), "w") as f:
        json.dump(m, f, indent=1)
    print(json.dumps({k: m[k] for k in ("rows", "softened")}))


if __name__ == "__main__":
    main()
