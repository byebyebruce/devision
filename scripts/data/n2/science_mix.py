"""r20 (if needed): a short science-heavy continuation from r19 (plan docs/experiments/2026-10-21-v04-plan.md, stage 4).

    uv run python scripts/data/n2/science_mix.py --root data [--replay 40000]

r19's ScienceQA dev fell while it trained (dev_sqa 0.804 -> 0.756 at step 20k; the step-28k snapshot lost the
"highlighted region on a map" questions: Thirteen Colonies 0.93 -> 0.57) -- real science questions were about 4% of
its mix. This mix takes, from data/n2/train_mix_fix.jsonl (the r19 rules, but drawn pictures held out only by identical
pixels -- r19's own mix had lost 2,291 of 4,948 ScienceQA questions to dHash collisions between maps):
- every ScienceQA / AI2D / TQA question x3 and every IconQA question once;
- the synthetic attract / repel and two-pair magnet questions (sqa-synth), but not the solution-concentration ones
  (on real beakers the model answered "neither" to all 45 dev questions at step 10k);
- --replay questions drawn at random from the rest of the r19 mix, so the other abilities keep being trained.
Writes <root>/n2/science_mix.jsonl and science_mix.MIX.json.
"""
import argparse
import json
import os
import random
from collections import Counter

SCIENCE = {"scienceqa": 3, "ai2d": 3, "tqa": 3, "iconqa": 1}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    p.add_argument("--replay", type=int, default=40000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--src", default="n2/train_mix_fix.jsonl",
                   help="the mix to draw from (train_mix_fix: drawn pictures held out by identical pixels only)")
    a = p.parse_args(argv)
    rng = random.Random(a.seed)
    seen, science, synth, rest = set(), [], [], []
    for line in open(os.path.join(a.root, a.src)):
        r = json.loads(line)
        if r["id"] in seen:          # the r19 mix already repeats ScienceQA / AI2D / TQA x2
            continue
        seen.add(r["id"])
        if r["source"] in SCIENCE:
            science += [r] * SCIENCE[r["source"]]
        elif r["source"] == "sqa-synth":
            if r["kind"] != "concentration":
                synth.append(r)
        else:
            rest.append(r)
    rng.shuffle(rest)
    rows = science + synth + rest[:a.replay]
    rng.shuffle(rows)
    out = os.path.join(a.root, "n2", "science_mix.jsonl")
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    m = {"built_by": "scripts/data/n2/science_mix.py", "args": vars(a), "rows": len(rows),
         "by_source": dict(Counter(r["source"] for r in rows).most_common())}
    with open(out.replace(".jsonl", ".MIX.json"), "w") as f:
        json.dump(m, f, indent=1)
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
