"""Training data shared by the three round-13 screens (plan revision 2, step 2): about 16,000 questions =
counterfactual synthetic 50% (whole scene families) + GQA real-photo questions 30% + replay 20%.

    uv run python scripts/r13/screen_mix.py --root data --out data/r13/screen

Replay is drawn from v12's own training mix (data/ext/train_mix.jsonl: the extension pack + every earlier ability),
whole pictures. Every sample gets a `group` (scene family, or the picture for GQA / replay) so paired questions stay
in one micro batch. Also writes dev_bind_cf_watch.jsonl: 150 families of dev_bind_cf for the periodic curve
(the full dev set is evaluated after training).
"""
import argparse
import json
import os
import random
from collections import defaultdict


def rows(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def take_groups(samples, key, target, rng):
    """Whole groups, in random order, until `target` questions."""
    groups = defaultdict(list)
    for s in samples:
        groups[key(s)].append(s)
    names = list(groups)
    rng.shuffle(names)
    out, n = [], 0
    for g in names:
        if n >= target:
            break
        out += groups[g]
        n += sum(len(s["questions"]) for s in groups[g])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    p.add_argument("--out", default="data/r13/screen")
    p.add_argument("--questions", type=int, default=16000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    rng = random.Random(a.seed)
    q = a.questions
    synth = take_groups(rows(os.path.join(a.root, "r13/synth/train.jsonl")), lambda s: s["group"], q * 0.5, rng)
    gqa = take_groups(rows(os.path.join(a.root, "r13/gqa/train.jsonl")), lambda s: s.get("group", s["image_id"]), q * 0.3, rng)
    replay = take_groups(rows(os.path.join(a.root, "ext/train_mix.jsonl")), lambda s: s["image_id"], q * 0.2, rng)
    for s in gqa + replay:
        s.setdefault("group", s["image_id"])
    mix = synth + gqa + replay
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "train.jsonl"), "w") as f:
        for s in mix:
            f.write(json.dumps(s) + "\n")
    dev = rows(os.path.join(a.root, "r13/synth/dev_bind_cf.jsonl"))
    fams = sorted({s["group"] for s in dev})
    keep = set(random.Random(1).sample(fams, 150))
    watch = [s for s in dev if s["group"] in keep]
    with open(os.path.join(a.out, "dev_bind_cf_watch.jsonl"), "w") as f:
        for s in watch:
            f.write(json.dumps(s) + "\n")
    count = lambda ss: sum(len(s["questions"]) for s in ss)
    manifest = {"questions": count(mix), "synthetic": count(synth), "gqa": count(gqa), "replay": count(replay),
                "pictures": len(mix), "watch_questions": count(watch), "seed": a.seed}
    json.dump(manifest, open(os.path.join(a.out, "MANIFEST.json"), "w"), indent=1)
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
