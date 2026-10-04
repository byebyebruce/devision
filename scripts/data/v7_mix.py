"""Training data for round 7: ScienceQA recovered by data-v4, plus replay (kept as data, built here):

    uv run python scripts/data/v7_mix.py --root data

data-v4 matches diagram pictures against the evaluation pictures by identical pixels and so keeps 2,095
ScienceQA training questions data-v3 had dropped (mostly state capitals, geography, economics -- the
categories where round 6 trails laya-vision most). Round 7 continues round 6 on: every data-v4 ScienceQA
training question (the recovered ones and the ones round 5 already saw), the TQA questions data-v4 adds,
a minimum per ability from data/v4/train_mix.jsonl (as in round 6), and --pairs left/right pairs from
round 6's mix so its gains are kept. One pass. Writes data/v4/train_sqa_mix.jsonl.
"""
import argparse
import json
import os
import random
from collections import Counter

from v6_mix import MAX_TOKENS, group_of, is_lr, shorten_hints

QUOTAS = [("count", 2000), ("color", 2000), ("relation_tb", 2000), ("aokvqa", 2500), ("ai2d", 1500), ("tqa", 1500),
          ("size", 1000), ("gqa", 1000), ("position_tb", 800), ("vqav2-yesno", 1500), ("coco-exist", 1200)]


def read(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    p.add_argument("--pairs", type=int, default=3000, help="left/right pairs (question + mirror) from round 6's mix")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    rng = random.Random(a.seed)
    v3_tqa = {r["id"] for r in read(os.path.join(a.root, "v3", "tqa.jsonl"))}
    sqa = read(os.path.join(a.root, "v4", "scienceqa.jsonl"))
    tqa_new = [r for r in read(os.path.join(a.root, "v4", "tqa.jsonl")) if r["id"] not in v3_tqa]
    have = {r["id"] for r in sqa + tqa_new}
    pool = [r for r in read(os.path.join(a.root, "v4", "train_mix.jsonl"))
            if not is_lr(r) and r["id"] not in have and r.get("source") != "scienceqa"]
    replay = []
    for name, n in QUOTAS:
        group = [r for r in pool if group_of(r) == name]
        rng.shuffle(group)
        replay += group[:n]
    v6 = read(os.path.join(a.root, "v6", "train_flip_mix.jsonl"))
    by_id = {r["id"]: r for r in v6}
    originals = [r for r in v6 if r.get("axis") == "lr" and r.get("kind") in ("position", "relation")
                 and not r["id"].startswith("flip:") and "flip:" + r["id"] in by_id]
    pairs = [x for r in rng.sample(originals, min(a.pairs, len(originals))) for x in (r, by_id["flip:" + r["id"]])]
    cut = shorten_hints(sqa + replay)   # long ScienceQA hints drive memory peaks (round 5)
    out = sqa + tqa_new + replay + pairs
    rng.shuffle(out)
    assert len({r["id"] for r in out}) == len(out), "duplicate ids"
    with open(os.path.join(a.root, "v4", "train_sqa_mix.jsonl"), "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in out)
    print("hints cut to fit %d tokens: %d" % (MAX_TOKENS, cut))
    print("train_sqa_mix.jsonl: %d = %d ScienceQA + %d new TQA + %d replayed (%s) + %d left/right pair questions" % (
        len(out), len(sqa), len(tqa_new), len(replay), dict(Counter(group_of(r) for r in replay)), len(pairs)))


if __name__ == "__main__":
    main()
