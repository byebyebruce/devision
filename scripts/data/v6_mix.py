"""Training data for round 6: left/right counterfactual pairs plus replay (kept out of the repo).

    uv run python scripts/data/v6_mix.py --root data [--pairs 12000] [--replay 24000]

Left/right is still at chance after every earlier fix; round 6 trains on pairs: a left/right question of
the COCO position / relation pools AND the same question on the mirrored picture with the answer swapped
(data/v6/flip_lr.jsonl, scripts/data/v6_flip.py). Only both together rule out answering from names or
option order. --pairs pairs are drawn (seeded, balanced between position and relation as the pools are),
plus --replay questions from data/v3/train_mix.jsonl without left/right ones, so other abilities are kept:
first a minimum per ability (QUOTAS: counting and colour, which round 5 lost; above/below relations; the
reasoning and diagram sets round 5 learned), then the rest at random.
One pass. Pairs: relation questions first, at most PER_PICTURE per picture; hints longer than MAX_TOKENS
are cut. Also writes data/v6/dev_lr_pairs.jsonl: the dev left/right questions and their mirrors, to
watch during training (accuracy on both halves of a pair is what has to rise).
Writes data/v6/train_flip_mix.jsonl.
"""
import argparse
import json
import os
import random
from collections import Counter


def read(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


QUOTAS = [("count", 3000), ("color", 3000), ("relation_tb", 3000), ("aokvqa", 2000), ("scienceqa", 1000),
          ("ai2d", 1000), ("tqa", 1000), ("size", 1500), ("gqa", 1500), ("position_tb", 1000), ("material", 300)]
PER_PICTURE = 3          # left/right questions per original picture (so 6 with their mirrors)
MAX_TOKENS = 150         # longest text sequence; longer ones (ScienceQA hints) get their hint cut


def group_of(r):
    if r.get("kind") in ("count", "color", "size", "gqa", "material"):
        return r["kind"]
    if r.get("kind") in ("relation", "position") and r.get("axis") == "tb":
        return r["kind"] + "_tb"
    return r.get("source", "")


def pick_pairs(originals, n, rng):
    """`n` originals, relation questions first (the two-object task that is really missing), at most
    PER_PICTURE per picture, then position questions to fill up."""
    per_pic: Counter = Counter()
    out = []
    for kind in ("relation", "position"):
        rows = [r for r in originals if r["kind"] == kind]
        rng.shuffle(rows)
        for r in rows:
            if len(out) >= n:
                return out
            if per_pic[r["image_id"]] < PER_PICTURE:
                per_pic[r["image_id"]] += 1
                out.append(r)
    return out


def shorten_hints(rows):
    """Cut the hint of questions whose text would exceed MAX_TOKENS (a few ScienceQA hints drive memory peaks)."""
    from transformers import AutoTokenizer
    from devision.infer import question_item
    from devision.model import ModelConfig
    tok = AutoTokenizer.from_pretrained("convaiinnovations/laya", subfolder="tokenizer")   # deVision's tokenizer
    cfg = ModelConfig()                                                                   # same token budgets
    length = lambda r, hint: len(question_item(tok, cfg, hint, r["questions"]["q"])["ids"])
    cut = 0
    for r in rows:
        hint = r.get("state_text", "")
        if not hint or length(r, hint) <= MAX_TOKENS:
            continue
        words = hint.split()
        lo, hi = 0, len(words)          # longest prefix of the hint that fits
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if length(r, " ".join(words[:mid])) <= MAX_TOKENS:
                lo = mid
            else:
                hi = mid - 1
        r["state_text"] = " ".join(words[:lo]) + " ..."
        r["state_text_cut"] = True
        cut += 1
    return cut


def is_lr(r):
    return (r.get("kind") in ("position", "relation") and r.get("axis") == "lr") or r.get("kind") == "spatial"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    p.add_argument("--pairs", type=int, default=12000)
    p.add_argument("--replay", type=int, default=24000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    rng = random.Random(a.seed)
    flips = {r["id"][len("flip:"):]: r for r in read(os.path.join(a.root, "v6", "flip_lr.jsonl"))}
    originals = [r for f in ("coco_position", "coco_relation") for r in read(os.path.join(a.root, "v2", f + ".jsonl"))
                 if r.get("axis") == "lr" and r["id"] in flips]
    picked = pick_pairs(originals, a.pairs, rng)
    pairs = [x for r in picked for x in (r, flips[r["id"]])]
    have = {r["id"] for r in pairs}
    pool = [r for r in read(os.path.join(a.root, "v3", "train_mix.jsonl")) if not is_lr(r) and r["id"] not in have]
    # minimum replay per ability first (round 5 lost 7.4 points on counting, 4.7 on colour, and is short of
    # round 4's above/below), then the rest at random
    replay, used = [], set()
    for name, n in QUOTAS:
        group = [r for r in pool if group_of(r) == name]
        rng.shuffle(group)
        take = group[:n]
        replay += take
        used |= {r["id"] for r in take}
    rest = [r for r in pool if r["id"] not in used]
    replay += rng.sample(rest, max(0, min(a.replay - len(replay), len(rest))))
    cut = shorten_hints(replay)
    out = pairs + replay
    rng.shuffle(out)
    assert len({r["id"] for r in out}) == len(out)
    with open(os.path.join(a.root, "v6", "train_flip_mix.jsonl"), "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in out)
    dev = [r for f in ("dev_coco_position", "dev_coco_relation") for r in read(os.path.join(a.root, "v2", f + ".jsonl"))
           if r.get("axis") == "lr"] + read(os.path.join(a.root, "v6", "dev_flip_lr.jsonl"))
    with open(os.path.join(a.root, "v6", "dev_lr_pairs.jsonl"), "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in dev)
    print("hints cut to fit %d tokens: %d" % (MAX_TOKENS, cut))
    print("train_flip_mix.jsonl: %d = %d pairs (%s) + %d replayed (%s)" % (
        len(out), len(picked), dict(Counter(r["kind"] for r in picked)), len(replay),
        dict(Counter(group_of(r) for r in replay).most_common(10))))
    print("dev_lr_pairs.jsonl: %d (%d mirrored)" % (len(dev), sum(r.get("flipped", False) for r in dev)))


if __name__ == "__main__":
    main()
