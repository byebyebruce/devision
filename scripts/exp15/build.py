"""Quick checks on the exp14 base (docs/experiments/2026-10-12-exp15-steps.md): the same 12,000-question replay subset
exp14-pixmo used (75% of every exp14-base block) plus 4,000 questions of ONE new block, so each arm compares with
exp14-base (pure replay, same seed / steps / learning rates) on one variable.

    uv run python scripts/exp15/build.py --root data --block bind     # -> data/exp15/train_bind.jsonl

Blocks:
  bind   real-photo binding: GQA binding 2,000 + VisMin 2,000 (whole pictures / original-edit pairs; round-13 train
         splits, pictures isolated from every dev / test set in round 13)
  count  0-10 counting from the v12 pool's real-photo sources (TallyQA, Objects365, PixMo-Count, VQAv2 choice) not in
         the replay, a third per answer bucket 0-2 / 3-5 / 6-10, half every candidate 0..10, half a run of 3-5
  sci    CoSyn closed questions (data/ext/cosyn/train.jsonl, built by scripts/data/pub/cosyn.py)
"""
import argparse
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "r13b"))
import build as r13b  # noqa: E402


def block_bind(root, rng):
    R = lambda rel: r13b.rows(os.path.join(root, rel))
    return (r13b.take(R("r13/gqa/train.jsonl"), lambda s: s["group"], 2000, rng)
            + r13b.take(R("r13/vismin/train.jsonl"), lambda s: s["group"], 2000, rng))


def narrow_run(gold, rng):
    k = rng.choice([3, 4, 5])
    pos = rng.randrange(k)
    lo = max(0, min(gold - pos, 10 - (k - 1)))
    return [str(n) for n in range(lo, lo + k)]


def block_count(root, rng, used):
    """0-10 counting from the v12 pool's real-photo sources not already in the replay, a third per answer bucket
    (0-2 / 3-5 / 6-10), half with every candidate 0..10 and half with a run of 3-5 holding the gold anywhere."""
    pool = r13b.rows(os.path.join(root, "ext/train_mix.jsonl"))
    first = lambda s: next(iter(zip(s["questions"].values(), s["gold"].values())))
    cand = [s for s in pool if s["id"] not in used and s.get("source") in r13b.OLD_COUNT and r13b.is_count(*first(s))
            and max(first(s)[1]["probabilities"].values()) == 1.0]
    by = {}
    for s in cand:
        by.setdefault(r13b.bucket(s), []).append(s)
    out = []
    for b in ("0-2", "3-5", "6-10"):
        for i, s in enumerate(r13b.sample_rows(by[b], 1334 if b != "6-10" else 1332, rng)):
            q, g = first(s)
            n = r13b.number(r13b.top(g))
            opts = [str(m) for m in range(11)] if i % 2 == 0 else narrow_run(n, rng)
            rng.shuffle(opts)
            k = next(iter(s["questions"]))
            out.append(dict(s, id=s["id"] + ":exp15", candidates="0..10" if len(opts) == 11 else "run",
                            questions={k: dict(q, criteria={o: None for o in opts})},
                            gold={k: {"probabilities": {o: float(o == str(n)) for o in opts}}}))
    return [r13b.as_picture_group(s) for s in out]


def block_sci(root, rng, used):
    """CoSyn closed questions (scripts/data/pub/cosyn.py -> data/ext/cosyn/train.jsonl), whole figures."""
    rows = r13b.rows(os.path.join(root, "ext/cosyn/train.jsonl"))
    return r13b.take(rows, lambda s: s["image_id"], 4000, rng)


BLOCKS = {"bind": lambda root, rng, used: block_bind(root, rng), "count": block_count, "sci": block_sci}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    p.add_argument("--block", required=True, choices=sorted(BLOCKS))
    a = p.parse_args()
    rng = random.Random(15)
    base = [s for s in r13b.rows(os.path.join(a.root, "exp14/train_pixmo.jsonl")) if s["block"] != "pixmo"]
    used = {s["id"].split(":rep")[0] for s in base}
    new = [dict(s, block=a.block) for s in BLOCKS[a.block](a.root, rng, used)]
    mix = base + new
    assert len({s["id"] for s in mix}) == len(mix), "duplicate ids"
    random.Random(1).shuffle(mix)
    os.makedirs(os.path.join(a.root, "exp15"), exist_ok=True)
    out = os.path.join(a.root, "exp15", "train_%s.jsonl" % a.block)
    r13b.write(out, mix)
    m = {"option_rules_new": r13b.option_rules(new), "replay": r13b.nq(base), "new": r13b.nq(new), "total": r13b.nq(mix),
         "new_by_source": dict(Counter(s["source"] for s in new for _ in s["questions"])),
         "new_pictures": len({s["image_id"] for s in new}), "blocks": dict(Counter(s["block"] for s in mix))}
    json.dump(m, open(out.replace(".jsonl", ".MANIFEST.json"), "w"), indent=1)
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
