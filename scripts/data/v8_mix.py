"""Training data for round 8: data-v5's VSR and Visual7W telling, plus round 7's mix of ScienceQA, replay and
left/right pairs (kept as data, built here):

    uv run python scripts/data/v8_mix.py --root data

- VSR (--vsr) and Visual7W telling (--v7w) questions from data/v5/{vsr,v7w}.jsonl (fewer if the pool is smaller);
- every data-v4 ScienceQA training question and the TQA questions data-v4 added (as round 7), each once;
- replay with round 7's minimum per ability plus counting +1,000 and GQA +500 (QUOTAS);
- --pairs left/right pairs from round 6's mix.
Replay never repeats a question already in the mix and never uses a picture of data-v5's test or dev sets.
Writes data/v5/train_mix.jsonl, data/v5/dev_mix.jsonl (data-v4's dev_mix + dev_vsr + dev_v7w),
data/v5/image_identity.json (data-v2's VG -> COCO map plus data-v5's VG pictures) and data/v5/MIX.json (counts per part and ability, unique questions and pictures, questions per picture).
"""
import argparse
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert import picture  # noqa: E402
from v6_mix import MAX_TOKENS, group_of, is_lr, shorten_hints  # noqa: E402
from v7_mix import QUOTAS as V7_QUOTAS  # noqa: E402

EXTRA = {"count": 1000, "gqa": 500}
QUOTAS = [(name, n + EXTRA.get(name, 0)) for name, n in V7_QUOTAS]


def read(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def replay_pool(rows, have, held, v2c):
    """Replay candidates: not a left/right question (pairs come separately), not a ScienceQA question, not a
    question already in the mix, and not on a picture of data-v5's test or dev sets."""
    return [r for r in rows if not is_lr(r) and r["id"] not in have and r.get("source") != "scienceqa"
            and picture(r["image_id"], v2c) not in held]


def pick_replay(pool, quotas, rng):
    replay = []
    for name, n in quotas:
        group = [r for r in pool if group_of(r) == name]
        rng.shuffle(group)
        replay += group[:n]
    return replay


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    p.add_argument("--vsr", type=int, default=6500)
    p.add_argument("--v7w", type=int, default=7500)
    p.add_argument("--pairs", type=int, default=3000, help="left/right pairs (question + mirror) from round 6's mix")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    rng = random.Random(a.seed)
    v5 = os.path.join(a.root, "v5")
    from fetch import vg_to_coco
    v2c = vg_to_coco(a.root)
    held = {picture(r["image_id"], v2c) for n in ("test_vsr", "test_v7w", "dev_vsr", "dev_v7w")
            for r in read(os.path.join(v5, n + ".jsonl"))}

    vsr, v7w = read(os.path.join(v5, "vsr.jsonl")), read(os.path.join(v5, "v7w.jsonl"))
    rng.shuffle(vsr)
    rng.shuffle(v7w)
    vsr, v7w = vsr[:a.vsr], v7w[:a.v7w]
    v3_tqa = {r["id"] for r in read(os.path.join(a.root, "v3", "tqa.jsonl"))}
    sqa = read(os.path.join(a.root, "v4", "scienceqa.jsonl"))
    tqa_new = [r for r in read(os.path.join(a.root, "v4", "tqa.jsonl")) if r["id"] not in v3_tqa]
    have = {r["id"] for r in vsr + v7w + sqa + tqa_new}
    pool = replay_pool(read(os.path.join(a.root, "v4", "train_mix.jsonl")), have, held, v2c)
    replay = pick_replay(pool, QUOTAS, rng)
    v6 = read(os.path.join(a.root, "v6", "train_flip_mix.jsonl"))
    by_id = {r["id"]: r for r in v6}
    originals = [r for r in v6 if r.get("axis") == "lr" and r.get("kind") in ("position", "relation")
                 and not r["id"].startswith("flip:") and "flip:" + r["id"] in by_id
                 and picture(r["image_id"], v2c) not in held]
    pairs = [x for r in rng.sample(originals, min(a.pairs, len(originals))) for x in (r, by_id["flip:" + r["id"]])]
    cut = shorten_hints(sqa + replay)   # long ScienceQA hints drive memory peaks (round 5)
    out = vsr + v7w + sqa + tqa_new + replay + pairs
    rng.shuffle(out)
    assert len({r["id"] for r in out}) == len(out), "duplicate ids"
    assert not {picture(r["image_id"], v2c) for r in out} & held, "a test / dev picture in training"
    with open(os.path.join(v5, "train_mix.jsonl"), "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in out)
    dev = read(os.path.join(a.root, "v4", "dev_mix.jsonl")) + read(os.path.join(v5, "dev_vsr.jsonl")) \
        + read(os.path.join(v5, "dev_v7w.jsonl"))
    assert len({r["id"] for r in dev}) == len(dev), "duplicate dev ids"
    with open(os.path.join(v5, "dev_mix.jsonl"), "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in dev)

    # picture identity for the evaluation roles: data-v2's map plus the Visual Genome ids data-v5 uses
    identity = json.load(open(os.path.join(a.root, "v2", "image_identity.json")))
    for n in ("vsr", "v7w", "test_vsr", "test_v7w", "dev_vsr", "dev_v7w"):
        for r in read(os.path.join(v5, n + ".jsonl")):
            if r["image_id"] in v2c:
                identity[r["image_id"]] = v2c[r["image_id"]]
    with open(os.path.join(v5, "image_identity.json"), "w") as f:
        json.dump(identity, f)

    got = Counter(group_of(r) for r in replay)
    short = {name: (got[name], n) for name, n in QUOTAS if got[name] < n}
    per_pic = Counter(picture(r["image_id"], v2c) for r in out)
    report = {"questions": len(out), "unique_ids": len({r["id"] for r in out}), "pictures": len(per_pic),
              "max_questions_per_picture": max(per_pic.values()),
              "parts": {"vsr": len(vsr), "v7w": len(v7w), "scienceqa": len(sqa), "tqa_new": len(tqa_new),
                        "replay": len(replay), "left_right_pair_questions": len(pairs)},
              "replay_by_ability": dict(got), "replay_quotas": dict(QUOTAS), "replay_short_of_quota": short,
              "hints_cut": cut, "max_hint_tokens": MAX_TOKENS, "dev_mix": len(dev),
              "steps_at_micro_batch_8": -(-len(out) // 8)}
    with open(os.path.join(v5, "MIX.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
