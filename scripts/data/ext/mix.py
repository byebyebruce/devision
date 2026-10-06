"""Extension pack, last step: mix the converted sources with replay into the training file of the next round.

    uv run python scripts/data/ext/mix.py --root data [--quota objects365=80000 ...] [--seed 0]

Inputs: data/ext/<source>/train.jsonl of every source in QUOTAS that has been built (a missing source is
reported and skipped), and data/ext/replay/train.jsonl (scripts/data/v9_mix.py: v9b's 2,488 left/right questions +
replay of every earlier ability in REPLAY_QUOTAS' proportions; built by build_ext.sh).
Per source, pictures are split by a hash of the picture id (v2_common.in_dev): --dev-share of them (at most
--dev-cap questions per source) go to data/ext/dev_ext.jsonl, a monitoring set of the new skills, and never to
training; from the rest a deterministic sample of the source's quota is taken (all of it when the source is
smaller). Writes data/ext/train_mix.jsonl (shuffled), data/ext/dev_ext.jsonl and data/ext/MIX.json (per source:
available, taken, dev; replay; SHA256s; steps at micro batch 8).
"""
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import base_args, sha256  # noqa: E402

# questions per source in the training mix (docs/research/data-pack-sources-2026-10-06.md); a smaller source gives all.
# DVQA is built but left out: its yes / no questions are all about chart style ("Are the bars horizontal?"), none compare.
QUOTAS = {"objects365": 80000, "pixmo_points": 50000, "pixmo_count": 20000, "tallyqa": 40000, "clevr": 35000,
          "clevr_math": 10000, "superclevr": 10000, "iconqa": 28000, "visonlyqa": 30000, "figureqa": 30000,
          "mapqa": 7500, "spatialsense": 15000, "snli_ve": 30000, "vision_flan": 30000}


def read(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def split_source(rows, share, cap, rng):
    """(train candidates, dev) by picture; dev capped at `cap` questions."""
    from v2_common import in_dev  # pyright: ignore[reportMissingImports]
    dev = [r for r in rows if in_dev(r["image_id"], share)]
    train = [r for r in rows if not in_dev(r["image_id"], share)]
    rng.shuffle(dev)
    return train, dev[:cap]


def main(argv=None):
    p = base_args(__doc__)
    p.add_argument("--quota", nargs="*", default=[], help="override, e.g. objects365=60000")
    p.add_argument("--dev-share", type=float, default=0.02)
    p.add_argument("--dev-cap", type=int, default=300)
    a = p.parse_args(argv)
    quotas = dict(QUOTAS, **{k: int(v) for k, v in (q.split("=") for q in a.quota)})
    rng = random.Random(a.seed)
    out_dir = os.path.join(a.root, "ext")
    train, dev, report, missing = [], [], {}, []
    for source, quota in quotas.items():
        path = os.path.join(out_dir, source, "train.jsonl")
        if not os.path.exists(path):
            missing.append(source)
            continue
        rows = read(path)
        cand, d = split_source(rows, a.dev_share, a.dev_cap, rng)
        cand.sort(key=lambda r: r["id"])
        rng.shuffle(cand)
        take = cand[:quota]
        train += take
        dev += d
        report[source] = {"available": len(rows), "dev": len(d), "quota": quota, "taken": len(take),
                          "short": max(0, quota - len(cand)), "sha256": sha256(path)}
    replay_path = os.path.join(out_dir, "replay", "train.jsonl")
    replay = read(replay_path) if os.path.exists(replay_path) else []
    if not replay:
        missing.append("replay")
    new = len(train)
    train += replay
    ids = Counter(r["id"] for r in train + dev)
    dup = [i for i, n in ids.items() if n > 1]
    if dup:
        raise SystemExit("duplicate ids across sources: %s" % dup[:5])
    rng.shuffle(train)
    for name, rows in (("train_mix", train), ("dev_ext", dev)):
        with open(os.path.join(out_dir, name + ".jsonl"), "w") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    mix = {"questions": len(train), "new": new, "replay": len(replay), "dev_ext": len(dev),
           "new_share": round(new / max(1, len(train)), 4), "sources": report, "missing": missing,
           "types": dict(Counter(r["questions"]["q"]["type"] for r in train)),
           "replay_sha256": sha256(replay_path) if replay else None,
           "train_sha256": sha256(os.path.join(out_dir, "train_mix.jsonl")),
           "dev_sha256": sha256(os.path.join(out_dir, "dev_ext.jsonl")),
           "steps_at_micro_batch_8": -(-len(train) // 8), "seed": a.seed}
    with open(os.path.join(out_dir, "MIX.json"), "w") as f:
        json.dump(mix, f, indent=1)
    print(json.dumps(mix, indent=1))


if __name__ == "__main__":
    main()
