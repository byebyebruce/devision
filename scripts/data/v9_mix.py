"""Training data for round 9 (critic/plan-round9-v9.md): real left/right pairs plus replay that protects every
earlier ability, for the short mixed screen (S-mix) and the round itself:

    uv run python scripts/data/v9_mix.py --root data --out data/v9/smix13 --position 600 --rel-choice 300 \\
        --rel-noul 300 --replay 4800
    uv run python scripts/data/v9_mix.py --root data --out data/v9/train13 --position 2000 --rel-choice 1000 \\
        --rel-noul 1000 --replay 16000

- Left/right pairs: a data-v2 COCO question plus its mirrored copy from data/v6/flip_lr.jsonl (answer swapped),
  per subtask -- position (choice), relation choice, relation yes/no. Pairs no earlier mix (rounds 6-8) used
  come first, then pairs those rounds trained on (only 349 relation pairs are unused; review R9-V9-01).
- Replay: REPLAY_QUOTAS scaled to --replay (largest remainder), every ability round 8 had, including the
  sources round 8 added outside its replay (VSR, ScienceQA, Visual7W; review R9-V9-02). An ability whose pool
  cannot fill its share stops the build.
- No picture of any evaluation, monitoring or calibration set (v8_mix.held_pictures: test_ / dev / bench_ /
  val / pope / align_val files, laya-vision sets, data-v5 test and dev, dev_lr_pairs included), by picture identity.
Writes <out>/train.jsonl and <out>/MIX.json (counts per part, subtask and ability, unused vs reused pairs,
unique questions and pictures, the overlap check, steps at micro batch 8).
"""
import argparse
import json
import os
import random
import sys
from collections import Counter
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert import picture  # noqa: E402
from v6_mix import group_of, shorten_hints  # noqa: E402
from v8_mix import held_pictures, read, replay_pool  # noqa: E402

# per 16,000 replay questions (the plan's table); round 8's counts in the comments
REPLAY_QUOTAS = [("vsr", 2400), ("scienceqa", 2100), ("v7w", 1500),          # 6,411 / 4,948 / 7,500
                 ("count", 1800), ("coco-exist", 1200), ("aokvqa", 1100),       # 3,000 / 1,200 / 2,500
                 ("gqa", 1000), ("color", 900), ("relation_tb", 900),           # 1,500 / 2,000 / 2,000
                 ("vqav2-yesno", 800), ("ai2d", 700), ("tqa", 700),            # 1,500 / 1,500 / 1,500
                 ("size", 500), ("position_tb", 400)]                           # 1,000 / 800
EARLIER_MIXES = ("v6/train_flip_mix.jsonl", "v4/train_sqa_mix.jsonl", "v5/train_mix.jsonl")
SUBTASKS = (("position", "choice"), ("relation", "choice"), ("relation", "noul"))


def scale(quotas: Sequence[Tuple[str, int]], total: int) -> List[Tuple[str, int]]:
    """`quotas` scaled to sum exactly to `total`, rounding by largest remainder, order kept."""
    base = sum(n for _, n in quotas)
    exact = [n * total / base for _, n in quotas]
    out = [int(x) for x in exact]
    for i in sorted(range(len(quotas)), key=lambda i: exact[i] - out[i], reverse=True)[:total - sum(out)]:
        out[i] += 1
    return [(name, n) for (name, _), n in zip(quotas, out)]


def subtask(r: dict) -> Tuple[str, str]:
    return r["kind"], r["questions"]["q"]["type"]


def origin(image_id: str) -> str:
    return image_id.split(":flip")[0]


def pick_pairs(flips: List[dict], originals: Dict[str, dict], wanted: Dict[Tuple[str, str], int],
               used: set, blocked: set, v2c: Dict[str, str], rng: random.Random):
    """(rows, report): per subtask, unused pairs first, then reused ones; never a blocked picture."""
    rows, report = [], {}
    pool = [f for f in flips if f["id"][5:] in originals and picture(origin(f["image_id"]), v2c) not in blocked]
    for key, n in wanted.items():
        fresh = [f for f in pool if subtask(f) == key and f["id"] not in used]
        old = [f for f in pool if subtask(f) == key and f["id"] in used]
        rng.shuffle(fresh)
        rng.shuffle(old)
        take = (fresh + old)[:n]
        if len(take) < n:
            raise SystemExit("not enough %s/%s pairs: %d of %d" % (*key, len(take), n))
        rows += [x for f in take for x in (originals[f["id"][5:]], f)]
        report["%s/%s" % key] = {"pairs": n, "unused": min(n, len(fresh)), "reused": max(0, n - len(fresh)),
                                 "pictures": len({picture(origin(f["image_id"]), v2c) for f in take})}
    return rows, report


def pick_replay(pools: Dict[str, List[dict]], quotas: Sequence[Tuple[str, int]], rng: random.Random):
    """(replay, count per ability); stops the build when an ability's pool cannot fill its share."""
    out, got, short = [], Counter(), {}
    for name, n in quotas:
        group = list(pools.get(name, []))
        if len(group) < n:
            short[name] = (len(group), n)
            continue
        rng.shuffle(group)
        out += group[:n]
        got[name] = n
    if short:
        raise SystemExit("replay below its share (have, wanted): %s" % short)
    return out, got


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    p.add_argument("--out", required=True)
    p.add_argument("--position", type=int, required=True, help="position pairs (question + mirror)")
    p.add_argument("--rel-choice", type=int, required=True, help="relation choice pairs")
    p.add_argument("--rel-noul", type=int, required=True, help="relation yes/no pairs")
    p.add_argument("--replay", type=int, required=True, help="replay questions, REPLAY_QUOTAS scaled to this")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    rng = random.Random(a.seed)
    from fetch import vg_to_coco
    v2c = vg_to_coco(a.root)
    held = held_pictures(a.root, v2c)

    originals = {r["id"]: r for f in ("coco_position.jsonl", "coco_relation.jsonl")
                 for r in read(os.path.join(a.root, "v2", f))}
    used = {r["id"] for m in EARLIER_MIXES for r in read(os.path.join(a.root, m))}
    wanted = dict(zip(SUBTASKS, (a.position, a.rel_choice, a.rel_noul)))
    pairs, pair_report = pick_pairs(read(os.path.join(a.root, "v6", "flip_lr.jsonl")), originals,
                                    {k: n for k, n in wanted.items() if n}, used, held, v2c, rng)

    have = {r["id"] for r in pairs}
    ok = lambda rows: [r for r in rows if picture(r["image_id"], v2c) not in held and r["id"] not in have]
    pools: Dict[str, List[dict]] = {"vsr": ok(read(os.path.join(a.root, "v5", "vsr.jsonl"))),
                                    "v7w": ok(read(os.path.join(a.root, "v5", "v7w.jsonl"))),
                                    "scienceqa": ok(read(os.path.join(a.root, "v4", "scienceqa.jsonl")))}
    for r in replay_pool(read(os.path.join(a.root, "v4", "train_mix.jsonl")), have, held, v2c):
        pools.setdefault(group_of(r), []).append(r)
    quotas = scale(REPLAY_QUOTAS, a.replay)
    replay, got = pick_replay(pools, quotas, rng)
    cut = shorten_hints(replay)

    out = pairs + replay
    rng.shuffle(out)
    assert len({r["id"] for r in out}) == len(out), "duplicate ids"
    leak = {picture(origin(r["image_id"]), v2c) for r in out} & held
    assert not leak, "held-out pictures in training: %s" % sorted(leak)[:5]
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "train.jsonl"), "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in out)
    per_pic = Counter(picture(origin(r["image_id"]), v2c) for r in out)
    report = {"questions": len(out), "unique_ids": len({r["id"] for r in out}), "pictures": len(per_pic),
              "max_questions_per_picture": max(per_pic.values()),
              "parts": {"left_right_pair_questions": len(pairs), "replay": len(replay)},
              "left_right_share": round(len(pairs) / len(out), 4), "pairs": pair_report,
              "replay_by_ability": dict(got),
              "replay_quotas": dict(quotas), "held_out_pictures": len(held), "held_out_overlap": len(leak),
              "hints_cut": cut, "seed": a.seed, "steps_at_micro_batch_8": -(-len(out) // 8)}
    with open(os.path.join(a.out, "MIX.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
