"""The v2 training mix (kept out of the repo): data/v2 training files -> data/v2/train_mix.jsonl.

    uv run python scripts/data/v2_mix.py --root data [--lr-share 0.05] [--out data/v2/train_mix.jsonl]

Per-ability quotas (docs/research/data-quality.md, "v2 训练实验计划"); left/right questions (position
lr, relations, GQA spatial) together at most --lr-share of the mix; colour and count capped so they do
not make up most of the multiple choice; at most --per-picture questions on one picture across all
files. The cap is applied before the final balancing, and the result is checked like the build's files.
"""
import argparse
import json
import os
import random
import sys
from collections import Counter
from typing import Callable, Dict, List, Tuple

from convert import picture
from fetch import vg_to_coco
from prepare import read_jsonl
from v2_build import BALANCE, finalize
from v2_common import check, file_stats, write

# (file, which rows, how many, lr question?)
QUOTAS: List[Tuple[str, Callable[[dict], bool], int, bool]] = [
    ("vqa_yesno", lambda r: True, 50000, False),
    ("coco_exist", lambda r: True, 40000, False),
    ("vqa_choice", lambda r: r["kind"] == "color", 8000, False),
    ("vqa_choice", lambda r: r["kind"] == "count", 8000, False),
    ("vqa_choice", lambda r: r["kind"] not in ("color", "count"), 100000, False),
    ("gqa", lambda r: r["kind"] == "gqa", 8000, False),
    ("coco_position", lambda r: r["axis"] == "tb", 5000, False),
    ("coco_size", lambda r: True, 100000, False),
    ("coco_position", lambda r: r["axis"] == "lr", 3000, True),
    ("coco_relation", lambda r: True, 2500, True),
    ("gqa", lambda r: r["kind"] == "spatial", 1500, True),
]


def is_lr(r: dict) -> bool:
    return (r.get("kind") in ("position", "relation") and r.get("axis") == "lr") or r.get("kind") == "spatial"


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="data")
    p.add_argument("--out", default=None)
    p.add_argument("--lr-share", type=float, default=0.05)
    p.add_argument("--per-picture", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    v2 = os.path.join(a.root, "v2")
    out_path = a.out or os.path.join(v2, "train_mix.jsonl")
    rng = random.Random(a.seed)
    mapping = vg_to_coco(a.root)
    pools = {stem: read_jsonl(os.path.join(v2, stem + ".jsonl")) for stem in {q[0] for q in QUOTAS}}
    picked: Dict[str, List[dict]] = {}
    for stem, which, n, _ in QUOTAS:
        rows = [r for r in pools[stem] if which(r)]
        rng.shuffle(rows)
        picked.setdefault(stem, []).extend(rows[:n])
    # cap per picture across all files, then rebalance each file's part and check it
    per_pic: Counter = Counter()
    every = [(stem, r) for stem, rows in picked.items() for r in rows]
    rng.shuffle(every)
    capped: Dict[str, List[dict]] = {}
    for stem, r in every:
        pid = picture(r["image_id"], mapping)
        if per_pic[pid] < a.per_picture:
            per_pic[pid] += 1
            capped.setdefault(stem, []).append(r)
    problems: List[str] = []
    mix: List[dict] = []
    for stem, rows in capped.items():
        rows = finalize(rows, BALANCE[stem])
        problems += check(rows, "mix:" + stem, BALANCE[stem])
        mix += rows
    # left/right share
    lr = [r for r in mix if is_lr(r)]
    limit = int(a.lr_share * (len(mix) - len(lr)) / (1 - a.lr_share))
    if len(lr) > limit:
        drop = {id(r) for r in rng.sample(lr, len(lr) - limit)}
        mix = [r for r in mix if id(r) not in drop]
        # dropping can unbalance the lr groups again: rebalance those files
        rebal: Dict[str, List[dict]] = {}
        for r in mix:
            rebal.setdefault(r["source"], []).append(r)
        mix = []
        for src, rows in rebal.items():
            stem = {"coco-position": "coco_position", "coco-relation": "coco_relation"}.get(src)
            if stem:
                rows = finalize(rows, BALANCE[stem])
                problems += check(rows, "mix:" + stem, BALANCE[stem])
            mix += rows
    rng.shuffle(mix)
    dup = [i for i, c in Counter(r["id"] for r in mix).items() if c > 1]
    if dup:
        problems.append("duplicate ids in the mix: %s" % dup[:3])
    write(out_path, mix)
    stats = file_stats(out_path)
    stats["sources"] = dict(Counter(r["source"] for r in mix))
    stats["lr_share"] = round(sum(map(is_lr, mix)) / len(mix), 4)
    stats["max_per_picture"] = max(Counter(picture(r["image_id"], mapping) for r in mix).values())
    stats["settings"] = {"lr_share": a.lr_share, "per_picture": a.per_picture, "seed": a.seed}
    with open(out_path[:-6] + ".stats.json", "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats, indent=1))
    if problems:
        print("PROBLEMS:\n  " + "\n  ".join(problems))
        sys.exit(1)


if __name__ == "__main__":
    main()
