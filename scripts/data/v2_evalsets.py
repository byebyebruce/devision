"""v2 evaluation sets (kept out of the repo). Run after the v2 training generators' --split val* runs:

    uv run python scripts/data/v2_coco_exist.py --root data --split val2014 --per-category 7 --out data/v2/eval_coco_exist.jsonl
    uv run python scripts/data/v2_coco_spatial.py --root data --split val2014 --out-dir data/v2 --prefix eval_ \
        --cap-position 4 --cap-relation 3 --cap-size 3
    uv run python scripts/data/v2_vqa_choice.py --root data --split Val --limit 1000 --out data/v2/eval_vqa_choice.jsonl
    uv run python scripts/data/v2_evalsets.py --root data

This script adds data/v2/eval_pope.jsonl -- all of POPE (random / popular / adversarial, 3,000 each,
500 COCO val2014 images), with the split as the sample source so results come out per split -- and
fetches the COCO val2014 images every data/v2/eval_*.jsonl needs.
"""
import argparse
import glob
import os
from collections import Counter

POPE_SHARDS = ["test-00000-of-00003.parquet", "test-00001-of-00003.parquet", "test-00002-of-00003.parquet"]


def main(argv=None) -> None:
    from convert import convert_pope
    from fetch import download_coco_images, pope_records
    from prepare import _write_jsonl, read_jsonl

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="data")
    a = p.parse_args(argv)
    pope = []
    for shard in POPE_SHARDS:
        for rec in pope_records(a.root, shard):
            s = convert_pope(rec)
            s["source"] = "pope-" + rec["category"]
            pope.append(s)
    path = os.path.join(a.root, "v2", "eval_pope.jsonl")
    print("%s: %d questions %s" % (path, _write_jsonl(path, pope), dict(Counter(s["source"] for s in pope))))
    for f in sorted(glob.glob(os.path.join(a.root, "v2", "eval_*.jsonl"))):
        rows = read_jsonl(f)
        failed = download_coco_images(a.root, [s for s in rows if s["image"].startswith("coco/")])
        print("%s: %d questions, %d images missing" % (f, len(rows), len(failed)))


if __name__ == "__main__":
    main()
