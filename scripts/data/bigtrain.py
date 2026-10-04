"""A larger stage-2 training set from data already on disk (kept out of the repo).

    uv run python scripts/data/bigtrain.py --root data --gqa 60000 --vqav2 40000 --out data/train_100k.jsonl

GQA from the train image shards already downloaded, VQAv2 yes/no on COCO train2014 (all images are
on disk after the full caption run; missing ones are fetched). Every evaluation image -- val*.jsonl,
pope.jsonl, under both its COCO and Visual Genome id -- is excluded.
"""
import argparse
import glob
import os
import random

from convert import convert_gqa, convert_vqav2, same_images, select
from fetch import download_coco_images, gqa_records, vg_to_coco, vqav2_records
from prepare import _write_jsonl, read_jsonl


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="data")
    p.add_argument("--gqa", type=int, default=60000)
    p.add_argument("--vqav2", type=int, default=40000)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    evals = [f for f in glob.glob(os.path.join(a.root, "*.jsonl"))
             if os.path.basename(f).startswith(("val", "pope", "test_", "bench_"))]
    held_out = same_images({s["image_id"] for f in evals for s in read_jsonl(f)}, vg_to_coco(a.root))
    print("held out: %d image ids from %s" % (len(held_out), sorted(map(os.path.basename, evals))))
    shards = sorted(os.path.basename(f) for f in glob.glob(
        os.path.join(a.root, "raw", "hf", "datasets--lmms-lab--GQA", "snapshots", "*", "train_balanced_images", "*.parquet")))
    gqa = select((convert_gqa(q, r) for q, r in gqa_records(a.root, "train", shards)), a.gqa, held_out, a.seed)
    vqa = select((convert_vqav2(q, ann, "train2014") for q, ann in vqav2_records(a.root)), a.vqav2, held_out, a.seed)
    failed = set(download_coco_images(a.root, vqa))
    vqa = [s for s in vqa if s["id"] not in failed]
    train = gqa + vqa
    random.Random(a.seed).shuffle(train)
    print("gqa %d (from %d shards), vqav2 %d (%d images failed), total %d" % (
        len(gqa), len(shards), len(vqa), len(failed), _write_jsonl(a.out, train)))


if __name__ == "__main__":
    main()
