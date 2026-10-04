"""Larger evaluation sets, and train.jsonl without samples on evaluation images (kept out of the repo).

    uv run python scripts/data/evalsets.py --root data --gqa 1000 --vqav2 1000

Writes <root>/val_gqa.jsonl (GQA testdev: yes/no + choice), <root>/val_vqav2.jsonl (VQAv2 val2014
yes/no, COCO images fetched one by one) and <root>/val_mix.jsonl (both; for --val, i.e. temperature
fitting). VQAv2 val images used by any training file are skipped. Training samples whose image is an
evaluation image -- also under its other id, Visual Genome being half COCO -- are dropped from
<root>/train.jsonl (the previous file is kept as train.pre-evalsets.jsonl).
"""
import argparse
import os
import random
import shutil

from convert import convert_gqa, convert_vqav2, same_images, select
from fetch import download_coco_images, gqa_records, vg_to_coco, vqav2_records
from prepare import _write_jsonl, read_jsonl

TRAINING = ["train.jsonl", "align_train.jsonl", "align_full_train.jsonl", "match_train.jsonl"]
EVALUATION = ["val.jsonl", "pope.jsonl"]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="data")
    p.add_argument("--gqa", type=int, default=1000)
    p.add_argument("--vqav2", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    path = lambda name: os.path.join(a.root, name)
    mapping = vg_to_coco(a.root)

    gqa = select((convert_gqa(q, r) for q, r in gqa_records(a.root, "testdev", ["testdev-00000-of-00001.parquet"])),
                 a.gqa, set(), a.seed)
    trained_on = same_images({s["image_id"] for f in TRAINING if os.path.exists(path(f))
                              for s in read_jsonl(path(f))}, mapping)
    vqa = select((convert_vqav2(q, ann, "val2014") for q, ann in vqav2_records(a.root, "Val")),
                 a.vqav2, trained_on, a.seed)
    failed = set(download_coco_images(a.root, vqa))
    vqa = [s for s in vqa if s["id"] not in failed]
    mix = gqa + vqa
    random.Random(a.seed).shuffle(mix)
    for name, samples in (("val_gqa", gqa), ("val_vqav2", vqa), ("val_mix", mix)):
        print("%s: %d samples" % (name, _write_jsonl(path(name + ".jsonl"), samples)))
    print("vqav2 images failed to download: %d" % len(failed))

    held_out = same_images({s["image_id"] for f in EVALUATION for s in read_jsonl(path(f))}
                           | {s["image_id"] for s in mix}, mapping)
    train = read_jsonl(path("train.jsonl"))
    kept = [s for s in train if s["image_id"] not in held_out]
    if len(kept) < len(train):
        shutil.copy(path("train.jsonl"), path("train.pre-evalsets.jsonl"))
        _write_jsonl(path("train.jsonl"), kept)
    print("train: dropped %d samples on evaluation images, %d left" % (len(train) - len(kept), len(kept)))


if __name__ == "__main__":
    main()
