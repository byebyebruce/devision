"""Prepare training / evaluation JSONL for devision (kept out of the repo).

    uv run python scripts/data/prepare.py fetch --root data --train-limit 10000 --vqav2-limit 10000 \\
        --train-shards train-00000-of-00021.parquet train-00001-of-00021.parquet
    uv run python scripts/data/prepare.py convert --gqa train_balanced_questions.json --limit 5000 --out x.jsonl
    uv run pytest scripts/data          # converter tests
"""
import argparse
import json
import os
import sys
from typing import Iterable, List

from convert import Sample, convert_gqa, convert_pope, convert_vqav2, same_images, select


def _write_jsonl(path: str, samples: Iterable[Sample]) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str) -> List[Sample]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def fetch_main(argv=None) -> None:
    """GQA train shards + VQAv2 yes/no -> train.jsonl, GQA testdev -> val.jsonl, POPE -> pope.jsonl."""
    import random

    from fetch import download_coco_images, gqa_records, pope_records, vg_to_coco, vqav2_records

    p = argparse.ArgumentParser(description=fetch_main.__doc__)
    p.add_argument("--root", default="data")
    p.add_argument("--train-limit", type=int, default=500)
    p.add_argument("--val-limit", type=int, default=200)
    p.add_argument("--pope-limit", type=int, default=300)
    p.add_argument("--train-shards", nargs="+", default=["train-00000-of-00021.parquet"],
                   help="GQA image shards (train-000NN-of-00021.parquet, ~490MB each)")
    p.add_argument("--vqav2-limit", type=int, default=0,
                   help="VQAv2 yes/no samples added to train (COCO images fetched one by one)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    val = select((convert_gqa(q, r) for q, r in gqa_records(a.root, "testdev", ["testdev-00000-of-00001.parquet"])),
                 a.val_limit, set(), a.seed)
    pope = select((convert_pope(r) for r in pope_records(a.root)), a.pope_limit, set(), a.seed)
    held_out = same_images({s["image_id"] for s in val + pope}, vg_to_coco(a.root))
    train = select((convert_gqa(q, r) for q, r in gqa_records(a.root, "train", a.train_shards)),
                   a.train_limit, held_out, a.seed)
    print("gqa train: %d samples" % len(train))
    if a.vqav2_limit:
        vqa = select((convert_vqav2(q, ann, "train2014") for q, ann in vqav2_records(a.root)),
                     a.vqav2_limit, held_out, a.seed)
        failed = set(download_coco_images(a.root, vqa))
        vqa = [s for s in vqa if s["id"] not in failed]
        print("vqav2 train: %d samples (%d images failed to download)" % (len(vqa), len(failed)))
        train = train + vqa
        random.Random(a.seed).shuffle(train)
    for name, samples in (("train", train), ("val", val), ("pope", pope)):
        print("%s: %d samples" % (name, _write_jsonl(os.path.join(a.root, name + ".jsonl"), samples)))


def convert_main(argv=None) -> None:
    """Convert official GQA / VQAv2 / POPE files into a selected, balanced JSONL."""
    p = argparse.ArgumentParser(description=convert_main.__doc__)
    p.add_argument("--gqa", nargs="*", default=[], help="GQA *_balanced_questions.json files")
    p.add_argument("--vqav2-questions", help="v2_OpenEnded_mscoco_<split>_questions.json")
    p.add_argument("--vqav2-annotations", help="v2_mscoco_<split>_annotations.json")
    p.add_argument("--vqav2-split", default="train2014")
    p.add_argument("--pope", nargs="*", default=[], help="POPE JSONL records")
    p.add_argument("--exclude", nargs="*", default=[], help="JSONL sample files whose images must not appear")
    p.add_argument("--limit", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    def samples():
        for path in a.gqa:
            with open(path) as f:
                for qid, rec in json.load(f).items():
                    yield convert_gqa(qid, rec)
        if a.vqav2_questions:
            with open(a.vqav2_questions) as f:
                questions = {q["question_id"]: q for q in json.load(f)["questions"]}
            with open(a.vqav2_annotations) as f:
                for ann in json.load(f)["annotations"]:
                    yield convert_vqav2(questions[ann["question_id"]], ann, a.vqav2_split)
        for path in a.pope:
            for rec in read_jsonl(path):
                yield convert_pope(rec)

    exclude = {s["image_id"] for path in a.exclude for s in read_jsonl(path)}
    print("wrote %d samples" % _write_jsonl(a.out, select(samples(), a.limit, exclude, a.seed)))


if __name__ == "__main__":
    commands = {"fetch": fetch_main, "convert": convert_main}
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        sys.exit("usage: prepare.py {fetch,convert} [options]")
    commands[sys.argv[1]](sys.argv[2:])
