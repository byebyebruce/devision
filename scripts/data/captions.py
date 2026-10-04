"""COCO caption samples for the alignment stage (kept out of the repo).

    uv run python scripts/data/captions.py --root data --images 30000 --val-images 200 \\
        --exclude data/val.jsonl data/pope.jsonl

Reads <root>/raw/coco/captions_train2014.json (from annotations_trainval2014.zip), picks
`--images` train2014 images with a fixed seed, drops any whose image_id appears in the
exclude files, downloads the missing images and writes <root>/align_{train,val}.jsonl.
"""
import argparse
import json
import os
import random
import zipfile

from fetch import download_coco_images
from prepare import _write_jsonl, read_jsonl

ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"


def caption_samples(captions_json: str):
    with open(captions_json) as f:
        data = json.load(f)
    by_image = {}
    for a in data["annotations"]:
        by_image.setdefault(a["image_id"], []).append(a["caption"].strip())
    for iid in sorted(by_image):
        yield {"id": "coco-cap:%d" % iid, "source": "coco-captions", "image_id": "coco:%d" % iid,
               "image": "coco/train2014/COCO_train2014_%012d.jpg" % iid, "captions": by_image[iid]}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="data")
    p.add_argument("--images", type=int, default=30000)
    p.add_argument("--val-images", type=int, default=200)
    p.add_argument("--exclude", nargs="*", default=[])
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    raw = os.path.join(a.root, "raw", "coco")
    captions_json = os.path.join(raw, "captions_train2014.json")
    if not os.path.exists(captions_json):
        zpath = os.path.join(raw, "annotations_trainval2014.zip")
        if not os.path.exists(zpath):
            import urllib.request
            os.makedirs(raw, exist_ok=True)
            urllib.request.urlretrieve(ANNOTATIONS_URL, zpath)
        with zipfile.ZipFile(zpath) as z, open(captions_json, "wb") as f:
            f.write(z.read("annotations/captions_train2014.json"))

    exclude = {s["image_id"] for path in a.exclude for s in read_jsonl(path)}
    samples = [s for s in caption_samples(captions_json) if s["image_id"] not in exclude]
    # Images already on disk first, so a small run needs no downloads.
    on_disk = [s for s in samples if os.path.exists(os.path.join(a.root, s["image"]))]
    rest = [s for s in samples if not os.path.exists(os.path.join(a.root, s["image"]))]
    rng = random.Random(a.seed)
    rng.shuffle(on_disk)
    rng.shuffle(rest)
    chosen = (on_disk + rest)[:a.images]
    failed = set(download_coco_images(a.root, chosen))
    chosen = [s for s in chosen if s["id"] not in failed]
    rng.shuffle(chosen)
    val, train = chosen[:a.val_images], chosen[a.val_images:]
    print("train %d images, val %d images, %d downloads failed" % (
        _write_jsonl(os.path.join(a.root, "align_train.jsonl"), train),
        _write_jsonl(os.path.join(a.root, "align_val.jsonl"), val), len(failed)))


if __name__ == "__main__":
    main()
