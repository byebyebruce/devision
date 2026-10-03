"""Download the pictures of LookFirst from their original hosts (the dataset does not redistribute them).

    pip install pyarrow
    python fetch_images.py --out images                 # every split of every config
    python fetch_images.py --out images --config exist --split test

Each row's `image_file` is the path under --out: coco/train2014/... and coco/val2014/... come from
images.cocodataset.org; gqa/images/<id>.jpg is Visual Genome picture <id>, whose URL is listed in Visual
Genome's image_data.json (downloaded once). Files already present are skipped.
"""
import argparse
import glob
import io
import json
import os
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

COCO = "http://images.cocodataset.org/%s"
VG_IMAGE_DATA = "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip"


def vg_urls(cache: str) -> dict:
    path = os.path.join(cache, "image_data.json.zip")
    if not os.path.exists(path):
        os.makedirs(cache, exist_ok=True)
        with urllib.request.urlopen(VG_IMAGE_DATA, timeout=120) as r, open(path, "wb") as f:
            f.write(r.read())
    with zipfile.ZipFile(path) as z:
        data = json.load(z.open(z.namelist()[0]))
    return {str(d["image_id"]): d["url"] for d in data}


def main(argv=None):
    import pyarrow.parquet as pq
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="images")
    p.add_argument("--config", default="*")
    p.add_argument("--split", default="*")
    p.add_argument("--workers", type=int, default=16)
    a = p.parse_args(argv)
    here = os.path.dirname(os.path.abspath(__file__))
    files = set()
    for f in glob.glob(os.path.join(here, "data", a.config, "%s.parquet" % a.split)):
        files |= set(pq.read_table(f, columns=["image_file"]).column(0).to_pylist())
    todo = sorted(f for f in files if not os.path.exists(os.path.join(a.out, f)))
    urls = vg_urls(os.path.join(a.out, ".cache")) if any(f.startswith("gqa/") for f in todo) else {}

    def one(rel):
        url = COCO % rel[len("coco/"):] if rel.startswith("coco/") else urls.get(os.path.basename(rel)[:-4])
        if not url:
            return rel
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
        except OSError:
            return rel
        path = os.path.join(a.out, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        return ""

    with ThreadPoolExecutor(a.workers) as ex:
        failed = [r for r in ex.map(one, todo) if r]
    print("%d pictures needed, %d downloaded, %d failed" % (len(files), len(todo) - len(failed), len(failed)))
    if failed:
        print("failed, e.g.:", failed[:5])


if __name__ == "__main__":
    main()
