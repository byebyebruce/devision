"""Exclusion index of every Decision Index public-vision board image -> data/v05/board_exclusion.json.

    uv run --no-sync python scripts/data/v05_board_exclusion.py --root data \\
        --board /home/bruce/ext/decision-index-data/vision [--audit data/v05/exp132/train.jsonl ...]

For each image under <board>/full/*/img (every image a board row can send, including the second image of two-image
BLINK / Winoground items): sha256 of the file bytes, sha256 of the decoded RGB pixels, a 64-bit dHash and a 64-bit
pHash of the grey image. Source ids: every board item id and image file name ("board:<set>:<name>"), CV-Bench's
original files (from the pinned CV-Bench parquet when it is in the HF cache: COCO val2017 -> "coco:<id>",
ADE20K val -> "ade20k:<name>", Omni3D test -> "omni3d:<path>"), Hateful Memes ids "hm:<id>" of the board rows and,
when data/raw/board/hateful is present, of all dev_seen / dev_unseen / test_seen / test_unseen rows, FUNSD test
documents "funsd:<doc>", Mind2Web annotation ids "m2w:<annotation_id>" and, when data/raw/board/charxiv_test has
CharXiv's val / test parquet, the arXiv paper of every CharXiv figure ("arxiv:<id>", so figures of the same paper
are excluded too). Upstream held-out images that are NOT board rows but belong to the board benchmark's held-out
split (CharXiv test) are hashed into the same index with paths "upstream/<split>/<n>". "upstream_splits" lists, per
board benchmark, the upstream splits the board draws from; builders never read them.

--audit FILE...: also report how many images of existing training files hit the index (written to the JSON).
"""
import argparse
import glob
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v05_board_common import BoardIndex, image_hashes  # noqa: E402

OUT = os.path.join("v05", "board_exclusion.json")
CHARXIV_GLOB = "raw/board/charxiv_test/*.parquet"
UPSTREAM_EXTRA = {"charxiv_test": "raw/board/charxiv_test/test.parquet"}   # held-out images that are not board rows
CVBENCH_GLOB = "raw/hf/datasets--nyu-visionx--CV-Bench/snapshots/bc284db50d036958861cb60cdd7b77612052ce0d/test_*.parquet"
UPSTREAM_SPLITS = {
    "CV-Bench": ["nyu-visionx/CV-Bench test", "COCO val2017 (subset of val2014)", "ADE20K validation",
                 "Omni3D test (SUN RGB-D, Hypersim, nuScenes)"],
    "BLINK": ["BLINK-Benchmark/BLINK val (and test)"],
    "RealWorldQA": ["xai-org/RealworldQA test (the only split)"],
    "CharXiv": ["princeton-nlp/CharXiv validation and test"],
    "InfographicVQA": ["InfographicVQA validation and test (lmms-lab DocVQA mirror)"],
    "Mind2Web": ["osunlp/Multimodal-Mind2Web test_website (also test_domain / test_task kept out of board families)"],
    "Winoground": ["facebook/winoground test (the only split)"],
    "KIE (CORD+FUNSD)": ["naver-clova-ix/cord-v2 validation and test", "FUNSD testing_data"],
    "Moderation (Hateful Memes)": ["Hateful Memes dev_seen, test_seen, dev_unseen, test_unseen"],
    "R-Bench-M": ["R-Bench rbench-m test"],
    "MMMU-Pro vision": ["MMMU/MMMU_Pro vision test (from MMMU validation / test)"],
}


def hash_file(path: str) -> Dict:
    with open(path, "rb") as f:
        data = f.read()
    try:
        return image_hashes(data)
    except Exception as e:   # pragma: no cover - a board file PIL cannot open is reported, not skipped silently
        return {"error": str(e)}


def hash_bytes(data: bytes) -> Dict:
    try:
        return image_hashes(data)
    except Exception as e:  # pragma: no cover
        return {"error": str(e)}


def upstream_images(root: str, workers: int) -> List[Dict]:
    out = []
    for name, rel in UPSTREAM_EXTRA.items():
        p = os.path.join(root, rel)
        if not os.path.exists(p):
            continue
        import pyarrow.parquet as pq
        blobs = [r["bytes"] for r in pq.read_table(p, columns=["image"]).column("image").to_pylist()]
        with ProcessPoolExecutor(workers) as ex:
            hs = list(ex.map(hash_bytes, blobs, chunksize=16))
        out += [{"path": "upstream/%s/%d" % (name, n), **h} for n, h in enumerate(hs) if "error" not in h]
    return out


def board_images(board: str) -> List[str]:
    return sorted(glob.glob(os.path.join(board, "full", "*", "img", "*")))


def source_ids(root: str, board: str) -> List[str]:
    ids = set()
    for items in sorted(glob.glob(os.path.join(board, "full", "*", "items.jsonl"))):
        st = os.path.basename(os.path.dirname(items))
        for line in open(items):
            it = json.loads(line)
            ids.add("board:" + it["id"])
            for im in it.get("images", []):
                ids.add("board:%s:%s" % (st, os.path.basename(im)))
            if st == "full_hateful_memes":
                ids.add("hm:" + it["id"].split(":", 1)[1])
            elif st == "full_funsd":
                ids.add("funsd:" + it["id"].split(":")[1])
            elif st == "full_mind2web" and it.get("annotation_id"):
                ids.add("m2w:" + it["annotation_id"])
    for p in sorted(glob.glob(os.path.join(root, CVBENCH_GLOB))):
        import pyarrow.parquet as pq
        for r in pq.read_table(p, columns=["source", "source_filename"]).to_pylist():
            fn = r["source_filename"]
            if r["source"] == "COCO":
                ids.add("coco:%d" % int(os.path.splitext(os.path.basename(fn))[0]))
            elif r["source"] == "ADE20K":
                ids.add("ade20k:" + os.path.splitext(os.path.basename(fn))[0])
            else:
                ids.add("omni3d:" + fn)
    for p in sorted(glob.glob(os.path.join(root, CHARXIV_GLOB))):
        import pyarrow.parquet as pq
        ids |= {"arxiv:" + str(x).strip() for x in pq.read_table(p, columns=["original_id"]).column(
            "original_id").to_pylist() if x}
    hm = os.path.join(root, "raw", "board", "hateful")
    for split in ("dev_seen", "dev_unseen", "test_seen", "test_unseen"):
        p = os.path.join(hm, split + ".jsonl")
        if os.path.exists(p):
            ids |= {"hm:%05d" % int(json.loads(l)["id"]) for l in open(p) if l.strip()}
    return sorted(ids)


def audit(index: BoardIndex, root: str, files: List[str], workers: int) -> Dict:
    out = {}
    for f in files:
        rows = [json.loads(l) for l in open(f)]
        rels = sorted({r["image"] for r in rows})
        ids = Counter(r.get("image_id", "") for r in rows if index.hit_id(r.get("image_id", "")))
        with ProcessPoolExecutor(workers) as ex:
            hs = list(ex.map(hash_file, [os.path.join(root, p) for p in rels], chunksize=64))
        hits = {}
        for rel, h in zip(rels, hs):
            if "error" in h:
                continue
            got = index.hit_hashes(h)
            if got:
                hits[rel] = list(got)
        out[f] = {"rows": len(rows), "images": len(rels), "image_hash_hits": len(hits),
                  "rows_with_hit_image": sum(r["image"] in hits for r in rows),
                  "id_hits": sum(ids.values()), "hit_kinds": dict(Counter(v[0] for v in hits.values())),
                  "hit_boards": dict(Counter(v[1].split("/")[1] for v in hits.values())),
                  "examples": dict(list(hits.items())[:20])}
        print(f, {k: v for k, v in out[f].items() if k != "examples"}, flush=True)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data")
    ap.add_argument("--board", default="/home/bruce/ext/decision-index-data/vision")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--audit", nargs="*", default=[])
    a = ap.parse_args(argv)
    t0 = time.time()
    paths = board_images(a.board)
    with ProcessPoolExecutor(a.workers) as ex:
        hs = list(ex.map(hash_file, paths, chunksize=16))
    images, errors = [], []
    for p, h in zip(paths, hs):
        rel = os.path.relpath(p, a.board)
        (errors.append({"path": rel, **h}) if "error" in h else images.append({"path": rel, **h}))
    n_board = len(images)
    images += upstream_images(a.root, a.workers)
    index = {"board": a.board, "rows_file_sha256": None, "n_images": n_board, "n_upstream_images": len(images) - n_board,
             "errors": errors, "per_set": dict(Counter(i["path"].split("/")[1] for i in images)),
             "near_rule": "dHash <= 6 bits AND pHash <= 8 bits (64-bit each, grey image)",
             "upstream_splits": UPSTREAM_SPLITS, "source_ids": source_ids(a.root, a.board), "images": images}
    rows = os.path.join(a.board, "vision-rows.jsonl")
    if os.path.exists(rows):
        import hashlib
        index["rows_file_sha256"] = hashlib.sha256(open(rows, "rb").read()).hexdigest()
    # how many board images are near-duplicates of each other (a sanity check that the near rule is not loose)
    bi = BoardIndex(index)
    import numpy as np
    self_near = 0
    for i in range(len(images)):
        dd = np.bitwise_count(bi.d ^ bi.d[i])
        pp = np.bitwise_count(bi.p ^ bi.p[i])
        self_near += int(((dd <= bi.dhash_bits) & (pp <= bi.phash_bits)).sum()) - 1 > 0
    index["board_images_with_a_near_twin"] = self_near
    if a.audit:
        index["audit"] = audit(bi, a.root, a.audit, a.workers)
    out = os.path.join(a.root, OUT)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(index, f)
    print("board images %d (errors %d), source ids %d, near twins %d, %.0f s -> %s" % (
        len(images), len(errors), len(index["source_ids"]), self_near, time.time() - t0, out))


if __name__ == "__main__":
    main()
