"""data-n2: one held-out rule for every training source built for v0.4 (plan docs/experiments/2026-10-21-v04-plan.md).

A training picture is held out when
1. its id (or the other id of the same picture: COCO <-> Visual Genome, convert.same_images) is the image_id of any
   evaluation / dev / benchmark file (`eval_files`), or
2. it is a COCO val2014 picture (all of val2014 is held out: test sets, POPE and test_count_coco sit on it), or
3. its 64-bit dHash equals the dHash of any picture of those files (benchmarks without traceable ids: MMBench,
   CV-Bench, MME, NaturalBench, CountBench, ...; also catches re-encoded copies).

    import held
    h = held.load("data")            # builds <root>/n2/held.json once (rebuilt when the eval files change)
    h.is_held(image_id, image_path)  # image_path relative to the root; hashed on demand
"""
import glob
import hashlib
import json
import os
import sys
from typing import Dict, List, Optional, Set

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

HELD_PREFIXES = ("test_", "dev", "bench_", "val", "pope", "align_val")
PREFIX_DIRS = ("", "v2", "v3", "v4", "v5", "v6", "v10", "v11", "ext", "n1", "n2")
WHOLE_DIRS = ("lv_bench", "n2/bench", "r13/bench")


# data-n2's own monitoring splits (written by n2/mix.py from the training blocks): their pictures are already kept
# out of training by the mix's dev rule; holding them out here too would empty them on the next mix
N2_OWN_DEV = ("dev_v7w", "dev_spatial", "dev_count", "dev_sqa_synth", "dev_iconqa", "dev_tallyqa", "dev_mapqa",
              "dev_clevr", "dev_spatialsense")


def eval_files(root: str) -> List[str]:
    out = []
    for d in PREFIX_DIRS:
        for f in sorted(glob.glob(os.path.join(root, d, "*.jsonl"))):
            name = os.path.basename(f)
            if d == "n2" and name[:-len(".jsonl")] in N2_OWN_DEV:
                continue
            if name.startswith(HELD_PREFIXES):
                out.append(f)
    for d in WHOLE_DIRS:
        out += sorted(f for f in glob.glob(os.path.join(root, d, "*.jsonl")) if not f.endswith(".details.jsonl"))
    return out


def dhash_file(path: str) -> Optional[int]:
    from PIL import Image
    from lv_bench import dhash  # pyright: ignore[reportMissingImports]
    try:
        with Image.open(path) as im:
            return dhash(im)
    except OSError:
        return None


class Held:
    def __init__(self, root: str, ids: Set[str], hashes: Set[int], vg2coco: Dict[str, str]):
        self.root, self.ids, self.hashes, self.vg2coco = root, ids, hashes, vg2coco

    def id_held(self, image_id: str, image_path: str = "") -> bool:
        if image_id in self.ids or self.vg2coco.get(image_id, image_id) in self.ids:
            return True
        return "val2014" in image_path

    def is_held(self, image_id: str, image_path: str) -> bool:
        if self.id_held(image_id, image_path):
            return True
        h = dhash_file(os.path.join(self.root, image_path))
        return h is not None and h in self.hashes


def _signature(files: List[str]) -> str:
    s = hashlib.sha256()
    for f in files:
        st = os.stat(f)
        s.update(("%s:%d:%d;" % (f, st.st_size, int(st.st_mtime))).encode())
    return s.hexdigest()


def load(root: str = "data") -> Held:
    from convert import same_images  # pyright: ignore[reportMissingImports]
    from fetch import vg_to_coco  # pyright: ignore[reportMissingImports]
    files = eval_files(root)
    sig = _signature(files)
    cache = os.path.join(root, "n2", "held.json")
    vg2coco = vg_to_coco(root)
    if os.path.exists(cache):
        c = json.load(open(cache))
        if c.get("signature") == sig:
            return Held(root, set(c["ids"]), set(c["hashes"]), vg2coco)
    ids, pics = set(), set()
    for f in files:
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                ids.add(r["image_id"])
                pics.add(r["image"])
    ids = same_images(ids, vg2coco)
    hashes = {h for h in (dhash_file(os.path.join(root, p)) for p in sorted(pics)) if h is not None}
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, "w") as fh:
        json.dump({"signature": sig, "files": [os.path.relpath(f, root) for f in files], "ids": sorted(ids),
                   "hashes": sorted(hashes)}, fh)
    return Held(root, ids, hashes, vg2coco)
