"""Held-out pictures for the public-data pack (scripts/data/pub/): everything common.HeldOut holds (every evaluation /
monitoring / calibration set up to round 12) plus the sets added since -- round 13's sealed binding tests and
transfer benches (CountBench, What'sUp, SugarCrepe, CoGenT B), the v12-retrain panels and calibration set, and the
exp14 PixMo dev panel. Same rules: picture identity (COCO = VG) for ids, dHash within NEAR_BITS for photos."""
import glob
import hashlib
import json
import os
import sys
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ext"))
sys.path.insert(0, os.path.join(HERE, ".."))
from common import HeldOut, raw_dir, sha256  # noqa: E402

EXTRA = ("r13/gqa/dev_bind_gqa.jsonl", "r13/gqa/test_bind_gqa.jsonl", "r13/bench/countbench.jsonl",
         "r13/bench/whatsup.jsonl", "r13/bench/sugarcrepe_att.jsonl", "r13/bench/sugarcrepe_rel.jsonl",
         "r13/bench/cogent_b.jsonl", "r13b/val.jsonl", "r13b/panel_exist.jsonl", "r13b/panel_science.jsonl",
         "r13b/count_wide.jsonl", "exp14/pixmo_count_wide.jsonl", "exp14/pixmo_exist.jsonl")


def extra_files(root: str) -> List[str]:
    return [os.path.join(root, f) for f in EXTRA if os.path.exists(os.path.join(root, f))]


class PubHeld(HeldOut):
    def __init__(self, root: str):
        super().__init__(root)
        from convert import picture  # pyright: ignore[reportMissingImports]
        from fetch import vg_to_coco  # pyright: ignore[reportMissingImports]
        from convert import same_images  # pyright: ignore[reportMissingImports]
        ids = {json.loads(l)["image_id"] for f in extra_files(root) for l in open(f) if l.strip()}
        self.ids |= {picture(i, self.v2c) for i in same_images(ids, self.v2c)}
        self.missing = [f for f in EXTRA if not os.path.exists(os.path.join(root, f))]

    def photo_hashes(self) -> Dict[str, int]:
        if self._hashes is None:
            base = super().photo_hashes()
            from v9_mix import hashes, is_photo  # pyright: ignore[reportMissingImports]
            files = extra_files(self.root)
            key = hashlib.sha256("".join(sha256(f) for f in files).encode()).hexdigest()
            cache = os.path.join(raw_dir(self.root, ""), "held_hashes_pub.json")
            if os.path.exists(cache) and json.load(open(cache)).get("key") == key:
                extra = {k: int(v) for k, v in json.load(open(cache))["hashes"].items()}
            else:
                photos = {r["image"] for f in files for r in map(json.loads, open(f)) if r.get("image") and is_photo(r["image"])}
                extra = hashes(self.root, photos)
                json.dump({"key": key, "hashes": {k: str(v) for k, v in extra.items()}}, open(cache, "w"))
            self._hashes = {**base, **extra}
        return self._hashes


def require_history(root: str) -> None:
    """The isolation needs the project's evaluation sets on this machine; refuse to build without them."""
    need = ["v2/test_exist.jsonl", "v5/dev_mix.jsonl", "lv_bench/pope.jsonl", "ext/dev_ext.jsonl", "r13/gqa/test_bind_gqa.jsonl"]
    missing = [n for n in need if not os.path.exists(os.path.join(root, n))]
    if missing:
        raise SystemExit("the held-out sets are missing (%s): build or copy the project's data first "
                         "(scripts/data/build_all.sh, scripts/data/ext/build_ext.sh, scripts/data/r13/...) -- "
                         "without them the new pictures cannot be checked against the evaluation sets" % missing)


def split_dev(root: str, source: str, rows, share: float = 0.03, key=lambda r: r["image_id"]):
    """Hold out `share` of the pictures (by a hash of `key`, stable across re-runs) as <root>/ext/<source>/dev.jsonl
    for monitoring; returns the training rows. Dev pictures never appear in train.jsonl."""
    from common import problems  # noqa: E402
    dev = [r for r in rows if int(hashlib.sha1(("dev:" + key(r)).encode()).hexdigest()[:8], 16) < share * 0x100000000]
    ids = {id(r) for r in dev}
    bad = [(r["id"], p) for r in dev for p in [problems(r, root)] if p]
    if bad:
        raise SystemExit("%s dev: %d malformed samples, e.g. %s" % (source, len(bad), bad[:3]))
    d = os.path.join(root, "ext", source)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "dev.jsonl.tmp"), "w") as f:
        for r in dev:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(os.path.join(d, "dev.jsonl.tmp"), os.path.join(d, "dev.jsonl"))
    return [r for r in rows if id(r) not in ids], len(dev)


def balance_noul(rows, rng):
    """As many yes as no among the yes/no questions (the majority side randomly cut; soft golds by their larger side);
    choice questions untouched. Round 13 lost POPE recall to a training set that was 15% yes."""
    noul = [r for r in rows if r["questions"]["q"]["type"] == "noul"]
    yes = [r for r in noul if r["answer_key"] == "true"]
    no = [r for r in noul if r["answer_key"] != "true"]
    big, small = (yes, no) if len(yes) > len(no) else (no, yes)
    keep = {id(r) for r in small} | {id(r) for r in rng.sample(big, len(small))}
    return [r for r in rows if r["questions"]["q"]["type"] != "noul" or id(r) in keep]
