"""exp30 real caption replay pool for the alignment stage (docs/experiments/2026-10-17-exp30-align-replay-plan.md;
critic V04-PREP-02): COCO train2014 pictures with their captions (data/align_full_train.jsonl), minus every picture
any monitoring, calibration, development, test or benchmark set uses -- by image id, with VG ids mapped to their
COCO ids (data/v5/image_identity.json), a mirror (":flip") counted as its original, and by image file path.

    uv run python scripts/exp30/isolate.py --root data --n 20000

Outputs <root>/exp30/real_captions.jsonl (n pictures, seed 30) and <root>/exp30/ISOLATION.json (the excluded
sets, how many pool pictures each removed, the counts before and after).
"""
import argparse
import glob
import hashlib
import json
import os
import random

EVAL_GLOBS = ["r13b/panel_exist.jsonl", "r13b/panel_science.jsonl", "r13b/count_wide.jsonl", "r13b/val.jsonl",
              "ext/dev_ext.jsonl", "v11/dev_count_transfer.jsonl", "v11/test_count_fresh.jsonl", "v6/dev_lr_pairs.jsonl",
              "v6/test_*.jsonl", "v5/dev_*.jsonl", "v5/test_*.jsonl", "v2/dev_*.jsonl", "v2/test_*.jsonl",
              "v2/bench_pope.jsonl", "lv_bench/*.jsonl", "r13/gqa/dev_*.jsonl", "r13/gqa/test_*.jsonl", "r13/bench/*.jsonl",
              "exp14/pixmo_*.jsonl",
              "v10/dev_sqa_*.jsonl", "exp21/dev_*.jsonl", "exp22/*.jsonl", "exp28/*.jsonl", "val_*.jsonl", "pope.jsonl"]


def norm(image_id: str, identity: dict) -> str:
    i = image_id.removesuffix(":flip")
    return identity.get(i, i)


def excluded(root: str, identity: dict):
    ids, paths, per = set(), set(), {}
    for pattern in EVAL_GLOBS:
        for f in sorted(glob.glob(os.path.join(root, pattern))):
            if f.endswith(".details.jsonl"):
                continue
            rel = os.path.relpath(f, root)
            these = set()
            for line in open(f):
                s = json.loads(line)
                if "image_id" in s:
                    these.add(norm(s["image_id"], identity))
                if "image" in s:
                    paths.add(s["image"])
            per[rel] = these
            ids |= these
    return ids, paths, per


def build(root: str, n: int, seed: int = 30) -> dict:
    identity = json.load(open(os.path.join(root, "v5", "image_identity.json")))
    ids, paths, per = excluded(root, identity)
    pool = [json.loads(x) for x in open(os.path.join(root, "align_full_train.jsonl"))]
    keep = [s for s in pool if norm(s["image_id"], identity) not in ids and s["image"] not in paths and s["captions"]]
    removed = {rel: sum(norm(s["image_id"], identity) in these for s in pool) for rel, these in per.items()}
    rng = random.Random(seed)
    rng.shuffle(keep)
    chosen = keep[:n]
    if len(chosen) < n:
        raise SystemExit("only %d isolated pictures, %d asked" % (len(chosen), n))
    out = os.path.join(root, "exp30")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "real_captions.jsonl")
    with open(path, "w") as fh:
        for s in chosen:
            fh.write(json.dumps(s) + "\n")
    report = {"pool": len(pool), "excluded_ids": len(ids), "excluded_paths": len(paths), "kept": len(keep), "chosen": n,
              "seed": seed, "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
              "removed_by_set": {k: v for k, v in removed.items() if v}, "sets_checked": sorted(per)}
    json.dump(report, open(os.path.join(out, "ISOLATION.json"), "w"), indent=1)
    return report


def mix(root: str, seed: int = 30) -> dict:
    """data/exp30/align_mix.jsonl: exp24's 20,000 text pictures + the 20,000 isolated real caption pictures,
    shuffled; one pass = 2,500 steps of 16, the same optimizer steps as exp24 phase A (2 passes over 20,000)."""
    text = [json.loads(x) for x in open(os.path.join(root, "exp24", "align_train.jsonl"))]
    real = [json.loads(x) for x in open(os.path.join(root, "exp30", "real_captions.jsonl"))]
    rows = text + real
    random.Random(seed).shuffle(rows)
    path = os.path.join(root, "exp30", "align_mix.jsonl")
    with open(path, "w") as fh:
        for s in rows:
            fh.write(json.dumps(s) + "\n")
    return {"text": len(text), "real": len(real), "rows": len(rows), "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    p.add_argument("--n", type=int, default=20000)
    a = p.parse_args()
    r = build(a.root, a.n)
    print(json.dumps({k: v for k, v in r.items() if k != "sets_checked"}, indent=1))
    print(json.dumps(mix(a.root), indent=1))

