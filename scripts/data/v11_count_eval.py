"""Round 11 counting evaluation sets (critic/plan-round11.md, revision 1, stage 0): counting questions on pictures
this project has never used, to tell whether v9b counts worse than v8 beyond the 136 test questions.

    uv run python scripts/data/v11_count_eval.py --root data

From VQAv2 val (COCO val2014), with data-v2's conversion (v2_vqa_choice: "how many", answers 0-10, >= 7 of 10
annotators agree, 2-5 options, distractors within 3 of the answer, recurring wordings and answers flattened):
  - data/v11/dev_count_transfer.jsonl  300 questions, 300 pictures: decides whether a recovery run is worth it;
  - data/v11/test_count_fresh.jsonl    600 questions, 600 pictures: sealed, scored only after a candidate passes;
  - data/v11/dev_count_existing.jsonl  the counting questions of data/v2/dev_vqa_choice.jsonl (COCO train2014,
    includes dev_mix's 288), unchanged ids, a second monitoring view.
No picture any project data file names (every JSONL under <root> except raw downloads and v11 itself: all
training files of v9b's lineage from caption alignment on, every held-out / monitoring / calibration set, the
diagnostic sets), by picture identity (COCO and VG ids of one picture are one), and no photo within NEAR_BITS of
any photo those files use (64-bit dHash; A-OKVQA's pictures are COCO pictures under hashed names). One question
per picture. Dev and test are split by stratum (answer 0 / 1-2 / 3-4 / 5-10 x option count), seed 20261006, never
looking at a model's answers. Stops if fewer than 900 pictures remain. Writes data/v11/EVAL_MANIFEST.json (ids,
answer and option distributions, filter counts, file SHA256s, the files scanned).
"""
import argparse
import glob
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SEED = 20261006
NEAR_BITS = 2
RAW_DIRS = ("raw", "coco", "gqa", "hf", "v11")
SYNTHETIC = ("binding/", "r9_binding/", "r9_paste/", "synthetic_spatial/", "v10/images/syn")


def bucket(answer: str) -> str:
    n = int(answer)
    return "0" if n == 0 else "1-2" if n <= 2 else "3-4" if n <= 4 else "5-10"


def base_picture(image_id: str) -> str:
    """"coco:N:flip" (data-v6's mirrored copy) is the picture coco:N."""
    return image_id[:-len(":flip")] if image_id.endswith(":flip") else image_id


def one_per_picture(rows: List[dict], rng: random.Random) -> List[dict]:
    by: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by[r["image_id"]].append(r)
    return [rng.choice(by[k]) for k in sorted(by)]


def split(rows: List[dict], dev: int, test: int, rng: random.Random) -> Tuple[List[dict], List[dict]]:
    """Take dev + test questions at random, then give each stratum (answer bucket x option count) its share of the
    dev part (largest remainder), so dev and test follow the same distribution."""
    rows = sorted(rows, key=lambda r: r["id"])
    rng.shuffle(rows)
    chosen = rows[:dev + test]
    if len(chosen) < dev + test:
        raise SystemExit("only %d pictures left, %d needed" % (len(chosen), dev + test))
    strata: Dict[tuple, List[dict]] = defaultdict(list)
    for r in chosen:
        strata[(bucket(r["answer_key"]), len(r["questions"]["q"]["criteria"]))].append(r)
    share = {k: len(v) * dev / len(chosen) for k, v in strata.items()}
    n = {k: int(s) for k, s in share.items()}
    for k in sorted(share, key=lambda k: (n[k] - share[k], k))[:dev - sum(n.values())]:
        n[k] += 1
    d, t = [], []
    for k in sorted(strata):
        d += strata[k][:n[k]]
        t += strata[k][n[k]:]
    return d, t


def project_files(root: str) -> List[str]:
    return sorted(f for f in glob.glob(os.path.join(root, "**", "*.jsonl"), recursive=True)
                  if os.path.relpath(f, root).split(os.sep)[0] not in RAW_DIRS)


def used_pictures(files: Sequence[str], v2c: Dict[str, str]) -> Tuple[Set[str], Set[str]]:
    """(picture identities, photo files) of every question in `files`."""
    from convert import picture, same_images
    from v9_mix import is_photo, original_image
    ids, photos = set(), set()
    for f in files:
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("image_id"):
                ids.add(base_picture(r["image_id"]))
            rel = r.get("image")
            if rel and is_photo(rel) and not rel.startswith(SYNTHETIC):
                photos.add(original_image(rel))
    return {picture(i, v2c) for i in same_images(ids, v2c)}, photos


def sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def write(path: str, rows: List[dict]) -> None:
    with open(path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


def describe(rows: List[dict]) -> dict:
    return {"questions": len(rows), "pictures": len({r["image_id"] for r in rows}),
            "answers": dict(sorted(Counter(r["answer_key"] for r in rows).items(), key=lambda kv: int(kv[0]))),
            "answer_buckets": dict(Counter(bucket(r["answer_key"]) for r in rows)),
            "options": dict(sorted(Counter(len(r["questions"]["q"]["criteria"]) for r in rows).items()))}


def main(argv=None) -> None:
    from convert import picture
    from fetch import download_coco_images, vg_to_coco, vqav2_records
    from v2_vqa_choice import convert, finish, problems, select, vocabularies
    from v9_mix import hashes, near_held

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    p.add_argument("--dev", type=int, default=300)
    p.add_argument("--test", type=int, default=600)
    p.add_argument("--seed", type=int, default=SEED)
    a = p.parse_args(argv)
    root, rng = a.root, random.Random(a.seed)
    out_dir = os.path.join(root, "v11")
    os.makedirs(out_dir, exist_ok=True)
    v2c = vg_to_coco(root)
    vocabs = vocabularies()

    files = project_files(root)
    used, photos = used_pictures(files, v2c)
    converted = [s for s in (convert(q, ann, "val2014", vocabs, rng) for q, ann in vqav2_records(root, "Val")) if s]
    counting = [s for s in converted if s["kind"] == "count"]
    fresh = [s for s in counting if picture(s["image_id"], v2c) not in used]
    flat = select(fresh, len(fresh), set(), rng)            # data-v2's wording / answer flattening
    single = one_per_picture(flat, rng)
    pool = finish(single, vocabs, rng)
    bad = [(s["id"], problems(s, vocabs)) for s in pool if problems(s, vocabs)]
    if bad:
        raise SystemExit("%d bad samples, e.g. %s" % (len(bad), bad[:3]))
    failed = set(download_coco_images(root, pool))
    pool = [s for s in pool if s["id"] not in failed]
    held_hash = hashes(root, photos)
    cand_hash = hashes(root, [s["image"] for s in pool])
    near = near_held(cand_hash, held_hash)
    clean = [s for s in pool if s["image"] in cand_hash and s["image"] not in near]
    for s in clean:
        s["id"] = s["id"].replace("v2-vqachoice:", "v11-count:")
    dev, test = split(clean, a.dev, a.test, rng)

    existing = [r for r in (json.loads(l) for l in open(os.path.join(root, "v2", "dev_vqa_choice.jsonl")) if l.strip())
                if r.get("kind") == "count"]
    outputs = {"dev_count_transfer": dev, "test_count_fresh": test, "dev_count_existing": existing}
    for name, rows in outputs.items():
        write(os.path.join(out_dir, name + ".jsonl"), rows)
    overlap = {picture(r["image_id"], v2c) for r in dev} & {picture(r["image_id"], v2c) for r in test}
    assert not overlap, "dev and test share pictures"
    manifest = {
        "seed": a.seed, "near_bits": NEAR_BITS,
        "steps": {"vqav2_val_converted": len(converted), "counting": len(counting),
                  "on_unused_pictures": len(fresh), "after_flattening": len(flat), "one_per_picture": len(single),
                  "with_options": len(pool) + len(failed), "image_download_failed": len(failed),
                  "near_duplicate_of_a_used_photo": len(near), "unreadable": len(pool) - len(cand_hash),
                  "eligible": len(clean)},
        "used_pictures": len(used), "used_photo_files": len(photos), "used_photos_hashed": len(held_hash),
        "files_scanned": [os.path.relpath(f, root) for f in files],
        "sets": {name: dict(describe(rows), file=os.path.join("v11", name + ".jsonl"),
                            sha256=sha256(os.path.join(out_dir, name + ".jsonl")), ids=[r["id"] for r in rows])
                 for name, rows in outputs.items()},
        "mismatched_control": "devision-eval --control mismatched: fixed-seed derangement of the set's pictures "
                              "(evaluate.picture_pairs, seed 0), the same for every checkpoint",
    }
    with open(os.path.join(out_dir, "EVAL_MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("files_scanned", "sets")}, indent=1))
    for name, s in manifest["sets"].items():
        print(name, {k: v for k, v in s.items() if k != "ids"})


if __name__ == "__main__":
    main()
