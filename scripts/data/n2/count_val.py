"""v0.4 evaluation: counting questions on COCO val2014 photos (`test_count_coco`, plan
docs/experiments/2026-10-21-v04-plan.md, stage 0).

    uv run python scripts/data/n2/count_val.py --root data [--per-count 100]
    uv run python scripts/data/n2/count_val.py --root data --split train2014 --per-count 3000   # data-n2 block C

With --split train2014 the same rules build training questions (<root>/n2/count_train.jsonl): pictures held out by
n2/held.py are skipped; counts with fewer candidates than --per-count keep all they have (the MANIFEST says how
many).

Every COCO val2014 picture is outside v0.3's training lineage (the project held out all of val2014: our test sets
and POPE sit on it), so this set is unseen by v0.3 for certain, unlike a rebuild of data/v11/test_count_fresh
(COCO train2014, whose held-out choice depended on files this machine never rebuilt). data-n2 must keep val2014 out.

Rules (pure functions below, tested in test_count_val.py):
- a (picture, category) pair is countable when the category has no crowd annotation in the picture and every
  instance covers at least `min_area` of the picture (tiny background instances make the count ambiguous);
- counts 1..10 are asked `per_count` times each, count 0 (a category COCO says is absent; COCO labels its 80
  categories exhaustively) `per_count` times too; one question per picture;
- options are always the eleven counts "0".."10" in a per-question shuffled order (gold-dependent windows let a
  picture-blind rule win, r13);
- category names with no clean plural (skis, scissors, ...) are not asked.

Writes <root>/n2/test_count_coco.jsonl and test_count_coco.MANIFEST.json (counts per gold, per category, the
picture-blind baselines).
"""
import argparse
import json
import os
import random
import sys
import zipfile
from collections import Counter, defaultdict
from typing import Dict, List, Optional

OPTIONS = [str(n) for n in range(11)]
PLURAL = {"person": "people", "mouse": "computer mice", "sheep": "sheep", "bus": "buses", "bench": "benches",
          "couch": "couches", "wine glass": "wine glasses", "knife": "knives", "sandwich": "sandwiches",
          "toothbrush": "toothbrushes", "skateboard": "skateboards", "potted plant": "potted plants",
          "dining table": "dining tables", "teddy bear": "teddy bears", "hair drier": "hair driers",
          "giraffe": "giraffes", "cell phone": "cell phones", "tv": "TVs", "glass": "glasses"}
NOT_ASKED = {"skis", "scissors", "sports ball", "broccoli", "carrot", "donut", "banana", "apple", "orange",
             "book", "kite"}  # no clean plural, or piles / clusters where instances are ambiguous


def plural(name: str) -> Optional[str]:
    if name in NOT_ASKED:
        return None
    return PLURAL.get(name, name + "s")


def countable(anns: List[dict], category: int, area: float, min_area: float) -> Optional[int]:
    """The number of `category` instances in a picture's annotations, or None when the count is ambiguous."""
    mine = [x for x in anns if x["category_id"] == category]
    if any(x.get("iscrowd") for x in mine):
        return None
    if any(x["area"] < min_area * area for x in mine):
        return None
    return len(mine)


def pick(images: List[dict], anns: Dict[int, List[dict]], names: Dict[int, str], per_count: int,
         min_area: float, rng: random.Random) -> List[dict]:
    """(image, category, count) triples: per_count for each count 0..10, one per picture."""
    by_count = defaultdict(list)
    for im in images:
        a = anns.get(im["id"], [])
        area = im["width"] * im["height"]
        present = {x["category_id"] for x in a}
        for c, name in names.items():
            if plural(name) is None:
                continue
            n = countable(a, c, area, min_area) if c in present else 0
            if n is not None and n <= 10 and (n > 0 or a):
                by_count[n].append((im, c))
    used, out = set(), []
    for n in range(11):
        cands = by_count[n][:]
        rng.shuffle(cands)
        k = 0
        for im, c in cands:
            if k == per_count:
                break
            if im["id"] in used:
                continue
            used.add(im["id"])
            out.append((im, c, n))
            k += 1
    return out


def row(im: dict, category: str, n: int, split: str, rng: random.Random) -> dict:
    opts = OPTIONS[:]
    rng.shuffle(opts)
    return {"id": "n2-count:%s:%d:%s" % (split, im["id"], category.replace(" ", "_")), "source": "coco-count-val",
            "kind": "count", "category": category, "group": "coco:%d" % im["id"], "answer_key": str(n),
            "image_id": "coco:%d" % im["id"], "image": "coco/%s/COCO_%s_%012d.jpg" % (split, split, im["id"]),
            "questions": {"q": {"type": "choice",
                                "instructions": "How many %s are there in the image?" % plural(category),
                                "criteria": {o: None for o in opts}}},
            "gold": {"q": {"probabilities": {o: float(o == str(n)) for o in opts}}}}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    p.add_argument("--per-count", type=int, default=100)
    p.add_argument("--min-area", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split", default="val2014", choices=["val2014", "train2014"])
    a = p.parse_args(argv)
    split = a.split
    with zipfile.ZipFile(os.path.join(a.root, "raw", "coco", "annotations_trainval2014.zip")) as z:
        inst = json.load(z.open("annotations/instances_%s.json" % split))
    names = {c["id"]: c["name"] for c in inst["categories"]}
    anns = defaultdict(list)
    for x in inst["annotations"]:
        anns[x["image_id"]].append(x)
    rng = random.Random(a.seed)
    images = sorted(inst["images"], key=lambda im: im["id"])
    images = [im for im in images
              if os.path.exists(os.path.join(a.root, "coco", split, "COCO_%s_%012d.jpg" % (split, im["id"])))]
    if split == "train2014":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import held
        h = held.load(a.root)
        images = [im for im in images if not h.is_held("coco:%d" % im["id"], "coco/%s/COCO_%s_%012d.jpg"
                                                        % (split, split, im["id"]))]
    rows = [row(im, names[c], n, split, rng) for im, c, n in pick(images, anns, names, a.per_count, a.min_area, rng)]
    out = os.path.join(a.root, "n2", "test_count_coco.jsonl" if split == "val2014" else "count_train.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    golds = Counter(r["answer_key"] for r in rows)
    best = max(golds, key=lambda g: golds[g])
    manifest = {"built_by": "scripts/data/n2/count_val.py", "split": split, "questions": len(rows),
                "per_gold": dict(sorted(golds.items(), key=lambda kv: int(kv[0]))),
                "per_category": dict(Counter(r["category"] for r in rows).most_common()),
                "min_area": a.min_area, "seed": a.seed,
                "option_baselines": {"uniform_guess": round(1 / len(OPTIONS), 4),
                                     "best_constant_answer": {"answer": best,
                                                              "accuracy": round(golds[best] / len(rows), 4)}},
                "note": ("COCO val2014 is outside v0.3's training lineage; " if split == "val2014" else "training; ") + "options are always 0..10"}
    with open(out.replace(".jsonl", ".MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
