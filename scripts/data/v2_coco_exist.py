"""v2 class 1: "is there a <thing>?" from COCO instance boxes, balanced per category (kept out of the repo).

    uv run python scripts/data/v2_coco_exist.py --root data --split train2014 --per-category 300 --out data/v2/coco_exist.jsonl

Fixes over cocoqa.py's exist questions (docs/research/data-quality.md):
  - every category is asked equally often as "yes" and as "no", so the category name alone says
    nothing (v1: "chair" was "no" 82% of the time, "person" "yes" 78%);
  - negatives are hard: images without the category that hold the things it usually comes with
    (POPE's adversarial rule), ranked by how strongly they co-occur;
  - a negative is dropped if any of the image's five captions mentions the category (COCO boxes miss
    objects; the captions catch many of those), or if a crowd box of it exists;
  - positives must be visible at 256 px: the largest box of the category covers >= 1% of the image;
  - at most 2 questions per image, several phrasings.
Images of every evaluation set are skipped (COCO and Visual Genome ids).
"""
import argparse
import json
import os
import random
import re
import zipfile
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Set

# Words that, in a caption, mean the category is (probably) in the picture. Plurals are added.
SYNONYMS = {
    "person": "person people man men woman women boy girl child children kid guy lady player skier surfer "
              "skateboarder snowboarder rider batter catcher pitcher crowd baby toddler someone adult",
    "bicycle": "bicycle bike cyclist", "car": "car taxi suv sedan", "motorcycle": "motorcycle motorbike scooter",
    "airplane": "airplane plane jet aircraft airliner", "bus": "bus", "train": "train locomotive tram",
    "truck": "truck pickup lorry", "boat": "boat ship canoe kayak sailboat yacht ferry", "traffic light": "traffic light stoplight",
    "fire hydrant": "hydrant", "stop sign": "stop sign", "parking meter": "parking meter meter", "bench": "bench",
    "bird": "bird duck goose pigeon seagull gull parrot swan", "cat": "cat kitten kitty", "dog": "dog puppy pup",
    "horse": "horse pony", "sheep": "sheep lamb", "cow": "cow cattle bull calf", "elephant": "elephant",
    "bear": "bear", "zebra": "zebra", "giraffe": "giraffe", "backpack": "backpack", "umbrella": "umbrella parasol",
    "handbag": "handbag purse bag", "tie": "tie necktie", "suitcase": "suitcase luggage", "frisbee": "frisbee",
    "skis": "ski skis", "snowboard": "snowboard", "sports ball": "ball football soccer baseball tennis",
    "kite": "kite", "baseball bat": "bat", "baseball glove": "glove mitt", "skateboard": "skateboard",
    "surfboard": "surfboard surf board", "tennis racket": "racket racquet", "bottle": "bottle",
    "wine glass": "wine glass glass", "cup": "cup mug", "fork": "fork", "knife": "knife", "spoon": "spoon",
    "bowl": "bowl", "banana": "banana", "apple": "apple", "sandwich": "sandwich burger hamburger sub",
    "orange": "orange", "broccoli": "broccoli", "carrot": "carrot", "hot dog": "hot dog hotdog", "pizza": "pizza",
    "donut": "donut doughnut", "cake": "cake cupcake", "chair": "chair seat stool", "couch": "couch sofa",
    "potted plant": "plant flower", "bed": "bed", "dining table": "table", "toilet": "toilet",
    "tv": "tv television monitor screen", "laptop": "laptop computer notebook", "mouse": "mouse",
    "remote": "remote controller", "keyboard": "keyboard", "cell phone": "phone cellphone smartphone",
    "microwave": "microwave", "oven": "oven stove", "toaster": "toaster", "sink": "sink",
    "refrigerator": "refrigerator fridge", "book": "book", "clock": "clock", "vase": "vase",
    "scissors": "scissors", "teddy bear": "teddy bear stuffed", "hair drier": "hair dryer drier blow",
    "toothbrush": "toothbrush",
}
TEMPLATES = ["Is there {a} {c} in the image?", "Is there {a} {c} in the image?", "Is there {a} {c} in this picture?",
             "Can you see {a} {c}?", "Does this image contain {a} {c}?", "Is {a} {c} visible in the photo?"]
MIN_POS_AREA = 0.01


def article(word: str) -> str:
    return "an" if word[0] in "aeiou" else "a"


def mention_patterns() -> Dict[str, re.Pattern]:
    pats = {}
    for cat, words in SYNONYMS.items():
        alts = set()
        for w in words.split():
            alts |= {w, w + "s", w + "es"}
        # multi-word names ("traffic light") are matched as phrases too
        alts.add(cat)
        pats[cat] = re.compile(r"\b(%s)\b" % "|".join(sorted(map(re.escape, alts), key=len, reverse=True)))
    return pats


def mentions(captions: Iterable[str], pattern: re.Pattern) -> bool:
    return any(pattern.search(c.lower()) for c in captions)


def build(images: List[dict], anns_by_image: Dict[int, List[dict]], names: Dict[int, str],
          captions: Dict[int, List[str]], split: str, per_category: int, held_out: Set[str],
          rng: random.Random) -> List[dict]:
    pats = mention_patterns()
    area = {im["id"]: im["width"] * im["height"] for im in images}
    present = {}       # image -> categories with any box (crowd included)
    big = {}           # image -> categories whose largest box is >= MIN_POS_AREA
    for im in images:
        anns = anns_by_image.get(im["id"], [])
        present[im["id"]] = {a["category_id"] for a in anns}
        big[im["id"]] = {a["category_id"] for a in anns if not a["iscrowd"] and a["area"] / area[im["id"]] >= MIN_POS_AREA}
    cooccur: Dict[int, Counter] = defaultdict(Counter)
    for cats in present.values():
        for c in cats:
            cooccur[c].update(cats - {c})
    usable = [im["id"] for im in images if "coco:%d" % im["id"] not in held_out and present[im["id"]]]
    used = Counter()
    out = []

    def add(iid, cat, truth):
        used[iid] += 1
        c = names[cat]
        q = rng.choice(TEMPLATES).format(a=article(c), c=c)
        out.append({"id": "v2-exist:%s:%d:%d" % (split, iid, cat), "source": "coco-exist", "kind": "exist",
                    "category": c, "group": c, "answer_key": "true" if truth else "false", "image_id": "coco:%d" % iid,
                    "image": "coco/%s/COCO_%s_%012d.jpg" % (split, split, iid),
                    "questions": {"q": {"type": "noul", "instructions": q}},
                    "gold": {"q": {"probabilities": {"false": float(not truth), "true": float(truth)}}}})

    for cat in sorted(names):
        pat = pats[names[cat]]
        pos = [i for i in usable if cat in big[i]]
        # hard negatives: absent, not mentioned, and the image holds the category's usual companions
        neg = [i for i in usable if cat not in present[i] and not mentions(captions.get(i, []), pat)]
        score = lambda i: sum(cooccur[cat][o] for o in present[i])
        rng.shuffle(pos)
        rng.shuffle(neg)
        neg.sort(key=score, reverse=True)
        neg = neg[:len(neg) // 4] or neg   # keep the top quarter by co-occurrence, then sample from it
        rng.shuffle(neg)
        k = min(per_category, len(pos), len(neg))
        for pool, truth in ((pos, True), (neg, False)):
            n = 0
            for i in pool:
                if n == k:
                    break
                if used[i] < 2:
                    add(i, cat, truth)
                    n += 1
    rng.shuffle(out)
    return out


def main(argv=None) -> None:
    from convert import same_images
    from fetch import coco_annotations, vg_to_coco
    from prepare import _write_jsonl, read_jsonl
    import glob

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="data")
    p.add_argument("--split", default="train2014")
    p.add_argument("--per-category", type=int, default=300)
    p.add_argument("--out", required=True)
    p.add_argument("--exclude", nargs="*", default=[], help="more JSONL files whose images must not be used")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    with zipfile.ZipFile(coco_annotations(a.root)) as z:
        inst = json.load(z.open("annotations/instances_%s.json" % a.split))
        caps_json = json.load(z.open("annotations/captions_%s.json" % a.split))
    names = {c["id"]: c["name"] for c in inst["categories"]}
    anns = defaultdict(list)
    for x in inst["annotations"]:
        anns[x["image_id"]].append(x)
    captions = defaultdict(list)
    for x in caps_json["annotations"]:
        captions[x["image_id"]].append(x["caption"])
    evals = [f for f in glob.glob(os.path.join(a.root, "*.jsonl")) + glob.glob(os.path.join(a.root, "v2", "*.jsonl"))
             if os.path.basename(f).startswith(("val", "pope", "eval", "dev", "test_", "bench_")) and os.path.abspath(f) != os.path.abspath(a.out)]
    held_out = same_images({s["image_id"] for f in evals + a.exclude for s in read_jsonl(f)}, vg_to_coco(a.root))
    rows = build(inst["images"], anns, names, captions, a.split, a.per_category, held_out, random.Random(a.seed))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    print("%s: %d questions, %d categories, evaluation sets held out: %s" % (
        a.out, _write_jsonl(a.out, rows), len({r["category"] for r in rows}), sorted(map(os.path.basename, evals))))


if __name__ == "__main__":
    main()
