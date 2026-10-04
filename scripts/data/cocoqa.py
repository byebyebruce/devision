"""Clean grounding questions generated from COCO instance boxes (kept out of the repo).

    uv run python scripts/data/cocoqa.py --root data --split train2014 --limit 60000 --out data/cocoqa_train.jsonl
    uv run python scripts/data/cocoqa.py --root data --split val2014 --limit 1000 --out data/val_cocoqa.jsonl

Question kinds, each answered from the boxes alone:
  exist     noul  "Is there a fork in the image?"  -- negatives are mostly categories that often
                  co-occur with what is in the image (as POPE's adversarial split), some popular/random
  absolute  choice "Is the dog on the left or on the right side of the image?" (also top/bottom)
  relative  choice "Is the cup to the left or to the right of the laptop?" (also above/below), and
            noul "Is the cup to the left of the laptop?"
  size      choice "Which looks bigger in the image, the dog or the chair?"
Only categories with a single instance in the image are asked about by name, so "the dog" is
unambiguous; positions are only asked when the gap is clear. Images of any evaluation set (also under
their Visual Genome id) are skipped.
"""
import argparse
import glob
import json
import os
import random
import zipfile
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence

Sample = Dict[str, Any]

MIN_EXIST_AREA = 0.005   # share of the image; smaller positives are hard to see at 256 px
MIN_NAMED_AREA = 0.01    # objects asked about by position or size
SIDE = 0.4               # absolute: box centre left of 0.4 W or right of 0.6 W
MIN_GAP = 0.05           # relative: boxes separated by at least this share of the image
MIN_SIZE_RATIO = 2.0


def _article(name: str) -> str:
    return ("an " if name[0] in "aeiou" else "a ") + name


def _sample(kind: str, image_id: int, split: str, n: int, question: Dict[str, Any],
            probs: Dict[str, float]) -> Sample:
    return {"id": "cocoqa:%s:%d:%d" % (kind, image_id, n), "source": "cocoqa", "kind": kind,
            "image_id": "coco:%d" % image_id, "image": "coco/%s/COCO_%s_%012d.jpg" % (split, split, image_id),
            "questions": {"q": question}, "gold": {"q": {"probabilities": probs}}}


def _noul(text: str, truth: bool) -> Dict[str, Any]:
    return {"type": "noul", "instructions": text}, {"false": float(not truth), "true": float(truth)}


def _choice(text: str, options: Sequence[str], answer: str):
    return ({"type": "choice", "instructions": text, "criteria": {o: None for o in options}},
            {o: float(o == answer) for o in options})


def image_questions(image: Dict[str, Any], anns: List[Dict[str, Any]], names: Dict[int, str],
                    cooccur: Dict[int, Counter], popular: List[int], split: str,
                    rng: random.Random) -> List[Sample]:
    """All questions for one image; `anns` are its non-crowd instance annotations."""
    W, H = float(image["width"]), float(image["height"])
    area = W * H
    count = Counter(a["category_id"] for a in anns)
    present = set(count)
    out: List[Sample] = []

    def add(kind, q_probs):
        out.append(_sample(kind, image["id"], split, len(out), *q_probs))

    # exist: one positive, one negative
    pos = [c for c in present if any(a["category_id"] == c and a["area"] / area >= MIN_EXIST_AREA for a in anns)]
    if pos:
        c = rng.choice(pos)
        add("exist", _noul("Is there %s in the image?" % _article(names[c]), True))
    r = rng.random()
    if r < 0.5:   # adversarial: what usually comes with these objects
        votes = Counter()
        for c in present:
            votes.update(cooccur[c])
        cands = [c for c, _ in votes.most_common() if c not in present][:5]
    elif r < 0.75:
        cands = [c for c in popular if c not in present][:5]
    else:
        cands = [c for c in names if c not in present]
    if cands:
        add("exist", _noul("Is there %s in the image?" % _article(names[rng.choice(cands)]), False))

    named = [a for a in anns if count[a["category_id"]] == 1 and a["area"] / area >= MIN_NAMED_AREA]
    rng.shuffle(named)

    def centre(a):
        x, y, w, h = a["bbox"]
        return (x + w / 2) / W, (y + h / 2) / H

    # absolute position
    for a in named[:1]:
        cx, cy = centre(a)
        n = names[a["category_id"]]
        axes = []
        if cx < SIDE or cx > 1 - SIDE:
            axes.append(("left", "right", "left" if cx < 0.5 else "right",
                         "Is the %s on the %s or on the %s side of the image?"))
        if cy < SIDE or cy > 1 - SIDE:
            axes.append(("top", "bottom", "top" if cy < 0.5 else "bottom",
                         "Is the %s in the %s or in the %s part of the image?"))
        if axes:
            a_, b_, ans, tmpl = rng.choice(axes)
            opts = [a_, b_] if rng.random() < 0.5 else [b_, a_]
            add("absolute", _choice(tmpl % (n, opts[0], opts[1]), opts, ans))

    # relative position on up to 3 pairs of named objects, size on the first pair
    pairs = [(named[i], named[j]) for i in range(len(named)) for j in range(i + 1, len(named))][:3]
    for k, (a, b) in enumerate(pairs):
        na, nb = names[a["category_id"]], names[b["category_id"]]
        ax, ay, aw, ah = a["bbox"]
        bx, by, bw, bh = b["bbox"]
        rel = []
        if bx - (ax + aw) > MIN_GAP * W:
            rel.append(("to the left of", "to the right of", "to the left of"))
        elif ax - (bx + bw) > MIN_GAP * W:
            rel.append(("to the left of", "to the right of", "to the right of"))
        if by - (ay + ah) > MIN_GAP * H:
            rel.append(("above", "below", "above"))
        elif ay - (by + bh) > MIN_GAP * H:
            rel.append(("above", "below", "below"))
        if rel:
            p, q, ans = rng.choice(rel)
            if rng.random() < 0.5:
                opts = [p, q] if rng.random() < 0.5 else [q, p]
                add("relative", _choice("Is the %s %s or %s the %s?" % (na, opts[0], opts[1], nb), opts, ans))
            else:
                asked = rng.choice([p, q])
                add("relative", _noul("Is the %s %s the %s?" % (na, asked, nb), asked == ans))
        big, small = (a, b) if a["area"] >= b["area"] else (b, a)
        if k == 0 and big["area"] >= MIN_SIZE_RATIO * small["area"]:
            opts = [na, nb] if rng.random() < 0.5 else [nb, na]
            add("size", _choice("Which looks bigger in the image, the %s or the %s?" % tuple(opts), opts,
                                names[big["category_id"]]))
    return out


def load_instances(root: str, split: str):
    from fetch import coco_annotations
    with zipfile.ZipFile(coco_annotations(root)) as z:
        d = json.load(z.open("annotations/instances_%s.json" % split))
    names = {c["id"]: c["name"] for c in d["categories"]}
    by_image = defaultdict(list)
    for a in d["annotations"]:
        if not a["iscrowd"]:
            by_image[a["image_id"]].append(a)
    return d["images"], by_image, names


def main(argv=None) -> None:
    from convert import same_images, select
    from fetch import coco_annotations, download_coco_images, vg_to_coco
    from prepare import _write_jsonl, read_jsonl

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="data")
    p.add_argument("--split", default="train2014")
    p.add_argument("--limit", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    rng = random.Random(a.seed)

    images, by_image, names = load_instances(a.root, a.split)
    cooccur: Dict[int, Counter] = defaultdict(Counter)
    freq = Counter()
    for anns in by_image.values():
        cats = {x["category_id"] for x in anns}
        freq.update(cats)
        for c in cats:
            cooccur[c].update(cats - {c})
    popular = [c for c, _ in freq.most_common()]

    evals = [f for f in glob.glob(os.path.join(a.root, "*.jsonl"))
             if os.path.basename(f).startswith(("val", "pope", "test_", "bench_")) and os.path.abspath(f) != os.path.abspath(a.out)]
    held_out = same_images({s["image_id"] for f in evals for s in read_jsonl(f)}, vg_to_coco(a.root))
    by_kind: Dict[str, List[Sample]] = defaultdict(list)
    for img in images:
        if "coco:%d" % img["id"] in held_out or not by_image.get(img["id"]):
            continue
        for s in image_questions(img, by_image[img["id"]], names, cooccur, popular, a.split, rng):
            by_kind[s["kind"]].append(s)
    print("generated:", {k: len(v) for k, v in by_kind.items()})
    # equal share per kind (exist is noul and gets yes/no balanced by select)
    share = a.limit // len(by_kind)
    picked: List[Sample] = []
    for kind, ss in sorted(by_kind.items()):
        picked += select(ss, share, set(), a.seed)
    rng.shuffle(picked)
    failed = set(download_coco_images(a.root, picked))
    picked = [s for s in picked if s["id"] not in failed]
    print("%s: %d samples (%s), %d images failed" % (a.out, _write_jsonl(a.out, picked),
          dict(Counter(s["kind"] for s in picked)), len(failed)))


if __name__ == "__main__":
    main()
