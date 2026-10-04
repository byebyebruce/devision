"""v2 class 2: size and position questions from COCO instance boxes, with no category prior (kept out of the repo).

    uv run python scripts/data/v2_coco_spatial.py --root data --split train2014 --out-dir data/v2

Writes size.jsonl, position.jsonl (where one named object is) and relation.jsonl (where one is
relative to another). Fixes over cocoqa.py (docs/research/data-quality.md): there, "the dining table
is at the bottom" held 96% of the time and "the person is bigger than the tennis racket" 100%, so the
category names answered the question. Here every group that shares a question -- a category for
position, an unordered category pair for size and relation -- gets each answer equally often, and
groups where one answer never occurs are dropped.

Only categories with exactly one instance in the image are named (so "the dog" is unambiguous; crowd
boxes count as instances), every named object covers >= 1% of the image, and positions are only asked
when clear: centre in the outer 40% for position, a gap of >= 5% of the image between boxes for
relations, an area ratio >= 2 for size with both objects >= 3% of the image; a position
only when >= 75% of the box lies in that half. Evaluation images are skipped (COCO and VG ids).
"""
import argparse
import json
import os
import random
import zipfile
from collections import Counter, defaultdict
from itertools import combinations
from typing import Dict, List, Set, Tuple

MIN_AREA = 0.01
SIDE = 0.4
MIN_GAP = 0.05
MIN_RATIO = 2.0
MIN_SIZE_AREA = 0.03   # size: both objects clearly visible
MIN_IN_HALF = 0.75     # position: share of the box inside the half it is said to be in
POSITION = {"lr": ("left", "right", ["Is the {a} on the {x} or on the {y} side of the image?",
                                     "Is the {a} on the {x} or the {y} of the picture?"]),
            "tb": ("top", "bottom", ["Is the {a} in the {x} or in the {y} part of the image?",
                                     "Is the {a} near the {x} or the {y} of the picture?"])}
RELATION = {"lr": ("to the left of", "to the right of"), "tb": ("above", "below")}
SIZE = ["Which looks bigger in the image, the {x} or the {y}?", "Which takes up more of the picture, the {x} or the {y}?"]


def named_objects(anns: List[dict], area: float) -> Dict[int, dict]:
    """Category -> its single box, for categories with exactly one instance that is big enough."""
    count = Counter(a["category_id"] for a in anns)
    return {a["category_id"]: a for a in anns
            if count[a["category_id"]] == 1 and not a["iscrowd"] and a["area"] / area >= MIN_AREA}


def candidates(images, anns_by_image, held_out: Set[str]):
    """(kind, group, answer, payload) for every clear question; balancing happens afterwards."""
    for im in images:
        if "coco:%d" % im["id"] in held_out:
            continue
        W, H = float(im["width"]), float(im["height"])
        objs = named_objects(anns_by_image.get(im["id"], []), W * H)
        for cat, a in objs.items():
            x, y, w, h = a["bbox"]
            cx, cy = (x + w / 2) / W, (y + h / 2) / H
            in_left = max(0.0, min(x + w, W / 2) - x) / max(w, 1e-9)
            in_top = max(0.0, min(y + h, H / 2) - y) / max(h, 1e-9)
            if (cx < SIDE and in_left >= MIN_IN_HALF) or (cx > 1 - SIDE and 1 - in_left >= MIN_IN_HALF):
                yield "position", ("lr", cat), "left" if cx < 0.5 else "right", (im, cat)
            if (cy < SIDE and in_top >= MIN_IN_HALF) or (cy > 1 - SIDE and 1 - in_top >= MIN_IN_HALF):
                yield "position", ("tb", cat), "top" if cy < 0.5 else "bottom", (im, cat)
        for c1, c2 in combinations(sorted(objs), 2):
            (ax, ay, aw, ah), (bx, by, bw, bh) = objs[c1]["bbox"], objs[c2]["bbox"]
            # answers are always stated for the lower category id as the subject
            if bx - (ax + aw) > MIN_GAP * W:
                yield "relation", ("lr", c1, c2), "to the left of", (im, c1, c2)
            elif ax - (bx + bw) > MIN_GAP * W:
                yield "relation", ("lr", c1, c2), "to the right of", (im, c1, c2)
            if by - (ay + ah) > MIN_GAP * H:
                yield "relation", ("tb", c1, c2), "above", (im, c1, c2)
            elif ay - (by + bh) > MIN_GAP * H:
                yield "relation", ("tb", c1, c2), "below", (im, c1, c2)
            a1, a2 = objs[c1]["area"], objs[c2]["area"]
            if max(a1, a2) >= MIN_RATIO * min(a1, a2) and min(a1, a2) >= MIN_SIZE_AREA * W * H:
                yield "size", (c1, c2), c1 if a1 > a2 else c2, (im, c1, c2)


def balance(cands: List[Tuple], cap: int, rng: random.Random) -> List[Tuple]:
    """Per group, the same number of candidates for each of its two answers (at most `cap` each)."""
    groups: Dict[tuple, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for c in cands:
        groups[c[1]][c[2]].append(c)
    out = []
    for g in sorted(groups):
        by_answer = groups[g]
        if len(by_answer) < 2:
            continue
        k = min(cap, *(len(v) for v in by_answer.values()))
        for v in by_answer.values():
            rng.shuffle(v)
            out += v[:k]
    return out


def to_sample(c: Tuple, names: Dict[int, str], split: str, rng: random.Random) -> dict:
    kind, group, answer, payload = c
    im = payload[0]
    if kind == "position":
        axis, cat = group
        x, y, tmpls = POSITION[axis]
        opts = [x, y] if rng.random() < 0.5 else [y, x]
        q = {"type": "choice", "instructions": rng.choice(tmpls).format(a=names[cat], x=opts[0], y=opts[1]),
             "criteria": {o: None for o in opts}}
        probs = {o: float(o == answer) for o in opts}
        cats = [cat]
    elif kind == "relation":
        axis, c1, c2 = group
        p1, p2 = RELATION[axis]
        subj, obj, ans = c1, c2, answer
        if rng.random() < 0.5:  # ask it the other way round
            subj, obj, ans = c2, c1, p2 if answer == p1 else p1
        if rng.random() < 0.5:
            opts = [p1, p2] if rng.random() < 0.5 else [p2, p1]
            q = {"type": "choice", "instructions": "Is the %s %s or %s the %s?" % (names[subj], opts[0], opts[1], names[obj]),
                 "criteria": {o: None for o in opts}}
            probs = {o: float(o == ans) for o in opts}
        else:
            asked = rng.choice([p1, p2])
            q = {"type": "noul", "instructions": "Is the %s %s the %s?" % (names[subj], asked, names[obj])}
            probs = {"false": float(asked != ans), "true": float(asked == ans)}
        cats = [c1, c2]
    else:
        c1, c2 = group
        opts = [names[c1], names[c2]] if rng.random() < 0.5 else [names[c2], names[c1]]
        q = {"type": "choice", "instructions": rng.choice(SIZE).format(x=opts[0], y=opts[1]),
             "criteria": {o: None for o in opts}}
        probs = {o: float(o == names[answer]) for o in opts}
        cats = [c1, c2]
    axis = group[0] if kind != "size" else ""
    return {"id": "v2-%s:%s:%d:%s:%s" % (kind, split, im["id"], axis, "-".join(str(c) for c in cats)),
            "source": "coco-" + kind, "kind": kind, "category": " / ".join(names[c] for c in cats),
            "group": "%s:%s" % (axis, "/".join(names[c] for c in cats)),
            "answer_key": names[answer] if kind == "size" else answer,
            "axis": group[0] if kind != "size" else "",
            "image_id": "coco:%d" % im["id"], "image": "coco/%s/COCO_%s_%012d.jpg" % (split, split, im["id"]),
            "questions": {"q": q}, "gold": {"q": {"probabilities": probs}}}


def build(images, anns_by_image, names, split, held_out, caps: Dict[str, int], rng) -> Dict[str, List[dict]]:
    by_kind: Dict[str, list] = defaultdict(list)
    for c in candidates(images, anns_by_image, held_out):
        by_kind[c[0]].append(c)
    out = {}
    for kind, cands in by_kind.items():
        picked = balance(cands, caps[kind], rng)
        rng.shuffle(picked)
        out[kind] = [to_sample(c, names, split, rng) for c in picked]
    return out


def main(argv=None) -> None:
    import glob

    from convert import same_images
    from fetch import coco_annotations, vg_to_coco
    from prepare import _write_jsonl, read_jsonl

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="data")
    p.add_argument("--split", default="train2014")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--prefix", default="", help="file name prefix, e.g. eval_")
    p.add_argument("--cap-position", type=int, default=150, help="per category and answer")
    p.add_argument("--cap-relation", type=int, default=40, help="per category pair and answer")
    p.add_argument("--cap-size", type=int, default=40, help="per category pair and answer")
    p.add_argument("--exclude", nargs="*", default=[], help="more JSONL files whose images must not be used")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    with zipfile.ZipFile(coco_annotations(a.root)) as z:
        inst = json.load(z.open("annotations/instances_%s.json" % a.split))
    names = {c["id"]: c["name"] for c in inst["categories"]}
    anns = defaultdict(list)
    for x in inst["annotations"]:
        anns[x["image_id"]].append(x)
    evals = [f for f in glob.glob(os.path.join(a.root, "*.jsonl")) + glob.glob(os.path.join(a.root, "v2", "*.jsonl"))
             if os.path.basename(f).startswith(("val", "pope", "eval", "dev", "test_", "bench_"))]
    held_out = same_images({s["image_id"] for f in evals + a.exclude for s in read_jsonl(f)}, vg_to_coco(a.root))
    out = build(inst["images"], anns, names, a.split, held_out,
                {"position": a.cap_position, "relation": a.cap_relation, "size": a.cap_size}, random.Random(a.seed))
    os.makedirs(a.out_dir, exist_ok=True)
    for kind, rows in sorted(out.items()):
        path = os.path.join(a.out_dir, "%scoco_%s.jsonl" % (a.prefix, kind))
        print("%s: %d questions" % (path, _write_jsonl(path, rows)), dict(Counter(r["axis"] for r in rows)))


if __name__ == "__main__":
    main()
