"""Round 13 counterfactual binding scenes (docs/experiments/2026-10-08-r13-binding-plan.md, revision 2, 1.2).

    uv run python scripts/data/r13/synth.py --root data --train 1400 --dev 300 --test 300

A scene family is one base scene with 2 kinds of shapes (shape + colour + size per kind, 1-4 of each, 3-6
objects) and its variants, all in one `group`:

  base       the scene
  recolour   one kind gets a new colour (its colour answer changes, the other kind's must not)
  recolour2  the other kind gets a new colour
  count      one object of one kind removed (or added): its count changes, nothing else

Every picture asks about EVERY kind (so "always answer about the big one" and "always answer about the small
one" both fail): colour (choice of 4 colours, the other kinds' colours among the options), count (choice of 4
neighbouring numbers, the other kinds' counts among them when they differ), "Is there a <colour> <shape>?" true
and false -- the false one uses another kind's colour (the salient-object trap) -- and "Are there more Xs than
Ys?" when counts differ.

Per question `meta` records the salience stratum (target kind's total area / the largest other kind's:
>= 1.5 "target_salient", <= 1/1.5 "target_minor", else "close"; families are drawn so the three are about equal)
and two name-blind baselines: the answer read off the most salient kind and off the least salient kind.
`boxes` holds the asked kind's boxes ([x, y, w, h] in picture pixels) for the grounding loss.

Backgrounds: half plain colours, half COCO train2014 photos (all train2014 pictures were seen by caption
alignment, so they are not claimed unseen); train, dev and test draw backgrounds from disjoint photo pools.
Colour x shape combinations in HELD_OUT never appear in train or dev scenes; every test family uses one.
"""
import argparse
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw

SHAPES = ["circle", "square", "triangle", "diamond"]
COLORS = {"red": (210, 50, 45), "green": (45, 150, 70), "blue": (45, 90, 210), "yellow": (235, 200, 30),
          "purple": (130, 60, 170), "orange": (240, 130, 30)}
HELD_OUT = {("purple", "triangle"), ("orange", "circle"), ("yellow", "diamond"), ("blue", "square")}
NUMBERS = ["zero", "one", "two", "three", "four", "five", "six"]
PLAIN = [(244, 232, 220), (236, 238, 230), (240, 240, 245), (250, 248, 240), (225, 230, 235), (60, 60, 64)]
W, H = 640, 420
SPLIT_SEED = {"train": 11, "dev": 22, "test": 33}


def plural(shape: str) -> str:
    return shape + "s"


def draw(d: ImageDraw.ImageDraw, shape: str, box: Tuple[int, int, int, int], fill, outline=None):
    x0, y0, x1, y1 = box
    if shape == "circle":
        d.ellipse(box, fill=fill, outline=outline, width=3)
    elif shape == "square":
        d.rectangle(box, fill=fill, outline=outline, width=3)
    elif shape == "triangle":
        d.polygon([((x0 + x1) / 2, y0), (x1, y1), (x0, y1)], fill=fill, outline=outline)
    else:
        d.polygon([((x0 + x1) / 2, y0), (x1, (y0 + y1) / 2), ((x0 + x1) / 2, y1), (x0, (y0 + y1) / 2)],
                  fill=fill, outline=outline)


def place(rng: random.Random, sizes: List[int]) -> Optional[List[Tuple[int, int, int, int]]]:
    """Non-overlapping boxes for objects of the given sizes, or None."""
    boxes: List[Tuple[int, int, int, int]] = []
    for size in sizes:
        for _ in range(300):
            x, y = rng.randint(8, W - 8 - size), rng.randint(8, H - 8 - size)
            b = (x, y, x + size, y + size)
            if all(b[2] + 10 < o[0] or o[2] + 10 < b[0] or b[3] + 10 < o[1] or o[3] + 10 < b[1] for o in boxes):
                boxes.append(b)
                break
        else:
            return None
    return boxes


def base_scene(rng: random.Random, split: str) -> Dict[str, Any]:
    """Two kinds with shape, colour, count and size. Two thirds of the families have a clear salience gap (one
    kind's total area 2-4 times the other's: one "target_salient" and one "target_minor" question per picture),
    one third are "close" (ratio 0.8-1.25), so the three strata come out about equal."""
    while True:
        shapes = rng.sample(SHAPES, 2)
        colours = rng.sample(list(COLORS), 2)
        combos = set(zip(colours, shapes))
        if (split == "test") != bool(combos & HELD_OUT):
            continue
        counts = [rng.randint(1, 4), rng.randint(1, 4)]
        if not 3 <= sum(counts) <= 6:
            continue
        ratio = rng.uniform(2, 4) if rng.random() < 2 / 3 else rng.uniform(0.8, 1.25)
        if rng.random() < 0.5:
            ratio = 1 / ratio
        size0 = rng.randint(40, 95)
        size1 = round(size0 * (counts[0] / (counts[1] * ratio)) ** 0.5)   # area1 / area0 = 1 / ratio
        if not 30 <= size1 <= 120:
            continue
        return {"kinds": [{"shape": sh, "colour": c, "count": n, "size": sz}
                          for sh, c, n, sz in zip(shapes, colours, counts, (size0, size1))]}


def render(kinds: List[Dict[str, Any]], rng: random.Random, background) -> Optional[Tuple[Image.Image, List[List[List[int]]]]]:
    sizes = [k["size"] for k in kinds for _ in range(k["count"])]
    boxes = place(rng, sizes)
    if boxes is None:
        return None
    if isinstance(background, str):
        with Image.open(background) as photo:
            im = photo.convert("RGB").resize((W, H))
    else:
        im = Image.new("RGB", (W, H), background)
    d = ImageDraw.Draw(im)
    per_kind: List[List[List[int]]] = []
    i = 0
    for k in kinds:
        bs = []
        for _ in range(k["count"]):
            b = boxes[i]
            i += 1
            draw(d, k["shape"], b, COLORS[k["colour"]], outline=(255, 255, 255) if isinstance(background, str) else None)
            bs.append([b[0], b[1], b[2] - b[0], b[3] - b[1]])
        per_kind.append(bs)
    return im, per_kind


def area(k: Dict[str, Any]) -> float:
    return k["count"] * k["size"] ** 2


def stratum(kinds, i) -> str:
    other = max(area(k) for j, k in enumerate(kinds) if j != i)
    r = area(kinds[i]) / other
    return "target_salient" if r >= 1.5 else "target_minor" if r <= 1 / 1.5 else "close"


def questions(kinds: List[Dict[str, Any]], rng: random.Random):
    """(questions, gold, meta) for one picture: every kind asked."""
    qs, gold, meta = {}, {}, {}
    order = sorted(range(len(kinds)), key=lambda j: area(kinds[j]))
    least, most = kinds[order[0]], kinds[order[-1]]
    for i, k in enumerate(kinds):
        sh, st = k["shape"], stratum(kinds, i)
        # colour
        others = [o["colour"] for j, o in enumerate(kinds) if j != i]
        pool = [c for c in COLORS if c != k["colour"] and c not in others]
        opts = [k["colour"]] + others[:2] + rng.sample(pool, 4 - 1 - len(others[:2]))
        rng.shuffle(opts)
        qid = "colour_%s" % sh
        qs[qid] = {"type": "choice", "instructions": "What color are the %s?" % plural(sh), "criteria": {o: None for o in opts}}
        gold[qid] = {"probabilities": {o: float(o == k["colour"]) for o in opts}}
        meta[qid] = {"task": "colour", "kind": i, "stratum": st, "salient": most["colour"], "minor": least["colour"]}
        # count: the true number, the other kinds' numbers, then the nearest others (1-6)
        n = k["count"]
        nums = [n] + [o["count"] for o in kinds if o is not k and o["count"] != n]
        for x in sorted(range(1, 7), key=lambda x: (abs(x - n), x)):
            if len(set(nums)) >= 4:
                break
            nums.append(x)
        words = [NUMBERS[x] for x in sorted(set(nums))[:4]] if n in sorted(set(nums))[:4] else \
            [NUMBERS[x] for x in ([n] + [y for y in sorted(set(nums)) if y != n][:3])]
        rng.shuffle(words)
        qid = "count_%s" % sh
        qs[qid] = {"type": "choice", "instructions": "How many %s are there?" % plural(sh), "criteria": {w: None for w in words}}
        gold[qid] = {"probabilities": {w: float(w == NUMBERS[n]) for w in words}}
        meta[qid] = {"task": "count", "kind": i, "stratum": st, "salient": NUMBERS[most["count"]], "minor": NUMBERS[least["count"]]}
        # composite existence, true and trap
        qid = "exists_%s" % sh
        qs[qid] = {"type": "noul", "instructions": "Is there a %s %s?" % (k["colour"], sh)}
        gold[qid] = {"probabilities": {"false": 0.0, "true": 1.0}}
        meta[qid] = {"task": "exists_true", "kind": i, "stratum": st}
        trap = rng.choice(others)
        qid = "trap_%s" % sh
        qs[qid] = {"type": "noul", "instructions": "Is there a %s %s?" % (trap, sh)}
        gold[qid] = {"probabilities": {"false": 1.0, "true": 0.0}}
        meta[qid] = {"task": "exists_trap", "kind": i, "stratum": st}
    # one comparison between the first two kinds when their counts differ
    a, b = kinds[0], kinds[1]
    if a["count"] != b["count"]:
        qid = "more_%s_%s" % (a["shape"], b["shape"])
        qs[qid] = {"type": "noul", "instructions": "Are there more %s than %s?" % (plural(a["shape"]), plural(b["shape"]))}
        truth = a["count"] > b["count"]
        gold[qid] = {"probabilities": {"false": float(not truth), "true": float(truth)}}
        meta[qid] = {"task": "compare", "kind": 0, "stratum": stratum(kinds, 0),
                     "salient": "true" if area(a) >= area(b) else "false"}
    return qs, gold, meta


def variants(base: Dict[str, Any], rng: random.Random, split: str) -> List[Tuple[str, List[Dict[str, Any]]]]:
    kinds = base["kinds"]
    out = [("base", kinds)]
    for v, idx in (("recolour", 0), ("recolour2", 1)):
        used = {k["colour"] for k in kinds}
        shape = kinds[idx]["shape"]
        choices = [c for c in COLORS if c not in used and (split == "test" or (c, shape) not in HELD_OUT)]
        new = [dict(k) for k in kinds]
        new[idx]["colour"] = rng.choice(choices)
        out.append((v, new))
    new = [dict(k) for k in kinds]
    j = rng.randrange(len(new))
    new[j]["count"] += -1 if new[j]["count"] > 1 and (rng.random() < 0.5 or sum(k["count"] for k in new) >= 6) else 1
    out.append(("count", new))
    return out


def backgrounds(root: str, split: str, n: int) -> List[str]:
    """Disjoint COCO train2014 photo pools per split (deterministic)."""
    folder = os.path.join(root, "coco", "train2014")
    names = sorted(os.listdir(folder)) if os.path.isdir(folder) else []
    if not names:
        return []
    third = len(names) // 3
    pool = {"train": names[:third], "dev": names[third:2 * third], "test": names[2 * third:]}[split]
    rng = random.Random(SPLIT_SEED[split])
    return [os.path.join(folder, x) for x in rng.sample(pool, min(n, len(pool)))]


def build_split(root: str, split: str, families: int) -> Dict[str, Any]:
    rng = random.Random(SPLIT_SEED[split])
    photos = backgrounds(root, split, families)
    out_dir = os.path.join(root, "r13", "synth")
    img_dir = os.path.join(out_dir, "images", split)
    os.makedirs(img_dir, exist_ok=True)
    rows, strata = [], {}
    f = 0
    while f < families:
        base = base_scene(rng, split)
        background = photos[f % len(photos)] if photos and f % 2 == 1 else rng.choice(PLAIN)
        vs = variants(base, rng, split)
        rendered = []
        for name, kinds in vs:
            r = render(kinds, random.Random(rng.random()), background)
            if r is None:
                break
            rendered.append((name, kinds, r))
        if len(rendered) != len(vs):
            continue
        group = "r13synth:%s:%05d" % (split, f)
        for name, kinds, (im, boxes) in rendered:
            rel = os.path.join("r13", "synth", "images", split, "%05d_%s.png" % (f, name))
            im.save(os.path.join(root, rel))
            qs, gold, meta = questions(kinds, rng)
            qboxes = {}
            for qid, m in meta.items():
                strata[m["stratum"]] = strata.get(m["stratum"], 0) + 1
                ks = [m["kind"]] + ([1] if m["task"] == "compare" else [])
                qboxes[qid] = [b for i in ks for b in boxes[i]]
            sid = "%s_%s" % (group, name)
            rows.append({"id": sid, "source": "r13-synth", "kind": "binding", "image_id": sid, "group": group,
                         "variant": name, "image": rel, "width": W, "height": H,
                         "background": "photo" if isinstance(background, str) else "plain",
                         "scene": kinds, "questions": qs, "gold": gold, "meta": meta, "boxes": qboxes})
        f += 1
    name = {"train": "train", "dev": "dev_bind_cf", "test": "test_bind_cf"}[split]
    with open(os.path.join(out_dir, name + ".jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return {"split": split, "file": name + ".jsonl", "families": families, "pictures": len(rows),
            "questions": sum(len(r["questions"]) for r in rows), "strata": strata}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    p.add_argument("--train", type=int, default=1400, help="scene families")
    p.add_argument("--dev", type=int, default=300)
    p.add_argument("--test", type=int, default=300)
    a = p.parse_args(argv)
    manifest = {"held_out_combinations": sorted(map(list, HELD_OUT)), "splits": []}
    for split, n in (("train", a.train), ("dev", a.dev), ("test", a.test)):
        info = build_split(a.root, split, n)
        manifest["splits"].append(info)
        print(json.dumps(info), flush=True)
    with open(os.path.join(a.root, "r13", "synth", "MANIFEST.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)


if __name__ == "__main__":
    main()
