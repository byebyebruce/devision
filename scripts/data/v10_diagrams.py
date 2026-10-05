"""Round 10 synthetic diagrams (critic/plan-round10.md, sections 2 and 4, stage A2): ScienceQA-style attract /
repel questions drawn natively, with labels computed from the scene, never from the picture.

    uv run python scripts/data/v10_diagrams.py --root data --type repel --train 2400 --test 300

A scene is two magnets in a row (left-right like ScienceQA's 500x53 strips, or top-bottom like its 186x500
ones), each of one shape (bar, diamond, split disc, ring) with two coloured halves labelled N and S. The answer
depends only on the two facing ends: different poles attract, same poles repel (`repel_answer`).
Each scene family has three pictures and the rule says which answer each one has:
  - original;
  - "flip_one": one magnet turned round -> the facing pole of that magnet changes -> the answer changes;
  - "mirror": the whole picture mirrored -> both magnets turn round together, the facing ends stay facing ->
    the answer does not change.
Train scenes use only the original picture (one question each); test scenes keep all three (data/v10/
syn_repel_test{,_flip_one,_mirror}.jsonl). Families never cross the train / test split. Pictures go to
data/v10/images/syn_repel/; data/v10/syn_repel.MANIFEST.json counts answers, shapes and layouts.
"""
import argparse
import json
import os
import random
from collections import Counter
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont

FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
QUESTION = "Will these magnets attract or repel each other?"
HINTS = ["Two magnets are placed as shown.",
         "Two magnets are placed as shown.\n\nHint: Magnets that attract pull together. Magnets that repel push apart."]
COLOURS = ["#2a78d6", "#6abe30", "#e05bd6", "#00b5e8", "#8a7fe0", "#ff8a2a", "#009a44"]
SHAPES = ["bar", "diamond", "disc", "ring"]


def repel_answer(facing_a: str, facing_b: str) -> str:
    """Different facing poles attract, the same ones repel."""
    return "attract" if facing_a != facing_b else "repel"


def facing(poles: Tuple[str, str], first: bool) -> str:
    """poles = (pole at the start, pole at the end) along the row; the first magnet faces with its end, the
    second with its start."""
    return poles[1] if first else poles[0]


def draw_magnet(d: ImageDraw.ImageDraw, box, shape: str, poles: Tuple[str, str], colours, font, vertical: bool):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    if vertical:
        halves = [(x0, y0, x1, y0 + h // 2), (x0, y0 + h // 2, x1, y1)]
    else:
        halves = [(x0, y0, x0 + w // 2, y1), (x0 + w // 2, y0, x1, y1)]
    if shape == "bar":
        for (bx0, by0, bx1, by1), c in zip(halves, colours):
            d.rectangle((bx0, by0, bx1, by1), fill=c)
    elif shape == "diamond":
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        if vertical:
            d.polygon([(cx, y0), (x1, cy), (x0, cy)], fill=colours[0])
            d.polygon([(x0, cy), (x1, cy), (cx, y1)], fill=colours[1])
        else:
            d.polygon([(x0, cy), (cx, y0), (cx, y1)], fill=colours[0])
            d.polygon([(cx, y0), (x1, cy), (cx, y1)], fill=colours[1])
    else:
        start, end = (180, 0) if vertical else (90, 270)
        d.pieslice(box, start, start + 180, fill=colours[0])
        d.pieslice(box, end, end + 180, fill=colours[1])
        if shape == "ring":
            r = int(min(w, h) * 0.22)
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            d.ellipse((cx - r, cy - r, cx + r, cy + r), fill="white")
    for (bx0, by0, bx1, by1), p in zip(halves, poles):
        if shape == "ring":   # label on the coloured band, away from the hole
            tx = (bx0 + bx1) / 2 if vertical else (bx0 + (bx1 - bx0) * (0.25 if p == poles[0] else 0.75))
            ty = (by0 + (by1 - by0) * (0.25 if p == poles[0] else 0.75)) if vertical else (by0 + by1) / 2
        else:
            tx, ty = (bx0 + bx1) / 2, (by0 + by1) / 2
        d.text((tx, ty), p, fill="white", font=font, anchor="mm")


def scene(rng: random.Random):
    vertical = rng.random() < 0.2
    shape_a = rng.choice(SHAPES)
    shape_b = shape_a if rng.random() < 0.8 else rng.choice(SHAPES)
    poles = [tuple(rng.sample(["N", "S"], 2)) for _ in range(2)]
    colours = [rng.sample(COLOURS, 2) for _ in range(2)]
    return {"vertical": vertical, "shapes": [shape_a, shape_b], "poles": poles, "colours": colours,
            "gap": rng.uniform(0.15, 0.45), "length": rng.randint(440, 530), "thick": rng.randint(50, 90),
            "font": rng.randint(13, 18)}


def render(s) -> Image.Image:
    length, thick = s["length"], s["thick"]
    magnet = int(length * (1 - s["gap"]) / 2)
    img = Image.new("RGB", (thick, length) if s["vertical"] else (length, thick), "white")
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT, s["font"])
    for k in range(2):
        start = 0 if k == 0 else length - magnet
        size = magnet if s["shapes"][k] in ("bar", "diamond") else min(magnet, thick)
        if k == 1:
            start = length - size
        box = (0, start, thick, start + size) if s["vertical"] else (start, 0, start + size, thick)
        draw_magnet(d, box, s["shapes"][k], s["poles"][k], s["colours"][k], font, s["vertical"])
    return img


def answer(s) -> str:
    return repel_answer(facing(s["poles"][0], True), facing(s["poles"][1], False))


def variants(s):
    """(name, scene, picture transform) for the original and the two controls."""
    flip_one = dict(s, poles=[s["poles"][0], s["poles"][1][::-1]])
    mirrored = dict(s, poles=[s["poles"][1][::-1], s["poles"][0][::-1]],
                    shapes=s["shapes"][::-1], colours=[c[::-1] for c in s["colours"][::-1]])
    return [("orig", s), ("flip_one", flip_one), ("mirror", mirrored)]


def sample(root: str, family: str, name: str, s, rng: random.Random) -> dict:
    rel = "v10/images/syn_repel/%s_%s.png" % (family, name)
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    render(s).save(path)   # the mirror control is drawn as a mirrored scene, so the N / S letters stay readable
    gold = answer(s)
    order = ["attract", "repel"]
    rng.shuffle(order)
    return {"id": "v10-syn-repel:%s:%s" % (family, name), "source": "syn-repel", "kind": "scienceqa",
            "image_id": "syn:%s" % family, "image": rel, "state_text": rng.choice(HINTS),
            "questions": {"q": {"type": "choice", "instructions": QUESTION, "criteria": {o: None for o in order}}},
            "gold": {"q": {"probabilities": {o: float(o == gold) for o in order}}}, "answer_key": gold,
            "diagram_type": "repel", "family": family, "variant": name}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    p.add_argument("--type", default="repel", choices=["repel"])
    p.add_argument("--train", type=int, default=2400)
    p.add_argument("--test", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    rng = random.Random(a.seed)
    out = os.path.join(a.root, "v10")
    train: List[dict] = []
    test = {"orig": [], "flip_one": [], "mirror": []}
    shapes, layouts = Counter(), Counter()
    for i in range(a.train + a.test):
        s = scene(rng)
        family = "f%05d" % i
        shapes[s["shapes"][0]] += 1
        layouts["vertical" if s["vertical"] else "horizontal"] += 1
        if i < a.train:
            train.append(sample(a.root, family, "orig", s, rng))
        else:
            for name, v in variants(s):
                test[name].append(sample(a.root, family, name, v, rng))
    with open(os.path.join(out, "syn_repel_train.jsonl"), "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in train)
    for name, rows in test.items():
        suffix = "" if name == "orig" else "_" + name
        with open(os.path.join(out, "syn_repel_test%s.jsonl" % suffix), "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
    flips = sum(o["answer_key"] != f["answer_key"] for o, f in zip(test["orig"], test["flip_one"]))
    mirrors = sum(o["answer_key"] == m["answer_key"] for o, m in zip(test["orig"], test["mirror"]))
    manifest = {"train": len(train), "test_families": len(test["orig"]),
                "answers_train": dict(Counter(r["answer_key"] for r in train)),
                "answers_test": dict(Counter(r["answer_key"] for r in test["orig"])),
                "flip_one_changes_answer": flips, "mirror_keeps_answer": mirrors,
                "first_shape": dict(shapes), "layout": dict(layouts), "seed": a.seed}
    assert flips == len(test["orig"]) and mirrors == len(test["orig"]), "label rule broken"
    with open(os.path.join(out, "syn_repel.MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
