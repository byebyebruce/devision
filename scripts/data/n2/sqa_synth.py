"""data-n2, block A: programmatic science diagrams for the ScienceQA skills v0.3 fails most (plan
docs/experiments/2026-10-21-v04-plan.md; stage-1 table from scripts/data/n2/slices.py on data/n2/dev_sqa.jsonl,
natural science: magnetic force strength 0.21, magnitude 0.47, attract / repel 0.60, solution concentration 0.51 --
two and three options).

    uv run python scripts/data/n2/sqa_synth.py --root data [--per-skill 6000]

Pictures are drawn here (PIL) in the layouts ScienceQA uses -- magnets as coloured bars / discs / horseshoes with N and
S letters, two "Pair" panels with a measured gap, beakers with solute balls and a solvent volume -- with random
colours, sizes, gaps and counts, and the answer is computed from what was drawn:
- attract / repel: the facing poles' letters (same -> repel);
- force strength / magnitude: same magnets -> the smaller gap is stronger; same gap -> the smaller magnets are weaker;
  same gap and magnets -> the same; the stated distance is drawn to scale;
- concentration: particles / volume; equal -> "neither".
The wording, the three-option sets and the hint text follow ScienceQA's train questions, so this teaches the skill in
the format the model is tested on; the pictures never come from ScienceQA. Writes <root>/n2/sqa_synth.jsonl,
pictures under <root>/n2/images/sqa_synth/, and sqa_synth.MANIFEST.json.
"""
import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter
from typing import List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

PALETTE = ["#00b0f0", "#f28e2b", "#d65fdb", "#66c33a", "#3d7fe6", "#e8504a", "#f2c12e", "#8f6fd8", "#29b3a5"]
BALL = {"pink": "#d65fdb", "green": "#66c33a", "blue": "#3d7fe6", "purple": "#8f6fd8", "orange": "#f28e2b"}


def font(size: int, bold: bool = False):
    from PIL import ImageFont
    from fonts import font_path  # pyright: ignore[reportMissingImports]
    return ImageFont.truetype(font_path("Arial Bold" if bold else "Arial"), size)


def centered(draw, xy, text, f, fill="white"):
    x, y = xy
    b = draw.textbbox((0, 0), text, font=f)
    draw.text((x - (b[2] - b[0]) / 2, y - (b[3] - b[1]) / 2 - b[1]), text, font=f, fill=fill)


def bar(draw, x, y, w, h, left: str, colors, f):
    """A horizontal bar magnet, `left` pole on the left half."""
    other = "S" if left == "N" else "N"
    draw.rectangle([x, y, x + w // 2, y + h], fill=colors[left])
    draw.rectangle([x + w // 2, y, x + w, y + h], fill=colors[other])
    centered(draw, (x + w // 4, y + h // 2), left, f)
    centered(draw, (x + 3 * w // 4, y + h // 2), other, f)


def disc(draw, cx, top, d, upper: str, colors, f):
    other = "S" if upper == "N" else "N"
    draw.pieslice([cx - d // 2, top, cx + d // 2, top + d], 180, 360, fill=colors[upper])
    draw.pieslice([cx - d // 2, top, cx + d // 2, top + d], 0, 180, fill=colors[other])
    centered(draw, (cx, top + d // 6), upper, f)
    centered(draw, (cx, top + 5 * d // 6), other, f)


# ---------------------------------------------------------------- attract / repel

def attract_repel(rng: random.Random):
    from PIL import Image, ImageDraw
    c1, c2 = rng.sample(PALETTE, 2)
    colors = {"N": c1, "S": c2}
    layout = rng.choice(["bars", "stack", "discs"])
    if layout == "bars":   # two bars side by side: right pole of the first faces left pole of the second
        w, h, gap = rng.randint(120, 200), rng.randint(40, 70), rng.randint(40, 160)
        img = Image.new("RGB", (2 * w + gap + 40, h + 40), "white")
        d, f = ImageDraw.Draw(img), font(max(14, h // 2))
        a, b = rng.choice("NS"), rng.choice("NS")
        bar(d, 20, 20, w, h, a, colors, f)
        bar(d, 20 + w + gap, 20, w, h, b, colors, f)
        facing = ("S" if a == "N" else "N", b)
    elif layout == "stack":  # two vertical bars, one above the other
        w, h, gap = rng.randint(60, 110), rng.randint(110, 180), rng.randint(40, 140)
        img = Image.new("RGB", (w + 40, 2 * h + gap + 40), "white")
        d, f = ImageDraw.Draw(img), font(max(16, w // 3))
        tops = []
        for k in range(2):
            up = rng.choice("NS")
            other = "S" if up == "N" else "N"
            y0 = 20 + k * (h + gap)
            d.rectangle([20, y0, 20 + w, y0 + h // 2], fill=colors[up])
            d.rectangle([20, y0 + h // 2, 20 + w, y0 + h], fill=colors[other])
            centered(d, (20 + w // 2, y0 + h // 4), up, f)
            centered(d, (20 + w // 2, y0 + 3 * h // 4), other, f)
            tops.append((up, other))
        facing = (tops[0][1], tops[1][0])
    else:
        dd, gap = rng.randint(120, 190), rng.randint(60, 160)
        img = Image.new("RGB", (dd + 40, 2 * dd + gap + 40), "white")
        d, f = ImageDraw.Draw(img), font(max(16, dd // 7))
        u1, u2 = rng.choice("NS"), rng.choice("NS")
        disc(d, 20 + dd // 2, 20, dd, u1, colors, f)
        disc(d, 20 + dd // 2, 20 + dd + gap, dd, u2, colors, f)
        facing = ("S" if u1 == "N" else "N", u2)
    gold = "repel" if facing[0] == facing[1] else "attract"
    q = {"instructions": "Will these magnets attract or repel each other?", "options": ["attract", "repel"],
         "gold": gold, "hint": rng.choice(["Two magnets are placed as shown.",
                                           "Two magnets are placed as shown.\n\nHint: Magnets that attract pull together. "
                                           "Magnets that repel push apart."])}
    return img, q, "attract_repel", layout


# ---------------------------------------------------------------- force between pairs

def pairs(rng: random.Random):
    from PIL import Image, ImageDraw
    c1, c2 = rng.sample(PALETTE, 2)
    colors = {"N": c1, "S": c2}
    unit = rng.choice(["in", "cm", "mm"])
    kind = rng.choice(["distance", "distance", "size", "same"])
    d_val = [rng.choice([1, 2, 3, 4, 5, 1.5, 2.5, 3.5, 4.5]) for _ in range(2)]
    scale = [1.0, 1.0]
    if kind == "distance":
        while d_val[0] == d_val[1]:
            d_val[1] = rng.choice([1, 2, 3, 4, 5, 1.5, 2.5, 3.5, 4.5])
    else:
        d_val[1] = d_val[0]
        if kind == "size":
            scale[rng.randrange(2)] = rng.choice([0.55, 0.65, 0.75])
    W, H = 340, 130
    px_per = min(rng.randint(28, 40), (W - 30 - 2 * 110) / max(d_val))   # both magnets and the gap fit the panel
    img = Image.new("RGB", (2 * W + 30, H + 40), "white")
    d = ImageDraw.Draw(img)
    for k in range(2):
        x0 = 10 + k * (W + 10)
        d.rounded_rectangle([x0, 28, x0 + W, 28 + H], radius=6, outline="#c0c0c0", width=2)
        centered(d, (x0 + W // 2, 12), "Pair %d" % (k + 1), font(16), fill="black")
        gap = int(d_val[k] * px_per)
        bw, bh = int(110 * scale[k]), int(36 * scale[k])
        a, b = rng.choice("NS"), rng.choice("NS")
        left = x0 + (W - (2 * bw + gap)) // 2
        y = 28 + 18 + (36 - bh) // 2
        f = font(max(9, int(13 * scale[k])))
        bar(d, left, y, bw, bh, a, colors, f)
        bar(d, left + bw + gap, y, bw, bh, b, colors, f)
        ly = 28 + H - 28
        lx0, lx1 = left + bw, left + bw + gap
        d.line([lx0, ly, lx1, ly], fill="#555555", width=1)
        d.line([lx0, ly - 5, lx0, ly + 5], fill="#555555", width=1)
        d.line([lx1, ly - 5, lx1, ly + 5], fill="#555555", width=1)
        label = ("%g %s" % (d_val[k], unit))
        lb = d.textbbox((0, 0), label, font=font(14))
        cx = (lx0 + lx1) // 2
        d.rectangle([cx - (lb[2] - lb[0]) // 2 - 3, ly - 9, cx + (lb[2] - lb[0]) // 2 + 3, ly + 9], fill="white")
        centered(d, (cx, ly), label, font(14), fill="black")
    stronger = None
    if kind == "distance":
        stronger = 0 if d_val[0] < d_val[1] else 1
    elif kind == "size":
        stronger = 0 if scale[0] > scale[1] else 1
    wording = rng.choice(["strength", "magnitude"])
    if wording == "strength":
        opts = ["The magnetic force is stronger in Pair 1", "The magnetic force is stronger in Pair 2",
                "The strength of the magnetic force is the same in both pairs"]
        gold = opts[2] if stronger is None else opts[stronger]
    else:
        opts = ["The magnitude of the magnetic force is smaller in Pair 1",
                "The magnitude of the magnetic force is smaller in Pair 2",
                "The magnitude of the magnetic force is the same in both pairs"]
        gold = opts[2] if stronger is None else opts[1 - stronger]
    same = "made of the same material" if kind != "size" else "made of the same material, but some of them are different sizes"
    hint = ("The images below show two pairs of magnets. The magnets in different pairs do not affect each other. "
            "All the magnets shown are %s." % same)
    q = {"instructions": "Think about the magnetic force between the magnets in each pair. Which of the following "
                         "statements is true?", "options": opts, "gold": gold, "hint": hint}
    return img, q, "magnet_pairs", kind


# ---------------------------------------------------------------- concentration

def beakers(rng: random.Random):
    from PIL import Image, ImageDraw
    color = rng.choice(sorted(BALL))
    vols = [rng.choice([20, 25, 30, 35, 40, 45, 50])] * 2
    if rng.random() < 0.3:
        vols[1] = rng.choice([v for v in [20, 25, 30, 35, 40, 45, 50] if v != vols[0]])
    n = [rng.randint(1, 9), rng.randint(1, 9)]
    if rng.random() < 0.3:
        n[1] = n[0] * vols[1] // vols[0] if (n[0] * vols[1]) % vols[0] == 0 else n[0]
        if vols[0] != vols[1] and (n[0] * vols[1]) % vols[0] != 0:
            vols[1] = vols[0]
    img = Image.new("RGB", (390, 260), "white")
    d = ImageDraw.Draw(img)
    for k in range(2):
        x0 = 20 + k * 200
        d.rectangle([x0, 10, x0 + 150, 160], outline="#9ab8c8", width=3, fill="#eaf6fb")
        level = 160 - int(55 * vols[k] / 50) - 10
        d.rectangle([x0 + 3, level, x0 + 147, 157], fill="#d3dde3")
        pts: List[Tuple[int, int]] = []
        tries = 0
        while len(pts) < n[k] and tries < 2000:
            tries += 1
            p = (rng.randint(x0 + 15, x0 + 135), rng.randint(level + 12, 145))
            if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 > 22 ** 2 for q in pts):
                pts.append(p)
        n[k] = len(pts)
        for px, py in pts:
            d.ellipse([px - 10, py - 10, px + 10, py + 10], fill=BALL[color])
        centered(d, (x0 + 75, 200), "Solvent volume: %d mL" % vols[k], font(15), fill="black")
        centered(d, (x0 + 75, 238), "Solution %s" % "AB"[k], font(16, bold=True), fill="black")
    ca, cb = n[0] / vols[0], n[1] / vols[1]
    opts = ["Solution A", "Solution B", "neither; their concentrations are the same"]
    gold = opts[2] if abs(ca - cb) < 1e-9 else opts[0] if ca > cb else opts[1]
    q = {"instructions": "Which solution has a higher concentration of %s particles?" % color, "options": opts,
         "gold": gold, "hint": "The diagram below is a model of two solutions. Each %s ball represents one particle "
                               "of solute." % color}
    return img, q, "concentration", "equal_volume" if vols[0] == vols[1] else "different_volume"


GENERATORS = {"attract_repel": attract_repel, "magnet_pairs": pairs, "concentration": beakers}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    p.add_argument("--per-skill", type=int, default=6000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    rng = random.Random(a.seed)
    rows, golds = [], Counter()
    out_img = os.path.join(a.root, "n2", "images", "sqa_synth")
    os.makedirs(out_img, exist_ok=True)
    for skill, gen in GENERATORS.items():
        for i in range(a.per_skill):
            img, q, sk, variant = gen(rng)
            import io
            buf = io.BytesIO()
            img.save(buf, "PNG")
            h = hashlib.sha1(buf.getvalue()).hexdigest()[:16]
            rel = "n2/images/sqa_synth/%s.png" % h
            with open(os.path.join(a.root, rel), "wb") as fh:
                fh.write(buf.getvalue())
            opts = q["options"][:]
            rng.shuffle(opts)
            rid = "n2-sqasynth:%s:%d" % (skill, i)
            rows.append({"id": rid, "source": "sqa-synth", "kind": skill, "category": variant,
                         "image_id": "sqasynth:%s" % h, "image": rel, "group": skill, "state_text": q["hint"],
                         "questions": {"q": {"type": "choice", "instructions": q["instructions"],
                                             "criteria": {o: None for o in opts}}},
                         "gold": {"q": {"probabilities": {o: float(o == q["gold"]) for o in opts}}},
                         "answer_key": q["gold"]})
            golds["%s / %s" % (skill, q["gold"][:40])] += 1
    out = os.path.join(a.root, "n2", "sqa_synth.jsonl")
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    m = {"built_by": "scripts/data/n2/sqa_synth.py", "questions": len(rows), "per_gold": dict(sorted(golds.items())),
         "per_variant": dict(Counter("%s/%s" % (r["kind"], r["category"]) for r in rows))}
    with open(out.replace(".jsonl", ".MANIFEST.json"), "w") as f:
        json.dump(m, f, indent=1)
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
