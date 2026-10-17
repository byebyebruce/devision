"""Real-layout bridge data (candidate for v0.4; critic EXP27-RESULT-04 / exp28 branch "real-layout reading /
attribute-binding bridge"): newly drawn pictures in the layout of ScienceQA's two diagram forms deVision misses,
with labels computed from the drawn values -- never a ScienceQA picture.

  - temperature: two jars side by side, coloured particles inside, under each "Sample A" / "Sample B", "Mass of
    each particle: M u" and "Average particle speed: V m/s" (picture about 560 x 405, as the real ones; the text
    ends up about 6-7 px high after the 256 letterbox);
  - force: two boxed pairs of magnets ("Pair 1" / "Pair 2"; bars, discs or diamonds with N / S halves, large or
    small) with a distance bar and a label ("3 in", "1.25 cm", "42.8 mm") under each pair (picture about
    600-700 x 120-175).

Each picture yields alignment captions (the drawn values in a fixed order, as exp26's panel captions) and decision
questions: reading questions (one shared candidate set per field, as exp28's fields) and the ScienceQA-style
question (temperature: higher M * V^2; force: closer or larger -> stronger, one factor at a time, six wordings).

    uv run python scripts/data/real_layout.py --root data --n 6000 --val 300
"""
import argparse
import hashlib
import json
import os
import random
import sys
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(__file__))
from rules_text import FORCE_HINT, FORCE_Q, TEMP_HINT, TEMP_Q, TEMP_SAME, UNITS, fmt  # noqa: E402

FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
COLOURS = ["#6abe30", "#6c63d9", "#e0533d", "#2a9fd6", "#e8a33a", "#9b59b6", "#16a085"]
MAG_COLOURS = [("#d65bd6", "#00913f"), ("#2a78d6", "#6abe30"), ("#00b5e8", "#ff8a2a"), ("#8a7fe0", "#6abe30"),
               ("#009a44", "#8a7fe0"), ("#ff8a2a", "#00b5e8")]


# ---------------------------------------------------------------- temperature

def temp_scene(rng: random.Random) -> Dict:
    case = rng.choice(["same mass", "same speed", "both differ", "equal"])
    while True:
        ma, mb = rng.randint(4, 46), rng.randint(4, 46)
        va, vb = rng.randrange(300, 2000, 10), rng.randrange(300, 2000, 10)
        if case == "same mass":
            mb = ma
        elif case == "same speed":
            vb = va
        elif case == "equal":
            mb, vb = ma, va
        if case == "equal" or ((ma * va * va) != (mb * vb * vb) and (case != "both differ" or (ma != mb and va != vb))):
            break
    return {"case": case, "A": {"mass": ma, "speed": va}, "B": {"mass": mb, "speed": vb},
            "colours": [rng.choice(COLOURS), rng.choice(COLOURS)], "count": rng.randint(5, 12),
            "width": rng.randint(540, 580), "font": rng.randint(15, 17), "seed": rng.random()}


def temp_render(s: Dict) -> Image.Image:
    w, h = s["width"], 405
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    rng = random.Random(s["seed"])
    font, bold = ImageFont.truetype(FONT, s["font"]), ImageFont.truetype(BOLD, s["font"])
    for k, name in enumerate(("A", "B")):
        cx = w // 4 + k * (w // 2)
        x0, x1, y0, y1 = cx - 95, cx + 95, 30, 300
        d.rounded_rectangle((x0, y0 + 25, x1, y1), radius=28, outline="#9a9a9a", width=2)
        d.rectangle((x0 + 10, y0, x1 - 10, y0 + 20), fill="#9a9a9a")
        mass = s[name]["mass"]
        r = 6 + mass // 5                      # heavier particles drawn bigger, as in the real diagrams
        for _ in range(s["count"]):
            px, py = rng.randint(x0 + 20 + r, x1 - 20 - r), rng.randint(y0 + 50 + r, y1 - 15 - r)
            d.ellipse((px - r, py - r, px + r, py + r), fill=s["colours"][k])
        d.text((cx, 330), "Sample " + name, fill="black", font=bold, anchor="mm")
        d.text((cx, 360), "Mass of each particle: %d u" % mass, fill="black", font=font, anchor="mm")
        d.text((cx, 390), "Average particle speed: %s m/s" % fmt(s[name]["speed"]), fill="black", font=font, anchor="mm")
    return img


def temp_items(s: Dict, k: int) -> Tuple[str, List[Dict]]:
    a, b = s["A"], s["B"]
    caption = "Sample A: mass of each particle %d u, average particle speed %s m/s. Sample B: mass of each particle %d u, average particle speed %s m/s." % (
        a["mass"], fmt(a["speed"]), b["mass"], fmt(b["speed"]))
    ka, kb = a["mass"] * a["speed"] ** 2, b["mass"] * b["speed"] ** 2
    gold = "sample A" if ka > kb else "sample B" if kb > ka else TEMP_SAME
    opts = ["sample A", "sample B", TEMP_SAME]
    random.Random(k).shuffle(opts)
    items = [{"q": TEMP_Q, "options": opts, "gold": gold, "kind": "temp-question"}]
    for field, unit, q in (("mass", "u", "What is the mass of each particle in sample %s?"),
                           ("speed", "m/s", "What is the average particle speed in sample %s?")):
        cands = shared(a[field], b[field], field, k)
        for j, name in enumerate(("A", "B")):
            items.append(reading(q % name, cands, s[name][field], unit, k + j, "temp-" + field))
    return caption, items


# ---------------------------------------------------------------- force

def force_scene(rng: random.Random) -> Dict:
    case = rng.choice(["distance", "size", "equal"])
    unit, values = rng.choice(UNITS)
    d1 = d2 = rng.choice(values)
    big = [[True, True], [True, True]]
    if case == "distance":
        while d2 == d1:
            d2 = rng.choice(values)
    elif case == "size":
        p = rng.randrange(2)
        big[p][rng.randrange(2)] = False
    shape = rng.choice(["bar", "disc", "diamond"])
    poles = [[rng.choice(["NS", "SN"]) for _ in range(2)] for _ in range(2)]
    return {"case": case, "unit": unit, "d": [d1, d2], "big": big, "shape": shape, "poles": poles,
            "colours": rng.choice(MAG_COLOURS), "font": rng.randint(13, 15), "tall": rng.choice([90, 130, 170]),
            "width": rng.randint(580, 700)}


def _magnet(d: ImageDraw.ImageDraw, box, shape, poles, colours, font):
    x0, y0, x1, y1 = box
    mx = (x0 + x1) // 2
    halves = [(x0, y0, mx, y1), (mx, y0, x1, y1)]
    if shape == "bar":
        for hb, c in zip(halves, colours):
            d.rectangle(hb, fill=c)
    elif shape == "disc":
        d.pieslice(box, 90, 270, fill=colours[0])
        d.pieslice(box, 270, 450, fill=colours[1])
    else:
        cy = (y0 + y1) // 2
        d.polygon([(x0, cy), (mx, y0), (mx, y1)], fill=colours[0])
        d.polygon([(mx, y0), (x1, cy), (mx, y1)], fill=colours[1])
    for (hx0, hy0, hx1, hy1), p in zip(halves, poles):
        d.text(((hx0 + hx1) / 2, (hy0 + hy1) / 2), p, fill="white", font=font, anchor="mm")


def force_render(s: Dict) -> Image.Image:
    w, h = s["width"], s["tall"] + 10
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    font, label = ImageFont.truetype(FONT, s["font"]), ImageFont.truetype(FONT, s["font"] + 3)
    pw = (w - 30) // 2
    mag = max(26, int((s["tall"] - 50) * 0.95))
    for p in range(2):
        bx0 = 10 + p * (pw + 10)
        d.text((bx0 + pw / 2, 12), "Pair %d" % (p + 1), fill="black", font=label, anchor="mm")
        d.rectangle((bx0, 24, bx0 + pw, h - 2), outline="#cccccc", width=2)
        cy = 24 + (h - 26 - 24) // 2
        sizes = [mag if s["big"][p][j] else int(mag * 0.6) for j in range(2)]
        widths = [z * (2 if s["shape"] == "bar" else 1) for z in sizes]
        room = pw - 16 - widths[0] - widths[1]               # what is left for the gap inside the box
        if room < 0.35 * pw:                                  # too big for the box: shrink the magnets
            f = (0.65 * pw - 16) / (widths[0] + widths[1])
            sizes, widths = [int(z * f) for z in sizes], [int(x * f) for x in widths]
            room = pw - 16 - widths[0] - widths[1]
        gap = int(room * (0.35 + 0.6 * s["d"][p] / max(s["d"])))   # wider gap for a longer distance, inside the box
        start = bx0 + 8 + (room - gap) // 2
        for j in range(2):
            x = start if j == 0 else start + widths[0] + gap
            _magnet(d, (x, cy - sizes[j] // 2, x + widths[j], cy + sizes[j] // 2), s["shape"], s["poles"][p][j], s["colours"], font)
        ax0, ax1 = start + widths[0], start + widths[0] + gap
        ty = h - 14
        text = "%s %s" % (fmt(s["d"][p]), s["unit"])
        d.line((ax0, ty, ax1, ty), fill="#555555", width=1)
        d.line((ax0, ty - 4, ax0, ty + 4), fill="#555555", width=1)
        d.line((ax1, ty - 4, ax1, ty + 4), fill="#555555", width=1)
        tb = d.textbbox(((ax0 + ax1) / 2, ty), text, font=font, anchor="mm")
        d.rectangle((tb[0] - 2, tb[1] - 1, tb[2] + 2, tb[3] + 1), fill="white")
        d.text(((ax0 + ax1) / 2, ty), text, fill="black", font=font, anchor="mm")
    return img


def force_items(s: Dict, k: int) -> Tuple[str, List[Dict]]:
    desc = []
    for p in range(2):
        b = s["big"][p]
        size = "two large magnets" if all(b) else "one large magnet and one small magnet"
        desc.append("Pair %d: %s, %s %s apart." % (p + 1, size, fmt(s["d"][p]), s["unit"]))
    caption = " ".join(desc)
    if s["case"] == "distance":
        stronger = 1 if s["d"][0] < s["d"][1] else 2
    elif s["case"] == "size":
        stronger = 1 if all(s["big"][0]) else 2
    else:
        stronger = 0
    rng = random.Random(k)
    style = rng.choice(["magnitude-greater", "magnitude-smaller", "strength-stronger", "strength-weaker"])
    end = "." if rng.random() < 0.5 else ""
    kind, word = style.split("-")
    if kind == "strength":
        opts = ["The magnetic force is %s in Pair 1%s" % (word, end), "The magnetic force is %s in Pair 2%s" % (word, end),
                "The strength of the magnetic force is the same in both pairs%s" % end]
    else:
        opts = ["The magnitude of the magnetic force is %s in Pair 1%s" % (word, end),
                "The magnitude of the magnetic force is %s in Pair 2%s" % (word, end),
                "The magnitude of the magnetic force is the same in both pairs%s" % end]
    if stronger == 0:
        gold = opts[2]
    else:
        pick = stronger if word in ("greater", "stronger") else 3 - stronger
        gold = opts[pick - 1]
    rng.shuffle(opts)
    items = [{"q": FORCE_Q, "options": opts, "gold": gold, "kind": "force-question"}]
    cands = shared(s["d"][0], s["d"][1], "distance", k)
    for p in range(2):
        items.append(reading("What is the distance between the magnets in Pair %d?" % (p + 1), cands, s["d"][p], s["unit"],
                             k + p, "force-distance"))
    return caption, items


# ---------------------------------------------------------------- shared

def shared(a: float, b: float, field: str, k: int) -> List[float]:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "exp28"))
    from build import candidates  # type: ignore
    return candidates(a, b, field, k)


def reading(q: str, cands: List[float], right: float, unit: str, slot: int, kind: str) -> Dict:
    others = [x for x in cands if x != right]
    order = list(others)
    order.insert(slot % 3, right)
    opts = ["%s %s" % (fmt(x), unit) for x in order]
    return {"q": q, "options": opts, "gold": "%s %s" % (fmt(right), unit), "kind": kind}


def build(root: str, n: int, n_val: int, seed: int = 31) -> Dict:
    out = os.path.join(root, "real_layout")
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    rng = random.Random(seed)
    files: Dict[str, List[Dict]] = {"captions_train": [], "captions_val": [], "questions_train": [], "questions_val": []}
    for i in range(n + n_val):
        split = "train" if i < n else "val"
        form = "temp" if i % 2 == 0 else "force"
        s = temp_scene(rng) if form == "temp" else force_scene(rng)
        img = temp_render(s) if form == "temp" else force_render(s)
        rel = "real_layout/images/%s_%05d.png" % (split, i)
        img.save(os.path.join(root, rel))
        iid = "real-layout:%05d" % i
        caption, items = (temp_items if form == "temp" else force_items)(s, i)
        hint = TEMP_HINT if form == "temp" else FORCE_HINT + (", but some of them are different sizes." if s["case"] == "size" else ".")
        files["captions_" + split].append({"id": iid, "source": "real-layout-" + form, "image_id": iid, "image": rel,
                                          "captions": [caption]})
        files["questions_" + split].append({
            "id": iid, "source": "real-layout-" + form, "kind": form, "axis": s["case"], "image_id": iid, "image": rel,
            "state_text": hint,
            "questions": {"q%d" % j: {"type": "choice", "instructions": it["q"], "criteria": {o: None for o in it["options"]}}
                          for j, it in enumerate(items)},
            "gold": {"q%d" % j: {"probabilities": {o: float(o == it["gold"]) for o in it["options"]}} for j, it in enumerate(items)}})
    manifest = {"seed": seed}
    for name, rows in files.items():
        path = os.path.join(out, name + ".jsonl")
        with open(path, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        manifest[name] = {"rows": len(rows), "questions": sum(len(r.get("questions", {"c": 0})) for r in rows),
                          "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}
    json.dump(manifest, open(os.path.join(out, "MANIFEST.json"), "w"), indent=1)
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    p.add_argument("--n", type=int, default=6000)
    p.add_argument("--val", type=int, default=300)
    a = p.parse_args()
    print(json.dumps(build(a.root, a.n, a.val), indent=1))
