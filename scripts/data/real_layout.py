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

    uv run python scripts/data/real_layout.py --root data --n 6000 --tune 300 --confirm 300
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
            "width": rng.randint(520, 600), "font": rng.choice([13, 14, 15, 15, 16, 16, 17, 18, 20, 22, 24]),
            "a_left": rng.random() < 0.7, "wording": int(rng.random() < 0.25), "spacing": rng.uniform(1.5, 2.1),
            "seed": rng.random()}


LINES = [("Mass of each particle: %d u", "Average particle speed: %s m/s"), ("Particle mass: %d u", "Average speed: %s m/s")]


def temp_twin(s: Dict) -> Dict:
    """The same picture with only values changed (critic: paired counterfactuals change values or the label
    mapping only): A and B swap their values; an equal pair instead gets a faster sample B."""
    if s["case"] == "equal":
        return dict(s, case="same mass", B=dict(s["B"], speed=s["B"]["speed"] + 100))
    return dict(s, A=s["B"], B=s["A"])


def temp_render(s: Dict) -> Image.Image:
    w = s["width"]
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    size = s["font"]
    longest = LINES[s["wording"]][1] % "1,999"
    while size > 11 and probe.textlength(longest, font=ImageFont.truetype(FONT, size)) > w / 2 - 12:
        size -= 1                             # the longest line must fit in its half
    line = int(size * s["spacing"])
    h = max(405, 315 + 3 * line + size)
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    rng = random.Random(s["seed"])
    font, bold = ImageFont.truetype(FONT, size), ImageFont.truetype(BOLD, size)
    for k, name in enumerate(("A", "B") if s.get("a_left", True) else ("B", "A")):
        cx = w // 4 + k * (w // 2)
        x0, x1, y0, y1 = cx - 95, cx + 95, 30, 300
        d.rounded_rectangle((x0, y0 + 25, x1, y1), radius=28, outline="#9a9a9a", width=2)
        d.rectangle((x0 + 10, y0, x1 - 10, y0 + 20), fill="#9a9a9a")
        mass = s[name]["mass"]
        r = 6 + mass // 5                      # heavier particles drawn bigger, as in the real diagrams
        for _ in range(s["count"]):
            px, py = rng.randint(x0 + 20 + r, x1 - 20 - r), rng.randint(y0 + 50 + r, y1 - 15 - r)
            d.ellipse((px - r, py - r, px + r, py + r), fill=s["colours"][k])
        top = 315 + line // 2
        mass_line, speed_line = LINES[s.get("wording", 0)]
        d.text((cx, top), "Sample " + name, fill="black", font=bold, anchor="mm")
        d.text((cx, top + line), mass_line % mass, fill="black", font=font, anchor="mm")
        d.text((cx, top + 2 * line), speed_line % fmt(s[name]["speed"]), fill="black", font=font, anchor="mm")
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
            "colours": rng.choice(MAG_COLOURS), "font": rng.choice([12, 13, 13, 14, 15, 16, 18, 20]),
            "tall": rng.choice([90, 130, 170]), "width": rng.randint(580, 700)}


def force_twin(s: Dict) -> Dict:
    """Only values change: the two pairs swap their distances and magnet sizes; an equal pair instead gets a longer
    distance in Pair 2."""
    if s["case"] == "equal":
        unit_values = dict(UNITS)[s["unit"]]
        longer = [v for v in unit_values if v > s["d"][0]] or [s["d"][0] * 2]
        return dict(s, case="distance", d=[s["d"][0], longer[0]])
    return dict(s, d=s["d"][::-1], big=s["big"][::-1])


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


def transcribed_keys() -> set:
    """Value combinations of the 24 diagnostic pictures (either order): never drawn (critic: no diagnostic picture
    or its transcription in training)."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "exp28"))
    from build import parse_force, parse_temp  # type: ignore
    notes = json.load(open(os.path.join(os.path.dirname(__file__), "..", "exp21", "sqa_transcribed.json")))
    keys = set()
    for k, v in notes.items():
        if k == "_note":
            continue
        if v.startswith("Sample"):
            t = parse_temp(v)
            a, b = (t["A"]["mass"], t["A"]["speed"]), (t["B"]["mass"], t["B"]["speed"])
            keys |= {("temp",) + a + b, ("temp",) + b + a}
        else:
            f = parse_force(v)
            keys |= {("force", f["1"]["distance"], f["2"]["distance"], f["1"]["unit"]),
                     ("force", f["2"]["distance"], f["1"]["distance"], f["1"]["unit"])}
    return keys


def scene_key(form: str, s: Dict) -> tuple:
    if form == "temp":
        return ("temp", float(s["A"]["mass"]), float(s["A"]["speed"]), float(s["B"]["mass"]), float(s["B"]["speed"]))
    return ("force", float(s["d"][0]), float(s["d"][1]), s["unit"])


SPLITS = ("train", "tune", "confirm")


def build(root: str, n: int, n_tune: int, n_confirm: int, seed: int = 31) -> Dict:
    """n / n_tune / n_confirm scene families (a scene and its value twin, both drawn), alternating temperature
    and force; families never cross splits. Captions for the alignment stage (train / tune), questions for every
    split; confirm is drawn once and never used to choose a recipe."""
    out = os.path.join(root, "real_layout")
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    rng = random.Random(seed)
    banned = transcribed_keys()
    files: Dict[str, List[Dict]] = {"%s_%s" % (kind, sp): [] for kind in ("captions", "questions") for sp in SPLITS}
    total, i, skipped = n + n_tune + n_confirm, 0, 0
    while i < total:
        split = "train" if i < n else "tune" if i < n + n_tune else "confirm"
        form = "temp" if i % 2 == 0 else "force"
        s = temp_scene(rng) if form == "temp" else force_scene(rng)
        twin = temp_twin(s) if form == "temp" else force_twin(s)
        if scene_key(form, s) in banned or scene_key(form, twin) in banned:
            skipped += 1
            continue
        fam = "rl%05d" % i
        for v, scene in enumerate((s, twin)):
            img = temp_render(scene) if form == "temp" else force_render(scene)
            rel = "real_layout/images/%s_v%d.png" % (fam, v)
            img.save(os.path.join(root, rel))
            iid = "real-layout:%s:v%d" % (fam, v)
            caption, items = (temp_items if form == "temp" else force_items)(scene, 2 * i + v)
            hint = TEMP_HINT if form == "temp" else FORCE_HINT + (", but some of them are different sizes." if scene["case"] == "size" else ".")
            files["captions_" + split].append({"id": iid, "source": "real-layout-" + form, "image_id": iid, "image": rel,
                                              "family": fam, "captions": [caption]})
            files["questions_" + split].append({
                "id": iid, "source": "real-layout-" + form, "kind": form, "axis": scene["case"], "image_id": iid, "image": rel,
                "family": fam, "group": fam, "state_text": hint,
                "questions": {"q%d" % j: {"type": "choice", "instructions": it["q"], "criteria": {o: None for o in it["options"]}}
                              for j, it in enumerate(items)},
                "gold": {"q%d" % j: {"probabilities": {o: float(o == it["gold"]) for o in it["options"]}} for j, it in enumerate(items)}})
        i += 1
    manifest: Dict = {"seed": seed, "families": {"train": n, "tune": n_tune, "confirm": n_confirm},
                      "skipped_transcribed_values": skipped}
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
    p.add_argument("--tune", type=int, default=300)
    p.add_argument("--confirm", type=int, default=300)
    a = p.parse_args()
    print(json.dumps(build(a.root, a.n, a.tune, a.confirm), indent=1))
