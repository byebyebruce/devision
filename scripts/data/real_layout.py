"""Real-layout bridge data (candidate for v0.4; critic EXP27-RESULT-04 / exp28 branch "real-layout reading /
attribute-binding bridge"): newly drawn pictures in the layout of ScienceQA's two diagram forms deVision misses,
with labels computed from the drawn values -- never a ScienceQA picture.

  - temperature: two jars side by side, coloured particles inside, under each "Sample A" / "Sample B", "Mass of
    each particle: M u" and "Average particle speed: V m/s" (picture about 560 x 405, as the real ones; the text
    ends up about 6-7 px high after the 256 letterbox);
  - force: two boxed pairs of magnets ("Pair 1" / "Pair 2"; bars, discs or diamonds with N / S halves, large or
    small) with a distance bar and a label ("3 in", "1.25 cm", "42.8 mm") under each pair (picture about
    600-700 x 120-175).

Each picture yields alignment captions (the drawn values by label: Sample A then B, Pair 1 then 2; temperature
fields in the order the rows are drawn) and decision questions: reading questions (one shared candidate set per
field, as exp28's fields) and the ScienceQA-style question (temperature: higher M * V^2; force: closer or larger ->
stronger, one factor at a time, six wordings).

Positions are decoupled from labels: Sample A is on the right in 30% of the pictures, Pair 1 in 30%, and the speed
row is above the mass row in 30%. A force picture uses one magnet scale and one distance-to-gap mapping for both
pairs (fixed per unit, growing with the distance), so equal distances are drawn as equal gaps and every large magnet has the same size
(critic EXP32-R1-01).

A family is a scene and its value twin. Within a family only the values change (critic EXP32-R1-04): the layout,
particle positions, question wording, candidate set and option arrangement are frozen; the candidates cover both
views' values. Splits are assigned by canonical case -- the unordered pair of (value, size) of the two objects --
so no case (and no family linked to it through a twin) is in two splits (critic EXP32-R1-03); the 24 diagnostic
cases are never drawn.

    uv run python scripts/data/real_layout.py --root data --n 6000 --tune 300 --confirm 300
"""
import argparse
import hashlib
import json
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(__file__))
from rules_text import FORCE_HINT, FORCE_Q, TEMP_HINT, TEMP_Q, TEMP_SAME, fmt  # noqa: E402

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
            "speed_first": rng.random() < 0.3, "seed": rng.random()}


R_MAX = 6 + 46 // 5                            # the largest particle radius (mass 46)
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
        for _ in range(s["count"]):            # centres do not depend on the mass (twins move no particle)
            px, py = rng.randint(x0 + 20 + R_MAX, x1 - 20 - R_MAX), rng.randint(y0 + 50 + R_MAX, y1 - 15 - R_MAX)
            d.ellipse((px - r, py - r, px + r, py + r), fill=s["colours"][k])
        top = 315 + line // 2
        mass_line, speed_line = LINES[s.get("wording", 0)]
        rows = [mass_line % mass, speed_line % fmt(s[name]["speed"])]
        if s.get("speed_first"):
            rows = rows[::-1]
        d.text((cx, top), "Sample " + name, fill="black", font=bold, anchor="mm")
        d.text((cx, top + line), rows[0], fill="black", font=font, anchor="mm")
        d.text((cx, top + 2 * line), rows[1], fill="black", font=font, anchor="mm")
    return img


def temp_items(s: Dict, k: int, views: List[Dict]) -> Tuple[str, List[Dict]]:
    """k: the family's number (the same for both views); views: the family's scenes, whose values the shared
    candidates cover."""
    a, b = s["A"], s["B"]
    def part(name):
        m, v = "mass of each particle %d u" % s[name]["mass"], "average particle speed %s m/s" % fmt(s[name]["speed"])
        return "Sample %s: %s." % (name, ", ".join([v, m] if s.get("speed_first") else [m, v]))
    caption = part("A") + " " + part("B")
    ka, kb = a["mass"] * a["speed"] ** 2, b["mass"] * b["speed"] ** 2
    gold = "sample A" if ka > kb else "sample B" if kb > ka else TEMP_SAME
    opts = ["sample A", "sample B", TEMP_SAME]
    random.Random(k).shuffle(opts)
    items = [{"q": TEMP_Q, "options": opts, "gold": gold, "kind": "temp-question"}]
    for f, (field, unit, q) in enumerate((("mass", "u", "What is the mass of each particle in sample %s?"),
                                          ("speed", "m/s", "What is the average particle speed in sample %s?"))):
        order = family_candidates([v[x][field] for v in views for x in "AB"], field, k + f)
        for name in ("A", "B"):
            items.append(reading(q % name, order, s[name][field], unit, "temp-" + field))
    return caption, items


# ---------------------------------------------------------------- force

FORCE_VALUES = {"in": [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5],
                "cm": [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 0.8, 1, 1.2, 1.25, 1.5, 1.6, 1.8, 2, 2.5, 3, 3.5, 4,
                       4.5, 5, 6, 7, 8],
                "mm": [5.5, 8.4, 9.6, 12.1, 13.2, 16.9, 18.3, 21.5, 24.7, 28.9, 31.6, 35.2, 42.8, 47.5, 51.3, 58.7,
                       63.4, 70.2]}


def force_scene(rng: random.Random) -> Dict:
    case = rng.choice(["distance", "size", "equal"])
    unit = rng.choice(list(FORCE_VALUES))
    values = FORCE_VALUES[unit]
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
            "tall": rng.choice([90, 130, 170]), "width": rng.randint(580, 700), "p1_left": rng.random() < 0.7}


def force_twin(s: Dict) -> Dict:
    """Only values change: the two pairs swap their distances and magnet sizes; an equal pair instead gets the next
    longer distance in Pair 2 (the next shorter one at the top of the range)."""
    if s["case"] == "equal":
        unit_values = FORCE_VALUES[s["unit"]]
        other = [v for v in unit_values if v > s["d"][0]] or [v for v in unit_values if v < s["d"][0]][-1:]
        return dict(s, case="distance", d=[s["d"][0], other[0]])
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


def force_geometry(s: Dict) -> Dict:
    """One layout for both pairs: the magnet scale fits two large magnets in a pair box, and the gap between the
    magnets depends only on the distance (its rank in the unit's value list, so it grows with the distance), the
    same mapping for both pairs."""
    w, h = s["width"], s["tall"] + 10
    pw = (w - 30) // 2
    big = max(26, int((s["tall"] - 50) * 0.95))
    wide = 2 if s["shape"] == "bar" else 1
    room = pw - 16 - 2 * big * wide                          # what is left for the gap with two large magnets
    if room < 0.35 * pw:                                     # too big for the box: shrink every magnet alike
        big = int((0.65 * pw - 16) / (2 * wide))
        room = pw - 16 - 2 * big * wide
    small = int(big * 0.6)
    values = FORCE_VALUES[s["unit"]]
    pairs = []
    for p in range(2):
        sizes = [big if s["big"][p][j] else small for j in range(2)]
        widths = [z * wide for z in sizes]
        gap = int(room * (0.3 + 0.65 * values.index(s["d"][p]) / (len(values) - 1)))
        box = 1 - p if not s.get("p1_left", True) else p           # Pair 1 drawn on the right in 30% of pictures
        bx0 = 10 + box * (pw + 10)
        start = bx0 + (pw - widths[0] - gap - widths[1]) // 2
        pairs.append({"box": bx0, "sizes": sizes, "widths": widths, "gap": gap, "start": start})
    return {"w": w, "h": h, "pw": pw, "pairs": pairs}


def force_render(s: Dict) -> Image.Image:
    g = force_geometry(s)
    w, h, pw = g["w"], g["h"], g["pw"]
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    font, label = ImageFont.truetype(FONT, s["font"]), ImageFont.truetype(FONT, s["font"] + 3)
    cy = 24 + (h - 26 - 24) // 2
    for p, q in enumerate(g["pairs"]):
        bx0 = q["box"]
        d.text((bx0 + pw / 2, 12), "Pair %d" % (p + 1), fill="black", font=label, anchor="mm")
        d.rectangle((bx0, 24, bx0 + pw, h - 2), outline="#cccccc", width=2)
        sizes, widths, gap, start = q["sizes"], q["widths"], q["gap"], q["start"]
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


def force_items(s: Dict, k: int, views: List[Dict]) -> Tuple[str, List[Dict]]:
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
    order = family_candidates([v["d"][p] for v in views for p in range(2)], "distance", k)
    for p in range(2):
        items.append(reading("What is the distance between the magnets in Pair %d?" % (p + 1), order, s["d"][p], s["unit"],
                             "force-distance"))
    return caption, items


# ---------------------------------------------------------------- shared

def shared(a: float, b: float, field: str, k: int) -> List[float]:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "exp28"))
    from build import candidates  # type: ignore
    return candidates(a, b, field, k)


def family_candidates(values: List[float], field: str, k: int) -> List[float]:
    """One candidate set for a field over both views of a family (at most two distinct values: twins swap values
    or change one), in a display order fixed for the family."""
    vals = sorted(set(values))
    assert len(vals) <= 2, vals
    cands = shared(vals[0], vals[-1], field, k)
    random.Random(k).shuffle(cands)
    return cands


def reading(q: str, order: List[float], right: float, unit: str, kind: str) -> Dict:
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


def case_key(form: str, s: Dict) -> tuple:
    """The canonical case of a view: the unordered pair of the two objects' values (force: distance and magnet
    sizes) -- what a split must not share."""
    if form == "temp":
        objs = sorted((s[x]["mass"], s[x]["speed"]) for x in "AB")
    else:
        objs = sorted((float(s["d"][p]), "large" if all(s["big"][p]) else "mixed") for p in range(2))
    return (form, s.get("unit", ""), tuple(objs))


def split_of(key: tuple, n: int, n_tune: int, n_confirm: int) -> str:
    """A fixed split for a new case, by hash, in proportion to the family counts."""
    u = int(hashlib.sha256(repr(key).encode()).hexdigest()[:12], 16) / 16 ** 12
    total = n + n_tune + n_confirm
    return "confirm" if u < n_confirm / total else "tune" if u < (n_confirm + n_tune) / total else "train"


SPLITS = ("train", "tune", "confirm")


def families(n: int, n_tune: int, n_confirm: int, seed: int = 31, stats: Optional[Dict] = None):
    """The scene families of `build`, in order: (number, split, form, scene, twin). The same seeded draws, split
    rules and skips as the drawn data, without drawing anything (exp37 re-renders the same scenes). `stats`
    receives the case owners, the per-split counts and the skip counts at the end."""
    rng = random.Random(seed)
    banned = transcribed_keys()
    want = {"train": n, "tune": n_tune, "confirm": n_confirm}
    have = {(sp, f): 0 for sp in SPLITS for f in ("temp", "force")}
    owner: Dict[tuple, str] = {}                              # case -> split
    i, skipped, attempts = 0, {"transcribed": 0, "case in another split": 0, "split full": 0}, 0
    while any(have[(sp, f)] < (want[sp] + (f == "temp")) // 2 for sp in SPLITS for f in ("temp", "force")):
        attempts += 1
        assert attempts < 200 * (n + n_tune + n_confirm) + 10000, "cannot fill the splits"
        form = "temp" if attempts % 2 else "force"
        s = temp_scene(rng) if form == "temp" else force_scene(rng)
        twin = temp_twin(s) if form == "temp" else force_twin(s)
        if scene_key(form, s) in banned or scene_key(form, twin) in banned:
            skipped["transcribed"] += 1
            continue
        keys = {case_key(form, s), case_key(form, twin)}
        known = {owner[k] for k in keys if k in owner}
        if len(known) > 1:
            skipped["case in another split"] += 1
            continue
        split = known.pop() if known else split_of(min(keys), n, n_tune, n_confirm)
        if have[(split, form)] >= (want[split] + (form == "temp")) // 2:
            skipped["split full"] += 1
            continue
        for k in keys:
            owner[k] = split
        have[(split, form)] += 1
        yield i, split, form, s, twin
        i += 1
    if stats is not None:
        stats.update(owner=owner, have=have, skipped=skipped)


def build(root: str, n: int, n_tune: int, n_confirm: int, seed: int = 31) -> Dict:
    """n / n_tune / n_confirm scene families (a scene and its value twin, both drawn), alternating temperature
    and force; families never cross splits. Captions for the alignment stage (train / tune), questions for every
    split; confirm is drawn once and never used to choose a recipe."""
    out = os.path.join(root, "real_layout")
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    files: Dict[str, List[Dict]] = {"%s_%s" % (kind, sp): [] for kind in ("captions", "questions") for sp in SPLITS}
    stats: Dict = {}
    for i, split, form, s, twin in families(n, n_tune, n_confirm, seed, stats):
        fam = "rl%05d" % i
        views = [s, twin]
        for v, scene in enumerate(views):
            img = temp_render(scene) if form == "temp" else force_render(scene)
            rel = "real_layout/images/%s_v%d.png" % (fam, v)
            img.save(os.path.join(root, rel))
            iid = "real-layout:%s:v%d" % (fam, v)
            caption, items = (temp_items if form == "temp" else force_items)(scene, i, views)
            hint = TEMP_HINT if form == "temp" else FORCE_HINT + (", but some of them are different sizes." if scene["case"] == "size" else ".")
            files["captions_" + split].append({"id": iid, "source": "real-layout-" + form, "image_id": iid, "image": rel,
                                              "family": fam, "captions": [caption]})
            files["questions_" + split].append({
                "id": iid, "source": "real-layout-" + form, "kind": form, "axis": scene["case"], "image_id": iid, "image": rel,
                "family": fam, "group": fam, "case": repr(case_key(form, scene)), "state_text": hint,
                "questions": {"q%d" % j: {"type": "choice", "instructions": it["q"], "criteria": {o: None for o in it["options"]}}
                              for j, it in enumerate(items)},
                "gold": {"q%d" % j: {"probabilities": {o: float(o == it["gold"]) for o in it["options"]}} for j, it in enumerate(items)}})
    owner, have, skipped = stats["owner"], stats["have"], stats["skipped"]
    cases = {sp: {f: sorted({repr(k) for k, o in owner.items() if o == sp and k[0] == f}) for f in ("temp", "force")} for sp in SPLITS}
    manifest: Dict = {"seed": seed, "families": {"%s_%s" % k: v for k, v in have.items()}, "skipped": skipped,
                      "cases": {sp: {f: len(v) for f, v in c.items()} for sp, c in cases.items()},
                      "case_overlap": {"%s-%s" % (a, b): sum(len(set(cases[a][f]) & set(cases[b][f])) for f in ("temp", "force"))
                                       for a, b in (("train", "tune"), ("train", "confirm"), ("tune", "confirm"))}}
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
