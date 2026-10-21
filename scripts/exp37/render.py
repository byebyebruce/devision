"""exp37 pictures (critic EXP37-PREP-01; docs/experiments/2026-10-17-exp37-digit-size-plan.md): the same real-layout
scenes (scripts/data/real_layout.py families(), replayed draw by draw) drawn in ONE shared layout with two digit
sizes:

  - S: every number at the scene's original text size (the size real_layout drew it at);
  - L: every number large enough that its digit ink is 16 px high (+-1) in the actual 256 letterbox input.

The shared layout reserves a fixed box for every number (its width is the widest possible string of that field at
the L size, its place depends only on the scene's layout parameters, never on a value or an answer); the label,
quantity name and unit are drawn outside the box at the original text size, identical in S and L. Temperature:
under each jar "Sample A", then per field (in the scene's row order) the quantity line ("Mass of each particle:")
and a row [number box][unit]. Force: the original magnets and distance bar (now without the text on it), and under
each pair a row [number box][unit]. So an S picture and its L picture differ only inside the number boxes -- every
pixel outside is identical (checked for every pair). Particles, magnets, colours, sides and twins are those of the
original scene.

Outputs data/exp37/images/{S,L}/<family>_v<k>.png for the exp36 training families, the 600 tune pictures and (from
them) the 200 fit pictures; per picture and number: box, digit ink height at 256 (measured: the L picture against
the same picture with the digits not drawn, letterboxed, rows with a difference >= 64 grey levels), variant.

    uv run python scripts/exp37/render.py --root data
"""
import argparse
import functools
import hashlib
import json
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
import real_layout as rl  # type: ignore  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from devision.image import letterbox  # noqa: E402

TARGET = 16            # digit ink height at 256 for L
WIDEST = {"mass": "00", "speed": "0,000", "distance": "00.00"}
UNITS = {"mass": "u", "speed": "m/s"}
LABEL = [("Mass of each particle:", "Average particle speed:"), ("Particle mass:", "Average speed:")]


@functools.lru_cache(maxsize=None)
def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(rl.BOLD if bold else rl.FONT, size)


@functools.lru_cache(maxsize=None)
def digit_ink(size: int) -> int:
    """Ink height of the digits at `size` in the original picture (from the glyph boxes)."""
    b = font(size).getbbox("0123456789")
    return int(b[3] - b[1])


def temp_small_size(s: Dict) -> int:
    """The text size real_layout.temp_render drew this scene with (the longest line fitted into half the width)."""
    w, size = s["width"], s["font"]
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    longest = rl.LINES[s["wording"]][1] % "1,999"
    while size > 11 and probe.textlength(longest, font=font(size)) > w / 2 - 12:
        size -= 1
    return size


@functools.lru_cache(maxsize=None)
def l_size(scale: float) -> int:
    """The digit size whose ink is TARGET px at this letterbox scale (nearest)."""
    # the glyph boxes overestimate the letterboxed ink by about half a pixel: aim at TARGET - 0.5
    return min(range(10, 120), key=lambda f: (abs(digit_ink(f) * scale - (TARGET - 0.5)), f))


# ---------------------------------------------------------------- temperature

def temp_layout(s: Dict) -> Dict:
    """Positions shared by S and L: canvas, rows and number boxes (no value is read here)."""
    w, small = s["width"], temp_small_size(s)
    line = int(small * s["spacing"])
    fields = ["speed", "mass"] if s.get("speed_first") else ["mass", "speed"]
    names = ("A", "B") if s.get("a_left", True) else ("B", "A")
    big, cols, h = 40, {}, 0
    for _ in range(5):                                   # the canvas height depends on the box height
        box_h = digit_ink(big) + int(0.35 * big) + 6   # room for the comma's descender
        cols = {}
        for k, name in enumerate(names):
            cx = w // 4 + k * (w // 2)
            y = 315 + line // 2
            title, rows = (cx, y), []
            for f in fields:
                label_y = y + line
                top = label_y + int(small * 0.6) + 4        # below the quantity line's ink
                bw = int(font(big).getlength(WIDEST[f])) + 6
                unit_w = int(font(small).getlength(" " + UNITS[f]))
                x0 = int(cx - (bw + unit_w) / 2)
                rows.append({"field": f, "label_y": label_y, "box": (x0, top, x0 + bw, top + box_h), "unit_x": x0 + bw + 2})
                y = top + box_h + 4 - line // 2
            cols[name] = {"cx": cx, "title": title, "rows": rows}
        h = max(r["box"][3] for c in cols.values() for r in c["rows"]) + 10
        nb = l_size(256 / max(w, h))
        if nb == big:
            break
        big = nb
    return {"w": w, "h": h, "small": small, "big": big, "line": line, "cols": cols, "scale": 256 / max(w, h)}


def temp_draw(s: Dict, variant: str, digits: bool = True, strip: bool = False) -> Tuple[Image.Image, List[Dict]]:
    L = temp_layout(s)
    img = Image.new("RGB", (L["w"], L["h"]), "white")
    d = ImageDraw.Draw(img)
    rng = random.Random(s["seed"])
    small, bold = font(L["small"]), font(L["small"], bold=True)
    num = font(L["big"] if variant == "L" else L["small"])
    numbers = []
    for k, name in enumerate(("A", "B") if s.get("a_left", True) else ("B", "A")):
        col = L["cols"][name]
        cx = col["cx"]
        x0, x1, y0, y1 = cx - 95, cx + 95, 30, 300           # the jar and particles exactly as real_layout draws them
        d.rounded_rectangle((x0, y0 + 25, x1, y1), radius=28, outline="#9a9a9a", width=2)
        d.rectangle((x0 + 10, y0, x1 - 10, y0 + 20), fill="#9a9a9a")
        mass = s[name]["mass"]
        r = 6 + mass // 5
        for _ in range(s["count"]):
            px, py = rng.randint(x0 + 20 + rl.R_MAX, x1 - 20 - rl.R_MAX), rng.randint(y0 + 50 + rl.R_MAX, y1 - 15 - rl.R_MAX)
            d.ellipse((px - r, py - r, px + r, py + r), fill=s["colours"][k])
        d.text(col["title"], "Sample " + name, fill="black", font=bold, anchor="mm")
        for row in col["rows"]:
            f = row["field"]
            label = LABEL[s.get("wording", 0)][0 if f == "mass" else 1]
            d.text((cx, row["label_y"]), label, fill="black", font=small, anchor="mm")
            bx = row["box"]
            d.text((row["unit_x"], (bx[1] + bx[3]) / 2), " " + UNITS[f], fill="black", font=small, anchor="lm")
            text = str(mass) if f == "mass" else rl.fmt(s[name]["speed"])
            if digits:
                shown = text.replace(",", "").replace(".", "") if strip else text
                d.text(((bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2), shown, fill="black", font=num, anchor="mm")
            numbers.append({"field": f, "object": "Sample " + name, "text": text, "box": list(bx)})
    return img, numbers


# ---------------------------------------------------------------- force

def force_layout(s: Dict) -> Dict:
    g = rl.force_geometry(s)
    small = s["font"]
    w = g["w"]
    big = 40
    for _ in range(4):
        box_h = digit_ink(big) + int(0.35 * big) + 6   # room for the comma's descender
        h = g["h"] + box_h + 12
        nb = l_size(256 / max(w, h))
        if nb == big:
            break
        big = nb
    pw = g["pw"]
    bw = int(font(big).getlength(WIDEST["distance"])) + 6
    rows = []
    for p, q in enumerate(g["pairs"]):
        unit_w = int(font(small).getlength(" " + s["unit"]))
        x0 = int(q["box"] + pw / 2 - (bw + unit_w) / 2)
        box = (x0, g["h"] + 4, x0 + bw, g["h"] + 4 + box_h)
        rows.append({"pair": p, "box": box, "unit_x": x0 + bw + 2})
    return {"g": g, "w": w, "h": h, "small": small, "big": big, "rows": rows, "scale": 256 / max(w, h)}


def force_draw(s: Dict, variant: str, digits: bool = True, strip: bool = False) -> Tuple[Image.Image, List[Dict]]:
    L = force_layout(s)
    g = L["g"]
    img = Image.new("RGB", (L["w"], L["h"]), "white")
    d = ImageDraw.Draw(img)
    small, label = font(L["small"]), font(L["small"] + 3)
    num = font(L["big"] if variant == "L" else L["small"])
    pw, gh = g["pw"], g["h"]
    cy = 24 + (gh - 26 - 24) // 2
    numbers = []
    for p, q in enumerate(g["pairs"]):                    # magnets and bar exactly as real_layout draws them
        bx0 = q["box"]
        d.text((bx0 + pw / 2, 12), "Pair %d" % (p + 1), fill="black", font=label, anchor="mm")
        d.rectangle((bx0, 24, bx0 + pw, gh - 2), outline="#cccccc", width=2)
        sizes, widths, gap, start = q["sizes"], q["widths"], q["gap"], q["start"]
        for j in range(2):
            x = start if j == 0 else start + widths[0] + gap
            rl._magnet(d, (x, cy - sizes[j] // 2, x + widths[j], cy + sizes[j] // 2), s["shape"], s["poles"][p][j], s["colours"], small)
        ax0, ax1, ty = start + widths[0], start + widths[0] + gap, gh - 14
        d.line((ax0, ty, ax1, ty), fill="#555555", width=1)
        d.line((ax0, ty - 4, ax0, ty + 4), fill="#555555", width=1)
        d.line((ax1, ty - 4, ax1, ty + 4), fill="#555555", width=1)
        row = L["rows"][p]
        bx = row["box"]
        d.text((row["unit_x"], (bx[1] + bx[3]) / 2), " " + s["unit"], fill="black", font=small, anchor="lm")
        text = rl.fmt(s["d"][p])
        if digits:
            shown = text.replace(",", "").replace(".", "") if strip else text
            d.text(((bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2), shown, fill="black", font=num, anchor="mm")
        numbers.append({"field": "distance", "object": "Pair %d" % (p + 1), "text": text, "box": list(bx)})
    return img, numbers


# ---------------------------------------------------------------- measuring and building

def ink_rows(a: Image.Image, b: Image.Image, box: List[int], scale_img: Image.Image) -> int:
    """Digit ink height at 256: rows inside the box's letterboxed region where the two letterboxed pictures differ
    by >= 64 grey levels."""
    la, lb = (np.asarray(letterbox(x, 256).convert("L"), dtype=int) for x in (a, b))
    w, h = scale_img.size
    sc = 256 / max(w, h)
    ox, oy = (256 - round(w * sc)) // 2, (256 - round(h * sc)) // 2
    y0, y1 = int(box[1] * sc) + oy - 1, int(box[3] * sc) + oy + 2
    x0, x1 = int(box[0] * sc) + ox - 1, int(box[2] * sc) + ox + 2
    diff = np.abs(la[y0:y1, x0:x1] - lb[y0:y1, x0:x1]) >= 64
    rows = np.where(diff.any(1))[0]
    return int(rows[-1] - rows[0] + 1) if len(rows) else 0


def render(form: str, s: Dict, variant: str, digits: bool = True, strip: bool = False):
    return (temp_draw if form == "temp" else force_draw)(s, variant, digits, strip)


def outside_boxes_equal(a: Image.Image, b: Image.Image, boxes: List[List[int]]) -> bool:
    x, y = np.asarray(a), np.asarray(b)
    if x.shape != y.shape:
        return False
    m = np.ones(x.shape[:2], bool)
    for b0, b1, b2, b3 in boxes:
        m[b1:b3, b0:b2] = False
    return bool((x[m] == y[m]).all())


def build(root: str, measure_every: int = 1) -> Dict:
    exp36 = json.load(open(os.path.join(root, "exp36", "MANIFEST.json")))
    wanted = {f for v in exp36["families_chosen"].values() for f in v}
    tune = {json.loads(x)["family"] for x in open(os.path.join(root, "real_layout", "captions_tune.jsonl"))}
    out = os.path.join(root, "exp37")
    for v in ("S", "L"):
        os.makedirs(os.path.join(out, "images", v), exist_ok=True)
    meta: Dict[str, Dict] = {}
    bad: List[str] = []
    heights: Dict[str, List[int]] = {"S": [], "L": []}
    for i, split, form, s, twin in rl.families(6000, 300, 300):
        fam = "rl%05d" % i
        if fam not in wanted and fam not in tune:
            continue
        for k, scene in enumerate((s, twin)):
            imgs, nums = {}, None
            for v in ("S", "L"):
                img, numbers = render(form, scene, v)
                img.save(os.path.join(out, "images", v, "%s_v%d.png" % (fam, k)))
                imgs[v], nums = img, numbers
            assert nums is not None
            boxes = [n["box"] for n in nums]
            if not outside_boxes_equal(imgs["S"], imgs["L"], boxes):
                bad.append("%s_v%d: pixels outside the number boxes differ" % (fam, k))
            w, h = imgs["L"].size
            for n in nums:
                b = n["box"]
                if not (0 <= b[0] < b[2] <= w and 0 <= b[1] < b[3] <= h):
                    bad.append("%s_v%d: box outside the picture" % (fam, k))
            if (i + k) % measure_every == 0:
                blank, _ = render(form, scene, "L", digits=False)
                for v in ("S", "L"):
                    bare, _ = render(form, scene, v, strip=True)        # digits without "," / "." for the height
                    for n in nums:
                        n["ink_256_" + v] = ink_rows(bare, blank, n["box"], bare)
                        heights[v].append(n["ink_256_" + v])
                    full = np.asarray(imgs[v].convert("L"), dtype=int) - np.asarray(blank.convert("L"), dtype=int)
                    inside = np.zeros(full.shape, bool)
                    for n in nums:
                        b = n["box"]
                        inside[b[1]:b[3], b[0]:b[2]] = True
                    if (np.abs(full[~inside]) > 0).any():            # every digit pixel inside its own box
                        bad.append("%s_v%d %s: digit ink outside the boxes" % (fam, k, v))
            meta["%s_v%d" % (fam, k)] = {"family": fam, "view": k, "form": form, "split": split, "numbers": nums,
                                        "size": [w, h], "letterbox_scale": 256 / max(w, h)}
    return {"meta": meta, "problems": bad, "heights": heights}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data")
    a = ap.parse_args(argv)
    r = build(a.root)
    out = os.path.join(a.root, "exp37")
    json.dump(r["meta"], open(os.path.join(out, "pictures.json"), "w"))
    hs = {v: np.array(x) for v, x in r["heights"].items()}
    summary = {"pictures": len(r["meta"]), "problems": r["problems"][:20], "n_problems": len(r["problems"]),
               "ink_256": {v: {"n": int(len(x)), "min": int(x.min()), "max": int(x.max()),
                               "hist": {int(k): int(c) for k, c in zip(*np.unique(x, return_counts=True))}} for v, x in hs.items()}}
    json.dump(summary, open(os.path.join(out, "render_summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
