"""exp21 synthetic reading probe (docs/experiments/2026-10-15-exp21-reading-plan.md): two labelled panels, each
showing one printed number; questions read panel A, read panel B, or compare them. Labels come from the
generated values, never from the picture.

    uv run python scripts/data/exp21_reading.py --root data --replay data/r14b/train.jsonl

A scene family fixes the layout (panels side by side or stacked, which panel comes first, colours), the quantity
and unit, the candidate set and the option order. Inside a family only two things vary:
  - the value view: view 0 holds the family's values; view 1 swaps A and B (or, for an equal pair, changes B to
    one of the candidates), so every one of the three answers that can change does, with the same candidates
    and the same option order (critic EXP21-R1-02 item 3);
  - the digit height: 6 / 8 / 11 / 16 px of ink in the 256 x 256 picture the model sees (drawn at 256, so no
    resize). Only the number changes size; the panel letters, quantity line and layout do not (item 4).
The candidates of the two reading questions are the same three values: A's, B's and one more value differing
from one of them in a single digit, all with the same number of digits (no length cue). Compare options are
"A", "B" and "neither; they are equal".

Outputs in <root>/exp21/ (pictures in images/):
  - train_syn.jsonl: 2,667 families, one view and one height each, the three questions on one picture;
  - dev_syn.jsonl: 200 other families x 2 views x 4 heights x 3 questions = 4,800 questions;
  - dev_syn_erased.jsonl: the first 100 dev families x 2 views with the numbers not drawn (same questions; a
    shortcut check: should be near chance);
  - dev_syn_text.jsonl: the same erased pictures with the two values given as text (state_text): can the model
    compare the numbers when it does not have to read them;
  - train_mix.jsonl: train_syn + r14b replay (seed 1) to at least 16,008 questions, shuffled;
  - MANIFEST.json: families, pictures, questions and question-uses per file, answer balance, measured heights.
"""
import argparse
import hashlib
import json
import os
import random
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
HEIGHTS = (6, 8, 11, 16)
QUANTITIES = [("length", "cm"), ("mass", "g"), ("speed", "m/s"), ("volume", "mL"), ("time", "s"), ("distance", "km")]
COLOURS = ["#dbeafe", "#dcfce7", "#fef3c7", "#fce7f3", "#ede9fe", "#e0f2f1", "#f3f4f6"]
EQUAL = "neither; they are equal"
DIGITS = [(1, 0.15), (2, 0.35), (3, 0.3), (4, 0.2)]
SIZE = 256


def fmt(v: int) -> str:
    return "{:,}".format(v)


def number(rng: random.Random, digits: int) -> int:
    lo = 0 if digits == 1 else 10 ** (digits - 1)
    return rng.randint(max(lo, 1), 10 ** digits - 1)


def one_digit_off(rng: random.Random, v: int, avoid) -> int:
    """A value with the same number of digits as v, differing in exactly one digit, not in `avoid`."""
    s = str(v)
    for _ in range(100):
        i = rng.randrange(len(s))
        d = rng.choice([c for c in "0123456789" if c != s[i] and not (i == 0 and c == "0")])
        w = int(s[:i] + d + s[i + 1:])
        if w not in avoid and w > 0:
            return w
    raise ValueError("no one-digit variant of %d" % v)


def family(rng: random.Random, fid: str) -> Dict:
    digits = rng.choices([d for d, _ in DIGITS], [w for _, w in DIGITS])[0]
    kind = rng.choice(["A", "B", "equal"])          # the compare answer of view 0, a third each
    while True:
        a, b = number(rng, digits), number(rng, digits)
        if a != b:
            break
    if kind == "equal":
        b = a
        c1 = one_digit_off(rng, a, {a})
        c2 = one_digit_off(rng, a, {a, c1})
        candidates = [a, c1, c2]
        views = [(a, a), (a, c1)]                    # view 1 changes B: read B and compare change, read A does not
    else:
        if (a > b) != (kind == "A"):
            a, b = b, a
        extra = one_digit_off(rng, rng.choice([a, b]), {a, b})
        candidates = [a, b, extra]
        views = [(a, b), (b, a)]                     # view 1 swaps: all three answers change
    rng.shuffle(candidates)
    compare_options = ["A", "B", EQUAL]
    rng.shuffle(compare_options)
    name, unit = rng.choice(QUANTITIES)
    return {"family": fid, "name": name, "unit": unit, "candidates": candidates, "compare_options": compare_options,
            "views": views, "stacked": rng.random() < 0.3, "a_first": rng.random() < 0.5,
            "colours": rng.sample(COLOURS, 2), "digits": digits}


def answers(f: Dict, view: int) -> Dict[str, str]:
    a, b = f["views"][view]
    return {"readA": fmt(a) + " " + f["unit"], "readB": fmt(b) + " " + f["unit"],
            "cmp": "A" if a > b else "B" if b > a else EQUAL}


def questions(f: Dict) -> Dict[str, Dict]:
    opts = {k: None for k in (fmt(v) + " " + f["unit"] for v in f["candidates"])}
    return {"readA": {"type": "choice", "instructions": "What %s is shown for panel A?" % f["name"], "criteria": dict(opts)},
            "readB": {"type": "choice", "instructions": "What %s is shown for panel B?" % f["name"], "criteria": dict(opts)},
            "cmp": {"type": "choice", "instructions": "Which panel shows the greater %s?" % f["name"],
                    "criteria": {k: None for k in f["compare_options"]}}}


_SIZES: Dict[int, int] = {}


def ink_height(size: int, text: str = "0123456789") -> int:
    """Rows of dark ink (< 128) of `text` drawn at Arial `size` -- pixels, not the font's point size or bbox."""
    font = ImageFont.truetype(FONT, size)
    im = Image.new("L", (40 * size, 4 * size), 255)
    ImageDraw.Draw(im).text((2, 2 * size), text, font=font, fill=0, anchor="ls")
    return int(dark_rows(im))


def dark_rows(im: Image.Image) -> int:
    a = np.asarray(im.convert("L"))
    rows = np.flatnonzero((a < 128).any(axis=1))
    return int(rows[-1] - rows[0] + 1) if len(rows) else 0


def font_for(height: int) -> ImageFont.FreeTypeFont:
    """The smallest Arial size whose digits are `height` px of ink."""
    if height not in _SIZES:
        _SIZES[height] = min(range(4, 60), key=lambda s: (abs(ink_height(s) - height), s))
    return ImageFont.truetype(FONT, _SIZES[height])


def digit_ink(img: Image.Image, box) -> int:
    """Measured ink height of a drawn number: dark rows inside `box` = (left, top, right, baseline), so a
    comma's tail below the baseline is not counted."""
    x0, y0, x1, base = (int(round(v)) for v in box)
    return dark_rows(img.convert("L").crop((x0 - 1, y0 - 2, x1 + 1, base)))


def boxes(f: Dict) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    """(panel letter, box) in drawing order."""
    m, g = 10, 12
    if f["stacked"]:
        h = (SIZE - 2 * m - g) // 2
        b = [(m, m, SIZE - m, m + h), (m, m + h + g, SIZE - m, SIZE - m)]
    else:
        w = (SIZE - 2 * m - g) // 2
        b = [(m, 40, m + w, SIZE - 40), (m + w + g, 40, SIZE - m, SIZE - 40)]
    order = ["A", "B"] if f["a_first"] else ["B", "A"]
    return list(zip(order, b))


def render(f: Dict, view: int, height: int, erased: bool = False) -> Tuple[Image.Image, Dict[str, Tuple]]:
    """The picture and, per panel, the ink box of the drawn number (empty when erased)."""
    a, b = f["views"][view]
    value = {"A": a, "B": b}
    img = Image.new("RGB", (SIZE, SIZE), "white")
    d = ImageDraw.Draw(img)
    title, label, num = ImageFont.truetype(BOLD, 22), ImageFont.truetype(FONT, 13), font_for(height)
    ink = {}
    for (p, (x0, y0, x1, y1)), c in zip(boxes(f), f["colours"]):
        d.rectangle((x0, y0, x1, y1), fill=c, outline="#555555", width=2)
        cx = (x0 + x1) / 2
        d.text((cx, y0 + 6), p, fill="black", font=title, anchor="mt")
        d.text((cx, y0 + 34), "%s (%s)" % (f["name"], f["unit"]), fill="#333333", font=label, anchor="mt")
        if not erased:
            base = int((y0 + 52 + y1) / 2 + height / 2)   # baseline: the digits sit above it
            d.text((cx, base), fmt(value[p]), fill="black", font=num, anchor="ms")
            ink[p] = d.textbbox((cx, base), fmt(value[p]), font=num, anchor="ms")[:3] + (base,)
    return img, ink


def sample(f: Dict, view: int, height: int, image: str, image_id: str, qids=("readA", "readB", "cmp"),
           erased=False, text=False) -> Dict:
    qs, ans = questions(f), answers(f, view)
    s = {"id": "exp21:%s:v%d:h%d%s" % (f["family"], view, height, ":erased" if erased else ":text" if text else ""),
         "source": "exp21-syn" + ("-erased" if erased else "-text" if text else ""), "kind": "exp21",
         "axis": "h%d" % height, "image_id": image_id, "image": image, "family": f["family"], "view": view,
         "height": height, "questions": {q: qs[q] for q in qids},
         "gold": {q: {"probabilities": {k: float(k == ans[q]) for k in qs[q]["criteria"]}} for q in qids}}
    if text:
        a, b = f["views"][view]
        s["state_text"] = "Panel A shows a %s of %s %s. Panel B shows a %s of %s %s." % (
            f["name"], fmt(a), f["unit"], f["name"], fmt(b), f["unit"])
    return s


def save(img: Image.Image, root: str, rel: str) -> None:
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)


def build(root: str, replay: str, seed: int = 21, n_train: int = 2667, n_dev: int = 200, n_controls: int = 100,
          total: int = 16008) -> Dict:
    out = os.path.join(root, "exp21")
    rng = random.Random(seed)
    train_f = [family(rng, "t%04d" % i) for i in range(n_train)]
    seen = {(f["name"], tuple(sorted(f["candidates"]))) for f in train_f}
    dev_f: List[Dict] = []
    while len(dev_f) < n_dev:   # a dev family never repeats a train family's quantity and candidate values
        f = family(rng, "d%04d" % len(dev_f))
        if (f["name"], tuple(sorted(f["candidates"]))) not in seen:
            dev_f.append(f)
    files: Dict[str, List[Dict]] = {"train_syn": [], "dev_syn": [], "dev_syn_erased": [], "dev_syn_text": []}
    heights: Dict[int, Counter] = {h: Counter() for h in HEIGHTS}
    for f in train_f:
        view, h = rng.randrange(2), HEIGHTS[rng.randrange(len(HEIGHTS))]
        rel = "exp21/images/train/%s_v%d_h%d.png" % (f["family"], view, h)
        save(render(f, view, h)[0], root, rel)
        files["train_syn"].append(sample(f, view, h, rel, "exp21:%s:v%d:h%d" % (f["family"], view, h)))
    for i, f in enumerate(dev_f):
        for view in (0, 1):
            for h in HEIGHTS:
                img, ink = render(f, view, h)
                heights[h].update(digit_ink(img, bx) for bx in ink.values())
                rel = "exp21/images/dev/%s_v%d_h%d.png" % (f["family"], view, h)
                save(img, root, rel)
                files["dev_syn"].append(sample(f, view, h, rel, "exp21:%s:v%d:h%d" % (f["family"], view, h)))
            if i < n_controls:
                rel = "exp21/images/dev/%s_v%d_erased.png" % (f["family"], view)
                save(render(f, view, HEIGHTS[0], erased=True)[0], root, rel)
                iid = "exp21:%s:v%d:erased" % (f["family"], view)
                files["dev_syn_erased"].append(sample(f, view, 0, rel, iid, erased=True))
                files["dev_syn_text"].append(sample(f, view, 0, rel, iid, text=True))
    mix = list(files["train_syn"])
    n = sum(len(s["questions"]) for s in mix)
    pool = [json.loads(line) for line in open(replay)]
    random.Random(1).shuffle(pool)
    replayed = 0
    for s in pool:
        if n >= total:
            break
        mix.append(s)
        n += len(s["questions"])
        replayed += len(s["questions"])
    random.Random(1).shuffle(mix)
    files["train_mix"] = mix
    manifest = {"seed": seed, "heights_target": list(HEIGHTS), "font_sizes": {str(h): _SIZES[h] for h in HEIGHTS},
                "dev_ink_height_px": {str(h): {str(k): v for k, v in sorted(heights[h].items())} for h in HEIGHTS},
                "replay": {"path": replay, "sha256": hashlib.sha256(open(replay, "rb").read()).hexdigest(),
                           "questions": replayed}, "files": {}}
    for name, rows in files.items():
        with open(os.path.join(out, name + ".jsonl"), "w") as fh:
            for s in rows:
                fh.write(json.dumps(s) + "\n")
        syn = [s for s in rows if s.get("source", "").startswith("exp21")]
        manifest["files"][name] = {
            "rows": len(rows), "questions": sum(len(s["questions"]) for s in rows),
            "synthetic_questions": sum(len(s["questions"]) for s in syn),
            "families": len({s["family"] for s in syn}), "pictures": len({s["image"] for s in syn}),
            "answers": dict(Counter("%s:%s" % (q, "equal" if v == EQUAL else v if q == "cmp" else "value")
                                    for s in syn for q, g in s["gold"].items()
                                    for v, p in g["probabilities"].items() if p == 1.0 and q == "cmp")),
            "heights": dict(Counter(str(s["height"]) for s in syn))}
    json.dump(manifest, open(os.path.join(out, "MANIFEST.json"), "w"), indent=1)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    p.add_argument("--replay", default="data/r14b/train.jsonl")
    a = p.parse_args()
    if os.path.exists(os.path.join(a.root, "exp21", "MANIFEST.json")):
        raise SystemExit("data/exp21 exists; remove it first")
    print(json.dumps(build(a.root, a.replay), indent=1))


if __name__ == "__main__":
    main()
