"""exp24 reading alignment data (docs/experiments/2026-10-16-exp24-read-align-plan.md): pictures with a few lines
of printed text -- numbers (with or without a unit) and short labels -- and, as the caption, exactly that text in
reading order (top to bottom). The alignment stage (src/devision/train/align.py) masks caption tokens and
ModernBERT has to fill them in from the visual tokens; the content is random, so the text around a mask does
not give it away.

    uv run python scripts/data/exp24_text.py --root data

A picture: light background, 1-4 lines stacked top to bottom at random x, each line one item drawn with digit ink
height 6 / 8 / 11 / 16 / 20 px (scripts/data/exp21_reading.font_for). Items: a number (1-4 digits, thousands
comma from 1,000) alone or with a unit, or a label (a quantity word, "Sample A", "Pair 2", a colour, ...).
Outputs in <root>/exp24/: align_train.jsonl (20,000 pictures), align_val.jsonl (1,000 others), images/,
MANIFEST.json (counts, heights, item kinds). Caption sample format: src/devision/train/samples.py.
"""
import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(__file__))
from exp21_reading import font_for, fmt  # noqa: E402

HEIGHTS = (6, 8, 11, 16, 20)
UNITS = ["cm", "m", "km", "g", "kg", "mL", "L", "s", "m/s", "u", "in", "mm", "N", "J", "K"]
WORDS = ["length", "mass", "speed", "volume", "time", "distance", "temperature", "force", "energy", "width",
         "height", "weight", "price", "area", "depth", "red", "blue", "green", "yellow", "orange", "purple",
         "north", "south", "east", "west", "left", "right", "top", "bottom", "solution", "particle", "magnet",
         "pair", "sample", "total", "average", "maximum", "minimum", "start", "finish"]
BACKGROUNDS = ["#ffffff", "#f8fafc", "#fefce8", "#f0fdf4", "#eff6ff", "#fdf2f8", "#f5f3ff"]


def item(rng: random.Random) -> Tuple[str, str]:
    """(kind, text)."""
    r = rng.random()
    if r < 0.6:
        digits = rng.choices([1, 2, 3, 4], [0.15, 0.35, 0.3, 0.2])[0]
        v = rng.randint(1 if digits == 1 else 10 ** (digits - 1), 10 ** digits - 1)
        if rng.random() < 0.5:
            return "number+unit", "%s %s" % (fmt(v), rng.choice(UNITS))
        return "number", fmt(v)
    if r < 0.8:
        return "label", rng.choice(["Sample", "Pair", "Panel", "Group"]) + " " + rng.choice("ABCD12")
    return "word", rng.choice(WORDS)


def scene(rng: random.Random) -> Dict:
    lines = [item(rng) + (rng.choice(HEIGHTS),) for _ in range(rng.randint(1, 4))]
    return {"lines": lines, "background": rng.choice(BACKGROUNDS), "ink": rng.choice(["#000000", "#1f2937", "#374151"])}


def render(s: Dict) -> Tuple[Image.Image, List[Tuple[int, int, int, int]]]:
    """The picture and each line's box; lines go top to bottom with gaps, so caption order = reading order."""
    img = Image.new("RGB", (256, 256), s["background"])
    d = ImageDraw.Draw(img)
    rng = random.Random(json.dumps(s))
    heights = [h for _, _, h in s["lines"]]
    free = 256 - 8 - sum(int(h * 1.6) for h in heights)
    y = 4 + rng.randint(0, max(0, free // 2))
    boxes = []
    for (kind, text, h), lh in zip(s["lines"], heights):
        font = font_for(h)
        w = int(d.textlength(text, font=font))
        x = rng.randint(4, max(4, 252 - w))
        base = y + h
        d.text((x, base), text, fill=s["ink"], font=font, anchor="ls")
        boxes.append(tuple(int(v) for v in d.textbbox((x, base), text, font=font, anchor="ls")))
        y = base + int(lh * 0.6) + rng.randint(2, max(2, free // (2 * len(heights))))
    return img, boxes


def caption(s: Dict) -> str:
    return ". ".join(text for _, text, _ in s["lines"])


def build(root: str, n_train: int = 20000, n_val: int = 1000, seed: int = 24) -> Dict:
    out = os.path.join(root, "exp24")
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    rng = random.Random(seed)
    seen, manifest = set(), {"seed": seed, "files": {}}
    for split, n in (("align_train", n_train), ("align_val", n_val)):
        rows, kinds, hs = [], Counter(), Counter()
        while len(rows) < n:
            s = scene(rng)
            img, boxes = render(s)
            if any(b[2] > 256 or b[3] > 256 or b[0] < 0 or b[1] < 0 for b in boxes):
                continue
            digest = hashlib.sha256(img.tobytes()).hexdigest()
            if digest in seen:          # never the same picture twice, in or across splits
                continue
            seen.add(digest)
            rel = "exp24/images/%s_%05d.png" % (split, len(rows))
            img.save(os.path.join(root, rel))
            rows.append({"id": "exp24:%s:%05d" % (split, len(rows)), "source": "exp24-text",
                         "image_id": "exp24:" + digest[:16], "image": rel, "captions": [caption(s)]})
            kinds.update(k for k, _, _ in s["lines"])
            hs.update(str(h) for _, _, h in s["lines"])
        with open(os.path.join(out, split + ".jsonl"), "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        manifest["files"][split] = {"pictures": len(rows), "lines": sum(kinds.values()), "kinds": dict(kinds),
                                    "heights": dict(hs)}
    json.dump(manifest, open(os.path.join(out, "MANIFEST.json"), "w"), indent=1)
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    a = p.parse_args()
    if os.path.exists(os.path.join(a.root, "exp24", "MANIFEST.json")):
        raise SystemExit("data/exp24 exists; remove it first")
    print(json.dumps(build(a.root), indent=1))
