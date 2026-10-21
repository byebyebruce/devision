"""exp49 input examples (critic EXP49-PREP-04): 12 questions by a fixed rule -- the first by key of each split (fit / tune)
x layout (original / L) x field -- each as the model's 256 letterbox with the box (red), the SigLIP 16 x 16 cells the head
reads (blue) and the projector 8 x 8 cells (green outline), plus the target string. runs/exp49-preview/sheet.png, notes.json.

    uv run python scripts/exp49/preview.py --root data
"""
import argparse
import importlib.util
import json
import os
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from devision.image import letterbox  # noqa: E402

_spec = importlib.util.spec_from_file_location("exp49_extract", os.path.join(os.path.dirname(__file__), "extract.py"))
assert _spec is not None and _spec.loader is not None
ex = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ex)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data")
    a = ap.parse_args(argv)
    qs = sorted((json.loads(x) for x in open(os.path.join(a.root, "exp49", "questions.jsonl"))), key=lambda q: q["key"])
    pick = []
    for split in ("fit", "tune"):
        for layout in ("original", "L"):
            for field in ("mass", "speed", "distance"):
                pick.append(next(q for q in qs if (q["split"], q["layout"], q["field"]) == (split, layout, field)))
    tile, pad = 256, 8
    sheet = Image.new("RGB", (4 * (tile + pad), 3 * (tile + 30 + pad)), "white")
    d = ImageDraw.Draw(sheet)
    notes = []
    for i, q in enumerate(pick):
        img = letterbox(Image.open(os.path.join(a.root, q["picture"])).convert("RGB"), tile)
        ov = ImageDraw.Draw(img, "RGBA")
        for side, colour in ((16, None), (8, (0, 160, 0, 255))):
            idx, _ = ex.region(q["box"], q["width"], q["height"], side)
            cell = tile // side
            for c in idx:
                y, x = divmod(c, side)
                box = (x * cell, y * cell, x * cell + cell - 1, y * cell + cell - 1)
                ov.rectangle(box, fill=(0, 120, 255, 70)) if colour is None else ov.rectangle(box, outline=colour, width=2)
        sc = tile / max(q["width"], q["height"])
        ox, oy = (tile - round(q["width"] * sc)) // 2, (tile - round(q["height"] * sc)) // 2
        x, y, w, h = q["box"]
        ov.rectangle((ox + x * sc, oy + y * sc, ox + (x + w) * sc, oy + (y + h) * sc), outline=(255, 0, 0, 255), width=2)
        col, row = i % 4, i // 4
        sheet.paste(img, (col * (tile + pad), row * (tile + 30 + pad)))
        d.text((col * (tile + pad), row * (tile + 30 + pad) + tile + 2), "%s %s %s %s -> %s" % (q["set"], q["field"], q["object"], q["id"].split(":")[1], q["target"]), fill="black")
        notes.append({k: q[k] for k in ("key", "picture", "box", "field", "object", "target", "twin", "twin_changed")})
    os.makedirs("runs/exp49-preview", exist_ok=True)
    sheet.save("runs/exp49-preview/sheet.png")
    json.dump(notes, open("runs/exp49-preview/notes.json", "w"), indent=1)
    print(len(pick))


if __name__ == "__main__":
    main()
