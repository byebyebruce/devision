"""exp45 region preview (critic EXP45-PREP-01): 12 training questions by a fixed rule -- in id order, the first of each
field x asked object (A / B, Pair 1 / 2) x side (left / right), per field O for A-left / B-right and L for A-right / B-left, a temperature
object on the right taken with the speed row on top -- so A / B, Pair 1 / 2, swapped sides, both row orders, O / L and
both aspect ratios appear -- each as the model's 256 letterbox with the box (red)
and the 16 x 16 target distribution (cell shading). runs/exp45-preview/sheet.png and notes.json.

    uv run python scripts/exp45/preview.py --root data
"""
import argparse
import json
import os
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from devision.image import letterbox  # noqa: E402
from devision.train.rlcd import REGION_SMOOTH, box_grid  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="data")
    a = p.parse_args(argv)
    regs = sorted((json.loads(x) for x in open(os.path.join(a.root, "exp45", "regions.jsonl"))), key=lambda r: r["id"])
    pick = []
    keys = [(f, o, sd) for f, objs in (("mass", "AB"), ("speed", "AB"), ("distance", "12")) for o in objs for sd in ("left", "right")]
    for i, (f, o, sd) in enumerate(keys):     # original / L alternating; temperature on the right: speed row on top
        lay = "original" if i % 4 in (0, 3) else "L"     # per field: O has A-left / B-right, L has A-right / B-left
        want_sf = True if (f != "distance" and sd == "right") else None
        cand = [r for r in regs if (r["field"], r["object"], r["side"], r["layout"]) == (f, o, sd, lay)
                and (want_sf is None or r["speed_first"] == want_sf)]
        pick.append(cand[0])
    tile, pad = 256, 8
    sheet = Image.new("RGB", (4 * (tile + pad), 3 * (tile + 30 + pad)), "white")
    d = ImageDraw.Draw(sheet)
    for i, r in enumerate(pick):
        img = letterbox(Image.open(os.path.join(a.root, r["picture"])).convert("RGB"), tile)
        ov = ImageDraw.Draw(img, "RGBA")
        grid = box_grid([r["box"]], r["width"], r["height"], tile, grid=16, smooth=REGION_SMOOTH)
        top = max(grid)
        for c, v in enumerate(grid):
            if v > REGION_SMOOTH / 256 + 1e-9:
                y, x = divmod(c, 16)
                ov.rectangle((x * 16, y * 16, x * 16 + 15, y * 16 + 15), fill=(0, 120, 255, int(120 * v / top)))
        sc = tile / max(r["width"], r["height"])
        ox, oy = (tile - round(r["width"] * sc)) // 2, (tile - round(r["height"] * sc)) // 2
        x, y, w, h = r["box"]
        ov.rectangle((ox + x * sc, oy + y * sc, ox + (x + w) * sc, oy + (y + h) * sc), outline=(255, 0, 0, 255), width=2)
        col, row = i % 4, i // 4
        sheet.paste(img, (col * (tile + pad), row * (tile + 30 + pad)))
        d.text((col * (tile + pad), row * (tile + 30 + pad) + tile + 2),
               "%s %s %s %s side=%s" % (r["layout"], r["field"], r["object"], r["id"].split(":")[1], r["side"]), fill="black")
    os.makedirs("runs/exp45-preview", exist_ok=True)
    sheet.save("runs/exp45-preview/sheet.png")
    json.dump(pick, open("runs/exp45-preview/notes.json", "w"), indent=1)
    print(len(pick), [(r["layout"], r["field"], r["object"], r["side"], r.get("speed_first")) for r in pick])


if __name__ == "__main__":
    main()
