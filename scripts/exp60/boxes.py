"""exp60 oracle field boxes (critic EXP60-PREP-01): one box per needed field of the 24 transcribed pictures, by a fixed
CPU rule on the original pixels -- never from reading results. Text = dark, unsaturated pixels (max channel < 110,
channel spread < 40); the picture is split at its middle column (sample A / Pair 1 left, sample B / Pair 2 right); in
each half the rows holding text form bands (gaps of >= 3 text-free rows separate them).
  temperature: the half's last two bands below its middle are the mass line and the speed line (EXP45 / 51 semantics:
               the whole field text line of that sample);
  magnets:     the half's last band is the distance label band (the "|- 42.8 mm -|" label of that pair).
Box = the band's text extent in that half, padded by 2 px and clipped: [x0, y0, w, h] in original pixels (reading's
pixel_box / region convention). Source: computed by this rule, previews looked at by Claude Code; not checked by a person.

    uv run python scripts/exp60/boxes.py [--preview DIR]     # writes scripts/exp60/oracle_boxes.json
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(__file__)
PAD = 2


def bands(mask_rows: np.ndarray, gap: int = 3):
    rows = np.flatnonzero(mask_rows)
    out = []
    for r in rows:
        if out and r - out[-1][1] <= gap:
            out[-1][1] = r
        else:
            out.append([r, r])
    return out


def text_mask(img: Image.Image) -> np.ndarray:
    a = np.asarray(img.convert("RGB")).astype(int)
    return (a.max(-1) < 110) & (a.max(-1) - a.min(-1) < 40)


def boxes_of(path: str, kind: str):
    img = Image.open(path)
    w, h = img.size
    m = text_mask(img)
    out = {}
    for side, (x0, x1) in (("left", (0, w // 2)), ("right", (w // 2, w))):
        half = m[:, x0:x1]
        bs = [b for b in bands(half.any(1)) if b[0] >= h // 2]
        need = 2 if kind == "temp" else 1
        if len(bs) < need:
            raise SystemExit("%s %s: %d text bands below the middle" % (path, side, len(bs)))
        for name, (r0, r1) in zip(("mass", "speed") if kind == "temp" else ("distance",), bs[-need:]):
            cols = np.flatnonzero(half[r0:r1 + 1].any(0)) + x0
            bx0, by0 = max(0, cols.min() - PAD), max(0, r0 - PAD)
            bx1, by1 = min(w, cols.max() + 1 + PAD), min(h, r1 + 1 + PAD)
            out[(side, name)] = [float(bx0), float(by0), float(bx1 - bx0), float(by1 - by0)]
    return out, (w, h)


TEMPLATE = {("left", "mass"): "What is the mass of each particle in sample A?", ("right", "mass"): "What is the mass of each particle in sample B?",
            ("left", "speed"): "What is the average particle speed in sample A?", ("right", "speed"): "What is the average particle speed in sample B?",
            ("left", "distance"): "What is the distance between the magnets in Pair 1?", ("right", "distance"): "What is the distance between the magnets in Pair 2?"}


def build(items, data_root="data", preview=""):
    out = {}
    for it in items:
        r = it["row"]
        p = os.path.join(data_root, r["image"])
        bx, size = boxes_of(p, it["kind"])
        out[r["id"]] = {"image": r["image"], "size": list(size), "boxes": {TEMPLATE[k]: v for k, v in bx.items()}}
        if preview:
            os.makedirs(preview, exist_ok=True)
            im = Image.open(p).convert("RGB")
            d = ImageDraw.Draw(im)
            for (side, name), (x, y, bw, bh) in bx.items():
                d.rectangle([x, y, x + bw, y + bh], outline=(255, 0, 0), width=2)
            im.save(os.path.join(preview, r["id"].split(":")[1] + ".png"))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preview", default="")
    a = ap.parse_args(argv)
    sys.path.insert(0, os.path.join(HERE, "..", "exp59"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("exp59", os.path.join(HERE, "..", "exp59", "exp59.py"))
    assert spec is not None and spec.loader is not None
    e59 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e59)
    res = build(e59.items(), preview=a.preview)
    path = os.path.join(HERE, "oracle_boxes.json")
    with open(path, "w") as fh:
        json.dump({"_note": (__doc__ or "").split("\n\n")[0].replace("\n", " "), **res}, fh, indent=1)
    print(json.dumps({"pictures": len(res), "boxes": sum(len(v["boxes"]) for v in res.values()),
                      "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
