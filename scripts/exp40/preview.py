"""exp40 pair preview (critic EXP40-PREP-01): for a fixed choice of families -- the first two temperature and the
first two force families of the selection, by id, among those exp37 drew and among those re-drawn here -- the
original picture and the L picture of each view as the model gets them (256 letterbox), side by side, with the
reading golds under each row. runs/exp40-preview/sheet.png and notes.json.

    uv run python scripts/exp40/preview.py --root data
"""
import argparse
import json
import os
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from devision.image import letterbox  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="data")
    p.add_argument("--out", default="runs/exp40-preview")
    a = p.parse_args(argv)
    rows = [json.loads(x) for x in open(os.path.join(a.root, "exp40", "train_M.jsonl"))]
    rl = [r for r in rows if r.get("source", "").startswith("real-layout") and r["exposure"] == 1]
    by_fam: dict = {}
    for r in rl:
        by_fam.setdefault(r["family"], []).append(r)
    pick = []
    for form in ("temp", "force"):
        for origin in ("exp37", "exp40"):
            fams = sorted(f for f, v in by_fam.items() if v[0]["kind"] == form and v[0]["image"].startswith(origin))
            pick += fams[:2]
    os.makedirs(a.out, exist_ok=True)
    tile, pad, text_h = 256, 8, 46
    sheet = Image.new("RGB", (4 * (tile + pad), len(pick) * (tile + text_h + pad)), "white")
    draw = ImageDraw.Draw(sheet)
    notes = []
    for i, fam in enumerate(pick):
        y = i * (tile + text_h + pad)
        golds = []
        for v, view in enumerate(("v0", "v1")):
            rs = sorted((r for r in by_fam[fam] if r["view"] == view), key=lambda r: r["orig_qid"])
            orig = rs[0]["orig_id"]
            src = next(json.loads(x) for x in open(os.path.join(a.root, "exp40", "train_O.jsonl")) if f'"id": "exp40:{fam}:{view}:q0:e1"' in x)
            for k, img in enumerate((src["image"], rs[0]["image"])):
                sheet.paste(letterbox(Image.open(os.path.join(a.root, img)).convert("RGB"), tile), ((2 * v + k) * (tile + pad), y))
            g = {"%s %s" % (r["orig_qid"], r["questions"][next(iter(r["questions"]))]["instructions"][-30:]):
                 next(o for o, q in r["gold"][next(iter(r["gold"]))]["probabilities"].items() if q == 1.0) for r in rs}
            golds.append(g)
            notes.append({"family": fam, "view": view, "original": src["image"], "L": rs[0]["image"], "row": orig, "golds": g})
            line = "; ".join(val[:24] for val in g.values())
            draw.text((2 * v * (tile + pad), y + tile + 4), "%s %s: %s" % (fam, view, line[:80]), fill="black")
    sheet.save(os.path.join(a.out, "sheet.png"))
    json.dump(notes, open(os.path.join(a.out, "notes.json"), "w"), indent=1)
    print("%d families: %s" % (len(pick), pick))


if __name__ == "__main__":
    main()
