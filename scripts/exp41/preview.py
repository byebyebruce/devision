"""exp41 preview (critic EXP41-PREP-01): 12 groups, each one (view, object) as the model gets it (256 letterbox):
original | keep_text | keep_object | zoom | other (force: keep_object is the keep picture itself), with the asked questions' golds. Fixed rule, per form, in the build's hash order
of families: F1 = the first family whose two objects differ in some reading value (in either view), F2 = the first such family with
the non-default layout (temperature: Sample A on the right; force: Pair 1 on the right); groups F1 v0 both objects,
F1 v1 first object (its value twin), F2 v0 both objects, F2 v1 first object. runs/exp41-preview/sheet.png, notes.json.

    uv run python scripts/exp41/preview.py --root data
"""
import argparse
import hashlib
import json
import os
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from devision.image import letterbox  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="data")
    p.add_argument("--out", default="runs/exp41-preview")
    a = p.parse_args(argv)
    rows = [json.loads(x) for x in open(os.path.join(a.root, "exp41", "questions.jsonl"))]
    pics = json.load(open(os.path.join(a.root, "exp41", "pictures.json")))
    tune = {json.loads(x)["id"]: json.loads(x) for x in open(os.path.join(a.root, "real_layout", "questions_tune.jsonl"))}
    gold = lambda r, q: next(o for o, v in r["gold"][q]["probabilities"].items() if v == 1.0)  # noqa: E731
    by = {}
    for r in rows:
        by.setdefault((r["family"], r["orig_id"].rsplit(":", 1)[1], r["object"]), {})[r["condition"]] = r

    def differs(fam):
        golds = {}
        for (f, v, o), c in by.items():
            if f == fam:
                for q in c["keep"]["questions"]:
                    golds.setdefault((v, c["keep"]["questions"][q]["instructions"].split(" in ")[0]), set()).add(gold(c["keep"], q))
        return any(len(g) > 1 for g in golds.values())

    def right_layout(fam):    # the first object (A / Pair 1) drawn right: its region starts right of the other's
        objs = sorted(o for (f, v, o) in by if f == fam and v == "v0")
        boxes = [pics["%s_v0_%s_keep" % (fam, o.replace(" ", ""))]["box"][0] for o in objs]
        return boxes[0] > boxes[1]
    groups = []
    for kind in ("temp", "force"):
        fams = sorted({r["family"] for r in rows if r["kind"] == kind}, key=lambda f: hashlib.sha256(f.encode()).hexdigest())
        f1 = next(f for f in fams if differs(f))
        f2 = next(f for f in fams if differs(f) and right_layout(f) and f != f1)
        for f in (f1, f2):
            objs = sorted(o for (g, v, o) in by if g == f and v == "v0")
            groups += [(f, "v0", objs[0]), (f, "v0", objs[1]), (f, "v1", objs[0])]
    os.makedirs(a.out, exist_ok=True)
    tile, pad, text_h = 256, 8, 40
    sheet = Image.new("RGB", (5 * (tile + pad), len(groups) * (tile + text_h + pad)), "white")
    d = ImageDraw.Draw(sheet)
    notes = []
    for i, (f, v, o) in enumerate(groups):
        y = i * (tile + text_h + pad)
        c = by[(f, v, o)]
        orig = tune[c["keep"]["orig_id"]]["image"]
        for k, img in enumerate([orig] + [c.get(x, c["keep"])["image"] for x in ("keep", "keep_object", "zoom", "other")]):
            sheet.paste(letterbox(Image.open(os.path.join(a.root, img)).convert("RGB"), tile), (k * (tile + pad), y))
        g = {c["keep"]["questions"][q]["instructions"]: gold(c["keep"], q) for q in c["keep"]["questions"]}
        notes.append({"family": f, "view": v, "object": o, "original": orig, **{x: c.get(x, c["keep"])["image"] for x in ("keep", "keep_object", "zoom", "other")},
                      "golds": g, "ink_256": {x: pics["%s_%s_%s_%s" % (f, v, o.replace(" ", ""), x)]["digit_ink_256"] for x in c}})
        d.text((0, y + tile + 4), "%s %s asked: %s | %s" % (f, v, o, "; ".join("%s -> %s" % (q[9:45], ans) for q, ans in g.items()))[:150], fill="black")
        d.text((0, y + tile + 18), "original | keep_text | keep_object%s | zoom | other (control)" % ("" if "keep_object" in c else " (= keep)"), fill="grey")
    sheet.save(os.path.join(a.out, "sheet.png"))
    json.dump(notes, open(os.path.join(a.out, "notes.json"), "w"), indent=1)
    print("%d groups: %s" % (len(groups), groups))


if __name__ == "__main__":
    main()
