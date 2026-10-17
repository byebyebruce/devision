"""exp41 target-region ORACLE diagnosis, data only (critic EXP41-PREP-01; docs/experiments/2026-10-18-exp41-region-plan.md;
NOT run, no model). Separates "cannot find the asked object" from "cannot read the asked number" on the real-layout
tune pictures, for one fixed endpoint (exp40-M). Not a product preprocessing, not a candidate, not a transfer result.

  - Families: 30 temperature and 30 force families of data/real_layout/questions_tune.jsonl, the lowest
    sha256(family) of each form (never by a model output); both views, every reading question (mass / speed /
    distance; the ScienceQA-style question is left out). Training families, confirm and the new frozen confirmation
    sets are not touched.
  - Regions come from the generator's geometry only (scripts/data/real_layout.py; each scene is re-drawn from
    real_layout.families() and checked pixel-identical to the original PNG before its box is used):
      temperature -- the object's text block: "Sample X", both field rows with their numbers and units (union of
        the three text boxes, padded 6 px; the jar and its particles are outside);
      force -- the object's pair box: "Pair N", the magnets, the distance bar and the distance with its unit.
  - Three pictures per (view, object), all cut from the ORIGINAL PNG (no number re-drawn or replaced; the
    original file is not overwritten):
      keep   -- the original canvas size, everything outside the asked object's region white;
      zoom   -- the asked object's region cropped (the model's own letterbox then enlarges it);
      other  -- the OTHER object's region cropped the same way (control: the asked number is not in the picture).
    Each row asks that object's reading questions with the original wording, options, order and gold. No value,
    answer or box is given to the model as text (state_text as in the original row).
  - Per picture: the box, what it covers, and the digit ink height at 256 estimated from the font's digit glyph box
    x letterbox scale (original / keep, and zoom), against exp37's validated 16 px.

Writes data/exp41/{images/,questions.jsonl,MANIFEST.json}.

    uv run python scripts/exp41/build.py --root data
"""
import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "data"))
import real_layout as rl  # type: ignore  # noqa: E402
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("exp40_build", os.path.join(HERE, "..", "exp40", "build.py"))
assert _spec is not None and _spec.loader is not None
_b40 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_b40)
field_of, object_of = _b40.field_of, _b40.object_of

PAD = 6
CONDITIONS = ("keep", "zoom", "other")


def sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def digit_ink(size: int) -> int:
    b = ImageFont.truetype(rl.FONT, size).getbbox("0123456789")
    return int(b[3] - b[1])


def temp_regions(s: Dict) -> Tuple[Dict[str, List[int]], int]:
    """Per object name: the text block box (label + both rows), as temp_render places them; and the text size."""
    w = s["width"]
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    size = s["font"]
    longest = rl.LINES[s["wording"]][1] % "1,999"
    while size > 11 and probe.textlength(longest, font=ImageFont.truetype(rl.FONT, size)) > w / 2 - 12:
        size -= 1
    line = int(size * s["spacing"])
    font, bold = ImageFont.truetype(rl.FONT, size), ImageFont.truetype(rl.BOLD, size)
    out = {}
    for k, name in enumerate(("A", "B") if s.get("a_left", True) else ("B", "A")):
        cx = w // 4 + k * (w // 2)
        top = 315 + line // 2
        mass_line, speed_line = rl.LINES[s.get("wording", 0)]
        rows = [mass_line % s[name]["mass"], speed_line % rl.fmt(s[name]["speed"])]
        boxes = [probe.textbbox((cx, top), "Sample " + name, font=bold, anchor="mm")]
        boxes += [probe.textbbox((cx, top + (j + 1) * line), t, font=font, anchor="mm") for j, t in enumerate(rows)]
        out["sample " + name] = [int(min(b[0] for b in boxes)) - PAD, int(min(b[1] for b in boxes)) - PAD,
                                 int(max(b[2] for b in boxes)) + PAD, int(max(b[3] for b in boxes)) + PAD]
    return out, size


def force_regions(s: Dict) -> Tuple[Dict[str, List[int]], int]:
    g = rl.force_geometry(s)
    return {"Pair %d" % (p + 1): [q["box"] - 4, 0, q["box"] + g["pw"] + 4, g["h"]] for p, q in enumerate(g["pairs"])}, s["font"]


def scenes(root: str, fams: set) -> Dict[str, Tuple[str, Dict, Dict]]:
    out = {}
    last = max(int(f[2:]) for f in fams)
    for i, split, form, s, twin in rl.families(6000, 300, 300):
        fam = "rl%05d" % i
        if fam in fams:
            assert split == "tune", fam
            out[fam] = (form, s, twin)
        if i >= last:
            break
    return out


def letterbox_scale(size: Tuple[int, int]) -> float:
    return 256 / max(size)


def build(root: str, per_form: int = 30) -> Dict:
    tune = [json.loads(x) for x in open(os.path.join(root, "real_layout", "questions_tune.jsonl"))]
    by_form: Dict[str, List[str]] = {}
    for r in tune:
        by_form.setdefault(r["kind"], [])
        if r["family"] not in by_form[r["kind"]]:
            by_form[r["kind"]].append(r["family"])
    chosen = {f for form in ("temp", "force") for f in sorted(by_form[form], key=lambda f: hashlib.sha256(f.encode()).hexdigest())[:per_form]}
    sc = scenes(root, chosen)
    out_dir = os.path.join(root, "exp41")
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    rows, pictures, problems = [], {}, []
    for r in sorted((r for r in tune if r["family"] in chosen), key=lambda r: r["id"]):
        form, s, twin = sc[r["family"]]
        view = int(r["id"].split(":")[2][1])
        scene = (s, twin)[view]
        orig = Image.open(os.path.join(root, r["image"])).convert("RGB")
        redrawn = (rl.temp_render if form == "temp" else rl.force_render)(scene)
        if redrawn.size != orig.size or not (np.asarray(redrawn) == np.asarray(orig)).all():
            problems.append("%s: re-drawn scene differs from the original PNG" % r["id"])
            continue
        regions, size = (temp_regions if form == "temp" else force_regions)(scene)
        ink = digit_ink(size)
        reading = {qid: q for qid, q in r["questions"].items() if field_of(q["instructions"]) != "cmp"}
        clip = lambda b: [max(0, b[0]), max(0, b[1]), min(orig.width, b[2]), min(orig.height, b[3])]  # noqa: E731
        regions = {o: clip(b) for o, b in regions.items()}       # inside the picture (a padded box can reach -1)
        for obj, box in regions.items():
            other = next(b for o, b in regions.items() if o != obj)
            qs = {qid: q for qid, q in reading.items() if object_of(q["instructions"]).lower() == obj.lower()}
            assert qs, (r["id"], obj)
            for cond in CONDITIONS:
                if cond == "keep":
                    img = Image.new("RGB", orig.size, "white")
                    img.paste(orig.crop((box[0], box[1], box[2], box[3])), (box[0], box[1]))
                else:
                    b = box if cond == "zoom" else other
                    img = orig.crop((b[0], b[1], b[2], b[3]))
                tag = "%s_%s_%s_%s" % (r["family"], "v%d" % view, obj.replace(" ", ""), cond)
                rel = "exp41/images/%s.png" % tag
                img.save(os.path.join(root, rel))
                pictures[tag] = {"box": box if cond != "other" else other, "covers": (
                    "label 'Sample X', both field rows, numbers, units (jar outside)" if form == "temp" else
                    "label 'Pair N', magnets, distance bar, distance number and unit"),
                    "size": list(img.size), "digit_ink_256": round(ink * letterbox_scale(img.size), 1),
                    "digit_ink_256_original": round(ink * letterbox_scale(orig.size), 1)}
                rows.append({"id": "exp41:%s:%s:%s:%s" % (r["family"], "v%d" % view, obj.replace(" ", ""), cond),
                             "source": r["source"], "kind": r["kind"], "axis": r.get("axis", ""), "family": r["family"],
                             "case": r["case"], "condition": cond, "object": obj, "orig_id": r["id"],
                             "image": rel, "image_id": "exp41:" + tag, "state_text": r["state_text"],
                             "questions": qs, "gold": {qid: r["gold"][qid] for qid in qs}})
    path = os.path.join(out_dir, "questions.jsonl")
    with open(path, "w") as fh:
        for x in rows:
            fh.write(json.dumps(x) + "\n")
    ink = {(c, f): [p["digit_ink_256"] for t, p in pictures.items() if t.endswith(c) and sc[t[:7]][0] == f] for c in CONDITIONS for f in ("temp", "force")}
    m = {"families": sorted(chosen), "per_form": dict(Counter(sc[f][0] for f in chosen)), "problems": problems,
         "rows": len(rows), "questions": sum(len(x["questions"]) for x in rows),
         "questions_by_condition": {c: sum(len(x["questions"]) for x in rows if x["condition"] == c) for c in CONDITIONS},
         "digit_ink_256": {"%s/%s" % k: [min(v), float(np.median(v)), max(v)] for k, v in ink.items() if v},
         "digit_ink_256_original": {f: [min(v), float(np.median(v)), max(v)] for f in ("temp", "force")
                                    for v in [[p["digit_ink_256_original"] for t, p in pictures.items() if sc[t[:7]][0] == f]]},
         "source_sha256": sha(os.path.join(root, "real_layout", "questions_tune.jsonl")), "sha256": sha(path)}
    json.dump(pictures, open(os.path.join(out_dir, "pictures.json"), "w"))
    json.dump(m, open(os.path.join(out_dir, "MANIFEST.json"), "w"), indent=1)
    return m


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    m = build(p.parse_args().root)
    print(json.dumps({k: v for k, v in m.items() if k != "families"}, indent=1))
