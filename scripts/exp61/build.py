"""exp61 layout variants for the selector (critic EXP61-PREP-01; docs/experiments/2026-10-18-exp61-selector-plan.md). CPU
only, no model. From the real-layout generator (scripts/data/real_layout.py, replayed read-only through exp43's scenes),
never from a ScienceQA / dev picture:

  - train variants: 100 temperature + 100 magnet families of the TRAIN split, by stable hash ("exp61-v1:" + family),
    excluding every family of data/exp49/questions.jsonl (fit_O / fit_L / rl_tune / rl_tune_L); both views;
  - new-layout dev slice: all 300 families of the TUNE split (the rl_tune families), both views -- frozen now, never
    trained; every view of a family in the same split.

A variant is the generator's own picture with only blank rows removed: top t px (0 .. the first ink row, at most 30) and
the bottom down to m px below the last ink row (m in 0 .. 6), t / m from random.Random(sha256(seed + picture id)). No
ink row is ever removed (checked). The field boxes are exp45's original_box (the EXP45 / 51 semantics: a temperature
row band of its sample's half; a magnet pair's wide label band), moved by -t and clipped to the new canvas; the drawn
value's text box must stay inside its box and the other object's text box must not touch it (checked, as exp45).
Targets are the drawn values (no thousands commas, as data/exp49). Writes data/exp61/{images/, questions.jsonl,
MANIFEST.json, audit.json} and, with --preview, contact sheets.

    uv run python scripts/exp61/build.py --root data [--preview runs/exp61_preview]
"""
import argparse
import hashlib
import importlib.util
import json
import os
import random
import statistics
import sys
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(__file__)
SEED = "exp61-v1:"
N_TEMP, N_FORCE = 100, 100
TOP_MAX, BOTTOM_MAX = 30, 6
TEMPLATE = {("mass", "A"): "What is the mass of each particle in sample A?", ("mass", "B"): "What is the mass of each particle in sample B?",
            ("speed", "A"): "What is the average particle speed in sample A?", ("speed", "B"): "What is the average particle speed in sample B?",
            ("distance", "1"): "What is the distance between the magnets in Pair 1?", ("distance", "2"): "What is the distance between the magnets in Pair 2?"}


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, rel))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


b45 = _load("exp45_build", "../exp45/build.py")
b43, rl = b45.b43, b45.rl


def scenes(fams: set):
    """exp43's scene replay; real_layout's exclusion of the 24 transcribed cases imports scripts/exp28/build.py by the
    bare name "build" at call time, so another module cached under that name is set aside for the call."""
    saved = sys.modules.pop("build", None)
    try:
        sys.path.insert(0, os.path.join(HERE, "..", "exp28"))
        return b43.scenes(fams)
    finally:
        sys.path.remove(os.path.join(HERE, "..", "exp28"))
        sys.modules.pop("build", None)
        if saved is not None:
            sys.modules["build"] = saved


def families_by_split():
    """rl.families with the same "build" module guard (it calls transcribed_keys)."""
    saved = sys.modules.pop("build", None)
    try:
        return list(rl.families(6000, 300, 300))
    finally:
        sys.modules.pop("build", None)
        if saved is not None:
            sys.modules["build"] = saved


def h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def ink_rows(img: Image.Image) -> Tuple[int, int]:
    a = np.asarray(img.convert("RGB"))
    rows = np.flatnonzero((a < 250).any(-1).any(-1))
    return int(rows[0]), int(rows[-1])


def crop_params(pid: str, img: Image.Image) -> Tuple[int, int]:
    """(top rows removed, the kept rows' end in the original picture) -- only blank rows; the new height is end - top."""
    first, last = ink_rows(img)
    r = random.Random(h(SEED + pid))
    t = r.randint(0, min(TOP_MAX, first))
    m = r.randint(0, BOTTOM_MAX)
    return t, min(img.size[1], last + 1 + m)


def variant(form: str, scene: Dict, pid: str):
    """(cropped picture, {(field, object): (box xywh, text xyxy)}, (t, new height, original height))."""
    img = rl.temp_render(scene) if form == "temp" else rl.force_render(scene)
    t, nh = crop_params(pid, img)
    first, last = ink_rows(img)
    if t > first or nh <= last:
        raise SystemExit("%s: the crop would remove ink" % pid)
    out = img.crop((0, t, img.size[0], nh))
    w, nh = img.size[0], nh - t                    # the new canvas height
    boxes = {}
    objs = ("A", "B") if form == "temp" else ("1", "2")
    fields = ("mass", "speed") if form == "temp" else ("distance",)
    for f in fields:
        for o in objs:
            box, text, other = b45.original_box(form, scene, o, f)
            mv = lambda b: [b[0], b[1] - t, b[2], b[3] - t]  # noqa: E731
            box, text, other = mv(box), mv(text), mv(other)
            box = [max(0.0, box[0]), max(0.0, box[1]), min(float(w), box[2]), min(float(nh), box[3])]
            if not b45.inside(text, box) or b45.overlaps(other, box) or text[3] > nh + 1 or text[1] < -1:
                raise SystemExit("%s %s %s: the value %s is not inside %s alone (other %s)" % (pid, f, o, text, box, other))
            boxes[(f, o)] = ([box[0], box[1], box[2] - box[0], box[3] - box[1]], text)
    if out.size != (w, nh):
        raise SystemExit("%s: crop size" % pid)
    return out, boxes, (t, nh, img.size[1])


def target_of(form: str, scene: Dict, f: str, o: str) -> str:
    if form == "temp":
        return str(scene[o]["mass"]) if f == "mass" else str(int(scene[o]["speed"]))
    return rl.fmt(scene["d"][int(o) - 1]).replace(",", "")


def pick(fams: Dict[str, str], n: int, form: str) -> List[str]:
    return sorted((f for f, k in fams.items() if k == form), key=lambda f: h(SEED + f))[:n]


def build(root: str = "data", preview: str = "") -> Dict:
    used = {json.loads(x)["family"] for x in open(os.path.join(root, "exp49", "questions.jsonl"))}
    train, tune = {}, {}
    for i, split, form, s, twin in families_by_split():
        fam = "rl%05d" % i
        if split == "train" and fam not in used:
            train[fam] = form
        elif split == "tune":
            tune[fam] = form
    chosen = {"train_V": pick(train, N_TEMP, "temp") + pick(train, N_FORCE, "force"), "tune_V": sorted(tune)}
    if set(chosen["train_V"]) & set(chosen["tune_V"]) or set(chosen["train_V"]) & used:
        raise SystemExit("family overlap")
    sc = scenes(set(chosen["train_V"]) | set(chosen["tune_V"]))
    dest = os.path.join(root, "exp61", "images")
    os.makedirs(dest, exist_ok=True)
    qs, crops = [], []
    for tag, fams in chosen.items():
        for fam in fams:
            for view in ("v0", "v1"):
                form, scene = sc[(fam, view)]
                pid = "%s_%s_%s" % (tag, fam, view)
                img, boxes, (t, nh, oh) = variant(form, scene, pid)
                rel = "exp61/images/%s.png" % pid
                img.save(os.path.join(root, rel))
                crops.append({"picture": rel, "top": t, "bottom_removed": oh - nh - t, "height": nh, "form": form})
                other_view = "v1" if view == "v0" else "v0"
                for (f, o), (box, text) in boxes.items():
                    key = "%s|%s:%s|%s:%s" % (tag, fam, view, f, o)
                    twin_scene = sc[(fam, other_view)][1]
                    qs.append({"key": key, "set": tag, "split": "train" if tag == "train_V" else "dev", "layout": "variant",
                               "id": "%s:%s" % (fam, view), "qid": "%s:%s" % (f, o), "text": TEMPLATE[(f, o)], "picture": rel,
                               "box": box, "width": img.size[0], "height": nh, "field": f, "object": o, "family": fam,
                               "case": str(rl.case_key(form, scene)), "view": view, "target": target_of(form, scene, f, o),
                               "twin": "%s|%s:%s|%s:%s" % (tag, fam, other_view, f, o),
                               "twin_changed": target_of(form, scene, f, o) != target_of(form, twin_scene, f, o),
                               "text_box": text})
    path = os.path.join(root, "exp61", "questions.jsonl")
    with open(path, "w") as fh:
        fh.write("".join(json.dumps(q) + "\n" for q in qs))
    aud = audit(root, qs)
    json.dump(aud, open(os.path.join(root, "exp61", "audit.json"), "w"), indent=1)
    pics = sorted({q["picture"] for q in qs})
    m = {"seed": SEED, "families": {k: len(v) for k, v in chosen.items()},
         "families_by_form": {k: dict(Counter(sc[(f, "v0")][0] for f in v)) for k, v in chosen.items()},
         "pictures": dict(Counter(q["set"] for q in {q["picture"]: q for q in qs}.values())),
         "questions": dict(Counter(q["set"] for q in qs)), "questions_by_field": {k: dict(Counter(q["field"] for q in qs if q["set"] == k)) for k in chosen},
         "crop": {"top_max": TOP_MAX, "bottom_margin_max": BOTTOM_MAX,
                  "top_removed": {k: dist([c["top"] for c in crops if c["picture"].split("/")[-1].startswith(k)]) for k in chosen},
                  "bottom_removed": {k: dist([c["bottom_removed"] for c in crops if c["picture"].split("/")[-1].startswith(k)]) for k in chosen}},
         "excluded_families": "every family of data/exp49/questions.jsonl (%d)" % len(used),
         "split_rule": "train_V from the generator's train split, tune_V = the tune split (rl_tune families), every view of a family in one split",
         "questions_sha256": h(open(path).read()),
         "pictures_sha256": h(json.dumps({p: hashlib.sha256(open(os.path.join(root, p), "rb").read()).hexdigest() for p in pics}, sort_keys=True)),
         "audit_sha256": h(open(os.path.join(root, "exp61", "audit.json")).read())}
    json.dump(m, open(os.path.join(root, "exp61", "MANIFEST.json"), "w"), indent=1)
    if preview:
        sheets(root, qs, preview)
    return m


def dist(xs: List[float]) -> Dict:
    xs = sorted(xs)
    return {"n": len(xs), "min": round(xs[0], 3), "p10": round(xs[len(xs) // 10], 3), "median": round(statistics.median(xs), 3),
            "p90": round(xs[9 * len(xs) // 10], 3), "max": round(xs[-1], 3)}


def audit(root: str, qs: List[Dict]) -> Dict:
    """Box geometry per set and field: centre-y fraction, width / height fractions, bottom margin (px below the box) --
    for the original fit_O / fit_L training boxes, the variants, and, as a reference only, EXP60's oracle boxes."""
    groups: Dict[str, List[Tuple[float, float, float, float]]] = {}

    def add(name, box, w, hh):
        x, y, bw, bh = box
        groups.setdefault(name, []).append(((y + bh / 2) / hh, bw / w, bh / hh, hh - (y + bh)))
    for q in map(json.loads, open(os.path.join(root, "exp49", "questions.jsonl"))):
        if q["set"] in ("fit_O", "fit_L"):
            add("%s/%s" % (q["set"], q["field"]), q["box"], q["width"], q["height"])
    for q in qs:
        add("%s/%s" % (q["set"], q["field"]), q["box"], q["width"], q["height"])
    ob = os.path.join(HERE, "..", "exp60", "oracle_boxes.json")
    if os.path.exists(ob):
        for k, v in json.load(open(ob)).items():
            if k.startswith("_"):
                continue
            for t, box in v["boxes"].items():
                f = "mass" if "mass" in t else "speed" if "speed" in t else "distance"
                add("reference_exp60_oracle/%s" % f, box, v["size"][0], v["size"][1])
    out = {}
    for k, xs in sorted(groups.items()):
        out[k] = {"centre_y": dist([x[0] for x in xs]), "width": dist([x[1] for x in xs]), "height": dist([x[2] for x in xs]),
                  "bottom_margin_px": dist([x[3] for x in xs])}
    return out


def sheets(root: str, qs: List[Dict], dest: str) -> None:
    os.makedirs(dest, exist_ok=True)
    by: Dict[str, List[Dict]] = {}
    for q in qs:
        by.setdefault(q["picture"], []).append(q)
    for tag in ("train_V", "tune_V"):
        pics = [p for p in sorted(by) if p.split("/")[-1].startswith(tag)]
        sample = sorted(pics, key=lambda p: h(SEED + "preview:" + p))[:12]
        sheet = Image.new("RGB", (3 * 620, 4 * 230), "white")
        for k, p in enumerate(sample):
            im = Image.open(os.path.join(root, p)).convert("RGB")
            d = ImageDraw.Draw(im)
            for q in by[p]:
                x, y, bw, bh = q["box"]
                d.rectangle([x, y, x + bw, y + bh], outline=(255, 0, 0), width=2)
            im.thumbnail((610, 225))
            sheet.paste(im, ((k % 3) * 620, (k // 3) * 230))
        sheet.save(os.path.join(dest, "%s.png" % tag))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data")
    ap.add_argument("--preview", default="")
    a = ap.parse_args(argv)
    print(json.dumps(build(a.root, a.preview), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
