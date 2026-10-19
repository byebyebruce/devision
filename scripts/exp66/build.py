"""exp66 layout variants for the selector (critic EXP66-PREP-01 items 2-3; docs/experiments/2026-10-19-exp66-layout-plan.md).
CPU only, no model. From the real-layout generator (scripts/data/real_layout.py, replayed read-only through exp43's scenes),
never from a ScienceQA / dev picture, its pixels, boxes or answers.

The new variable is where the field rows sit on the canvas (scripts/exp66/audit.py: the generator always centres a
temperature column at w / 4 -- 6,000 / 6,000 training scenes at 0.247-0.254 of the width -- and a magnet label at least
7-10 px above the bottom edge):

  temperature  the whole column of each sample (jar, particles, header, both field rows) moves outward by `shift` px, the
               same for both samples: shift uniform in [-s_in, s_max], s_max = the largest outward move that keeps every
               drawn text >= 1 px and the jar outline >= 2 px inside the canvas in both views, s_in = the largest inward
               move (<= 8) keeping every text >= 2 px from the middle (physical limits of the picture, not dev24
               coordinates). Then EXP61's blank-row crop (top t 0..min(30, first ink row), bottom
               m 0..6 px below the last ink row);
  magnets      EXP61's top crop and b 0..8 blank rows added below the canvas (the label's bottom margin).

The row pitch (spacing 1.5-2.1 x font) and the font sizes are the generator's own; nothing else changes. A family's two
views (the scene and its value twin) share every layout parameter (random.Random(sha256("exp66-v1:" + family))), so
within a family only the values change.

Labels keep the EXP45 / 51 semantics and move with what is drawn: a temperature field's box is its sample's half-width
row band (width w / 2 - 4, one pitch high) centred on the shifted column and cut at the picture's middle, a magnet pair's box its wide label band; then
moved by -t and clipped to the canvas. Checked on every question, from the drawing metadata and the pixels: the value's
text box inside its box and inside the canvas (no cut glyph), the other object's text box, the same sample's other row
and its header not touching the box, every text-coloured pixel in the box inside the target's text box (magnets: or its own dimension bar), the crop
removing blank rows only, the twins' canvases equal. Targets are the drawn values (no thousands commas, as data/exp49).

Sets: train_C = 100 temperature + 100 magnet families of the generator's TRAIN split (stable hash order), excluding every
family of data/exp49 and of EXP61 / EXP62 (train_V = EXP62's parents); each with its two original pictures as parents
(train_C_parent: the generator's own files, EXP45 boxes, paired by key). tune_C = all 300 TUNE-split families (the
rl_tune / tune_V families) with the new layouts, frozen now, never trained. Writes data/exp66/{images/, questions.jsonl,
coverage.json, MANIFEST.json}; --preview writes contact sheets.

    uv run python scripts/exp66/build.py --root data [--preview runs/exp66_preview]
"""
import argparse
import hashlib
import importlib.util
import json
import os
import random
import statistics
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(__file__)
SEED = "exp66-v1:"
N_TEMP, N_FORCE = 100, 100
SHIFT_IN, TOP_MAX, BOTTOM_MAX, PAD_MAX = 8, 30, 6, 8


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, rel))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


b61 = _load("exp61_build", "../exp61/build.py")
b45, rl = b61.b45, b61.rl
TEMPLATE = b61.TEMPLATE


def h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def jsonl(p: str) -> List[Dict]:
    return [json.loads(x) for x in open(p) if x.strip()]


# ------------------------------------------------------------------ the generator with a moved column

def centres(s: Dict, shift: int) -> List[int]:
    """The column centre x of the left / right drawn sample (generator: w // 4 + k * (w // 2)), moved outward by shift."""
    w = s["width"]
    return [w // 4 - shift, w // 4 + w // 2 + shift]


def temp_render_at(s: Dict, shift: int) -> Image.Image:
    """rl.temp_render with the column centres moved (shift 0: pixel-identical to rl.temp_render, checked)."""
    w = s["width"]
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    size = s["font"]
    longest = rl.LINES[s["wording"]][1] % "1,999"
    while size > 11 and probe.textlength(longest, font=ImageFont.truetype(rl.FONT, size)) > w / 2 - 12:
        size -= 1
    line = int(size * s["spacing"])
    hh = max(405, 315 + 3 * line + size)
    img = Image.new("RGB", (w, hh), "white")
    d = ImageDraw.Draw(img)
    rng = random.Random(s["seed"])
    font, bold = ImageFont.truetype(rl.FONT, size), ImageFont.truetype(rl.BOLD, size)
    cs = centres(s, shift)
    for k, name in enumerate(("A", "B") if s.get("a_left", True) else ("B", "A")):
        cx = cs[k]
        x0, x1, y0, y1 = cx - 95, cx + 95, 30, 300
        d.rounded_rectangle((x0, y0 + 25, x1, y1), radius=28, outline="#9a9a9a", width=2)
        d.rectangle((x0 + 10, y0, x1 - 10, y0 + 20), fill="#9a9a9a")
        mass = s[name]["mass"]
        r = 6 + mass // 5
        for _ in range(s["count"]):
            px, py = rng.randint(x0 + 20 + rl.R_MAX, x1 - 20 - rl.R_MAX), rng.randint(y0 + 50 + rl.R_MAX, y1 - 15 - rl.R_MAX)
            d.ellipse((px - r, py - r, px + r, py + r), fill=s["colours"][k])
        top = 315 + line // 2
        mass_line, speed_line = rl.LINES[s.get("wording", 0)]
        rows = [mass_line % mass, speed_line % rl.fmt(s[name]["speed"])]
        if s.get("speed_first"):
            rows = rows[::-1]
        d.text((cx, top), "Sample " + name, fill="black", font=bold, anchor="mm")
        d.text((cx, top + line), rows[0], fill="black", font=font, anchor="mm")
        d.text((cx, top + 2 * line), rows[1], fill="black", font=font, anchor="mm")
    return img


def temp_rows_at(s: Dict, shift: int) -> Dict[str, Dict]:
    """b45.temp_rows with the moved centres (shift 0: equal to b45.temp_rows, checked)."""
    out = b45.temp_rows(s)
    cs = centres(s, shift)
    for r in out.values():
        r["cx"] = cs[r["k"]]
    return out


def temp_texts(s: Dict, shift: int) -> Dict[str, Dict[str, List[float]]]:
    """Per object: the drawn text boxes of its header and both field rows (PIL text boxes, as exp45's original_box)."""
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    rows = temp_rows_at(s, shift)
    out = {}
    for o, r in rows.items():
        font, bold = ImageFont.truetype(rl.FONT, r["size"]), ImageFont.truetype(rl.BOLD, r["size"])
        mass_line, speed_line = rl.LINES[s.get("wording", 0)]
        top = min(r["y"].values()) - r["line"]
        out[o] = {"header": list(probe.textbbox((r["cx"], top), "Sample " + o, font=bold, anchor="mm")),
                  "mass": list(probe.textbbox((r["cx"], r["y"]["mass"]), mass_line % s[o]["mass"], font=font, anchor="mm")),
                  "speed": list(probe.textbbox((r["cx"], r["y"]["speed"]), speed_line % rl.fmt(s[o]["speed"]), font=font, anchor="mm"))}
    return out


def temp_band(s: Dict, shift: int, obj: str, field: str) -> List[float]:
    """EXP45's row band (half width less 4, one pitch high) centred on the moved column, never past the picture's middle
    (shift 0: exp45's band, which already ends 2 px before it)."""
    r = temp_rows_at(s, shift)[obj]
    w = s["width"]
    x0, x1 = r["cx"] - r["width"] / 2, r["cx"] + r["width"] / 2
    x0, x1 = (x0, min(x1, w / 2)) if r["k"] == 0 else (max(x0, w / 2), x1)
    return [x0, r["y"][field] - r["line"] / 2, x1, r["y"][field] + r["line"] / 2]


def max_shift(views: List[Dict]) -> int:
    """The largest outward move keeping every text >= 1 px and the jar outline >= 2 px inside the canvas in every view."""
    best = None
    for s in views:
        w = s["width"]
        lim = w // 4 - 95 - 2                                         # jar x0 = cx - 95 >= 2
        for o, t in temp_texts(s, 0).items():
            for b in t.values():
                left = (b[0] + b[2]) / 2 < w / 2
                lim = min(lim, int(b[0] - 1) if left else int(w - 1 - b[2]))
        best = lim if best is None else min(best, lim)
    return max(0, int(best or 0))


def min_shift(views: List[Dict]) -> int:
    """The largest inward move (<= SHIFT_IN) keeping every text >= 2 px from the picture's middle in every view."""
    best = SHIFT_IN
    for s in views:
        w = s["width"]
        for o, t in temp_texts(s, 0).items():
            for b in t.values():
                left = (b[0] + b[2]) / 2 < w / 2
                best = min(best, int(w / 2 - 2 - b[2]) if left else int(b[0] - w / 2 - 2))
    return max(0, best)


def ink_rows(img: Image.Image) -> Tuple[int, int]:
    return b61.ink_rows(img)


def text_mask(a: np.ndarray) -> np.ndarray:
    a = a.astype(int)
    return (a.max(-1) < 110) & (a.max(-1) - a.min(-1) < 40)


# ------------------------------------------------------------------ one family

def params(fam: str, form: str, views: List[Dict]) -> Dict:
    """The family's layout parameters, shared by its views."""
    r = random.Random(h(SEED + fam))
    imgs = [rl.temp_render(v) if form == "temp" else rl.force_render(v) for v in views]
    first = min(ink_rows(i)[0] for i in imgs)
    p = {"top": r.randint(0, min(TOP_MAX, first))}
    if form == "temp":
        sm = max_shift(views)
        si = min_shift(views)
        p["shift"] = r.randint(-si, sm)
        p["shift_max"], p["shift_in_max"] = sm, si
        p["bottom_margin"] = r.randint(0, BOTTOM_MAX)
    else:
        p["pad"] = r.randint(0, PAD_MAX)
    return p


def render(form: str, s: Dict, p: Dict) -> Tuple[Image.Image, Dict[Tuple[str, str], Tuple[List[float], List[float]]], Dict]:
    """(variant picture, {(field, object): (box xywh, text xyxy)}, crop facts) with every check of the module doc."""
    if form == "temp":
        full = temp_render_at(s, p["shift"])
        first, last = ink_rows(full)
        end = min(full.size[1], last + 1 + p["bottom_margin"])
    else:
        base = rl.force_render(s)
        full = Image.new("RGB", (base.size[0], base.size[1] + p["pad"]), "white")
        full.paste(base, (0, 0))
        first, last = ink_rows(full)
        end = full.size[1]
    t = p["top"]
    a = np.asarray(full)
    if t > first or end <= last or (a[:t] < 250).any() or (a[end:] < 250).any():
        raise SystemExit("the crop would remove ink")
    img = full.crop((0, t, full.size[0], end))
    w, nh = img.size
    pix = np.asarray(img)
    mask = text_mask(pix)
    boxes = {}
    mv = lambda b: [b[0], b[1] - t, b[2], b[3] - t]  # noqa: E731
    objs = ("A", "B") if form == "temp" else ("1", "2")
    fields = ("mass", "speed") if form == "temp" else ("distance",)
    texts = temp_texts(s, p["shift"]) if form == "temp" else None
    for f in fields:
        for o in objs:
            other_o = objs[1] if o == objs[0] else objs[0]
            if form == "temp":
                assert texts is not None
                band, text, other = temp_band(s, p["shift"], o, f), texts[o][f], texts[other_o][f]
                own_rest = [texts[o][k] for k in ("header", "mass", "speed") if k != f]
                other_all = list(texts[other_o].values())
            else:
                band, text, other = b45.original_box(form, s, o, f)
                own_rest, other_all = [], [other]
            band, text = mv(band), mv(text)
            box = [max(0.0, band[0]), max(0.0, band[1]), min(float(w), band[2]), min(float(nh), band[3])]
            if text[0] < 0 or text[1] < 0 or text[2] > w or text[3] > nh:
                raise SystemExit("%s %s: the value's text %s is cut by the canvas %s" % (f, o, text, (w, nh)))
            if not b45.inside(text, box):
                raise SystemExit("%s %s: the value %s is not inside %s" % (f, o, text, box))
            for b in own_rest + other_all:
                if b45.overlaps(mv(b), box):
                    raise SystemExit("%s %s: another text %s touches the box %s" % (f, o, mv(b), box))
            allowed = [text] if form == "temp" else [text, mv(bar(s, int(o) - 1))]
            x0, y0, x1, y1 = int(box[0]), int(box[1]), int(np.ceil(box[2])), int(np.ceil(box[3]))
            ys, xs = np.nonzero(mask[y0:y1, x0:x1])
            xs, ys = xs + x0, ys + y0
            ok = np.zeros(len(xs), bool)
            for ax0, ay0, ax1, ay1 in allowed:
                ok |= (xs >= ax0 - 1) & (xs < ax1 + 1) & (ys >= ay0 - 1) & (ys < ay1 + 1)
            if len(xs) == 0 or not ok.all():
                raise SystemExit("%s %s: text-coloured pixels in the box outside the value's text box" % (f, o))
            boxes[(f, o)] = ([box[0], box[1], box[2] - box[0], box[3] - box[1]], text)
    return img, boxes, {"top": t, "height": nh, "full_height": full.size[1]}


def bar(s: Dict, p: int) -> List[float]:
    """The dimension bar of pair p as rl.force_render draws it (line and end ticks, #555555): its extent [x0, y0, x1, y1)."""
    g = rl.force_geometry(s)
    q = g["pairs"][p]
    ax0, ax1 = q["start"] + q["widths"][0], q["start"] + q["widths"][0] + q["gap"]
    ty = g["h"] - 14
    return [ax0, ty - 4, ax1 + 1, ty + 5]


def target_of(form: str, scene: Dict, f: str, o: str) -> str:
    return b61.target_of(form, scene, f, o)


def parent_questions(tag: str, fam: str, form: str, sc: Dict, pair_tag: str) -> List[Dict]:
    """The generator's own two pictures of a family with exp45 boxes, each paired with its variant question."""
    out = []
    for view in ("v0", "v1"):
        scene = sc[(fam, view)][1]
        rel = "real_layout/images/%s_%s.png" % (fam, view)
        img = Image.open(os.path.join(ROOT, rel))
        w, hh = img.size
        objs = ("A", "B") if form == "temp" else ("1", "2")
        fields = ("mass", "speed") if form == "temp" else ("distance",)
        for f in fields:
            for o in objs:
                band, text, other = b45.original_box(form, scene, o, f)
                box = [max(0.0, band[0]), max(0.0, band[1]), min(float(w), band[2]), min(float(hh), band[3])]
                if not b45.inside(text, box) or b45.overlaps(other, box):
                    raise SystemExit("%s %s %s %s: parent box" % (fam, view, f, o))
                key = "%s|%s:%s|%s:%s" % (tag, fam, view, f, o)
                out.append({"key": key, "set": tag, "split": "train", "layout": "original", "id": "%s:%s" % (fam, view),
                            "qid": "%s:%s" % (f, o), "text": TEMPLATE[(f, o)], "picture": rel,
                            "box": [box[0], box[1], box[2] - box[0], box[3] - box[1]], "width": w, "height": hh, "field": f,
                            "object": o, "family": fam, "case": str(rl.case_key(form, scene)), "view": view,
                            "target": target_of(form, scene, f, o), "text_box": text,
                            "pair": "%s|%s:%s|%s:%s" % (pair_tag, fam, view, f, o)})
    return out


ROOT = "data"


# ------------------------------------------------------------------ coverage

def dist(xs: List[float]) -> Dict:
    xs = sorted(xs)
    if not xs:
        return {"n": 0}
    q = lambda p: round(xs[min(len(xs) - 1, int(p * len(xs)))], 3)  # noqa: E731
    return {"n": len(xs), "min": round(xs[0], 3), "p05": q(0.05), "p50": round(statistics.median(xs), 3), "p95": q(0.95), "max": round(xs[-1], 3)}


def coverage(root: str, qs: List[Dict]) -> Dict:
    """Ink geometry of the new pictures (audit.py's measures: column centre over width from the outer edge, outer side
    margin, bottom margin, row centre over height; temperature row pitch over ink height)."""
    g: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    ink: Dict[Tuple, List[float]] = {}
    for q in qs:
        a = np.asarray(Image.open(os.path.join(root, q["picture"])).convert("RGB"))
        x, y, bw, bh = q["box"]
        x0, y0 = int(x), int(y)
        m = text_mask(a[y0:int(np.ceil(y + bh)), x0:int(np.ceil(x + bw))])
        ys, xs = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
        t = [float(x0 + xs[0]), float(y0 + ys[0]), float(x0 + xs[-1] + 1), float(y0 + ys[-1] + 1)]
        w, hh = q["width"], q["height"]
        left = (t[0] + t[2]) / 2 < w / 2
        k = g["%s/%s" % (q["set"], q["field"])]
        k["col_x"].append((t[0] + t[2]) / 2 / w if left else 1 - (t[0] + t[2]) / 2 / w)
        k["side_margin"].append(t[0] if left else w - t[2])
        k["bottom_margin"].append(hh - t[3])
        k["row_y"].append((t[1] + t[3]) / 2 / hh)
        ink[(q["set"], q["picture"], q["object"], q["field"])] = t
    for (st, pic, o, f), t in ink.items():
        if f == "mass" and (st, pic, o, "speed") in ink:
            s_ = ink[(st, pic, o, "speed")]
            g["%s/temp" % st]["row_pitch"].append(abs((s_[1] + s_[3]) - (t[1] + t[3])) / 2 / (t[3] - t[1]))
    return {k: {m: dist(v) for m, v in d.items()} for k, d in sorted(g.items())}


# ------------------------------------------------------------------ build

def pick(fams: Dict[str, str], n: int, form: str) -> List[str]:
    return sorted((f for f, k in fams.items() if k == form), key=lambda f: h(SEED + f))[:n]


def build(root: str = "data", preview: str = "") -> Dict:
    global ROOT
    ROOT = root
    used = {q["family"] for q in jsonl(os.path.join(root, "exp49", "questions.jsonl"))}
    used61 = {q["family"] for q in jsonl(os.path.join(root, "exp61", "questions.jsonl")) if q["set"] == "train_V"}
    train, tune, cases = {}, {}, {}
    for i, split, form, s, twin in b61.families_by_split():
        fam = "rl%05d" % i
        cases[fam] = {str(rl.case_key(form, s)), str(rl.case_key(form, twin))}
        if split == "train" and fam not in used and fam not in used61:
            train[fam] = form
        elif split == "tune":
            tune[fam] = form
    chosen = {"train_C": pick(train, N_TEMP, "temp") + pick(train, N_FORCE, "force"), "tune_C": sorted(tune)}
    tr, tu = set(chosen["train_C"]), set(chosen["tune_C"])
    if tr & tu or tr & used or tr & used61:
        raise SystemExit("family overlap")
    if set().union(*(cases[f] for f in tr)) & set().union(*(cases[f] for f in tu)):
        raise SystemExit("case overlap between train_C and tune_C")
    sc = b61.scenes(tr | tu)
    dest = os.path.join(root, "exp66", "images")
    os.makedirs(dest, exist_ok=True)
    qs, lay = [], []
    for tag, fams in chosen.items():
        for fam in fams:
            form = sc[(fam, "v0")][0]
            views = [sc[(fam, "v0")][1], sc[(fam, "v1")][1]]
            p = params(fam, form, views)
            sizes = set()
            for view in ("v0", "v1"):
                scene = sc[(fam, view)][1]
                img, boxes, crop = render(form, scene, p)
                sizes.add(img.size)
                pid = "%s_%s_%s" % (tag, fam, view)
                rel = "exp66/images/%s.png" % pid
                img.save(os.path.join(root, rel))
                other_view = "v1" if view == "v0" else "v0"
                for (f, o), (box, text) in boxes.items():
                    twin_scene = sc[(fam, other_view)][1]
                    qs.append({"key": "%s|%s:%s|%s:%s" % (tag, fam, view, f, o), "set": tag, "split": "train" if tag == "train_C" else "dev",
                               "layout": "column", "id": "%s:%s" % (fam, view), "qid": "%s:%s" % (f, o), "text": TEMPLATE[(f, o)],
                               "picture": rel, "box": box, "width": img.size[0], "height": img.size[1], "field": f, "object": o,
                               "family": fam, "case": str(rl.case_key(form, scene)), "view": view, "target": target_of(form, scene, f, o),
                               "twin": "%s|%s:%s|%s:%s" % (tag, fam, other_view, f, o),
                               "twin_changed": target_of(form, scene, f, o) != target_of(form, twin_scene, f, o), "text_box": text,
                               "layout_params": p})
            if len(sizes) != 1:
                raise SystemExit("%s: the twins' canvases differ %s" % (fam, sizes))
            lay.append({"set": tag, "family": fam, "form": form, **p})
    for fam in chosen["train_C"]:
        qs.extend(parent_questions("train_C_parent", fam, sc[(fam, "v0")][0], sc, "train_C"))
    path = os.path.join(root, "exp66", "questions.jsonl")
    with open(path, "w") as fh:
        fh.write("".join(json.dumps(q) + "\n" for q in qs))
    cov = coverage(root, qs)
    json.dump(cov, open(os.path.join(root, "exp66", "coverage.json"), "w"), indent=1)
    pics = sorted({q["picture"] for q in qs if q["set"] != "train_C_parent"})
    temp = [x for x in lay if x["form"] == "temp"]
    m = {"seed": SEED, "families": {k: len(v) for k, v in chosen.items()},
         "families_by_form": {k: dict(Counter(sc[(f, "v0")][0] for f in v)) for k, v in chosen.items()},
         "pictures": dict(Counter(q["set"] for q in {q["picture"]: q for q in qs}.values())),
         "questions": dict(Counter(q["set"] for q in qs)),
         "layout": {k: {"shift": dist([x["shift"] for x in temp if x["set"] == k]), "shift_max": dist([x["shift_max"] for x in temp if x["set"] == k]),
                        "shift_in_max": dist([x["shift_in_max"] for x in temp if x["set"] == k]),
                        "top": dist([x["top"] for x in lay if x["set"] == k]),
                        "bottom_margin": dist([x["bottom_margin"] for x in temp if x["set"] == k]),
                        "pad": dist([x["pad"] for x in lay if x["set"] == k and x["form"] == "force"])} for k in chosen},
         "rules": {"shift": [-SHIFT_IN, "s_max"], "top_max": TOP_MAX, "bottom_margin_max": BOTTOM_MAX, "pad_max": PAD_MAX},
         "excluded_families": {"exp49": len(used), "exp61_train_V": len(used61)},
         "isolation": {"family_overlap": 0, "case_overlap": 0},
         "questions_sha256": h(open(path).read()),
         "pictures_sha256": h(json.dumps({p_: hashlib.sha256(open(os.path.join(root, p_), "rb").read()).hexdigest() for p_ in pics}, sort_keys=True)),
         "coverage_sha256": h(open(os.path.join(root, "exp66", "coverage.json")).read())}
    json.dump(m, open(os.path.join(root, "exp66", "MANIFEST.json"), "w"), indent=1)
    if preview:
        sheets(root, qs, preview)
    return m


def sheets(root: str, qs: List[Dict], dest: str) -> None:
    os.makedirs(dest, exist_ok=True)
    by: Dict[str, List[Dict]] = defaultdict(list)
    for q in qs:
        by[q["picture"]].append(q)
    for tag in ("train_C", "tune_C"):
        for form in ("temp", "force"):
            pics = [p for p in sorted(by) if p.split("/")[-1].startswith(tag + "_") and (by[p][0]["field"] == "distance") == (form == "force")]
            sample = sorted(pics, key=lambda p: h(SEED + "preview:" + p))[:9]
            sheet = Image.new("RGB", (3 * 620, 3 * 460 if form == "temp" else 3 * 200), "white")
            for k, p in enumerate(sample):
                im = Image.open(os.path.join(root, p)).convert("RGB")
                d = ImageDraw.Draw(im)
                for q in by[p]:
                    x, y, bw, bh = q["box"]
                    d.rectangle([x, y, x + bw, y + bh], outline=(255, 0, 0), width=1)
                    t = q["text_box"]
                    d.rectangle(t, outline=(0, 0, 255), width=1)
                im.thumbnail((610, 450 if form == "temp" else 190))
                sheet.paste(im, ((k % 3) * 620, (k // 3) * (460 if form == "temp" else 200)))
            sheet.save(os.path.join(dest, "%s_%s.png" % (tag, form)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data")
    ap.add_argument("--preview", default="")
    a = ap.parse_args(argv)
    print(json.dumps(build(a.root, a.preview), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
