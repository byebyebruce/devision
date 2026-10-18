"""exp45 region sidecar (critic EXP45-PREP-01; docs/experiments/2026-10-18-exp45-region-plan.md). For every question
of data/exp42/train.jsonl (the training rows; the JSONL itself is not changed) and, for observation only, of fit_O /
fit_L / rl_tune / rl_tune_L: the box of the asked object's asked field -- fixed by the layout, never by the answer:

  - original pictures (real_layout's layout; the scene re-drawn read-only from real_layout.families()):
      temperature: the field's row under the object's jar -- the row's centre line (temp_render: top + k x line,
        k from the row order speed_first), one line high, the object's half of the picture wide less 2 px each side (rows are fitted into w / 2 - 12);
      force: the band of the distance label under the object's pair -- the whole pair box wide, the label's
        line (h - 14) +- the font size high;
  - L pictures (exp37's shared layout): exp37's fixed number box for that object and field (pictures.json /
    exp40's pictures_redrawn.json), whose size depends only on the layout.

Each box is checked against what was actually drawn: the text of the asked value (temperature: the whole row; force:
the distance label) lies inside the box, and the other object's same field does not overlap it. Writes
data/exp45/regions.jsonl (training rows: id, box [x, y, w, h], width, height, picture, object, field, side, layout)
and data/exp45/regions_readout.jsonl (the four readout sets, keyed by set and question), with MANIFEST.json.

    uv run python scripts/exp45/build.py --root data
"""
import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "data"))
import real_layout as rl  # type: ignore  # noqa: E402
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("exp43_build", os.path.join(HERE, "..", "exp43", "build.py"))
assert _spec is not None and _spec.loader is not None
b43 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b43)


def sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def temp_rows(s: Dict) -> Dict[str, Dict]:
    """Per object name ("A" / "B"): its centre x, the row y per field, the line height, the row width, the text size."""
    w = s["width"]
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    size = s["font"]
    longest = rl.LINES[s["wording"]][1] % "1,999"
    while size > 11 and probe.textlength(longest, font=ImageFont.truetype(rl.FONT, size)) > w / 2 - 12:
        size -= 1
    line = int(size * s["spacing"])
    top = 315 + line // 2
    order = ["speed", "mass"] if s.get("speed_first") else ["mass", "speed"]
    out = {}
    for k, name in enumerate(("A", "B") if s.get("a_left", True) else ("B", "A")):
        cx = w // 4 + k * (w // 2)
        out[name] = {"cx": cx, "y": {f: top + (j + 1) * line for j, f in enumerate(order)}, "line": line, "size": size,
                     "width": w / 2 - 4, "k": k}     # the object's half less 2 px each side (the row is fitted into w / 2 - 12)
    return out


def original_box(form: str, s: Dict, obj: str, field: str) -> Tuple[List[float], List[float], List[float]]:
    """(box [x0, y0, x1, y1] from the layout, the drawn text's box of this value, the other object's text box)."""
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    if form == "temp":
        rows = temp_rows(s)
        font = ImageFont.truetype(rl.FONT, rows[obj]["size"])

        def text_box(o):
            r = rows[o]
            mass_line, speed_line = rl.LINES[s.get("wording", 0)]
            t = mass_line % s[o]["mass"] if field == "mass" else speed_line % rl.fmt(s[o]["speed"])
            return list(probe.textbbox((r["cx"], r["y"][field]), t, font=font, anchor="mm"))
        r = rows[obj]
        box = [r["cx"] - r["width"] / 2, r["y"][field] - r["line"] / 2, r["cx"] + r["width"] / 2, r["y"][field] + r["line"] / 2]
        return box, text_box(obj), text_box("B" if obj == "A" else "A")
    g = rl.force_geometry(s)
    font = ImageFont.truetype(rl.FONT, s["font"])

    def label_box(p):
        q = g["pairs"][p]
        ax0, ax1 = q["start"] + q["widths"][0], q["start"] + q["widths"][0] + q["gap"]
        return list(probe.textbbox(((ax0 + ax1) / 2, g["h"] - 14), "%s %s" % (rl.fmt(s["d"][p]), s["unit"]), font=font, anchor="mm"))
    p = int(obj) - 1
    bx0 = g["pairs"][p]["box"]
    box = [bx0, g["h"] - 14 - s["font"], bx0 + g["pw"], min(g["h"], g["h"] - 14 + s["font"])]
    return box, label_box(p), label_box(1 - p)


def inside(inner, outer, tol=1.0) -> bool:
    return inner[0] >= outer[0] - tol and inner[1] >= outer[1] - tol and inner[2] <= outer[2] + tol and inner[3] <= outer[3] + tol


def overlaps(a, b) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def regions(rows: List[Dict], sc, meta, tag: str) -> List[Dict]:
    out = []
    for r in rows:
        view = r.get("view") or r["id"].split(":")[2]
        form, scene = sc[(r["family"], view)]
        is_l = "/L/" in r["image"]
        for q, v in r["questions"].items():
            field = b43.field_of(v["instructions"])
            if not field:
                continue
            _, obj = b43.split_question(v["instructions"])
            img = Image.open(os.path.join(ROOT, r["image"]))
            W, H = img.size
            if is_l:
                m = meta["%s_%s" % (r["family"], view)]
                name = ("Sample " if form == "temp" else "Pair ") + obj
                (n,) = [n for n in m["numbers"] if n["object"] == name and n["field"] == field]
                box = [float(x) for x in n["box"]]
                text = box
                other = [n2["box"] for n2 in m["numbers"] if n2["object"] != name and n2["field"] == field][0]
                gold = next(o for o, p in r["gold"][q]["probabilities"].items() if p == 1.0)
                if n["text"] != gold.split()[0]:
                    raise SystemExit("%s %s: L box holds %s, gold %s" % (r["id"], q, n["text"], gold))
            else:
                box, text, other = original_box(form, scene, obj, field)
            box = [max(0.0, box[0]), max(0.0, box[1]), min(float(W), box[2]), min(float(H), box[3])]
            if not inside(text, box) or overlaps(other, box):
                raise SystemExit("%s %s: the drawn value %s is not inside %s alone (other %s)" % (r["id"], q, text, box, other))
            out.append({"id": r["id"], "set": tag, "qid": q, "box": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                        "width": W, "height": H, "picture": r["image"], "object": obj, "field": field,
                        "side": b43.side(form, scene, obj), "layout": "L" if is_l else "original",
                        "speed_first": bool(scene.get("speed_first")) if form == "temp" else None})
    return out


ROOT = "data"


def build(root: str) -> Dict:
    global ROOT
    ROOT = root
    src = {"train": os.path.join(root, "exp42", "train.jsonl"), "fit_O": os.path.join(root, "exp42", "fit_O.jsonl"),
           "fit_L": os.path.join(root, "exp42", "fit_L.jsonl"), "rl_tune": os.path.join(root, "real_layout", "questions_tune.jsonl"),
           "rl_tune_L": os.path.join(root, "exp40", "rl_tune_L.jsonl")}
    rows = {k: [json.loads(x) for x in open(v)] for k, v in src.items()}
    sc = b43.scenes({r["family"] for rs in rows.values() for r in rs})
    meta = json.load(open(os.path.join(root, "exp37", "pictures.json")))
    meta.update(json.load(open(os.path.join(root, "exp40", "pictures_redrawn.json"))))
    out_dir = os.path.join(root, "exp45")
    os.makedirs(out_dir, exist_ok=True)
    train = regions(rows["train"], sc, meta, "train")
    if len(train) != 4608 or len({r["id"] for r in train}) != 4608:
        raise SystemExit("train regions: %d rows" % len(train))
    readout = [x for k in ("fit_O", "fit_L", "rl_tune", "rl_tune_L") for x in regions(rows[k], sc, meta, k)]
    for name, rs in (("regions", train), ("regions_readout", readout)):
        with open(os.path.join(out_dir, name + ".jsonl"), "w") as fh:
            for x in rs:
                fh.write(json.dumps(x) + "\n")
    m = {"sources": {k: sha(v) for k, v in src.items()},
         "regions": {"rows": len(train), "sha256": sha(os.path.join(out_dir, "regions.jsonl")),
                     "by": {"%s/%s/%s/%s" % k: n for k, n in sorted(Counter((r["layout"], r["field"], r["object"], r["side"]) for r in train).items())},
                     "speed_first": dict(Counter(str(r["speed_first"]) for r in train))},
         "regions_readout": {"rows": len(readout), "by_set": dict(Counter(r["set"] for r in readout)),
                             "sha256": sha(os.path.join(out_dir, "regions_readout.jsonl"))}}
    json.dump(m, open(os.path.join(out_dir, "MANIFEST.json"), "w"), indent=1)
    return m


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    print(json.dumps(build(p.parse_args().root), indent=1))
