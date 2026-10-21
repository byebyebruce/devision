"""exp66 CPU region-coverage audit (critic EXP66-PREP-01 item 1; docs/experiments/2026-10-19-exp66-layout-plan.md). No
model, no forward: registered boxes, generator metadata and pixels only.

For every field read of the measured sets -- training (data/exp64/train.jsonl: fit_O / fit_L / train_V / train_P with the
EXP62 selector's registered boxes), the tune_V and rl_tune dev slices (EXP62's registered auto boxes) and the 24
transcribed pictures (EXP62's auto boxes; a development diagnostic only) -- the drawn text is measured on the pixels:
ink = dark, unsaturated pixels (EXP60's rule: max channel < 110, spread < 40) inside the field's reference box (training /
tune: its field band or text box padded 3 px; dev24: the EXP60 oracle box). Per read:

  canvas    ink height; ink bottom margin (canvas height - ink bottom) and outer side margin (ink x0 for the left
            half's object, width - ink x1 for the right one); temperature: row pitch (speed - mass ink centre of the same
            object) over ink height; row centre over canvas height;
  offset    predicted box centre - ink centre (y in ink heights, x in half-widths); box height over ink height; whether
            the box holds the ink;
  relative  the ink as the head sees it through region()'s geom: its centre y and height in box units, its x0 / x1 in
            box units -- where in the box the text sits;
  tokens    region()'s selected cells: full / local token counts and the share of the ink they cover (EXP64's rule).

"selected tokens cover the ink" and "the ink sits where training put it inside the box" are reported apart; a box can
cover the text with every token and still place it at an unseen relative position.

Plus the generator's own range over every TRAIN-split picture of data/real_layout (metadata text boxes, no pixels): the
bottom / side margins and the row pitch, with and without EXP61's blank-row crop (top t 0..30, bottom m 0..6).

    uv run python scripts/exp66/audit.py --root data [--preview runs/exp66_audit]
"""
import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(__file__)
E62, E60 = "runs/exp62", "runs/exp60"
PAD = 3


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, rel))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


b61 = _load("exp61_build", "../exp61/build.py")
b64 = _load("exp64_build", "../exp64/build.py")
b45, rl = b61.b45, b61.rl


def jsonl(p: str) -> List[Dict]:
    return [json.loads(x) for x in open(p) if x.strip()]


def sha(p: str) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def text_mask(a: np.ndarray) -> np.ndarray:
    a = a.astype(int)
    return (a.max(-1) < 110) & (a.max(-1) - a.min(-1) < 40)


_pix: Dict[str, np.ndarray] = {}


def pixels(path: str) -> np.ndarray:
    if path not in _pix:
        if len(_pix) > 64:
            _pix.clear()
        _pix[path] = np.asarray(Image.open(path).convert("RGB"))
    return _pix[path]


def ink(path: str, ref_xyxy: List[float]) -> Optional[List[float]]:
    """The ink extent [x0, y0, x1, y1) inside the reference box (clipped to the canvas); None when it holds no ink."""
    a = pixels(path)
    h, w = a.shape[:2]
    x0, y0 = max(0, int(ref_xyxy[0])), max(0, int(ref_xyxy[1]))
    x1, y1 = min(w, int(np.ceil(ref_xyxy[2]))), min(h, int(np.ceil(ref_xyxy[3])))
    m = text_mask(a[y0:y1, x0:x1])
    if not m.any():
        return None
    ys, xs = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
    return [float(x0 + xs[0]), float(y0 + ys[0]), float(x0 + xs[-1] + 1), float(y0 + ys[-1] + 1)]


def measure(path: str, w: int, h: int, box: List[float], ref: List[float]) -> Optional[Dict]:
    """One read: canvas / offset / relative / token measures of the ink inside `ref` (xyxy) against the predicted box
    (xywh, devision.reading's convention)."""
    t = ink(path, ref)
    if t is None:
        return None
    ih, icx, icy = t[3] - t[1], (t[0] + t[2]) / 2, (t[1] + t[3]) / 2
    bx, by, bw, bh = box
    left = icx < w / 2
    cov = b64.coverage(box, t, w, h)
    return {"ink": t, "ink_h": ih, "bottom_margin": h - t[3], "side_margin": t[0] if left else w - t[2], "row_y": icy / h,
            "col_x": icx / w if left else 1 - icx / w,
            "dy": (by + bh / 2 - icy) / ih, "dx": (bx + bw / 2 - icx) / (w / 2), "box_h": bh / ih, "holds": cov["inside"],
            "rel_cy": (icy - by) / bh, "rel_h": ih / bh, "rel_x0": (t[0] - bx) / bw, "rel_x1": (t[2] - bx) / bw,
            "full_tokens": cov["full"]["tokens"], "local_tokens": cov["local"]["tokens"], "local_covered": cov["local"]["covered"]}


def xyxy(b: List[float]) -> List[float]:
    return [b[0], b[1], b[0] + b[2], b[1] + b[3]]


def padded(b: List[float], p: float = PAD) -> List[float]:
    return [b[0] - p, b[1] - p, b[2] + p, b[3] + p]


# ------------------------------------------------------------------ the measured sets

def reads(root: str) -> List[Dict]:
    """(set, field, object, picture, size, predicted box, reference box) of every measured read, with its row key."""
    out = []
    for r in jsonl(os.path.join(root, "exp64", "train.jsonl")):
        ref = padded(r["text_box"]) if r["text_box"] else xyxy(r["field_box"])
        out.append({"set": "train/" + r["set"], "field": r["field"], "object": r["object"], "picture": os.path.join(root, r["picture"]),
                    "w": r["width"], "h": r["height"], "box": r["box"], "ref": ref})
    reg = json.load(open(os.path.join(E62, "registry.json")))
    v = {(q["picture"], q["text"]): q for q in jsonl(os.path.join(root, "exp61", "questions.jsonl")) if q["set"] == "tune_V"}
    f = os.path.join(E62, "tune_V_new.jsonl")
    if sha(f) != reg["reads_tune_V_new"]["tune_V_new.jsonl"]:
        raise SystemExit("tune_V_new is not registered")
    for x in jsonl(f)[1:]:
        for rd in x["reads"]:
            q = v[(x["picture"], rd["question"])]
            out.append({"set": "dev/tune_V", "field": q["field"], "object": q["object"], "picture": os.path.join(root, q["picture"]),
                        "w": q["width"], "h": q["height"], "box": rd["box"], "ref": padded(q["text_box"])})
    tune = {r["id"]: r for r in jsonl(os.path.join(root, "real_layout", "questions_tune.jsonl"))}
    qs = {(q["id"], tune[q["id"]]["questions"][q["qid"]]["instructions"]): q for q in jsonl(os.path.join(root, "exp49", "questions.jsonl"))
          if q["set"] == "rl_tune"}
    f = os.path.join(E62, "rl_tune_new.jsonl")
    if sha(f) != reg["reads_rl_tune_new"]["rl_tune_new.jsonl"]:
        raise SystemExit("rl_tune_new is not registered")
    for x in jsonl(f)[1:]:
        for rd in x["reads"]:
            q = qs[(x["id"], rd["question"])]
            out.append({"set": "dev/rl_tune", "field": q["field"], "object": q["object"], "picture": os.path.join(root, q["picture"]),
                        "w": q["width"], "h": q["height"], "box": rd["box"], "ref": xyxy(q["box"])})
    ob = {k: v for k, v in json.load(open(os.path.join("scripts", "exp60", "oracle_boxes.json"))).items() if not k.startswith("_")}
    for r in jsonl(os.path.join(E62, "details_dev24_new.jsonl"))[1:]:
        o = ob[r["sample_id"]]
        w, h = o["size"]
        for rd in r["diagnostics"]["reads"]:
            fld = "mass" if "mass" in rd["question"] else "speed" if "speed" in rd["question"] else "distance"
            obj = rd["question"].rstrip("?").split()[-1]
            out.append({"set": "dev24/auto", "field": fld, "object": obj, "picture": os.path.join(root, o["image"]), "w": w, "h": h,
                        "box": rd["box"], "ref": xyxy(o["boxes"][rd["question"]]), "sample": r["sample_id"]})
    return out


def dist(xs: List[float]) -> Dict:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return {"n": 0}
    q = lambda p: round(xs[min(len(xs) - 1, int(p * len(xs)))], 3)  # noqa: E731
    return {"n": len(xs), "min": round(xs[0], 3), "p05": q(0.05), "p50": round(statistics.median(xs), 3), "p95": q(0.95), "max": round(xs[-1], 3)}


KEYS = ("col_x", "half_fill", "ink_h", "bottom_margin", "side_margin", "row_y", "dy", "dx", "box_h", "rel_cy", "rel_h", "rel_x0", "rel_x1",
        "full_tokens", "local_tokens", "local_covered")


def summarise(rows: List[Dict]) -> Dict:
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        groups["%s/%s" % (r["set"], r["field"])].append(r)
    out = {}
    for k, v in sorted(groups.items()):
        m = [x["m"] for x in v if x["m"]]
        out[k] = {"n": len(v), "no_ink": len(v) - len(m), "holds": round(sum(x["holds"] for x in m) / max(1, len(m)), 3),
                  **{key: dist([x[key] for x in m]) for key in KEYS}}
        if v[0]["field"] in ("mass", "speed"):
            out[k]["row_pitch"] = dist([x["pitch"] for x in v if x.get("pitch") is not None])
            out[k]["pitch_px"] = dist([x["pitch_px"] for x in v if x.get("pitch_px") is not None])
            out[k]["header_gap"] = dist([x["header_gap"] for x in v if x.get("header_gap") is not None])
            out[k]["jar_gap"] = dist([x["jar_gap"] for x in v if x.get("jar_gap") is not None])
    return out


def pitches(rows: List[Dict]) -> None:
    """Temperature row pitch: |speed ink centre - mass ink centre| of the same object and picture, over the mass ink height."""
    by: Dict[Tuple, Dict[str, Dict]] = defaultdict(dict)
    for r in rows:
        if r["m"] and r["field"] in ("mass", "speed"):
            by[(r["set"], r["picture"], r["object"])][r["field"]] = r
    for d in by.values():
        if len(d) == 2:
            ms, sp = d["mass"]["m"], d["speed"]["m"]
            p = abs((sp["ink"][1] + sp["ink"][3]) / 2 - (ms["ink"][1] + ms["ink"][3]) / 2) / ms["ink_h"]
            d["mass"]["pitch"] = d["speed"]["pitch"] = p


def bands(rows_mask: np.ndarray, gap: int = 3) -> List[List[int]]:
    out: List[List[int]] = []
    for r in np.flatnonzero(rows_mask):
        if out and r - out[-1][1] <= gap:
            out[-1][1] = int(r)
        else:
            out.append([int(r), int(r)])
    return out


def header_gaps(rows: List[Dict]) -> None:
    """Temperature: the "Sample X" header's ink band is the last text band of the object's half above its first field
    row (EXP60's band rule); header gap = (first row ink centre - header centre) / row pitch in pixels (the generator
    draws the header one pitch above the first row, so 1.0 there)."""
    by: Dict[Tuple, Dict[str, Dict]] = defaultdict(dict)
    for r in rows:
        if r["m"] and r["field"] in ("mass", "speed"):
            by[(r["set"], r["picture"], r["object"])][r["field"]] = r
    for d in by.values():
        if len(d) != 2:
            continue
        ms, sp = d["mass"]["m"], d["speed"]["m"]
        first, second = (ms, sp) if ms["ink"][1] < sp["ink"][1] else (sp, ms)
        a = pixels(d["mass"]["picture"])
        w = a.shape[1]
        x0, x1 = (0, w // 2) if (first["ink"][0] + first["ink"][2]) / 2 < w / 2 else (w // 2, w)
        top = int(first["ink"][1])
        bs = [b for b in bands(text_mask(a[:top, x0:x1]).any(1)) if b[1] < top - 1]
        pitch_px = (second["ink"][1] + second["ink"][3] - first["ink"][1] - first["ink"][3]) / 2
        if not bs or pitch_px <= 0:
            continue
        hc = (bs[-1][0] + bs[-1][1] + 1) / 2
        g = ((first["ink"][1] + first["ink"][3]) / 2 - hc) / pitch_px
        d["mass"]["header_gap"] = d["speed"]["header_gap"] = g
        d["mass"]["pitch_px"] = d["speed"]["pitch_px"] = pitch_px
        drawn = (a.min(-1) < 235) & ~text_mask(a)                    # the jar: non-white, non-text pixels above the header
        jar = np.flatnonzero(drawn[:bs[-1][0] - 1, x0:x1].any(1))
        if len(jar):
            d["mass"]["jar_gap"] = d["speed"]["jar_gap"] = (hc - jar[-1]) / pitch_px


# ------------------------------------------------------------------ the generator's own range (metadata only)

def generator_range() -> Dict:
    """Every TRAIN-split scene of data/real_layout (the twin shares the layout): text-box bottom / side margins, row
    centre over canvas height and the temperature row pitch over text-box height, from the generator's metadata (PIL
    text boxes, a little taller than ink), as drawn (EXP61's crop changes only the bottom margin, to 0..6 px below ink)."""
    acc: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for _, split, form, s, _twin in b61.families_by_split():
        if split != "train":
            continue
        if form == "temp":
            r = b45.temp_rows(s)
            any_row = next(iter(r.values()))
            w, h = s["width"], max(405, 315 + 3 * any_row["line"] + any_row["size"])
            fields, objs = ("mass", "speed"), ("A", "B")
        else:
            g = rl.force_geometry(s)
            w, h = g["w"], g["h"]
            fields, objs = ("distance",), ("1", "2")
        for f in fields:
            for o in objs:
                _, t, _ = b45.original_box(form, s, o, f)
                left = (t[0] + t[2]) / 2 < w / 2
                a = acc[f]
                a["text_h"].append(t[3] - t[1])
                a["bottom_margin"].append(h - t[3])
                a["side_margin"].append(t[0] if left else w - t[2])
                a["row_y"].append((t[1] + t[3]) / 2 / h)
                a["half_fill"].append((t[2] - t[0]) / (w / 2))
                a["col_x"].append((t[0] + t[2]) / 2 / w if left else 1 - (t[0] + t[2]) / 2 / w)
        if form == "temp":
            for o in objs:
                _, tm, _ = b45.original_box(form, s, o, "mass")
                _, ts, _ = b45.original_box(form, s, o, "speed")
                acc["temp"]["row_pitch"].append(abs((ts[1] + ts[3]) - (tm[1] + tm[3])) / 2 / (tm[3] - tm[1]))
    return {f: {k: dist(v) for k, v in d.items()} for f, d in acc.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="data/exp66/audit.json")
    a = ap.parse_args(argv)
    rows = reads(a.root)
    for r in rows:
        r["m"] = measure(r["picture"], r["w"], r["h"], r["box"], r["ref"])
        if r["m"]:
            r["holds"] = r["m"]["holds"]
    pitches(rows)
    header_gaps(rows)
    for r in rows:
        if r["m"]:
            r["m"]["half_fill"] = (r["m"]["ink"][2] - r["m"]["ink"][0]) / (r["w"] / 2)
    res = {"measured": summarise(rows), "generator_train": generator_range()}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    with open(a.out.replace(".json", ".details.jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps({k: r.get(k) for k in ("set", "field", "object", "picture", "w", "h", "box", "ref", "sample", "pitch", "pitch_px",
                                                       "header_gap", "jar_gap", "m")}) + "\n")
    print("wrote %s (%d reads)" % (a.out, len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
