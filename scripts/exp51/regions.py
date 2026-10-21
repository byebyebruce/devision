"""exp51 region tokens on EXP50's local views (critic EXP51-PREP-01), model-free. For each exp49 question (its stable
field box [x, y, w, h], the row EXP45 checked), the tokens a frozen SigLIP gives the full picture and its 12 fixed views
(n12 / overlap 0.25, devision.image.view_rects; the full picture first, then the views in their order -- exactly
model.visual_inputs' key / value order) whose patch overlaps the field box with positive area, counted only over the
view's own content (a patch's part over the grey letterbox margin is not picture). No view is chosen and nothing falls
back to the full picture: every view's overlapping patches are kept, as separate tokens (no averaging over overlaps).

Per token: its index in visual_inputs' flat order (view * 256 + patch), its view, its (row, col) relative to the first
selected cell of the same view (0..15), and its continuous geometry relative to the field box in original-picture
coordinates: ((cx - bx) / bw, (cy - by) / bh, pw / bw, ph / bh) from image.patch_geometry. The view is recorded, not
given to the head.

    uv run python scripts/exp51/regions.py --root data      # writes data/exp51/regions.json(.gz) and MANIFEST.json
"""
import argparse
import gzip
import hashlib
import json
import os
import sys
from collections import Counter
from typing import Dict, List

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from devision.image import patch_geometry, view_rects  # noqa: E402

N, OVERLAP, SIZE, SIDE = 12, 0.25, 256, 16


def region(box, w: int, h: int, n: int = N, overlap: float = OVERLAP, size: int = SIZE, side: int = SIDE) -> Dict:
    """{"index", "view", "rel", "geom"} lists for one field box (see the module docstring)."""
    bx, by, bw, bh = box
    views = [(0, 0, w, h)] + view_rects(w, h, n, overlap)
    out: Dict[str, List] = {"index": [], "view": [], "rel": [], "geom": []}
    for k, (x0, y0, x1, y1) in enumerate(views):
        g = patch_geometry((x0, y0, x1, y1), w, h, size, side).tolist()
        cells = []
        for i, (cx, cy, pw, ph) in enumerate(g):
            px0, px1 = cx * w - pw * w / 2, cx * w + pw * w / 2
            py0, py1 = cy * h - ph * h / 2, cy * h + ph * h / 2
            ix = min(px1, x1, bx + bw) - max(px0, x0, bx)          # patch ∩ view content ∩ field box
            iy = min(py1, y1, by + bh) - max(py0, y0, by)
            if ix > 1e-9 and iy > 1e-9:
                cells.append(i)
        if not cells:
            continue
        r0, c0 = min(i // side for i in cells), min(i % side for i in cells)
        for i in cells:
            cx, cy, pw, ph = g[i]
            out["index"].append(k * side * side + i)
            out["view"].append(k)
            out["rel"].append([i // side - r0, i % side - c0])
            out["geom"].append([(cx * w - bx) / bw, (cy * h - by) / bh, pw * w / bw, ph * h / bh])
    return out


def checks(qs: List[Dict], regs: List[Dict]) -> Dict:
    """Token counts per set; rel inside 0..15; value-changed twins (the donor pairs) with the same indices, views, rel
    and geometry (float32); questions of one picture whose inputs coincide while their targets differ."""
    import numpy as np
    key_i = {q["key"]: i for i, q in enumerate(qs)}
    sig = [(tuple(r["index"]), np.asarray(r["geom"], np.float32).tobytes()) for r in regs]
    twins = [(i, key_i[q["twin"]]) for i, q in enumerate(qs) if q["twin_changed"]]
    by: Dict = {}
    for i, q in enumerate(qs):
        by.setdefault((q["set"], q["picture"], sig[i]), []).append(i)
    counts: Dict = {}
    for s in sorted({q["set"] for q in qs}):
        c = [len(regs[i]["index"]) for i, q in enumerate(qs) if q["set"] == s]
        counts[s] = {"min": min(c), "mean": sum(c) / len(c), "max": max(c)}
    return {"tokens": counts, "tokens_total": sum(len(r["index"]) for r in regs),
            "rel_in_range": all(0 <= a < SIDE and 0 <= b < SIDE for r in regs for a, b in r["rel"]),
            "empty": sum(not r["index"] for r in regs),
            "views_per_question": dict(Counter(len(set(r["view"])) for r in regs)),
            "full_picture_in_every_region": all(0 in r["view"] for r in regs),
            "changed_twins": len(twins), "changed_twins_same_input": sum(sig[i] == sig[j] for i, j in twins),
            "collisions": sum(any(qs[j]["target"] != qs[i]["target"] for j in by[(q["set"], q["picture"], sig[i])]) for i, q in enumerate(qs))}


def build(root: str, questions: str = "data/exp49/questions.jsonl") -> Dict:
    qs = [json.loads(x) for x in open(questions) if x.strip()]
    regs = [region(q["box"], q["width"], q["height"]) for q in qs]
    ck = checks(qs, regs)
    dest = os.path.join(root, "exp51")
    os.makedirs(dest, exist_ok=True)
    path = os.path.join(dest, "regions.json.gz")
    with gzip.open(path, "wt") as fh:
        json.dump([{"key": q["key"], **r} for q, r in zip(qs, regs)], fh)
    m = {"questions_sha256": hashlib.sha256(open(questions, "rb").read()).hexdigest(), "grid": {"n": N, "overlap": OVERLAP,
         "size": SIZE, "side": SIDE}, "checks": ck, "regions_sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}
    json.dump(m, open(os.path.join(dest, "MANIFEST.json"), "w"), indent=1)
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data")
    print(json.dumps(build(ap.parse_args().root), indent=1))
