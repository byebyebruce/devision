"""exp69 CPU data (critic EXP69-PREP-01 / -R1-ACCEPT-01; docs/experiments/2026-10-19-exp69-speed-region-plan.md). No model.

From the bound per-read audit (data/exp66/audit.details.jsonl a94f8f35...: registered c92 boxes and each drawn text's ink by
EXP60's text-pixel rule) and data/exp64/train.jsonl (the training records with their registered c92 boxes):

  train    the SPEED fields of the training records (train_V 400, train_P 400, fit_O 192 = 992). Target = the ink extent
           padded 2 px, clipped to the canvas (EXP60's oracle rule on the training picture's own drawn text). Inputs: 8
           jittered c92 boxes per record, records in key order, for each record 8 draws of (dy, dx) in that order from
           one random.Random(0): dy ~ U(-0.5, 0.5) c92 box heights, dx ~ U(-8, 8) px; the input box is clipped to the
           canvas. Window = the input box grown by 0.5 of ITS height above and below and 8 px left and right, clipped to
           the canvas. Label policy (one rule for every training sample, fixed now): the raw target is kept; the training
           label is the target intersected with the window (empty -> the sample is dropped and counted); the label in
           window fractions (cx, cy, w, h). Every window's local-view token count (region() of the window, views > 0).
  dev      the tight targets of the dev speed fields (rl_tune / tune_V from the same audit; dev24 = EXP60's oracle boxes),
           used only by the target-region control path, never trained on.
Writes data/exp69/{train.jsonl, dev_targets.jsonl, MANIFEST.json} (reach rates, token counts, cache size).

    uv run python scripts/exp69/build.py --root data
"""
import argparse
import hashlib
import json
import os
import random
import statistics
import sys
from collections import Counter
from typing import Dict, List

AUDIT = "data/exp66/audit.details.jsonl"
AUDIT_SHA = "a94f8f355040f8147236a4a72abea6ef3ffd90dbf1fa8a87af9fde56b3c5a85d"
DRAWS, SEED, DY, DX, GROW_Y, GROW_X, PAD = 8, 0, 0.5, 8.0, 0.5, 8.0, 2.0
SPEED = {"A": "What is the average particle speed in sample A?", "B": "What is the average particle speed in sample B?"}


def sha(p: str) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def jsonl(p: str) -> List[Dict]:
    return [json.loads(x) for x in open(p) if x.strip()]


def clip(b: List[float], w: int, h: int) -> List[float]:
    return [max(0.0, b[0]), max(0.0, b[1]), min(float(w), b[2]), min(float(h), b[3])]


def tight(ink: List[float], w: int, h: int) -> List[float]:
    return clip([ink[0] - PAD, ink[1] - PAD, ink[2] + PAD, ink[3] + PAD], w, h)


def window(box_xyxy: List[float], w: int, h: int) -> List[float]:
    bh = box_xyxy[3] - box_xyxy[1]
    return clip([box_xyxy[0] - GROW_X, box_xyxy[1] - GROW_Y * bh, box_xyxy[2] + GROW_X, box_xyxy[3] + GROW_Y * bh], w, h)


def inside(a: List[float], b: List[float]) -> bool:
    return a[0] >= b[0] and a[1] >= b[1] and a[2] <= b[2] and a[3] <= b[3]


def to_window(t: List[float], win: List[float]) -> List[float]:
    """(cx, cy, w, h) of t in win's fractions."""
    ww, wh = win[2] - win[0], win[3] - win[1]
    return [((t[0] + t[2]) / 2 - win[0]) / ww, ((t[1] + t[3]) / 2 - win[1]) / wh, (t[2] - t[0]) / ww, (t[3] - t[1]) / wh]


def local_index(win: List[float], w: int, h: int) -> List[int]:
    """region()'s token indices of the window in the local views (view > 0), in region's order."""
    from devision.reading import region
    r = region([win[0], win[1], win[2] - win[0], win[3] - win[1]], w, h)
    return [i for i, v in zip(r["index"], r["view"]) if v > 0]


def dist(xs: List[float]) -> Dict:
    xs = sorted(xs)
    q = lambda p: round(xs[min(len(xs) - 1, int(p * len(xs)))], 3)  # noqa: E731
    return {"n": len(xs), "min": round(xs[0], 3), "p50": round(statistics.median(xs), 3), "p95": q(0.95), "max": round(xs[-1], 3)}


def build(root: str = "data") -> Dict:
    if sha(AUDIT) != AUDIT_SHA:
        raise SystemExit("the audit details are not the bound a94f8f35")
    det = {}
    for d in jsonl(AUDIT):
        if d["field"] == "speed":
            det[(d["set"], d["picture"], d["object"])] = d
    recs = sorted((r for r in jsonl(os.path.join(root, "exp64", "train.jsonl")) if r["field"] == "speed" and r["set"] in ("train_V", "train_P", "fit_O")),
                  key=lambda r: r["key"])
    rng = random.Random(SEED)
    out, dropped, reach, toks, unions = [], 0, [], [], []
    for r in recs:
        union: set = set()
        d = det[("train/" + r["set"], os.path.join(root, r["picture"]), r["object"])]
        w, h = r["width"], r["height"]
        if [w, h] != [d["w"], d["h"]] or [round(v, 6) for v in d["box"]] != [round(v, 6) for v in r["box"]]:
            raise SystemExit("%s: the audit row is not this record" % r["key"])
        t = tight(d["m"]["ink"], w, h)
        x, y, bw, bh = r["box"]
        for k in range(DRAWS):
            dy, dx = rng.uniform(-DY, DY), rng.uniform(-DX, DX)
            inp = clip([x + dx, y + dy * bh, x + dx + bw, y + dy * bh + bh], w, h)
            win = window(inp, w, h)
            lab = [max(t[0], win[0]), max(t[1], win[1]), min(t[2], win[2]), min(t[3], win[3])]
            if lab[2] - lab[0] < 1 or lab[3] - lab[1] < 1:
                dropped += 1
                continue
            reach.append(inside(t, win))
            idx = local_index(win, w, h)
            n = len(idx)
            union |= set(idx)
            toks.append(n)
            out.append({"key": "%s|j%d" % (r["key"], k), "record": r["key"], "set": r["set"], "picture": r["picture"], "width": w, "height": h,
                        "object": r["object"], "template": SPEED[r["object"]], "c92": r["box"], "dy": dy, "dx": dx, "input": inp, "window": win,
                        "target": t, "label_px": lab, "label": to_window(lab, win), "reached": reach[-1], "local_tokens": n})
        unions.append(len(union))
    dev = []
    tune = {os.path.join(root, r["image"]): r["id"] for r in jsonl(os.path.join(root, "real_layout", "questions_tune.jsonl"))}
    for (s, pic, obj), d in sorted(det.items()):
        if s in ("dev/rl_tune", "dev/tune_V"):
            k = tune[pic] if s == "dev/rl_tune" else os.path.relpath(pic, root)
            dev.append({"set": s.split("/")[1], "key": k, "template": SPEED[obj], "target": tight(d["m"]["ink"], d["w"], d["h"]),
                        "width": d["w"], "height": d["h"]})
    dest = os.path.join(root, "exp69")
    os.makedirs(dest, exist_ok=True)
    for name, rows in (("train.jsonl", out), ("dev_targets.jsonl", dev)):
        with open(os.path.join(dest, name), "w") as fh:
            fh.write("".join(json.dumps(x) + "\n" for x in rows))
    m = {"records": len(recs), "by_set": dict(Counter(r["set"] for r in recs)), "draws": DRAWS, "samples": len(out), "dropped_empty_label": dropped,
         "reached": {"all": sum(reach), "of": len(reach), "share": round(sum(reach) / len(reach), 4)},
         "rule": {"seed": SEED, "order": "records by key; per record 8 x (dy, then dx)", "dy_box_heights": [-DY, DY], "dx_px": [-DX, DX],
                  "window": {"grow_y_box_heights": GROW_Y, "grow_x_px": GROW_X, "clip": "canvas"}, "target": "ink + 2 px, clipped (EXP60's rule)",
                  "label": "target intersected with the window; window fractions (cx, cy, w, h)"},
         "local_tokens": dist(toks), "max_tokens_per_window": max(toks),
         "cache": {"per_record_union_tokens": dist(unions), "float16_bytes": sum(unions) * 768 * 2,
                   "note": "one cache row per record: the union of its 8 windows' local tokens; each sample indexes into it"},
         "dev_targets": dict(Counter(x["set"] for x in dev)),
         "sources": {"audit_details": AUDIT_SHA, "exp64_train": sha(os.path.join(root, "exp64", "train.jsonl"))},
         "sha256": {n: sha(os.path.join(dest, n)) for n in ("train.jsonl", "dev_targets.jsonl")}}
    json.dump(m, open(os.path.join(dest, "MANIFEST.json"), "w"), indent=1)
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data")
    print(json.dumps(build(ap.parse_args().root), indent=1))
    sys.exit(0)
