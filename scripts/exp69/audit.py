"""exp69 feasibility audit (critic EXP69-PREP-01; docs/experiments/2026-10-19-exp69-speed-region-plan.md). CPU only, no
model, no forward: the existing per-read audit (data/exp66/audit.details.jsonl: registered c92 boxes and the drawn text's
ink, EXP60's text-pixel rule) on SPEED fields only.

For each speed read: the c92 box; the tight target = the ink extent padded 2 px and clipped (EXP60's oracle rule, applied
to the picture's own pixels -- on training pictures this is a label from drawn text, never from a dev answer); then
  offset     target centre - c92 centre in c92 box heights; IoU; whether c92 holds the ink;
  window     whether the target's rows lie inside the c92 box grown vertically by 0.5 / 1.0 box heights (the search
             window a local corrector would look in); apart, whether its columns lie inside c92's (+-1 px) and by how
             many px it sticks out left / right;
  package    region() of both boxes: token counts, local-token counts, Jaccard of the token index sets -- how different
             the S package of a tight speed box is from c92's.
Groups: train (EXP64 / EXP62 training records), dev tune_V and rl_tune (generated dev slices), dev24 (diagnostic only).

    uv run python scripts/exp69/audit.py [--out data/exp69/audit.json]
"""
import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from typing import Dict, List


def dist(xs: List[float]) -> Dict:
    xs = sorted(xs)
    if not xs:
        return {"n": 0}
    q = lambda p: round(xs[min(len(xs) - 1, int(p * len(xs)))], 3)  # noqa: E731
    return {"n": len(xs), "p05": q(0.05), "p50": round(statistics.median(xs), 3), "p95": q(0.95)}


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def main(argv=None) -> int:
    from devision.reading import region
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--details", default="data/exp66/audit.details.jsonl")
    ap.add_argument("--out", default="data/exp69/audit.json")
    a = ap.parse_args(argv)
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for line in open(a.details):
        d = json.loads(line)
        if d["field"] != "speed" or not d["m"]:
            continue
        g = "train" if d["set"].startswith("train/") and d["set"] != "train/fit_L" else d["set"]
        if g == "train/fit_L":
            continue
        w, h = d["w"], d["h"]
        x, y, bw, bh = d["box"]
        c92 = [x, y, x + bw, y + bh]
        ink = d["m"]["ink"]
        t = [max(0.0, ink[0] - 2), max(0.0, ink[1] - 2), min(float(w), ink[2] + 2), min(float(h), ink[3] + 2)]
        r_c, r_t = region(d["box"], w, h), region([t[0], t[1], t[2] - t[0], t[3] - t[1]], w, h)
        sc, st = set(r_c["index"]), set(r_t["index"])
        win = lambda k: t[1] >= y - k * bh and t[3] <= y + bh + k * bh  # noqa: E731
        horiz = t[0] >= x - 1 and t[2] <= x + bw + 1
        groups[g].append({"dy": ((t[1] + t[3]) / 2 - (y + bh / 2)) / bh, "iou": iou(c92, t), "holds": d["m"]["holds"],
                          "win05": win(0.5), "win10": win(1.0), "horiz": horiz, "left_out": max(0.0, x - t[0]), "right_out": max(0.0, t[2] - (x + bw)), "tok_c92": len(sc), "tok_tight": len(st),
                          "loc_c92": sum(v > 0 for v in r_c["view"]), "loc_tight": sum(v > 0 for v in r_t["view"]),
                          "jaccard": len(sc & st) / len(sc | st), "tight_h": t[3] - t[1], "c92_h": bh})
    res = {}
    for g, v in sorted(groups.items()):
        res[g] = {"n": len(v), "holds": round(sum(x["holds"] for x in v) / len(v), 3),
                  "window_0.5h": round(sum(x["win05"] for x in v) / len(v), 3), "window_1.0h": round(sum(x["win10"] for x in v) / len(v), 3),
                  "horizontal_inside": round(sum(x["horiz"] for x in v) / len(v), 3), "left_out_px": dist([x["left_out"] for x in v]),
                  "right_out_px": dist([x["right_out"] for x in v]),
                  **{k: dist([x[k] for x in v]) for k in ("dy", "iou", "tok_c92", "tok_tight", "loc_c92", "loc_tight", "jaccard", "tight_h", "c92_h")}}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    for g, v in res.items():
        print(g, v["n"], "holds", v["holds"], "win.5", v["window_0.5h"], "win1", v["window_1.0h"], "horiz", v["horizontal_inside"], "L", v["left_out_px"]["p50"], "R", v["right_out_px"]["p50"], "dy", v["dy"], "iou", v["iou"]["p50"],
              "tok", v["tok_c92"]["p50"], "->", v["tok_tight"]["p50"], "jac", v["jaccard"]["p50"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
