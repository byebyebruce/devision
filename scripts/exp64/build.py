"""exp64 CPU preparation (critic EXP64-PREP-01; docs/experiments/2026-10-19-exp64-head-plan.md): the character head's
cross-layout adaptation with everything else frozen -- EXP62's selector c92b7b7e (its registered boxes), EXP50-T / EXP57
vision, grid n12 / 0.25, the region rule, the decision. No model, no feature: boxes come from runs/exp62's registered
files (boxes.jsonl / boxes_parents.jsonl, selector "new"; equal to the live product boxes to < 0.001 px on rl_tune).

  train   the 3,552 records of EXP62 (fit_O + fit_L 1,152, train_P 1,200, train_V 1,200), each with EXP62's new box,
          its own target and its region (devision.reading.region: tokens, views); written to data/exp64/train.jsonl.
  plan    1,000 updates of 64 = 24 parent / crop pairs (adjacent, never split) + 16 replay rows, the same generators as
          EXP62 (scripts/exp62/build.plan, seeds 0 / 1), 64,000 exposures; data/exp64/plan.jsonl.
  audit   per set and field: the predicted box against the drawn text box (variants / parents) or the field box
          (exp49 sets) -- the share fully inside, and the share of the text box covered by the selected cells in the
          full view and in the local views apart; tokens per question (full / local) and their geometry; the feature
          bytes upper bound (tokens x 768 x float16). The dev slices (rl_tune, tune_V auto / oracle, dev24 auto / EXP60
          oracle) use their registered product-path boxes. Description only, never a training input.

    uv run python scripts/exp64/build.py --root data
"""
import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(__file__)
E62, E61, E60 = "runs/exp62", "runs/exp61", "runs/exp60"
STEPS = 1000


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, rel))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


b62 = _load("exp62_build", "../exp62/build.py")


def jsonl(p: str) -> List[Dict]:
    return [json.loads(x) for x in open(p) if x.strip()]


def sha(p: str) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def e62_boxes() -> Dict[str, List[float]]:
    """EXP62's new-selector boxes of every selector question, after checking both files against EXP62's registry."""
    reg = json.load(open(os.path.join(E62, "registry.json")))["selector"]
    out = {}
    for f in ("boxes.jsonl", "boxes_parents.jsonl"):
        if sha(os.path.join(E62, f)) != reg[f]:
            raise SystemExit("runs/exp62/%s is not the registered one" % f)
        for r in jsonl(os.path.join(E62, f)):
            if r["selector"] == "new":
                out[r["key"]] = r["box"]
    return out


def train_records(root: str = "data") -> List[Dict]:
    boxes = e62_boxes()
    fit = [q for q in jsonl(os.path.join(root, "exp49", "questions.jsonl")) if q["set"] in ("fit_O", "fit_L")]
    var = [q for q in jsonl(os.path.join(root, "exp61", "questions.jsonl")) if q["set"] == "train_V"]
    par = jsonl(os.path.join(root, "exp62", "parents.jsonl"))
    out = []
    for q in fit + var + par:
        out.append({"key": q["key"], "set": q["set"], "picture": q["picture"], "width": q["width"], "height": q["height"],
                    "field": q["field"], "object": q["object"], "family": q["family"], "target": q["target"],
                    "box": boxes[q["key"]], "field_box": q["box"], "text_box": q.get("text_box")})
    if len(out) != 3552 or len({r["key"] for r in out}) != 3552:
        raise SystemExit("training records: %d" % len(out))
    return out


def coverage(box: List[float], target_xyxy: List[float], w: int, h: int) -> Dict:
    """Selected cells (devision.reading.region) of `box`: tokens in the full view / the local views, and the share of
    the target rectangle covered by the union of the selected cells' pixel extents, full and local apart."""
    from devision.image import patch_geometry, view_rects
    from devision.reading import region
    reg = region(box, w, h)
    views = [(0, 0, w, h)] + view_rects(w, h, 12, 0.25)
    tx0, ty0, tx1, ty1 = target_xyxy
    area = max(1e-9, (tx1 - tx0) * (ty1 - ty0))
    res = {}
    for name, keep in (("full", lambda k: k == 0), ("local", lambda k: k > 0)):
        rects = []
        for idx, k in zip(reg["index"], reg["view"]):
            if not keep(k):
                continue
            g = patch_geometry(views[k], w, h, 256, 16).tolist()[idx - k * 256]
            cx, cy, pw, ph = g
            rects.append((cx * w - pw * w / 2, cy * h - ph * h / 2, cx * w + pw * w / 2, cy * h + ph * h / 2))
        # covered share of the target on a 1-px grid of pixel centres (exact enough for an audit)
        import numpy as np
        xs = np.arange(int(tx0), int(tx1) + 1) + 0.5
        ys = np.arange(int(ty0), int(ty1) + 1) + 0.5
        xs, ys = xs[(xs >= tx0) & (xs <= tx1)], ys[(ys >= ty0) & (ys <= ty1)]
        cov = np.zeros((len(ys), len(xs)), bool)
        for a, b_, c, d in rects:
            cov |= ((ys >= b_) & (ys <= d))[:, None] & ((xs >= a) & (xs <= c))[None, :]
        covered = int(cov.sum())
        area = max(1, len(xs) * len(ys))
        res[name] = {"tokens": len(rects), "covered": min(1.0, covered / area)}
    x, y, bw, bh = box
    res["inside"] = bool(x <= tx0 + 1 and y <= ty0 + 1 and x + bw >= tx1 - 1 and y + bh >= ty1 - 1)
    return res


def dist(xs: List[float]) -> Dict:
    xs = sorted(xs)
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "min": round(xs[0], 3), "p10": round(xs[len(xs) // 10], 3), "median": round(statistics.median(xs), 3),
            "p90": round(xs[9 * len(xs) // 10], 3), "max": round(xs[-1], 3)}


def audit(root: str, recs: List[Dict], dev: Dict[str, List[Tuple[Dict, List[float], List[float]]]]) -> Dict:
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in recs:
        tb = r["text_box"]
        fb = r["field_box"]
        target = tb if tb else [fb[0], fb[1], fb[0] + fb[2], fb[1] + fb[3]]
        groups["train/%s/%s" % (r["set"], r["field"])].append(coverage(r["box"], target, r["width"], r["height"]))
    for name, items in dev.items():
        for meta, box, target in items:
            groups["dev/%s/%s" % (name, meta["field"])].append(coverage(box, target, meta["width"], meta["height"]))
    out = {}
    for k, v in sorted(groups.items()):
        out[k] = {"n": len(v), "box_holds_target": round(sum(x["inside"] for x in v) / len(v), 4),
                  "full_tokens": dist([x["full"]["tokens"] for x in v]), "local_tokens": dist([x["local"]["tokens"] for x in v]),
                  "full_covered": dist([x["full"]["covered"] for x in v]), "local_covered": dist([x["local"]["covered"] for x in v])}
    return out


def dev_items(root: str) -> Dict[str, List[Tuple[Dict, List[float], List[float]]]]:
    """(question meta, product-path box, target rectangle) for the dev slices, from the registered EXP62 / EXP60 reads."""
    reg62 = json.load(open(os.path.join(E62, "registry.json")))
    out: Dict[str, List] = {}
    v = {q["key"]: q for q in jsonl(os.path.join(root, "exp61", "questions.jsonl")) if q["set"] == "tune_V"}
    by_pic: Dict[str, Dict[str, Dict]] = defaultdict(dict)
    for q in v.values():
        by_pic[q["picture"]][q["text"]] = q
    for name in ("tune_V_new", "tune_V_oracle"):
        f = os.path.join(E62, name + ".jsonl")
        if sha(f) != reg62["reads_" + name][name + ".jsonl"]:
            raise SystemExit("%s is not registered" % f)
        items = []
        for x in jsonl(f)[1:]:
            for rd in x["reads"]:
                q = by_pic[x["picture"]][rd["question"]]
                t = q["text_box"]
                items.append(({"field": q["field"], "width": q["width"], "height": q["height"]}, rd["box"], t))
        out[name] = items
    tune = {r["id"]: r for r in jsonl(os.path.join(root, "real_layout", "questions_tune.jsonl"))}
    qs = {(q["id"], tune[q["id"]]["questions"][q["qid"]]["instructions"]): q for q in jsonl(os.path.join(root, "exp49", "questions.jsonl"))
          if q["set"] == "rl_tune"}
    f = os.path.join(E62, "rl_tune_new.jsonl")
    if sha(f) != reg62["reads_rl_tune_new"]["rl_tune_new.jsonl"]:
        raise SystemExit("rl_tune_new is not registered")
    items = []
    for x in jsonl(f)[1:]:
        for rd in x["reads"]:
            q = qs[(x["id"], rd["question"])]
            fb = q["box"]
            items.append(({"field": q["field"], "width": q["width"], "height": q["height"]}, rd["box"], [fb[0], fb[1], fb[0] + fb[2], fb[1] + fb[3]]))
    out["rl_tune_new"] = items
    ob = {k: v for k, v in json.load(open(os.path.join("scripts", "exp60", "oracle_boxes.json"))).items() if not k.startswith("_")}
    for name, path in (("dev24_auto", os.path.join(E62, "details_dev24_new.jsonl")), ("dev24_exp60_oracle", os.path.join(E60, "details_oracle.jsonl"))):
        items = []
        for r in jsonl(path)[1:]:
            w, h = ob[r["sample_id"]]["size"]
            for rd in r["diagnostics"]["reads"]:
                fb = ob[r["sample_id"]]["boxes"][rd["question"]]
                f_ = "mass" if "mass" in rd["question"] else "speed" if "speed" in rd["question"] else "distance"
                items.append(({"field": f_, "width": w, "height": h}, rd["box"], [fb[0], fb[1], fb[0] + fb[2], fb[1] + fb[3]]))
        out[name] = items
    return out


def build(root: str = "data") -> Dict:
    recs = train_records(root)
    par = jsonl(os.path.join(root, "exp62", "parents.jsonl"))
    replay = [q for q in jsonl(os.path.join(root, "exp49", "questions.jsonl")) if q["set"] in ("fit_O", "fit_L")]
    plan = b62.plan(sorted(q["pair"] for q in par), sorted(q["key"] for q in replay), steps=STEPS)
    st = b62.plan_stats(plan, par, replay)
    if st["split_pairs"] or st["exposures"] != STEPS * 64 or st["batch_sizes"] != {64: STEPS}:
        raise SystemExit("plan: %s" % st)
    dest = os.path.join(root, "exp64")
    os.makedirs(dest, exist_ok=True)
    with open(os.path.join(dest, "train.jsonl"), "w") as fh:
        fh.write("".join(json.dumps(r) + "\n" for r in recs))
    with open(os.path.join(dest, "plan.jsonl"), "w") as fh:
        fh.write("".join(json.dumps(b) + "\n" for b in plan))
    dev = dev_items(root)
    aud = audit(root, recs, dev)
    json.dump(aud, open(os.path.join(dest, "audit.json"), "w"), indent=1)
    from devision.reading import region
    train_tokens = sum(len(region(r["box"], r["width"], r["height"])["index"]) for r in recs)
    dev_tokens = sum(len(region(box, meta["width"], meta["height"])["index"]) for items in dev.values() for meta, box, _ in items)
    pics = {"train": len({r["picture"] for r in recs}), "rl_tune": 600, "tune_V": 600, "dev24": 24}
    m = {"records": dict(Counter(r["set"] for r in recs)), "total": len(recs), "plan": st,
         "pictures": {**pics, "total": sum(pics.values())},
         "features": {"train_tokens": train_tokens, "dev_tokens": dev_tokens, "dim": 768, "bytes_float16": 2 * 768 * (train_tokens + dev_tokens)},
         "exp62": {"registry": sha(os.path.join(E62, "registry.json")), "report": sha(os.path.join(E62, "report.json"))},
         "train_sha256": sha(os.path.join(dest, "train.jsonl")), "plan_sha256": sha(os.path.join(dest, "plan.jsonl")),
         "audit_sha256": sha(os.path.join(dest, "audit.json"))}
    json.dump(m, open(os.path.join(dest, "MANIFEST.json"), "w"), indent=1)
    return m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data")
    print(json.dumps(build(ap.parse_args(argv).root), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
