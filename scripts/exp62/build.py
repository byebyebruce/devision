"""exp62 paired parent / crop supervision for the selector (critic EXP62-PREP-01; docs/experiments/2026-10-19-exp62-paired-plan.md).
CPU only, no model. For the 200 train_V families of EXP61 (data/exp61, unchanged) the 400 original parent pictures are
the generator's own files (data/real_layout/images/<family>_<view>.png); every EXP61 crop must be exactly rows
[t, t + height) of its parent (t = exp61 build.crop_params, checked pixel for pixel). Each parent question takes exp45's
original_box (the EXP45 / 51 field band; magnets the wide label band) clipped to the parent canvas, and its crop
twin's box must be that box moved by -t and clipped to the crop canvas (checked). Questions, fields, objects and values
are copied from the crop question one to one (pair = crop key).

The batch plan (fixed now, seed 0): every update = 24 parent / crop pairs (48 rows, a pair never split) + 16 rows of the
old fit_O / fit_L replay; the pair pool (1,200) and the replay pool (1,152) each go through seeded permutations
(torch.Generator seeded 0, one per pool), refilled when short; 2,000 updates = 128,000 exposures (48,000 parent,
48,000 crop, 32,000 replay). Writes data/exp62/{parents.jsonl, plan.jsonl, MANIFEST.json}.

    uv run python scripts/exp62/build.py --root data
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
from collections import Counter
from typing import Dict, List

import numpy as np
from PIL import Image

HERE = os.path.dirname(__file__)
PAIRS_PER_BATCH, REPLAY_PER_BATCH, STEPS, SEED = 24, 16, 2000, 0


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, rel))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


b61 = _load("exp61_build", "../exp61/build.py")
b45 = b61.b45


def sha_file(p: str) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def parent_of(q: Dict) -> str:
    return "real_layout/images/%s_%s.png" % (q["family"], q["view"])


def parents(root: str = "data") -> List[Dict]:
    crops = [json.loads(x) for x in open(os.path.join(root, "exp61", "questions.jsonl"))]
    crops = [q for q in crops if q["set"] == "train_V"]
    sc = b61.scenes({q["family"] for q in crops})
    out, checked = [], {}
    for q in crops:
        pp = parent_of(q)
        par = Image.open(os.path.join(root, pp)).convert("RGB")
        W, H = par.size
        if q["picture"] not in checked:
            tag = q["picture"].split("/")[-1].rsplit(".", 1)[0]
            t, end = b61.crop_params(tag, par)
            crop = np.asarray(Image.open(os.path.join(root, q["picture"])).convert("RGB"))
            if end - t != crop.shape[0] or not np.array_equal(np.asarray(par)[t:end], crop):
                raise SystemExit("%s is not rows [%d, %d) of %s" % (q["picture"], t, end, pp))
            checked[q["picture"]] = t
        t = checked[q["picture"]]
        form, scene = sc[(q["family"], q["view"])]
        box, text, other = b45.original_box(form, scene, q["object"], q["field"])
        box = [max(0.0, box[0]), max(0.0, box[1]), min(float(W), box[2]), min(float(H), box[3])]
        if not b45.inside(text, box) or b45.overlaps(other, box):
            raise SystemExit("%s %s: the drawn value is not inside its parent box alone" % (pp, q["key"]))
        moved = [box[0], max(0.0, box[1] - t), box[2], min(float(q["height"]), box[3] - t)]
        cx = [q["box"][0], q["box"][1], q["box"][0] + q["box"][2], q["box"][1] + q["box"][3]]
        if any(abs(a - b) > 1e-6 for a, b in zip(moved, cx)):
            raise SystemExit("%s: crop box %s is not the parent box %s moved by -%d and clipped" % (q["key"], cx, box, t))
        out.append({**{k: q[k] for k in ("id", "qid", "text", "field", "object", "family", "case", "view", "target", "twin_changed")},
                    "key": q["key"].replace("train_V|", "train_P|", 1), "set": "train_P", "split": "train", "layout": "parent",
                    "picture": pp, "box": [box[0], box[1], box[2] - box[0], box[3] - box[1]], "width": W, "height": H,
                    "twin": q["twin"].replace("train_V|", "train_P|", 1), "pair": q["key"], "crop_top": t,
                    "text_box": [float(v) for v in text]})
    return out


def plan(pair_keys: List[str], replay_keys: List[str], steps: int = STEPS) -> List[List[str]]:
    """2,000 batches of keys: 24 pairs as (parent, crop) rows next to each other, then 16 replay rows."""
    import torch
    gp, gr = torch.Generator().manual_seed(SEED), torch.Generator().manual_seed(SEED + 1)
    qp: List[int] = []
    qr: List[int] = []
    out = []
    for _ in range(steps):
        while len(qp) < PAIRS_PER_BATCH:
            qp += torch.randperm(len(pair_keys), generator=gp).tolist()
        while len(qr) < REPLAY_PER_BATCH:
            qr += torch.randperm(len(replay_keys), generator=gr).tolist()
        pk, qp = qp[:PAIRS_PER_BATCH], qp[PAIRS_PER_BATCH:]
        rk, qr = qr[:REPLAY_PER_BATCH], qr[REPLAY_PER_BATCH:]
        rows = []
        for i in pk:
            crop = pair_keys[i]
            rows += [crop.replace("train_V|", "train_P|", 1), crop]
        out.append(rows + [replay_keys[i] for i in rk])
    return out


def plan_stats(batches: List[List[str]], parents_: List[Dict], replay: List[Dict]) -> Dict:
    flat = [k for b in batches for k in b]
    expo = Counter(flat)
    pkeys = {q["key"] for q in parents_}
    split_pairs = sum(1 for b in batches for k in b if k.startswith("train_P|") and k.replace("train_P|", "train_V|", 1) not in b)
    field = {q["key"]: q["field"] for q in parents_ + replay}
    for q in parents_:
        field[q["pair"]] = q["field"]
    fam = {q["key"]: q["family"] for q in parents_ + replay}
    for q in parents_:
        fam[q["pair"]] = q["family"]
    return {"updates": len(batches), "batch_sizes": dict(Counter(map(len, batches))), "exposures": len(flat),
            "parent": sum(1 for k in flat if k.startswith("train_P|")), "crop": sum(1 for k in flat if k.startswith("train_V|")),
            "replay": sum(1 for k in flat if k.split("|")[0] in ("fit_O", "fit_L")), "split_pairs": split_pairs,
            "parent_exposure_range": [min(expo[k] for k in pkeys), max(expo[k] for k in pkeys)],
            "replay_exposure_range": [min(expo[q["key"]] for q in replay), max(expo[q["key"]] for q in replay)],
            "every_row_seen": all(expo[q["key"]] > 0 for q in parents_ + replay),
            "by_field": dict(Counter(field[k] for k in flat)), "families_seen": len({fam[k] for k in flat})}


def build(root: str = "data") -> Dict:
    ps = parents(root)
    replay = [json.loads(x) for x in open(os.path.join(root, "exp49", "questions.jsonl"))]
    replay = [q for q in replay if q["set"] in ("fit_O", "fit_L")]
    batches = plan(sorted(q["pair"] for q in ps), sorted(q["key"] for q in replay))
    st = plan_stats(batches, ps, replay)
    if st["split_pairs"] or st["exposures"] != 128000 or st["parent"] != 48000 or st["crop"] != 48000 or st["replay"] != 32000:
        raise SystemExit("plan is not 2,000 x (24 whole pairs + 16 replay): %s" % st)
    dest = os.path.join(root, "exp62")
    os.makedirs(dest, exist_ok=True)
    with open(os.path.join(dest, "parents.jsonl"), "w") as fh:
        fh.write("".join(json.dumps(q) + "\n" for q in ps))
    with open(os.path.join(dest, "plan.jsonl"), "w") as fh:
        fh.write("".join(json.dumps(b) + "\n" for b in batches))
    used61 = {q["family"] for q in map(json.loads, open(os.path.join(root, "exp61", "questions.jsonl"))) if q["set"] == "tune_V"}
    cases61 = {q["case"] for q in map(json.loads, open(os.path.join(root, "exp61", "questions.jsonl"))) if q["set"] == "tune_V"}
    pics = sorted({q["picture"] for q in ps})
    m = {"parents": {"pictures": len(pics), "questions": len(ps), "families": len({q["family"] for q in ps}),
                     "by_field": dict(Counter(q["field"] for q in ps)), "crop_top": dict(Counter(q["crop_top"] for q in ps))},
         "isolation": {"families_in_tune_V": len({q["family"] for q in ps} & used61), "cases_in_tune_V": len({q["case"] for q in ps} & cases61),
                       "families_in_exp49": len({q["family"] for q in ps} & {q["family"] for q in map(json.loads, open(os.path.join(root, "exp49", "questions.jsonl")))})},
         "plan": st, "records": {"replay": len(replay), "train_V": len(ps), "train_P": len(ps), "total": len(replay) + 2 * len(ps)},
         "parents_sha256": sha_file(os.path.join(dest, "parents.jsonl")), "plan_sha256": sha_file(os.path.join(dest, "plan.jsonl")),
         "pictures_sha256": hashlib.sha256(json.dumps({p: sha_file(os.path.join(root, p)) for p in pics}, sort_keys=True).encode()).hexdigest(),
         "exp61": {"questions": sha_file(os.path.join(root, "exp61", "questions.jsonl")), "manifest": sha_file(os.path.join(root, "exp61", "MANIFEST.json"))}}
    if any(m["isolation"].values()):
        raise SystemExit("isolation: %s" % m["isolation"])
    json.dump(m, open(os.path.join(dest, "MANIFEST.json"), "w"), indent=1)
    return m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data")
    print(json.dumps(build(ap.parse_args(argv).root), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
