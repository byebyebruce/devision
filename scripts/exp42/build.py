"""exp42 data (critic EXP42-PREP-01; docs/experiments/2026-10-18-exp42-objective-plan.md). One question file for
both arms (R = RLCD + CE, C = gold CE only), from exp40-M's own real-layout training rows (data/exp40/train_M.jsonl):

  - reading units (family, field) for mass / speed / distance whose two objects' golds differ in view v0, each with
    its 8 questions (A / B or Pair 1 / 2 x v0 / v1 x original / L picture; text, options, gold and pictures as in
    exp40); per field the first 48 by sha256("<family>:<field>") -- 144 units, 1,152 questions; never chosen by a
    model output; no tune / confirm row;
  - repeated 4 times in one epoch with the repeat in the group ("r<k>:<family>") and the id, so every (group, qid)
    unit is still exactly 8 questions: 4,608 presentations, 576 updates of 8;
  - fit sets for the readout: fit_O (the 576 original-picture questions, exposure 0) and fit_L (the 576 L-picture
    ones), each question once;
  - val_sub: every 4th row of data/r13b/val.jsonl (973 questions), the in-training monitor and temperature fit
    (the budget: the full 3,892-question val would take about 7 minutes per pass; the readout is at temperature 1).

Writes data/exp42/{train,fit_O,fit_L,val_sub}.jsonl and MANIFEST.json (counts, available units, sha256).

    uv run python scripts/exp42/build.py --root data
"""
import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, cast

FIELDS = ("mass", "speed", "distance")
PER_FIELD, REPEATS = 48, 4


def sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def gold_of(r: Dict) -> str:
    (g,) = r["gold"].values()
    return next(o for o, p in g["probabilities"].items() if p == 1.0)


def units(rows: List[Dict]) -> Dict[tuple, List[Dict]]:
    out: Dict[tuple, List[Dict]] = defaultdict(list)
    for r in rows:
        if r.get("source", "").startswith("real-layout"):
            (q,) = r["questions"]
            if q in FIELDS:
                out[(r["family"], q)].append(r)
    return out


def eligible(u: List[Dict]) -> bool:
    """8 questions, and the two objects' golds differ in view v0."""
    if len(u) != 8:
        return False
    return len({gold_of(r) for r in u if r["view"] == "v0" and r["exposure"] == 0}) == 2


def write(path: str, rows: List[Dict]) -> str:
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return sha(path)


def build(root: str) -> Dict:
    src = os.path.join(root, "exp40", "train_M.jsonl")
    us = units([json.loads(x) for x in open(src)])
    avail = {f: sorted((k for k, u in us.items() if k[1] == f and eligible(u)),
                       key=lambda k: hashlib.sha256(("%s:%s" % k).encode()).hexdigest()) for f in FIELDS}
    chosen = [k for f in FIELDS for k in avail[f][:PER_FIELD]]
    base = [r for k in chosen for r in sorted(us[k], key=lambda r: r["id"])]
    for r in base:
        assert (r["exposure"] == 0) == r["image"].startswith("real_layout/"), r["id"]
    train = [dict(r, id="%s:r%d" % (r["id"], k), group="r%d:%s" % (k, r["group"]), repeat=k)
             for k in range(REPEATS) for r in base]
    out = os.path.join(root, "exp42")
    os.makedirs(out, exist_ok=True)
    val = [x for i, x in enumerate(open(os.path.join(root, "r13b", "val.jsonl"))) if i % 4 == 0]
    with open(os.path.join(out, "val_sub.jsonl"), "w") as fh:
        fh.writelines(val)
    m = {"source_sha256": sha(src), "available_units": {f: len(v) for f, v in avail.items()},
         "units": [list(k) for k in chosen], "questions": len(base), "presentations": len(train),
         "group_sizes": dict(Counter(Counter((r["group"], next(iter(cast(Dict, r["questions"])))) for r in train).values())),
         "files": {"train": write(os.path.join(out, "train.jsonl"), train),
                   "fit_O": write(os.path.join(out, "fit_O.jsonl"), [r for r in base if r["exposure"] == 0]),
                   "fit_L": write(os.path.join(out, "fit_L.jsonl"), [r for r in base if r["exposure"] == 1]),
                   "val_sub": sha(os.path.join(out, "val_sub.jsonl"))},
         "val_sub_questions": len(val), "val_source_sha256": sha(os.path.join(root, "r13b", "val.jsonl"))}
    json.dump(m, open(os.path.join(out, "MANIFEST.json"), "w"), indent=1)
    return m


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    m = build(p.parse_args().root)
    print(json.dumps({k: v for k, v in m.items() if k != "units"}, indent=1))
