"""exp65 CPU data (critic EXP65-PREP-01; docs/experiments/2026-10-19-exp65-weight-plan.md). No model.

  weights  the W arm's per-question CE weights for data/facts/train.jsonl: every temperature comparison is put in one of
           six classes from its row's four gold numbers and the generator's rule (higher m * v^2; equal when both are
           equal) -- equal, same mass, same speed, same direction, opposite / mass wins, opposite / speed wins -- and
           weighted 200 / n_class (n over the train set only), every other question 1. The temperature comparisons then
           weigh 1,200 together and the whole set 9,600, as in the C arm (all 1). data/exp65/weights.jsonl.
  diag     a new rule diagnostic: 40 canonical cases per class (seed 65), each asked twice with A / B swapped (equal cases:
           the same table with the options in another order) = 480 requests of one question. Values from the train
           set's own mass / speed values; every canonical case of data/real_layout (all splits), data/cmp and
           data/facts excluded. EXP63's G text interface: the shared temperature state text + the gold table
           (devision.reading.serialise / fmt), the generator's question and options, options ordered by
           random.Random(sha256(id)); the label by the generator's rule. A development diagnostic designed after
           EXP63's errors -- never trained on, never used to pick a checkpoint, not a confirmation set.
           data/exp65/diag.jsonl.

    uv run python scripts/exp65/build.py --root data
"""
import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
from collections import Counter
from typing import Dict, List, Tuple

HERE = os.path.dirname(__file__)
CLASSES = ("equal", "same_mass", "same_speed", "same_direction", "opposite_mass_wins", "opposite_speed_wins")
PER_CLASS, SEED = 40, 65


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, rel))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fb = _load("facts_build", "../facts/build.py")
rl = _load("real_layout", "../data/real_layout.py")


def h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def values_of(r: Dict) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """((mass A, speed A), (mass B, speed B)) from a temperature row's own gold reading answers."""
    t = {(o, q): float(v.replace(",", "")) for o, q, v in fb.gold_table(r)}
    return ((t[("Sample A", "mass of each particle")], t[("Sample A", "average particle speed")]),
            (t[("Sample B", "mass of each particle")], t[("Sample B", "average particle speed")]))


def state_values(state_text: str) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """((mass A, speed A), (mass B, speed B)) from a diagnostic row's own table lines (its only source of numbers)."""
    t = {}
    for line in state_text.split("\n"):
        if " = " in line and ": " in line:
            obj, rest = line.split(": ", 1)
            q, v = rest.rsplit(" = ", 1)
            t[(obj, q)] = float(v.replace(",", ""))
    return ((t[("Sample A", "mass of each particle")], t[("Sample A", "average particle speed")]),
            (t[("Sample B", "mass of each particle")], t[("Sample B", "average particle speed")]))


def ke(p) -> float:
    return p[0] * p[1] * p[1]


def class_of(a, b) -> str:
    (ma, va), (mb, vb) = a, b
    if ma == mb and va == vb:
        return "equal"
    if ma == mb:
        return "same_mass"
    if va == vb:
        return "same_speed"
    if (ma > mb) == (va > vb):
        return "same_direction"
    heavy, light = (a, b) if ma > mb else (b, a)
    return "opposite_mass_wins" if ke(heavy) > ke(light) else "opposite_speed_wins"


def winner(a, b) -> str:
    return "neither" if ke(a) == ke(b) else "sample A" if ke(a) > ke(b) else "sample B"


def is_temp_cmp(q: Dict) -> bool:
    return q["instructions"] not in fb.TEMPLATES


def canonical(a, b) -> Tuple:
    return tuple(sorted([tuple(a), tuple(b)]))


def weights(root: str) -> Tuple[List[Dict], Dict]:
    rows = fb.jsonl(os.path.join(root, "facts", "train.jsonl"))
    cls = {}
    for r in rows:
        if r["kind"] != "temp":
            continue
        a, b = values_of(r)
        c = class_of(a, b)
        for qid, q in r["questions"].items():
            if is_temp_cmp(q):
                g = next(o for o, p in r["gold"][qid]["probabilities"].items() if p == 1.0)
                if fb.gold_class("temp", g) != {"neither": "same"}.get(winner(a, b), winner(a, b)):
                    raise SystemExit("%s: the gold answer is not the generator's rule" % r["id"])
                cls[(r["id"], qid)] = c
    n = Counter(cls.values())
    if set(n) != set(CLASSES):
        raise SystemExit("classes in train: %s" % dict(n))
    out = []
    for r in rows:
        for qid in r["questions"]:
            k = (r["id"], qid)
            out.append({"id": r["id"], "qid": qid, "weight": 200.0 / n[cls[k]] if k in cls else 1.0, **({"class": cls[k]} if k in cls else {})})
    tot = sum(x["weight"] for x in out)
    tcmp = sum(x["weight"] for x in out if "class" in x)
    if abs(tcmp - 1200) > 1e-6 or abs(tot - 9600) > 1e-6:
        raise SystemExit("weights: temperature comparisons %.6f, all %.6f" % (tcmp, tot))
    return out, {"n_class": dict(n), "weight_class": {c: 200.0 / n[c] for c in CLASSES}, "total": tot, "temperature_comparisons": tcmp,
                 "questions": len(out)}


def existing_cases(root: str) -> set:
    seen = set()
    paths = [os.path.join(root, "real_layout", "questions_%s.jsonl" % s) for s in ("train", "tune", "confirm")] + \
            [os.path.join(root, "cmp", "train.jsonl"), os.path.join(root, "facts", "train.jsonl")]
    for p in paths:
        for r in fb.jsonl(p):
            if r.get("kind") == "temp":
                seen.add(canonical(*values_of(r)))
    return seen


def diag(root: str) -> Tuple[List[Dict], Dict]:
    rows = [r for r in fb.jsonl(os.path.join(root, "facts", "train.jsonl")) if r["kind"] == "temp"]
    masses = sorted({p[0] for r in rows for p in values_of(r)})
    speeds = sorted({p[1] for r in rows for p in values_of(r)})
    state = rows[0]["state_text"].split("\n")[0]
    if any(r["state_text"].split("\n")[0] != state for r in rows):
        raise SystemExit("temperature rows do not share one state text")
    q_text = next(q["instructions"] for q in rows[0]["questions"].values() if is_temp_cmp(q))
    opts = sorted(next(q for q in rows[0]["questions"].values() if is_temp_cmp(q))["criteria"])
    same_opt = next(o for o in opts if o.startswith("neither"))
    seen = existing_cases(root)
    rng = random.Random(SEED)
    cases: Dict[str, List[Tuple]] = {c: [] for c in CLASSES}
    picked = set()
    tries = 0
    while any(len(v) < PER_CLASS for v in cases.values()):
        tries += 1
        if tries > 2_000_000:
            raise SystemExit("not enough new cases: %s" % {c: len(v) for c, v in cases.items()})
        a = (rng.choice(masses), rng.choice(speeds))
        c = rng.choice(CLASSES)
        if c == "equal":
            b = a
        elif c == "same_mass":
            b = (a[0], rng.choice(speeds))
        elif c == "same_speed":
            b = (rng.choice(masses), a[1])
        else:
            b = (rng.choice(masses), rng.choice(speeds))
        if class_of(a, b) != c or (c != "equal" and ke(a) == ke(b)):
            continue
        key = canonical(a, b)
        if key in seen or key in picked or len(cases[c]) >= PER_CLASS:
            continue
        picked.add(key)
        cases[c].append(key)
    out = []
    for c in CLASSES:
        for k, (p1, p2) in enumerate(cases[c]):
            for order, (a, b) in enumerate(((p1, p2), (p2, p1))):
                rid = "exp65-diag:%s:%02d:%d" % (c, k, order)
                table = [["Sample A", "mass of each particle", fb.fmt("%g" % a[0])], ["Sample B", "mass of each particle", fb.fmt("%g" % b[0])],
                         ["Sample A", "average particle speed", fb.fmt("%g" % a[1])], ["Sample B", "average particle speed", fb.fmt("%g" % b[1])]]
                o = list(opts)
                random.Random(h(rid)).shuffle(o)
                g = winner(a, b)
                gold = same_opt if g == "neither" else g
                out.append({"id": rid, "kind": "temp", "axis": c, "family": "%s:%02d" % (c, k), "case": str((p1, p2)), "source": "exp65-diag",
                            "image": "", "image_id": rid, "state_text": fb.serialise(state, table),
                            "questions": {"q0": {"type": "choice", "instructions": q_text, "criteria": {x: None for x in o}}},
                            "gold": {"q0": {"probabilities": {x: float(x == gold) for x in o}}}})
    return out, {"per_class": PER_CLASS, "requests": len(out), "seed": SEED, "masses": len(masses), "speeds": len(speeds),
                 "excluded_cases": len(seen), "gold": dict(Counter(next(o for o, p in r["gold"]["q0"]["probabilities"].items() if p == 1.0)
                                                                for r in out))}


def build(root: str = "data") -> Dict:
    dest = os.path.join(root, "exp65")
    os.makedirs(dest, exist_ok=True)
    w, wm = weights(root)
    d, dm = diag(root)
    for name, rows in (("weights.jsonl", w), ("diag.jsonl", d)):
        with open(os.path.join(dest, name), "w") as fh:
            fh.write("".join(json.dumps(x) + "\n" for x in rows))
    m = {"weights": wm, "diag": dm, "facts": {"train": h(open(os.path.join(root, "facts", "train.jsonl")).read()),
                                             "PLAN": h(open(os.path.join(root, "facts", "PLAN.json")).read())},
         "sha256": {n: h(open(os.path.join(dest, n)).read()) for n in ("weights.jsonl", "diag.jsonl")}}
    json.dump(m, open(os.path.join(dest, "MANIFEST.json"), "w"), indent=1)
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data")
    print(json.dumps(build(ap.parse_args().root), indent=1))
    sys.exit(0)
