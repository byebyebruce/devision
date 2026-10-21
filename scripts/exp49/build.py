"""exp49 question map (critic EXP49-PREP-02; docs/experiments/2026-10-18-exp49-region-readout-plan.md). Every reading
question of exp42's fit_O / fit_L (the readout heads' training set: 576 + 576, each picture x object x field once) and of
the real-layout tune sets rl_tune / rl_tune_L (1,800 + 1,800, readout only): its picture, the asked object's asked-field
box (EXP45's readout sidecar, already checked against the drawn values), the target string (the gold number with the
thousands commas removed, e.g. "1,360 m/s" -> "1360"), field, object, family, case, view, and its twin: the same family,
object and field in the other view, used as the donor (its region features with this question's label); a twin whose
number is the same is marked so it is left out of the donor comparison. Writes data/exp49/questions.jsonl, MANIFEST.json.

    uv run python scripts/exp49/build.py --root data
"""
import argparse
import hashlib
import json
import os
import re
from collections import Counter

SETS = {"fit_O": "exp42/fit_O.jsonl", "fit_L": "exp42/fit_L.jsonl", "rl_tune": "real_layout/questions_tune.jsonl",
        "rl_tune_L": "exp40/rl_tune_L.jsonl"}
NUMBER = re.compile(r"^(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]+)?$")


def sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def target_of(gold: str) -> str:
    """The gold answer's number, thousands commas removed; anything not a strict number is refused."""
    num = gold.split(" ")[0]
    if not NUMBER.match(num):
        raise SystemExit("%r is not a number answer" % gold)
    return num.replace(",", "")


def build(root: str) -> dict:
    regs = {}
    for line in open(os.path.join(root, "exp45", "regions_readout.jsonl")):
        r = json.loads(line)
        regs[(r["set"], r["id"], r["qid"])] = r
    out = []
    for name, rel in SETS.items():
        rows = [json.loads(x) for x in open(os.path.join(root, rel))]
        for r in rows:
            view = r.get("view") or r["id"].split(":")[2]
            for q, v in r["questions"].items():
                g = regs.get((name, r["id"], q))
                if g is None:
                    if any(w in v["instructions"] for w in ("mass", "speed", "distance")):
                        raise SystemExit("no region for %s %s %s" % (name, r["id"], q))
                    continue
                gold = next(o for o, p in r["gold"][q]["probabilities"].items() if p == 1.0)
                out.append({"key": "%s|%s|%s" % (name, r["id"], q), "set": name, "split": "fit" if name.startswith("fit") else "tune",
                            "layout": "L" if name.endswith("_L") else "original", "id": r["id"], "qid": q, "picture": r["image"],
                            "box": g["box"], "width": g["width"], "height": g["height"], "field": g["field"], "object": g["object"],
                            "family": r["family"], "case": r["case"], "view": view, "target": target_of(gold)})
    # uniqueness: one question per (set, picture, object, field)
    seen = Counter((q["set"], q["picture"], q["object"], q["field"]) for q in out)
    if any(n > 1 for n in seen.values()):
        raise SystemExit("a (set, picture, object, field) asked twice")
    by = {(q["set"], q["family"], q["object"], q["field"], q["view"]): q for q in out}
    for q in out:
        t = by.get((q["set"], q["family"], q["object"], q["field"], "v1" if q["view"] == "v0" else "v0"))
        q["twin"] = t["key"] if t else None
        q["twin_changed"] = bool(t and t["target"] != q["target"])
    dest = os.path.join(root, "exp49")
    os.makedirs(dest, exist_ok=True)
    path = os.path.join(dest, "questions.jsonl")
    with open(path, "w") as fh:
        for q in out:
            fh.write(json.dumps(q) + "\n")
    lens = Counter(len(q["target"]) for q in out)
    m = {"sources": {**{n: sha(os.path.join(root, p)) for n, p in SETS.items()},
                     "regions_readout": sha(os.path.join(root, "exp45", "regions_readout.jsonl"))},
         "questions": dict(Counter(q["set"] for q in out)), "target_lengths": dict(sorted(lens.items())), "max_length": max(lens),
         "twins_changed": dict(Counter(q["set"] for q in out if q["twin_changed"])),
         "twins_missing": dict(Counter(q["set"] for q in out if q["twin"] is None)),
         "pictures": len({q["picture"] for q in out}), "families": len({q["family"] for q in out}), "sha256": sha(path)}
    json.dump(m, open(os.path.join(dest, "MANIFEST.json"), "w"), indent=1)
    return m


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    print(json.dumps(build(p.parse_args().root), indent=1))
