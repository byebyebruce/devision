"""exp46 fixed selection (critic EXP46-PREP-02): 48 picture x field units, each with both of its reading questions (A / B
or Pair 1 / 2), 96 questions -- field (mass / speed / distance) x layout (original / L) x split (fit / tune) = 12 cells,
4 units per cell, 2 with the first object (A / Pair 1) drawn on the left and 2 on the right; only units whose two golds
differ and whose two questions share the picture and state text; in each cell and side, the lowest
sha256("<set>|<picture>|<field>") first -- never by a model output. Sources: exp42's four readout sets (EXP45-G's frozen
records are on exactly these), EXP45's readout sidecar for the boxes. Writes data/exp46/units.jsonl and MANIFEST.json
(the rule, every key, source sha256, the distribution).

    uv run python scripts/exp46/select.py --root data
"""
import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict

SETS = {"fit_O": ("exp42/fit_O.jsonl", "fit", "original"), "fit_L": ("exp42/fit_L.jsonl", "fit", "L"),
        "rl_tune": ("real_layout/questions_tune.jsonl", "tune", "original"), "rl_tune_L": ("exp40/rl_tune_L.jsonl", "tune", "L")}
FIELDS = ("mass", "speed", "distance")
PER_SIDE = 2


def sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def build(root: str) -> dict:
    regs = defaultdict(dict)
    for line in open(os.path.join(root, "exp45", "regions_readout.jsonl")):
        r = json.loads(line)
        regs[r["set"]][(r["id"], r["qid"])] = r
    units = []
    for name, (rel, split, layout) in SETS.items():
        rows = {r["id"]: r for r in (json.loads(x) for x in open(os.path.join(root, rel)))}
        by_unit = defaultdict(list)
        for (rid, q), g in regs[name].items():
            by_unit[(g["picture"], g["field"])].append((rid, q, g))
        for (pic, field), qs in by_unit.items():
            if len(qs) != 2:
                continue
            (r1, q1, g1), (r2, q2, g2) = sorted(qs, key=lambda x: x[2]["object"])          # A / Pair 1 first
            a, b = rows[r1], rows[r2]
            gold = lambda r, q: next(o for o, p in r["gold"][q]["probabilities"].items() if p == 1.0)  # noqa: E731
            if a["image"] != b["image"] or a.get("state_text", "") != b.get("state_text", "") or gold(a, q1) == gold(b, q2):
                continue
            key = "%s|%s|%s" % (name, pic, field)
            units.append({"key": key, "hash": hashlib.sha256(key.encode()).hexdigest(), "set": name, "split": split,
                          "layout": layout, "field": field, "picture": pic, "first_side": g1["side"],
                          "questions": [{"id": r1, "qid": q1, "object": g1["object"], "side": g1["side"], "gold": gold(a, q1)},
                                        {"id": r2, "qid": q2, "object": g2["object"], "side": g2["side"], "gold": gold(b, q2)}]})
    chosen, short = [], []
    for split in ("fit", "tune"):
        for layout in ("original", "L"):
            for field in FIELDS:
                for side in ("left", "right"):
                    cell = sorted((u for u in units if (u["split"], u["layout"], u["field"], u["first_side"]) == (split, layout, field, side)),
                                  key=lambda u: u["hash"])
                    if len(cell) < PER_SIDE:
                        short.append("%s/%s/%s/%s: %d" % (split, layout, field, side, len(cell)))
                    chosen += cell[:PER_SIDE]
    if short:
        raise SystemExit("cells short of units (report, do not replace): %s" % short)
    out = os.path.join(root, "exp46")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "units.jsonl")
    with open(path, "w") as fh:
        for u in chosen:
            fh.write(json.dumps(u) + "\n")
    m = {"rule": (__doc__ or "").split("\n\n")[0], "units": len(chosen), "questions": 2 * len(chosen), "keys": [u["key"] for u in chosen],
         "distribution": {"%s/%s/%s/%s" % k: n for k, n in sorted(Counter((u["split"], u["layout"], u["field"], u["first_side"]) for u in chosen).items())},
         "sources": {**{n: sha(os.path.join(root, rel)) for n, (rel, _, _) in SETS.items()},
                     "regions_readout": sha(os.path.join(root, "exp45", "regions_readout.jsonl"))},
         "pictures_sha256": {u["picture"]: sha(os.path.join(root, u["picture"])) for u in chosen}, "sha256": sha(path)}
    json.dump(m, open(os.path.join(out, "MANIFEST.json"), "w"), indent=1)
    return m


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    m = build(p.parse_args().root)
    print(json.dumps({k: v for k, v in m.items() if k not in ("keys", "pictures_sha256")}, indent=1))
