"""exp48 four-corner sidecar (critic EXP48-PREP-02 / 03; docs/experiments/2026-10-18-exp48-interaction-plan.md). From
data/exp42/train.jsonl (not modified): every (repeat, family, field, layout) quad -- the A / Pair 1 and B / Pair 2
questions of views v0 and v1, on the original (exposure 0) or L (exposure 1) picture. a = the gold of the first object
in v0, b = the other object's gold in v0; v1 must have them exchanged (first object b, other a); the four questions'
options are the same strings in the same order; a question is the same text in both views; the state text is the
same; the pictures are the stated view and layout. Each training row belongs to exactly one quad; anything else is
refused. The object is read from the question's text, never assumed from the qid. Writes data/exp48/quads.jsonl
(per row id: quad, view 0 / 1, object 0 / 1, a and b option indices) and MANIFEST.json.

The same rule gives the quads of a readout set (quads_of), used by the report for "all four right".

    uv run python scripts/exp48/build.py --root data
"""
import argparse
import hashlib
import importlib.util
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List

HERE = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("exp40_build", os.path.join(HERE, "..", "exp40", "build.py"))
assert _spec is not None and _spec.loader is not None
b40 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b40)


def sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def gold_of(r: Dict, q: str) -> str:
    ones = [o for o, p in r["gold"][q]["probabilities"].items() if p == 1.0]
    if len(ones) != 1:
        raise SystemExit("%s %s: not a single gold" % (r["id"], q))
    return ones[0]


def first_object(obj: str) -> bool:
    if obj not in ("sample A", "sample B", "Pair 1", "Pair 2"):
        raise SystemExit("unknown object %r" % obj)
    return obj in ("sample A", "Pair 1")


def quads_of(rows: List[Dict], key_of, strict: bool = True) -> Dict:
    """quad key -> {"corners": {(view, obj): (row id, qid)}, "a", "b", "options"} for the reading questions of `rows`;
    key_of(row) gives the quad key without the view. Malformed quads are refused when strict, else left out (counted)."""
    corners: Dict[tuple, Dict] = defaultdict(dict)
    info: Dict[tuple, Dict] = {}
    for r in rows:
        view = r.get("view") or r["id"].split(":")[2]
        for q, v in r["questions"].items():
            field = b40.field_of(v["instructions"])
            if field == "cmp":
                continue
            obj = 0 if first_object(b40.object_of(v["instructions"])) else 1
            k = key_of(r) + (field,)
            vi = {"v0": 0, "v1": 1}[view]
            if (vi, obj) in corners[k]:
                raise SystemExit("quad %s: corner %s repeated" % (k, (vi, obj)))
            corners[k][(vi, obj)] = (r, q)
    out, bad = {}, Counter()
    for k, cs in corners.items():
        why = ""
        if set(cs) != {(0, 0), (0, 1), (1, 0), (1, 1)}:
            why = "corners"
        else:
            rs = {c: cs[c] for c in cs}
            opts = [list(r["questions"][q]["criteria"]) for r, q in rs.values()]
            a, b = gold_of(*rs[(0, 0)]), gold_of(*rs[(0, 1)])
            if any(o != opts[0] for o in opts):
                why = "options"
            elif a == b or gold_of(*rs[(1, 0)]) != b or gold_of(*rs[(1, 1)]) != a:
                why = "no exchange"
            elif any(rs[(0, o)][0]["questions"][rs[(0, o)][1]]["instructions"] != rs[(1, o)][0]["questions"][rs[(1, o)][1]]["instructions"] for o in (0, 1)):
                why = "question text"
            elif len({r.get("state_text", "") for r, _ in rs.values()}) != 1:
                why = "state text"
            elif rs[(0, 0)][0]["image"] != rs[(0, 1)][0]["image"] or rs[(1, 0)][0]["image"] != rs[(1, 1)][0]["image"]:
                why = "pictures"
            else:
                out[k] = {"corners": {c: (r["id"], q) for c, (r, q) in rs.items()}, "a": a, "b": b, "options": opts[0]}
        if why:
            if strict:
                raise SystemExit("quad %s: %s" % (k, why))
            bad[why] += 1
    if not strict:
        return {"quads": out, "excluded": dict(bad)}
    return out


def build(root: str) -> Dict:
    src = os.path.join(root, "exp42", "train.jsonl")
    rows = [json.loads(x) for x in open(src)]
    for r in rows:
        if (r["exposure"] == 0) != r["image"].startswith("real_layout/"):
            raise SystemExit("%s: picture does not match its layout" % r["id"])
    quads = quads_of(rows, lambda r: (r["repeat"], r["family"], "O" if r["exposure"] == 0 else "L"))
    side = []
    seen = Counter()
    for k, qd in sorted(quads.items()):
        qid = "%d|%s|%s|%s" % k
        for (view, obj), (rid, q) in sorted(qd["corners"].items()):
            seen[rid] += 1
            side.append({"id": rid, "qid": q, "quad": qid, "view": view, "object": obj, "repeat": k[0], "family": k[1],
                         "layout": k[2], "field": k[3], "a_index": qd["options"].index(qd["a"]), "b_index": qd["options"].index(qd["b"])})
    if len(seen) != len(rows) or any(n != 1 for n in seen.values()):
        raise SystemExit("rows in quads: %d of %d, repeated %d" % (len(seen), len(rows), sum(n > 1 for n in seen.values())))
    out = os.path.join(root, "exp48")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "quads.jsonl")
    with open(path, "w") as fh:
        for x in side:
            fh.write(json.dumps(x) + "\n")
    m = {"source_sha256": sha(src), "rows": len(side), "quads": len(quads),
         "by": {"%s/%s" % k: n for k, n in sorted(Counter((k[2], k[3]) for k in quads).items())},
         "families": len({k[1] for k in quads}), "repeats": sorted({k[0] for k in quads}), "sha256": sha(path)}
    json.dump(m, open(os.path.join(out, "MANIFEST.json"), "w"), indent=1)
    return m


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    print(json.dumps(build(p.parse_args().root), indent=1))
