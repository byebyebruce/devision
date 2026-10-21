"""exp26 panel-binding alignment data (docs/experiments/2026-10-16-exp26-bind-align-plan.md): exp21-style pictures
of two labelled panels (scripts/data/exp21_reading.render: panel letter, quantity line, one number), captioned
"A: <quantity> <value> <unit>. B: <quantity> <value> <unit>." -- always A first, whichever side or row panel A is
drawn in. The alignment stage masks caption tokens, so filling "A: length [MASK] cm" needs the number of the panel
labelled A, not just a number in the picture (exp25: the model answered "A?" and "B?" with the same number).

    uv run python scripts/data/exp26_panels.py --root data

Families: seed 26; none shares with an exp21 dev family the quantity and candidate values, nor, in either value
view, the quantity / unit / A-B value pair (the exp25 rule, critic EXP25-I02). One random value view and one digit
height (6 / 8 / 11 / 16 / 20 px) per picture. Outputs in <root>/exp26/: align_train.jsonl (20,000),
align_val.jsonl (1,000 other families), images/, MANIFEST.json.
"""
import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from exp21_reading import family, fmt, render, save  # noqa: E402

HEIGHTS = (6, 8, 11, 16, 20)


def answer(s: dict, qid: str) -> str:
    probs = s["gold"][qid]["probabilities"]
    return max(probs, key=probs.get)


def pair_key(name: str, unit: str, a: int, b: int) -> tuple:
    return (name, unit) + tuple(sorted((a, b)))


def caption(f: dict, view: int) -> str:
    a, b = f["views"][view]
    return "A: %s %s %s. B: %s %s %s." % (f["name"], fmt(a), f["unit"], f["name"], fmt(b), f["unit"])


def dev_keys(root: str):
    """(quantity, candidate set) and (quantity, unit, value pair) of every exp21 dev family."""
    cands, pairs = set(), set()
    for s in map(json.loads, open(os.path.join(root, "exp21", "dev_syn.jsonl"))):
        name = s["questions"]["cmp"]["instructions"].split("greater ")[1].rstrip("?")
        cands.add((name, tuple(sorted(s["questions"]["readA"]["criteria"]))))
        a, b = answer(s, "readA"), answer(s, "readB")
        unit = a.split(" ", 1)[1]
        pairs.add(pair_key(name, unit, int(a.split()[0].replace(",", "")), int(b.split()[0].replace(",", ""))))
    return cands, pairs


def allowed(f: dict, cands, pairs) -> bool:
    opts = tuple(sorted("%s %s" % (fmt(v), f["unit"]) for v in f["candidates"]))
    return (f["name"], opts) not in cands and not any(
        pair_key(f["name"], f["unit"], *f["views"][v]) in pairs for v in (0, 1))


def build(root: str, n_train: int = 20000, n_val: int = 1000, seed: int = 26) -> dict:
    out = os.path.join(root, "exp26")
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    cands, pairs = dev_keys(root)
    rng = random.Random(seed)
    seen, manifest, k = set(), {"seed": seed, "files": {}}, 0
    for split, n in (("align_train", n_train), ("align_val", n_val)):
        rows, hs, firsts = [], Counter(), Counter()
        while len(rows) < n:
            f = family(rng, "b%05d" % k)
            k += 1
            if not allowed(f, cands, pairs):
                continue
            view, h = rng.randrange(2), rng.choice(HEIGHTS)
            img = render(f, view, h)[0]
            digest = hashlib.sha256(img.tobytes()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            rel = "exp26/images/%s_%05d.png" % (split, len(rows))
            save(img, root, rel)
            rows.append({"id": "exp26:%s:%05d" % (split, len(rows)), "source": "exp26-panels", "family": f["family"],
                         "image_id": "exp26:" + digest[:16], "image": rel, "captions": [caption(f, view)]})
            hs[str(h)] += 1
            firsts["A drawn first" if f["a_first"] else "B drawn first"] += 1
        with open(os.path.join(out, split + ".jsonl"), "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        manifest["files"][split] = {"pictures": len(rows), "families": len({r["family"] for r in rows}),
                                    "heights": dict(hs), "order": dict(firsts)}
    json.dump(manifest, open(os.path.join(out, "MANIFEST.json"), "w"), indent=1)
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    a = p.parse_args()
    if os.path.exists(os.path.join(a.root, "exp26", "MANIFEST.json")):
        raise SystemExit("data/exp26 exists; remove it first")
    print(json.dumps(build(a.root), indent=1))
