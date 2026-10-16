"""exp25 data (docs/experiments/2026-10-16-exp25-read-compare-plan.md): exp22's 16 px reading questions plus
text comparison questions -- the two values given as text (state_text, as exp21's text control), the question
"Which panel shows the greater <quantity>?" on the picture with the numbers not drawn -- so comparing two numbers
is taught without having to read them.

    uv run python scripts/exp25/build.py --root data

Text-comparison families are new (seed 25, ids "c0000".."c1999"); none shares with any exp21 dev family the
quantity and candidate values, nor -- in either value view -- the quantity, unit and A / B value pair a text
comparison actually shows (critic EXP25-I02); both value views, compare question only: 4,000 questions on 4,000
pictures.
Outputs in <root>/exp25/: text_cmp.jsonl, train.jsonl (data/exp22/train.jsonl + text_cmp, shuffled with seed 1),
MANIFEST.json.
"""
import argparse
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
from exp21_reading import family, render, sample, save  # type: ignore  # noqa: E402


def answer(s: dict, qid: str) -> str:
    probs = s["gold"][qid]["probabilities"]
    return max(probs, key=probs.get)


def pair_key(instructions: str, a: str, b: str) -> tuple:
    """Quantity question + the two values with their unit, unordered."""
    return (instructions,) + tuple(sorted((a, b)))


def build(root: str, n: int = 2000, seed: int = 25) -> dict:
    out = os.path.join(root, "exp25")
    os.makedirs(out, exist_ok=True)
    dev = [json.loads(x) for x in open(os.path.join(root, "exp21", "dev_syn.jsonl"))]
    taken = {(s["questions"]["cmp"]["instructions"], tuple(sorted(s["questions"]["readA"]["criteria"]))) for s in dev}
    # what a text comparison actually gives the model is the quantity, the unit and the A / B values: no family of
    # the exp21 dev set (whose first 100 families are the held-out text control) may share them, in either order
    # and in either view (critic EXP25-I02)
    pairs = {pair_key(s["questions"]["cmp"]["instructions"], answer(s, "readA"), answer(s, "readB")) for s in dev}
    rng = random.Random(seed)
    rows = []
    while len(rows) < 2 * n:
        f = family(rng, "c%04d" % (len(rows) // 2))
        q = (f"Which panel shows the greater {f['name']}?",
             tuple(sorted("{:,} {}".format(v, f["unit"]) for v in f["candidates"])))
        if q in taken or any(pair_key(q[0], *("{:,} {}".format(v, f["unit"]) for v in f["views"][view])) in pairs
                             for view in (0, 1)):
            continue
        for view in (0, 1):
            rel = "exp25/images/%s_v%d_erased.png" % (f["family"], view)
            save(render(f, view, 6, erased=True)[0], root, rel)
            row = sample(f, view, 0, rel, "exp25:%s:v%d:erased" % (f["family"], view), qids=("cmp",), text=True)
            row["candidates"] = list(q[1])   # the family's values, for the dev-exclusion check
            rows.append(row)
    train = [json.loads(x) for x in open(os.path.join(root, "exp22", "train.jsonl"))] + rows
    random.Random(1).shuffle(train)
    manifest = {"seed": seed, "files": {}}
    for name, data in (("text_cmp", rows), ("train", train)):
        path = os.path.join(out, name + ".jsonl")
        with open(path, "w") as fh:
            for s in data:
                fh.write(json.dumps(s) + "\n")
        manifest["files"][name] = {"rows": len(data), "questions": sum(len(s["questions"]) for s in data),
                                   "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}
    json.dump(manifest, open(os.path.join(out, "MANIFEST.json"), "w"), indent=1)
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    a = p.parse_args()
    if os.path.exists(os.path.join(a.root, "exp25", "MANIFEST.json")):
        raise SystemExit("data/exp25 exists; remove it first")
    print(json.dumps(build(a.root), indent=1))
