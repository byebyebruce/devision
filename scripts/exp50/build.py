"""exp50 data (critic EXP50-PREP-03): the validation subset -- every 4th row of data/exp42/val_sub.jsonl in file order
(244 of 973; only monitoring and the saved temperature, which the temperature-1 readout does not use; cut because each
validation question now encodes 17 images) -- and data/exp50/MANIFEST.json binding the training data, quads, plan,
geometry and the view rule. Nothing here looks at answers or models.

    uv run python scripts/exp50/build.py --root data
"""
import argparse
import hashlib
import json
import os

RULE = {"local_views": 16, "local_overlap": 0.5, "views": "devision.image.view_rects", "pixels": "devision.image.local_pixels"}


def sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def build(root: str) -> dict:
    dest = os.path.join(root, "exp50")
    rows = open(os.path.join(root, "exp42", "val_sub.jsonl")).read().splitlines()
    val = os.path.join(dest, "val_quarter.jsonl")
    with open(val, "w") as fh:
        fh.write("".join(r + "\n" for r in rows[::4]))
    geo = json.load(open(os.path.join(dest, "geometry.json")))
    chosen = geo["candidates"][geo["chosen"]]
    if (chosen["n"], chosen["overlap"]) != (RULE["local_views"], RULE["local_overlap"]):
        raise SystemExit("the geometry report chose %s, not the rule" % geo["chosen"])
    m = {"rule": RULE, "files": {"train": sha(os.path.join(root, "exp42", "train.jsonl")),
                                 "val_sub": sha(os.path.join(root, "exp42", "val_sub.jsonl")), "val_quarter": sha(val),
                                 "quads": sha(os.path.join(root, "exp48", "quads.jsonl")),
                                 "plan": sha(os.path.join(dest, "PLAN.json")), "geometry": sha(os.path.join(dest, "geometry.json"))},
         "val_quarter_rows": len(rows[::4])}
    json.dump(m, open(os.path.join(dest, "MANIFEST.json"), "w"), indent=1)
    return m


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    print(json.dumps(build(p.parse_args().root), indent=1))
