"""exp22 fit test data (docs/experiments/2026-10-16-exp22-fit-plan.md): exp21's 2,667 training families
(scripts/data/exp21_reading.py, same seed, so the same families), every one drawn at 16 px in both value views,
three questions each -- synthetic only, no replay.

    uv run python scripts/exp22/build.py --root data

Outputs in <root>/exp22/:
  - train.jsonl: 2,667 families x 2 views x 3 questions = 16,002 questions;
  - fit.jsonl: the first 200 of those families (1,200 questions), i.e. training questions: did training fit them;
  - dev16.jsonl: exp21's dev_syn at 16 px only (200 other families, 1,200 questions);
  - MANIFEST.json.
exp21's dev_syn (all heights), dev_syn_erased and dev_syn_text are reused as they are.
"""
import argparse
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
from exp21_reading import family, render, sample, save  # type: ignore  # noqa: E402

HEIGHT = 16


def build(root: str, seed: int = 21, n_train: int = 2667, n_fit: int = 200) -> dict:
    out = os.path.join(root, "exp22")
    os.makedirs(out, exist_ok=True)
    rng = random.Random(seed)
    fams = [family(rng, "t%04d" % i) for i in range(n_train)]   # exp21's build() draws these first, same rng
    train = []
    for f in fams:
        for view in (0, 1):
            rel = "exp22/images/%s_v%d_h%d.png" % (f["family"], view, HEIGHT)
            save(render(f, view, HEIGHT)[0], root, rel)
            train.append(sample(f, view, HEIGHT, rel, "exp21:%s:v%d:h%d" % (f["family"], view, HEIGHT)))
    fit = [s for s in train if int(s["family"][1:]) < n_fit]
    random.Random(1).shuffle(train)
    dev16 = [json.loads(x) for x in open(os.path.join(root, "exp21", "dev_syn.jsonl"))]
    dev16 = [s for s in dev16 if s["height"] == HEIGHT]
    files = {"train": train, "fit": fit, "dev16": dev16}
    manifest = {"seed": seed, "height": HEIGHT, "files": {}}
    for name, rows in files.items():
        path = os.path.join(out, name + ".jsonl")
        with open(path, "w") as fh:
            for s in rows:
                fh.write(json.dumps(s) + "\n")
        manifest["files"][name] = {"families": len({s["family"] for s in rows}), "pictures": len({s["image"] for s in rows}),
                                   "questions": sum(len(s["questions"]) for s in rows),
                                   "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}
    json.dump(manifest, open(os.path.join(out, "MANIFEST.json"), "w"), indent=1)
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    a = p.parse_args()
    if os.path.exists(os.path.join(a.root, "exp22", "MANIFEST.json")):
        raise SystemExit("data/exp22 exists; remove it first")
    print(json.dumps(build(a.root), indent=1))
