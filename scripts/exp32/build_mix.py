"""exp32 training mixes (docs/experiments/2026-10-17-exp32-bridge-plan.md):

  - align_mix.jsonl (stage A2, masked captions): the 12,000 real-layout caption pictures (6,000 families x scene
    and value twin) + 12,000 isolated real COCO caption pictures (exp30's pool, data/exp30/real_captions.jsonl),
    shuffled -- 1 pass = 1,500 steps of 16;
  - train_mix.jsonl (stage B, decisions): r14b replay (old decision tasks, r15's own mix) 12,000 questions +
    real-layout questions of 1,200 training families (both pictures, ~9,600 questions; a family's two pictures
    share a micro batch through "group") + exp25's synthetic panel questions 2,400 (regression), shuffled.

    uv run python scripts/exp32/build_mix.py --root data
"""
import argparse
import hashlib
import json
import os
import random


def read(p):
    return [json.loads(x) for x in open(p)]


def take_questions(rows, n, rng):
    rows = list(rows)
    rng.shuffle(rows)
    out, k = [], 0
    for r in rows:
        if k >= n:
            break
        out.append(r)
        k += len(r["questions"])
    return out


def build(root: str, seed: int = 32) -> dict:
    out = os.path.join(root, "exp32")
    os.makedirs(out, exist_ok=True)
    rng = random.Random(seed)
    caps = read(os.path.join(root, "real_layout", "captions_train.jsonl"))
    real = read(os.path.join(root, "exp30", "real_captions.jsonl"))
    rng.shuffle(real)
    align = caps + real[:len(caps)]
    random.Random(seed).shuffle(align)
    qs = read(os.path.join(root, "real_layout", "questions_train.jsonl"))
    fams = sorted({r["family"] for r in qs})
    rng.shuffle(fams)
    keep = set(fams[:1200])
    layout = [r for r in qs if r["family"] in keep]
    replay = take_questions(read(os.path.join(root, "r14b", "train.jsonl")), 12000, rng)
    panels = take_questions(read(os.path.join(root, "exp25", "train.jsonl")), 2400, rng)
    train = replay + layout + panels
    random.Random(seed + 1).shuffle(train)
    manifest: dict = {"seed": seed}
    for name, rows in (("align_mix", align), ("train_mix", train)):
        path = os.path.join(out, name + ".jsonl")
        with open(path, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        manifest[name] = {"rows": len(rows), "questions": sum(len(r.get("questions", {"c": 0})) for r in rows),
                          "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}
    manifest["train_parts"] = {"r14b_replay": sum(len(r["questions"]) for r in replay),
                               "real_layout": sum(len(r["questions"]) for r in layout), "real_layout_families": len(keep),
                               "exp25_panels": sum(len(r["questions"]) for r in panels)}
    manifest["align_parts"] = {"real_layout_captions": len(caps), "coco_captions": len(caps)}
    json.dump(manifest, open(os.path.join(out, "MANIFEST.json"), "w"), indent=1)
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    print(json.dumps(build(p.parse_args().root), indent=1))
