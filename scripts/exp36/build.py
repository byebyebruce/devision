"""exp36 data (critic EXP36-PREP-01 item 2; docs/experiments/2026-10-17-exp36-number-mask-plan.md): one 16,000-row
alignment list shared by both arms, so one pass is exactly 1,000 steps of 16:

  - real-layout: 2,000 temperature and 2,000 force families of data/real_layout/captions_train.jsonl (seed 36, by
    family, both views kept) = 8,000 pictures;
  - COCO: 8,000 of the 12,000 isolated COCO caption rows of exp32's A2 mix (data/exp32/align_mix.jsonl, seed 36);
  - shuffled together (seed 36). Nothing redrawn, no other source; tune / confirm / exp31 / confirmation items
    never enter (train and tune were split by canonical case when the data was made).

Also the fixed fit slice for the readout (EXP36-READOUT-01): 50 temperature and 50 force families of the chosen
ones, the lowest sha256(family id), both views adjacent (v0, v1) as in captions_tune = 200 pictures.

    uv run python scripts/exp36/build.py --root data
"""
import argparse
import hashlib
import json
import os
import random
from collections import defaultdict


def sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def build(root: str, n_fam: int = 2000, n_coco: int = 8000, n_fit: int = 50, seed: int = 36) -> dict:
    rl_path = os.path.join(root, "real_layout", "captions_train.jsonl")
    mix_path = os.path.join(root, "exp32", "align_mix.jsonl")
    rows = [json.loads(x) for x in open(rl_path)]
    fams = defaultdict(list)
    for r in rows:
        fams[(r["source"], r["family"])].append(r)
    rng = random.Random(seed)
    chosen = {}
    for src in ("real-layout-temp", "real-layout-force"):
        names = sorted(f for s, f in fams if s == src)
        chosen[src] = sorted(rng.sample(names, n_fam))
    rl = [r for src in chosen for f in chosen[src] for r in sorted(fams[(src, f)], key=lambda r: r["id"])]
    assert all(len(fams[(src, f)]) == 2 for src in chosen for f in chosen[src])
    coco = [r for r in map(json.loads, open(mix_path)) if r.get("source") == "coco-captions"]
    coco = rng.sample(coco, n_coco)
    mix = rl + coco
    rng.shuffle(mix)
    fit = []
    for src in chosen:
        for f in sorted(chosen[src], key=lambda f: hashlib.sha256(f.encode()).hexdigest())[:n_fit]:
            fit += sorted(fams[(src, f)], key=lambda r: r["id"])
    out = os.path.join(root, "exp36")
    os.makedirs(out, exist_ok=True)
    for name, data in (("align_mix.jsonl", mix), ("fit_slice.jsonl", fit)):
        with open(os.path.join(out, name), "w") as fh:
            for r in data:
                fh.write(json.dumps(r) + "\n")
    m = {"seed": seed, "sources": {"real_layout_captions_train": sha(rl_path), "exp32_align_mix": sha(mix_path)},
         "align_mix": {"rows": len(mix), "real_layout": len(rl), "coco": len(coco),
                       "families": {s: len(v) for s, v in chosen.items()}, "sha256": sha(os.path.join(out, "align_mix.jsonl"))},
         "fit_slice": {"rows": len(fit), "families": 2 * n_fit, "sha256": sha(os.path.join(out, "fit_slice.jsonl"))},
         "families_chosen": chosen}
    json.dump(m, open(os.path.join(out, "MANIFEST.json"), "w"), indent=1)
    return m


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    m = build(p.parse_args().root)
    print(json.dumps({k: v for k, v in m.items() if k != "families_chosen"}, indent=1))
