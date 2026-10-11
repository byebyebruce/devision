"""Round 13 hard-slice fitting diagnostic (plan section 6, proposal 2, approved by the owner 2026-10-09).

    uv run python scripts/r13/hard_slice.py --root data --families 60

Can the S2 recipe fit the hardest synthetic slice -- photo backgrounds with a kind of 3-4 objects -- when the same
questions are trained on repeatedly, and does what it fits carry over to unseen families of the same slice?

  hard_train.jsonl        whole families (4 pictures each, all questions) that S2 was trained on (the synthetic
                          part of data/r13/screen/train.jsonl): photo background and at least one kind with 3 or 4
                          instances in the base scene -- every qualifying family, up to --families
  hard_dev.jsonl          60 families of the same slice from dev_bind_cf (unseen scenes)
  hard_{train,dev}_count.jsonl   the same pictures, only the count questions whose answer is three or four
Families are chosen with seed 0 among those that qualify; nothing about the backgrounds or the generator changes.
"""
import argparse
import json
import os
import random
from collections import defaultdict


def rows(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def families(samples):
    fam = defaultdict(list)
    for s in samples:
        fam[s["group"]].append(s)
    return fam


def qualifies(fam):
    base = next(s for s in fam if s["variant"] == "base")
    return base["background"] == "photo" and any(k["count"] in (3, 4) for k in base["scene"])


def count_only(samples):
    out = []
    for s in samples:
        keep = [q for q, m in s["meta"].items() if m["task"] == "count"
                and max(s["gold"][q]["probabilities"], key=s["gold"][q]["probabilities"].get) in ("three", "four")]
        if keep:
            out.append(dict(s, id=s["id"] + ":count34", questions={q: s["questions"][q] for q in keep},
                            gold={q: s["gold"][q] for q in keep}, meta={q: s["meta"][q] for q in keep},
                            boxes={q: s["boxes"][q] for q in keep if q in s["boxes"]}))
    return out


def pick(path, n, seed=0, only=None):
    fam = families(rows(path))
    if only is not None:
        fam = {g: f for g, f in fam.items() if g in only}
    names = sorted(g for g, f in fam.items() if qualifies(f))
    chosen = random.Random(seed).sample(names, min(n, len(names)))
    return [s for g in chosen for s in fam[g]], len(names)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    p.add_argument("--families", type=int, default=60)
    p.add_argument("--out", default="data/r13/hard")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    manifest = {}
    trained = {s["group"] for s in rows(os.path.join(a.root, "r13/screen/train.jsonl")) if s.get("source") == "r13-synth"}
    n_train = None
    for split, src in (("train", "r13/synth-v2/train.jsonl"), ("dev", "r13/synth-v2/dev_bind_cf.jsonl")):
        chosen, pool = pick(os.path.join(a.root, src), n_train or a.families, only=trained if split == "train" else None)
        n_train = n_train or len({s["group"] for s in chosen})   # dev gets as many families as train
        for name, ss in (("hard_%s" % split, chosen), ("hard_%s_count" % split, count_only(chosen))):
            with open(os.path.join(a.out, name + ".jsonl"), "w") as f:
                for s in ss:
                    f.write(json.dumps(s) + "\n")
            manifest[name] = {"pictures": len(ss), "questions": sum(len(s["questions"]) for s in ss),
                              "families": len({s["group"] for s in ss}), "qualifying_pool": pool}
    json.dump(manifest, open(os.path.join(a.out, "MANIFEST.json"), "w"), indent=1)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
