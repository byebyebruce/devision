"""exp13-long training mix (plan section 6, proposal 1; owner approved 2026-10-09). About 86,000 questions, one pass:

  synthetic v2p75 (photo backgrounds in 75% of families)  24,000   whole scene families
  GQA binding train                                        16,000   whole pictures
  VisMin (object / attribute / relation)                   12,000   whole pairs (original + edit)
  TallyQA complex                                           8,000
  VAW                                                       all (6,135)
  PixMo two-kind counting                                   4,000
  v12 replay (data/ext/train_mix.jsonl)                    16,000   whole pictures; floors: A-OKVQA >= 1,500,
                                                                     ScienceQA >= 1,500, "not there" (a false
                                                                     noul from coco-exist / objects365) >= 2,000,
                                                                     the rest drawn at random

    uv run python scripts/r13/long_mix.py --root data --out data/r13/long
"""
import argparse
import json
import os
import random
from collections import defaultdict


def rows(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def nq(ss):
    return sum(len(s["questions"]) for s in ss)


def take(samples, key, target, rng, pred=None):
    groups = defaultdict(list)
    for s in samples:
        if pred is None or pred(s):
            groups[key(s)].append(s)
    names = sorted(groups)
    rng.shuffle(names)
    out = []
    for g in names:
        if nq(out) >= target:
            break
        out += groups[g]
    return out


def absent(s):
    return s.get("source") in ("coco-exist", "objects365") and any(
        q["type"] == "noul" and s["gold"][k]["probabilities"].get("false", 0) > 0.5 for k, q in s["questions"].items())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    p.add_argument("--out", default="data/r13/long")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    rng = random.Random(a.seed)
    r = lambda rel: rows(os.path.join(a.root, rel))
    pic = lambda s: s.get("group", s["image_id"])
    parts = {
        "synthetic": take(r("r13/synth-v2p75/train.jsonl"), lambda s: s["group"], 24000, rng),
        "gqa": take(r("r13/gqa/train.jsonl"), pic, 16000, rng),
        "vismin": take(r("r13/vismin/train.jsonl"), pic, 12000, rng),
        "tallyqa_complex": take(r("r13/tallyqa_complex/train.jsonl"), pic, 8000, rng),
        "vaw": r("r13/vaw/train.jsonl"),
        "pixmo_two_kind": take(r("r13/pixmo_two_kind/train.jsonl"), pic, 4000, rng),
    }
    replay_pool = r("ext/train_mix.jsonl")
    floors = [take(replay_pool, lambda s: s["image_id"], 1500, rng, lambda s: s.get("source") == "aokvqa"),
              take(replay_pool, lambda s: s["image_id"], 1500, rng, lambda s: s.get("source") == "scienceqa"),
              take(replay_pool, lambda s: s["image_id"], 2000, rng, absent)]
    used = {s["id"] for part in floors for s in part}
    rest = take([s for s in replay_pool if s["id"] not in used], lambda s: s["image_id"], 16000 - sum(map(nq, floors)), rng)
    parts["replay"] = [s for part in floors for s in part] + rest
    mix = []
    for name, ss in parts.items():
        for s in ss:
            mix.append(dict(s, group=pic(s)) if "group" not in s else s)
    ids = [s["id"] for s in mix]
    assert len(ids) == len(set(ids)), "duplicate sample ids"
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "train.jsonl"), "w") as f:
        for s in mix:
            f.write(json.dumps(s) + "\n")
    total = nq(mix)
    manifest = {"questions": total, "pictures": len(mix), "seed": a.seed,
                "parts": {k: {"questions": nq(v), "share": round(nq(v) / total, 3)} for k, v in parts.items()},
                "replay_floors": {"aokvqa": nq(floors[0]), "scienceqa": nq(floors[1]), "absent_samples": nq(floors[2])}}
    json.dump(manifest, open(os.path.join(a.out, "MANIFEST.json"), "w"), indent=1)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
