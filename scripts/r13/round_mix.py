"""Round 13 (r13-binding) training mix: the exp13-long recipe (owner adopted it as a trade-off, 2026-10-10) with
every round-13 source in full and a larger replay (plan step 3). One pass.

  synthetic r13p75 (photo backgrounds in 75% of families, seed 46)   ~40,000   whole scene families
  GQA binding train                                                   26,309   all
  VisMin (object / attribute / relation)                              29,820   all (14,910 pairs)
  CoGenT A                                                            20,000   all
  PixMo two-kind counting                                             15,000   all
  CounterCurate (mirror pairs)                                        14,974   all
  TallyQA complex                                                     14,831   all
  VAW                                                                  6,135   all
  v12 replay (data/ext/train_mix.jsonl)                               40,000   whole pictures; floors: A-OKVQA all
                                                                               (2,475), ScienceQA >= 3,000, "not
                                                                               there" (false noul, coco-exist /
                                                                               objects365) >= 10,000

    uv run python scripts/r13/round_mix.py --root data --out data/r13/round
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
from long_mix import absent, nq, rows, take  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    p.add_argument("--out", default="data/r13/round")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    rng = random.Random(a.seed)
    r = lambda rel: rows(os.path.join(a.root, rel))
    pic = lambda s: s.get("group", s["image_id"])
    parts = {"synthetic": take(r("r13/synth-r13p75/train.jsonl"), lambda s: s["group"], 40000, rng)}
    for name in ("gqa", "vismin", "cogent_a", "pixmo_two_kind", "countercurate", "tallyqa_complex", "vaw"):
        parts[name] = r("r13/%s/train.jsonl" % name)
    pool = r("ext/train_mix.jsonl")
    floors = [[s for s in pool if s.get("source") == "aokvqa"],
              take(pool, lambda s: s["image_id"], 3000, rng, lambda s: s.get("source") == "scienceqa"),
              take(pool, lambda s: s["image_id"], 10000, rng, absent)]
    used = {s["id"] for part in floors for s in part}
    rest = take([s for s in pool if s["id"] not in used], lambda s: s["image_id"], 40000 - sum(map(nq, floors)), rng)
    parts["replay"] = [s for part in floors for s in part] + rest
    mix = [s if "group" in s else dict(s, group=pic(s)) for ss in parts.values() for s in ss]
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
