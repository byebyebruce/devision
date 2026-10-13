"""Round 14 (r14-count) training mix: the blocks that passed the exp14 / exp15 quick checks, scaled up, with a large
replay of v12's own data (docs/experiments/2026-10-12-r14-count.md).

    uv run python scripts/r14/build.py --root data      # -> data/r14/train.jsonl + MANIFEST.json

  new (passed)  balanced 0-10 counting  12,000  v12 pool real-photo counting not in the replay, 4,000 per answer
                                                bucket 0-2 / 3-5 / 6-10, half every candidate 0..10, half a run of 3-5
                PixMo-Points            12,000  existence 6,000 (yes = no) + counting 6,000 with runs of 3-5 only
                                                (exp14: the 0..10 format on PixMo, whose counts are 0 or >= 4, hurt)
  replay        existence                10,000  v12 pool, yes = no inside every object category
                counting, original       6,000  v12 pool, its own option format
                science                  all     ScienceQA / AI2D / TQA of the v12 pool (7,875)
                general                 10,000  as the v12 retrain (small sources in full)
                v12-ext                 24,000  2,400 per source (twice the v12 retrain's)
Binding and CoSyn did not pass their checks and are left out.
"""
import argparse
import json
import os
import random
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "r13b"))
import build as r13b  # noqa: E402
import importlib.util  # noqa: E402


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


exp14 = load("exp14_build", os.path.join(HERE, "..", "exp14", "build.py"))
exp15 = load("exp15_build", os.path.join(HERE, "..", "exp15", "build.py"))


def build_blocks(root, ext_per=2400):
    """The r14 blocks (rng seed 14); `ext_per` questions per v12-ext source. Returns (pool, rng, blocks, info)."""
    rng = random.Random(14)
    pool = r13b.rows(os.path.join(root, "ext/train_mix.jsonl"))
    first = lambda s: next(iter(zip(s["questions"].values(), s["gold"].values())))
    is_count = lambda s: s.get("source") in r13b.OLD_COUNT and r13b.is_count(*first(s))

    blocks = {}
    blocks["existence"], ex = r13b.build_existence(pool, rng, 10000)
    blocks["count_original"] = r13b.sample_rows([s for s in pool if is_count(s)], 6000, rng)
    blocks["science"] = sorted((s for s in pool if s.get("source") in r13b.SCIENCE), key=lambda s: s["id"])
    used = {s["id"] for s in blocks["existence"] + blocks["count_original"]}
    gen_ok = lambda s: s["id"] not in used and not is_count(s) and \
        not (s.get("source") == "objects365" and first(s)[0]["type"] == "noul")
    gen = [s for s in pool if s.get("source") in r13b.GENERAL_FULL and gen_ok(s)]
    rest = max(0, 10000 - len(gen))
    for src in r13b.GENERAL_SHARED:
        gen += r13b.sample_rows([s for s in pool if s.get("source") == src and gen_ok(s)], rest // len(r13b.GENERAL_SHARED), rng)
    blocks["general"] = gen
    blocks["ext"] = [s for src in r13b.EXT for s in r13b.take([s for s in pool if s.get("source") == src],
                                                               lambda s: s["image_id"], ext_per, rng)]

    # balanced 0-10 counting (exp15-count's rule, 4,000 per bucket)
    cand = [s for s in pool if s["id"] not in used and is_count(s) and max(first(s)[1]["probabilities"].values()) == 1.0]
    by = {}
    for s in cand:
        by.setdefault(r13b.bucket(s), []).append(s)
    count_new = []
    for b in ("0-2", "3-5", "6-10"):
        if len(by[b]) < 4000:
            raise SystemExit("bucket %s: %d < 4000" % (b, len(by[b])))
        for i, s in enumerate(r13b.sample_rows(by[b], 4000, rng)):
            q, g = first(s)
            n = r13b.number(r13b.top(g))
            opts = [str(m) for m in range(11)] if i % 2 == 0 else exp15.narrow_run(n, rng)
            rng.shuffle(opts)
            k = next(iter(s["questions"]))
            count_new.append(dict(s, id=s["id"] + ":r14", candidates="0..10" if len(opts) == 11 else "run",
                                  questions={k: dict(q, criteria={o: None for o in opts})},
                                  gold={k: {"probabilities": {o: float(o == str(n)) for o in opts}}}))
    blocks["count_balanced"] = count_new

    # PixMo-Points: existence + counting with runs only, on exp14's training pictures (dev panel pictures excluded)
    split = exp14.pixmo(root, rng)
    tr = split["train"]
    blocks["pixmo"] = [exp14.count_q(*c, fmt="narrow", rng=rng) for c in exp14.pick(tr["count"], 6000, rng, 3)] + \
        [exp14.exist_q(*e[:4], e[4]) for e in exp14.pick(tr["exist_yes"], 3000, rng, 2) + exp14.pick(tr["exist_no"], 3000, rng, 2)]
    dev_imgs = {json.loads(l)["image_id"] for f in ("pixmo_count_wide", "pixmo_exist")
                for l in open(os.path.join(root, "exp14", f + ".jsonl"))}
    assert not dev_imgs & {s["image_id"] for s in blocks["pixmo"]}, "an exp14 dev picture in training"
    return pool, rng, blocks, {"existence_categories": ex, "count_balanced": count_new}


REPLAY = ("existence", "count_original", "science", "general", "ext")


def write_mix(out, blocks, info, replay=REPLAY, rules=("count_balanced", "count_original", "pixmo")):
    mix = []
    for name, ss in blocks.items():
        for s in ss:
            s = dict(s, block=name)
            mix.append(s if name == "pixmo" else r13b.as_picture_group(s))
    ids = Counter(s["id"] for s in mix)
    assert all(c == 1 for c in ids.values()), "duplicate ids"
    random.Random(1).shuffle(mix)
    os.makedirs(out, exist_ok=True)
    r13b.write(os.path.join(out, "train.jsonl"), mix)
    yes = sum(1 for s in mix for k, q in s["questions"].items() if q["type"] == "noul" and r13b.top(s["gold"][k]) == "true")
    noul = sum(1 for s in mix for q in s["questions"].values() if q["type"] == "noul")
    m = {"questions": r13b.nq(mix), "steps_at_micro_batch_8": r13b.nq(mix) // 8,
         "blocks": {k: r13b.nq(v) for k, v in blocks.items()},
         "replay_share": round(sum(r13b.nq(blocks[k]) for k in replay)
                               / r13b.nq(mix), 3),
         "noul_yes_share": round(yes / noul, 4), "existence_categories": info["existence_categories"],
         "option_rules": {k: r13b.option_rules(blocks[k]) for k in rules},
         "count_balanced_gold": dict(sorted(Counter(r13b.bucket(s) for s in info["count_balanced"]).items()))}
    json.dump(m, open(os.path.join(out, "MANIFEST.json"), "w"), indent=1)
    print(json.dumps(m, indent=1))
    return mix


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    p.add_argument("--out", default="data/r14")
    a = p.parse_args()
    _, _, blocks, info = build_blocks(a.root)
    write_mix(a.out, blocks, info)


if __name__ == "__main__":
    main()
