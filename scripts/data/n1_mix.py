"""data-n1 training mix: the new VQAv2 and GQA questions (n1_vqav2.py, n1_gqa.py) plus a replay of every other
ability v0.3 was trained on (docs/experiments/2026-10-20-r16-cuda-plan.md, "data-n1").

    uv run python scripts/data/n1_mix.py --root data [--gqa 300000]
    uv run python scripts/data/n1_mix.py --root data --extra n1/exist_train.jsonl --repeat scienceqa=2 ai2d=2 tqa=2 \
        --smooth 0.1 --out n1/r18_mix.jsonl                                        # round 18

Replay = the union of data-A's training mixes (v2 train_mix, v3 train_cont, v4 train_sqa_mix, v5 train_mix, v6
train_flip_mix, v9 x1_r2500), one copy of each question id, without their VQAv2 / GQA questions (n1 has every
usable one of those, with soft gold): COCO size / relation (and their mirrors) / position / existence, VG
relations, A-OKVQA, ScienceQA, AI2D, TQA, VSR, Visual7W, left/right pairs. `--gqa` caps the GQA questions (the
largest part). dev_mix.jsonl (calibration and monitoring) = n1's VQAv2 and GQA dev plus data/v5/dev_mix.jsonl
without its VQAv2 / GQA questions. Writes <root>/n1/train_mix.jsonl, dev_mix.jsonl, MIX.json.

Options (round 18): `--extra` adds more new files (e.g. n1_exist.py's existence questions); `--repeat SOURCE=N`
puts every replayed question of SOURCE in N times; `--smooth E` moves every training question whose gold is one
option with certainty to (1 - E) on it and E spread evenly over all options (label smoothing), so hard 0/1 targets
do not push the logits to extremes (r16 / r17: 0/1 existence and GQA yes/no made p > 0.99 common, and POPE wanted a
temperature near 6 while VQAv2 wanted 2). Soft gold (VQAv2 with split votes) is left as it is. dev_mix is never
smoothed or repeated.
"""
import argparse
import json
import os
import random
from collections import Counter

REPLAY_FILES = ("v2/train_mix.jsonl", "v3/train_cont.jsonl", "v4/train_sqa_mix.jsonl", "v5/train_mix.jsonl",
                "v6/train_flip_mix.jsonl", "v9/x1_r2500/train.jsonl")
SUPERSEDED = ("vqav2-yesno", "vqav2-choice", "gqa-v2", "vqav2", "gqa")


def rows(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def replay(root: str):
    seen, out = set(), []
    for name in REPLAY_FILES:
        for s in rows(os.path.join(root, name)):
            if s["source"] in SUPERSEDED or s["id"] in seen:
                continue
            seen.add(s["id"])
            out.append(s)
    return out


def smooth(sample: dict, eps: float) -> dict:
    """Label smoothing of every question whose gold puts all its mass on one option (others are left as they are)."""
    if eps <= 0:
        return sample
    gold = {}
    for qid, g in sample["gold"].items():
        p = g["probabilities"]
        if max(p.values()) == 1.0:
            p = {o: (1 - eps) * v + eps / len(p) for o, v in p.items()}
        gold[qid] = dict(g, probabilities=p)
    return dict(sample, gold=gold)


def main(argv=None) -> None:
    from prepare import _write_jsonl

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    p.add_argument("--gqa", type=int, default=300000, help="at most this many GQA questions")
    p.add_argument("--extra", nargs="*", default=[], help="more new training files under --root")
    p.add_argument("--repeat", nargs="*", default=[], help="SOURCE=N: each replayed question of SOURCE N times")
    p.add_argument("--smooth", type=float, default=0.0, help="label smoothing of one-hot gold (0: none)")
    p.add_argument("--out", default="n1/train_mix.jsonl", help="training mix, under --root")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    rng = random.Random(a.seed)
    n1 = os.path.join(a.root, "n1")
    vqa, gqa = rows(os.path.join(n1, "vqav2_train.jsonl")), rows(os.path.join(n1, "gqa_train.jsonl"))
    rng.shuffle(gqa)
    gqa = gqa[:a.gqa]
    rep = replay(a.root)
    extra = [x for f in a.extra for x in rows(os.path.join(a.root, f))]
    times = {k: int(v) for k, v in (r.split("=") for r in a.repeat)}
    rep = [x for x in rep for _ in range(times.get(x["source"], 1))]
    mix = [smooth(x, a.smooth) for x in vqa + gqa + extra + rep]
    rng.shuffle(mix)
    dev = rows(os.path.join(n1, "dev.jsonl")) + rows(os.path.join(n1, "gqa_dev.jsonl"))
    dev += [s for s in rows(os.path.join(a.root, "v5", "dev_mix.jsonl")) if s["source"] not in SUPERSEDED]
    out = os.path.join(a.root, a.out)
    info = {"train": _write_jsonl(out, mix), "extra": {f: sum(1 for _ in open(os.path.join(a.root, f))) for f in a.extra},
            "repeat": times, "smooth": a.smooth,
            "dev": _write_jsonl(os.path.join(n1, "dev_mix.jsonl"), dev),
            "parts": {"vqav2": len(vqa), "gqa": len(gqa), "replay": len(rep)},
            "replay_by_source": dict(Counter(s["source"] for s in rep).most_common()),
            "dev_by_source": dict(Counter(s["source"] for s in dev).most_common())}
    with open(out[:-len(".jsonl")] + ".MIX.json" if a.out != "n1/train_mix.jsonl" else os.path.join(n1, "MIX.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
