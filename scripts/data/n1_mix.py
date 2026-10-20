"""data-n1 training mix: the new VQAv2 and GQA questions (n1_vqav2.py, n1_gqa.py) plus a replay of every other
ability v0.3 was trained on (docs/experiments/2026-10-20-r16-cuda-plan.md, "data-n1").

    uv run python scripts/data/n1_mix.py --root data [--gqa 300000]

Replay = the union of data-A's training mixes (v2 train_mix, v3 train_cont, v4 train_sqa_mix, v5 train_mix, v6
train_flip_mix, v9 x1_r2500), one copy of each question id, without their VQAv2 / GQA questions (n1 has every
usable one of those, with soft gold): COCO size / relation (and their mirrors) / position / existence, VG
relations, A-OKVQA, ScienceQA, AI2D, TQA, VSR, Visual7W, left/right pairs. `--gqa` caps the GQA questions (the
largest part). dev_mix.jsonl (calibration and monitoring) = n1's VQAv2 and GQA dev plus data/v5/dev_mix.jsonl
without its VQAv2 / GQA questions. Writes <root>/n1/train_mix.jsonl, dev_mix.jsonl, MIX.json.
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


def main(argv=None) -> None:
    from prepare import _write_jsonl

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    p.add_argument("--gqa", type=int, default=300000, help="at most this many GQA questions")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    rng = random.Random(a.seed)
    n1 = os.path.join(a.root, "n1")
    vqa, gqa = rows(os.path.join(n1, "vqav2_train.jsonl")), rows(os.path.join(n1, "gqa_train.jsonl"))
    rng.shuffle(gqa)
    gqa = gqa[:a.gqa]
    rep = replay(a.root)
    mix = vqa + gqa + rep
    rng.shuffle(mix)
    dev = rows(os.path.join(n1, "dev.jsonl")) + rows(os.path.join(n1, "gqa_dev.jsonl"))
    dev += [s for s in rows(os.path.join(a.root, "v5", "dev_mix.jsonl")) if s["source"] not in SUPERSEDED]
    info = {"train": _write_jsonl(os.path.join(n1, "train_mix.jsonl"), mix),
            "dev": _write_jsonl(os.path.join(n1, "dev_mix.jsonl"), dev),
            "parts": {"vqav2": len(vqa), "gqa": len(gqa), "replay": len(rep)},
            "replay_by_source": dict(Counter(s["source"] for s in rep).most_common()),
            "dev_by_source": dict(Counter(s["source"] for s in dev).most_common())}
    with open(os.path.join(n1, "MIX.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
