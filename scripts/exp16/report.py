"""exp16 readout (docs/experiments/2026-10-13-exp16-seedpair.md, rules written before training): r14b - r14 at step
2000 on every monitor panel, per seed (seed 1 = exp16, seed 0 = r14-count / r14b-mirror) and pooled over the seeds,
plus each seed-1 arm against step 0 by r14b's frozen stop rules.

    uv run python scripts/exp16/report.py      # -> runs/exp16/report.json, text on stdout
"""
import json
import os
import subprocess
from collections import defaultdict

import numpy as np

from devision.train.compare import load_records, paired_metric

PANELS = ("count_narrow", "flip", "vsr", "count_wide", "exist", "science", "bind_gqa", "ext", "px_narrow", "px_exist")
PAIRS = {1: ("runs/exp16-r14-s1/stage2", "runs/exp16-r14b-s1/stage2"),
         0: ("runs/r14-count/stage2", "runs/r14b-mirror/stage2")}


def details(stage, panel, step=2000):
    p = os.path.join(stage, "monitor", "step%d" % step, panel + ".details.jsonl")
    return load_records(p) if os.path.exists(p) else None


def pooled(pairs, resamples=2000, seed=0):
    """Mean of (b - a) correctness over every question of every seed pair; pictures resampled, a picture carrying
    all of its questions in every seed. pairs: list of (a_records, b_records) on the same questions."""
    by_pic = defaultdict(lambda: [0.0, 0])
    for a, b in pairs:
        ia = {(r["sample_id"], r["qid"]): r for r in a}
        if len(ia) != len(b):
            raise ValueError("the two arms do not hold the same questions")
        for r in b:
            x = ia[(r["sample_id"], r["qid"])]
            if x["image_id"] != r["image_id"]:
                raise ValueError("picture mismatch on %s" % r["sample_id"])
            by_pic[r["image_id"]][0] += float(r["correct"]) - float(x["correct"])
            by_pic[r["image_id"]][1] += 1
    d = np.array([v[0] for v in by_pic.values()])
    n = np.array([v[1] for v in by_pic.values()])
    w = np.random.default_rng(seed).multinomial(len(d), [1 / len(d)] * len(d), size=resamples)
    boot = (w @ d) / (w @ n)
    return {"difference": float(d.sum() / n.sum()), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "questions": int(n.sum()), "pictures": len(d), "seeds": len(pairs)}


def main():
    out = {"panels": {}, "stop_rules": {}}
    for panel in PANELS:
        row = {}
        got = []
        for seed, (a_dir, b_dir) in PAIRS.items():
            a, b = details(a_dir, panel), details(b_dir, panel)
            if a is None or b is None:
                continue
            row["seed%d" % seed] = {"accuracy": paired_metric(a, b, "accuracy"), "nll": paired_metric(a, b, "nll")}
            got.append((a, b))
        if got:
            row["pooled"] = pooled(got)
        out["panels"][panel] = row
        line = "%-13s" % panel
        for k in ("seed1", "seed0"):
            if k in row:
                m = row[k]["accuracy"]
                line += "  %s %+.1f [%+.1f, %+.1f]" % (k, 100 * m["difference"], 100 * m["ci95"][0], 100 * m["ci95"][1])
        if "pooled" in row:
            m = row["pooled"]
            line += "  pooled(%d) %+.1f [%+.1f, %+.1f]" % (m["seeds"], 100 * m["difference"], 100 * m["ci95"][0], 100 * m["ci95"][1])
        print(line)
    print()
    for arm in ("r14", "r14b"):
        stage = "runs/exp16-%s-s1/stage2" % arm
        rc = subprocess.call(["uv", "run", "python", "scripts/r13b/judge.py", "--rules", "scripts/r14b/rules.json",
                              "monitor", stage])
        out["stop_rules"][arm] = {"stop": rc == 3, "exit": rc}
        print("seed-1 %s arm, r14b's frozen stop rules at step 2000: %s" % (arm, "STOP" if rc == 3 else "no stop"))
    os.makedirs("runs/exp16", exist_ok=True)
    json.dump(out, open("runs/exp16/report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
