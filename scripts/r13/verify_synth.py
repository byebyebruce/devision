"""Whole-data checks of the counterfactual sets for critic R13-I01..I03 (round 13).

    uv run python scripts/r13/verify_synth.py --root data --dir r13/synth-v2 [--out runs/r13/verify_synth_v2.json]

I01  per family: recolour / recolour2 keep the base instances exactly and change no pixel outside the edited
     kind's boxes; the count variant keeps every base instance but at most one; all pictures share one question
     set; target edits flip the own / new statements, distractor edits keep the other kind's answers.
I02  plan_batches (the training packer) on the train file, micro batch 8, seeds 0-2: counterfactual units
     (same group + question) split across micro batches, and batches used vs ceil(questions / 8).
I03  questions with a false "Is there a <colour> <shape>?" that carry a grounding box (must be 0).
"""
import argparse
import json
import os
import random
from collections import defaultdict

import numpy as np
from PIL import Image

from devision.train.rlcd import plan_batches


def rows(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def answer(g):
    return max(g["probabilities"], key=g["probabilities"].get)


def check_split(root, path):
    fams = defaultdict(dict)
    for r in rows(path):
        fams[r["group"]][r["variant"]] = r
    out = defaultdict(int)
    for fam in fams.values():
        base = fam["base"]
        out["families"] += 1
        out["question_sets_differ"] += sum(r["questions"] != base["questions"] for r in fam.values())
        a = np.asarray(Image.open(os.path.join(root, base["image"])).convert("RGB"))
        k0, k1 = (k["shape"] for k in base["scene"])
        for v, edited in (("recolour", 0), ("recolour2", 1)):
            r = fam[v]
            out["recolour_instances_moved"] += r["instances"] != base["instances"]
            b = np.asarray(Image.open(os.path.join(root, r["image"])).convert("RGB"))
            mask = np.ones(a.shape[:2], bool)
            for x in base["instances"]:
                if x["kind"] == edited:
                    x0, y0, x1, y1 = x["box"]
                    mask[y0:y1 + 1, x0:x1 + 1] = False
            out["recolour_pixels_changed_outside"] += int((a[mask] != b[mask]).any(-1).sum())
            own, other = (k0, k1) if edited == 0 else (k1, k0)
            out["target_own_flipped"] += answer(r["gold"]["own_" + own]) != answer(base["gold"]["own_" + own])
            out["target_new_flipped"] += answer(r["gold"]["new_" + own]) != answer(base["gold"]["new_" + own])
            out["target_colour_changed"] += answer(r["gold"]["colour_" + own]) != answer(base["gold"]["colour_" + own])
            for q in ("colour_", "count_", "own_", "new_"):
                out["distractor_edit_changed_other_kind"] += answer(r["gold"][q + other]) != answer(base["gold"][q + other])
            out["recolour_pairs"] += 1
        c = fam["count"]
        kept = sum(x in c["instances"] for x in base["instances"])
        out["count_variant_moved_instances"] += kept < len(base["instances"]) - 1
        changed = [q for q in base["questions"] if q.startswith("count_") and answer(c["gold"][q]) != answer(base["gold"][q])]
        out["count_variant_one_count_changed"] += len(changed) == 1
        for r in fam.values():
            for qid, m in r["meta"].items():
                if m["task"].startswith("exists") and answer(r["gold"][qid]) == "false":
                    out["false_statements"] += 1
                    out["false_statements_with_box"] += qid in r["boxes"]
    return dict(out)


def check_batches(path, seeds=(0, 1, 2), mb=8):
    items = [{"group": r["group"], "qid": q} for r in rows(path) for q in r["questions"]]
    units = defaultdict(set)
    for i, it in enumerate(items):
        units[(it["group"], it["qid"])].add(i)
    res = []
    for seed in seeds:
        batches = plan_batches(items, mb, random.Random(seed))
        where = {i: n for n, b in enumerate(batches) for i in b}
        split = sum(len({where[i] for i in idx}) > 1 for idx in units.values())
        res.append({"seed": seed, "items": len(items), "units": len(units), "units_split": split,
                    "batches": len(batches), "minimum_batches": -(-len(items) // mb),
                    "unit_sizes": sorted({len(u) for u in units.values()})})
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    p.add_argument("--dir", default="r13/synth-v2")
    p.add_argument("--out")
    a = p.parse_args()
    report = {"dir": a.dir}
    for name in ("train", "dev_bind_cf", "test_bind_cf"):
        report[name] = check_split(a.root, os.path.join(a.root, a.dir, name + ".jsonl"))
        print(name, json.dumps(report[name]), flush=True)
    report["batches_train"] = check_batches(os.path.join(a.root, a.dir, "train.jsonl"))
    print("batches", json.dumps(report["batches_train"]))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
