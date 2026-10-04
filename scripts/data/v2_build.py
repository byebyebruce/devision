"""Build the whole v2 dataset reproducibly (kept out of the repo):

    uv run python scripts/data/v2_build.py --root data

The previous data/v2 is moved to data/v2.prev-<time>. Steps, in this order:
 1. evaluation sets (COCO val2014, VQAv2 val, POPE), never on a picture any v1 training file used
    (so v1 and v2 models can be compared on them): test_* (our splits) and bench_pope, renamed from eval_* at the end
    (+ pope_full_9000, all of POPE, for outside comparisons only);
 2. training pools (COCO train2014, VQAv2 train, GQA/VG), never on a picture of any evaluation set
    (old or new) or of data/align_val.jsonl (stage-1 validation);
 3. dev: ~3% of the training pictures, chosen by picture hash (so a picture is on one side in every
    file), split off into dev_*.jsonl -- for model selection and temperature fitting, apart from the
    eval_* final test;
 4. final balancing of every file after all filtering (equal answers per group, or no dominant
    answer per wording), then acceptance: unique ids, the promised balance, multiple-choice option
    sanity, no shared picture between train and dev / eval / align_val under any COCO or VG id,
    text-only baselines; a MANIFEST.json with counts, hashes and checks. Any failed check stops the build.
"""
import argparse
import datetime
import hashlib
import json
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from typing import Dict, List

import v2_coco_exist
import v2_coco_spatial
import v2_gqa
import v2_vqa_choice
import v2_vqa_yesno
from convert import picture as picture_of, same_images
from fetch import download_coco_images, vg_to_coco
from prepare import read_jsonl
from v2_common import check, equalize, file_stats, flatten_top, in_dev, write

V1_TRAIN = ["train_100k.jsonl", "train_cocoqa_mix.jsonl", "train.jsonl", "cocoqa_train.jsonl",
            "align_full_train.jsonl", "align_train.jsonl", "match_train.jsonl", "train_small.jsonl"]
ALIGN_VAL = "align_val.jsonl"
DEV_SHARE = 0.03
# Thin slices get extra dev pictures (taken whole, so they leave every training file): (file, which rows, at least)
DEV_TARGETS = [("coco_size", lambda r: True, 300),
               ("coco_relation", lambda r: r.get("axis") == "tb", 150),
               ("coco_relation", lambda r: r.get("axis") == "lr", 150),
               ("gqa", lambda r: r.get("kind") == "gqa", 300)] + \
              [("vqa_choice", (lambda k: lambda r: r.get("kind") == k)(k), 40)
               for k in ["sport", "fruit", "vegetable", "meat", "animal", "room", "vehicle", "weather", "material",
                         "activity", "food"]]
# file stem -> how its final rows are balanced
BALANCE = {"coco_exist": "equal", "coco_position": "equal", "coco_relation": "equal", "coco_size": "equal",
           "vqa_yesno": "equal", "vqa_choice": "flat", "gqa": "flat"}


def finalize(rows: List[dict], how: str) -> List[dict]:
    if how.startswith("equal"):
        rows = equalize(rows)
    if how == "flat":
        rows = flatten_top(rows)
    return rows


def text_baseline(train: List[dict], evals: List[dict], rule: str = "all_files") -> Dict[str, dict]:
    """Accuracy without the image, overall and per kind. Two rules, kept apart because they differ:
      all_files    statistics from every v2 training file; among the options pick the one most often
                   right for the same wording (lower-cased, trimmed), break ties by how often each option
                   is right anywhere, then score remaining ties as a uniform pick;
      review_2026_10_02  statistics from the matching training file only (eval_x / dev_x -> x); a seen
                   wording picks its most frequent answer among the options with ties scored as a
                   uniform pick, an unseen wording falls back to the file's overall answer frequency."""
    norm = lambda t: t.lower().strip()
    by_text: Dict[str, Counter] = defaultdict(Counter)
    overall: Counter = Counter()
    for s in train:
        g = s["gold"]["q"]["probabilities"]
        a = max(g, key=g.get)
        by_text[norm(s["questions"]["q"]["instructions"])][a] += 1
        overall[a] += 1
    acc: Dict[str, List[float]] = defaultdict(list)
    chance: Dict[str, List[float]] = defaultdict(list)
    seen: Dict[str, List[int]] = defaultdict(list)
    for s in evals:
        g = s["gold"]["q"]["probabilities"]
        gold, opts = max(g, key=g.get), list(g)
        c = by_text.get(norm(s["questions"]["q"]["instructions"]))
        if rule == "all_files":
            score = {o: ((c or {}).get(o, 0), overall.get(o, 0)) for o in opts}
        else:
            score = {o: (c.get(o, 0) if c else overall.get(o, 0)) for o in opts}
        best = max(score.values())
        tied = [o for o in opts if score[o] == best]
        for key in ("all", s.get("kind", "")):
            acc[key].append(1 / len(tied) if gold in tied else 0.0)
            chance[key].append(1 / len(opts))
            seen[key].append(int(c is not None))
    return {k: {"n": len(v), "text_only": round(sum(v) / len(v), 4), "chance": round(sum(chance[k]) / len(v), 4),
                "wording_seen": round(sum(seen[k]) / len(v), 4)} for k, v in acc.items()}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    root, out = a.root, os.path.join(a.root, "v2")
    path = lambda *x: os.path.join(*x)
    if os.path.exists(out):
        prev = out + ".prev-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        shutil.move(out, prev)
        print("previous data/v2 moved to", prev)
    os.makedirs(out)
    v1_train = [path(root, f) for f in V1_TRAIN if os.path.exists(path(root, f))]
    mapping = vg_to_coco(root)
    with open(path(out, "image_identity.json"), "w") as fh:   # for devision-pipeline's set roles
        json.dump(mapping, fh)
    pic = lambda i: picture_of(i, mapping)

    # 1. evaluation sets
    v2_coco_exist.main(["--root", root, "--split", "val2014", "--per-category", "7",
                        "--out", path(out, "eval_coco_exist.jsonl"), "--exclude"] + v1_train)
    v2_coco_spatial.main(["--root", root, "--split", "val2014", "--out-dir", out, "--prefix", "eval_",
                          "--cap-position", "4", "--cap-relation", "3", "--cap-size", "3", "--exclude"] + v1_train)
    v2_vqa_choice.main(["--root", root, "--split", "Val", "--limit", "1500", "--stratify", "150",
                        "--out", path(out, "eval_vqa_choice.jsonl"), "--exclude"] + v1_train)
    v2_vqa_yesno.main(["--root", root, "--split", "Val", "--limit", "1000",
                       "--out", path(out, "eval_vqa_yesno.jsonl"), "--exclude"] + v1_train)
    v2_gqa.main(["--root", root, "--split", "val", "--limit", "800", "--spatial-share", "0.25",
                 "--out", path(out, "eval_gqa.jsonl"), "--exclude"] + v1_train)
    from convert import convert_pope
    from fetch import pope_records
    from v2_evalsets import POPE_SHARDS
    pope = []
    for shard in POPE_SHARDS:
        for rec in pope_records(root, shard):
            s = convert_pope(rec)
            s["source"] = "pope-" + rec["category"]
            pope.append(s)
    v1_pictures = same_images({s["image_id"] for f in v1_train for s in read_jsonl(f)}, mapping)
    write(path(out, "pope_full_9000.jsonl"), pope)
    write(path(out, "eval_pope.jsonl"), [s for s in pope if s["image_id"] not in v1_pictures])
    eval_files = sorted(path(out, f) for f in os.listdir(out) if f.startswith(("eval_", "pope")))
    for f in eval_files:
        download_coco_images(root, read_jsonl(f))

    # 2. training pools
    excl = ["--exclude", path(root, ALIGN_VAL)]
    v2_coco_exist.main(["--root", root, "--per-category", "300", "--out", path(out, "coco_exist.jsonl")] + excl)
    v2_coco_spatial.main(["--root", root, "--out-dir", out] + excl)
    v2_vqa_yesno.main(["--root", root, "--limit", "60000", "--out", path(out, "vqa_yesno.jsonl")] + excl)
    v2_vqa_choice.main(["--root", root, "--limit", "40000", "--out", path(out, "vqa_choice.jsonl")] + excl)
    v2_gqa.main(["--root", root, "--limit", "40000", "--out", path(out, "gqa.jsonl")] + excl)

    # 3 + 4. dev split, final balancing (eval sets too), acceptance
    problems: List[str] = []
    rng = random.Random(a.seed)
    vocabs = v2_vqa_choice.vocabularies()
    pools = {stem: read_jsonl(path(out, stem + ".jsonl")) for stem in BALANCE}
    dev_pics = {pic(r["image_id"]) for rows in pools.values() for r in rows if in_dev(pic(r["image_id"]), DEV_SHARE)}
    for stem, which, target in DEV_TARGETS:
        have = sum(1 for r in pools[stem] if which(r) and pic(r["image_id"]) in dev_pics)
        for r in sorted((r for r in pools[stem] if which(r)), key=lambda r: hashlib.sha1(pic(r["image_id"]).encode()).hexdigest()):
            if have >= target:
                break
            if pic(r["image_id"]) not in dev_pics:
                dev_pics.add(pic(r["image_id"]))
                have += sum(1 for x in pools[stem] if which(x) and pic(x["image_id"]) == pic(r["image_id"]))
    for stem, how in BALANCE.items():
        rows = pools[stem]
        dev = [r for r in rows if pic(r["image_id"]) in dev_pics]
        train = [r for r in rows if pic(r["image_id"]) not in dev_pics]
        rng.shuffle(dev)
        write(path(out, stem + ".jsonl"), finalize(train, how))
        write(path(out, "dev_%s.jsonl" % stem), finalize(dev, how))
    # one dev file for training's --val (monitoring, checkpoint choice, temperature fitting)
    dev_mix = [r for stem in BALANCE for r in read_jsonl(path(out, "dev_%s.jsonl" % stem))]
    rng.shuffle(dev_mix)
    write(path(out, "dev_mix.jsonl"), dev_mix)
    for f in eval_files:
        stem = os.path.basename(f)[5:-6] if os.path.basename(f).startswith("eval_") else None
        if stem in BALANCE:
            write(f, finalize(read_jsonl(f), BALANCE[stem]))

    files = sorted(path(out, f) for f in os.listdir(out) if f.endswith(".jsonl"))
    train_files = [f for f in files if not os.path.basename(f).startswith(("eval_", "dev_", "pope"))]
    held_files = [f for f in files if f not in train_files]
    manifest = {"built": datetime.datetime.now().isoformat(timespec="seconds"), "seed": a.seed,
                "dev_share": DEV_SHARE, "files": {}, "text_baselines": {}, "problems": []}
    for f in files:
        name = os.path.basename(f)
        rows = read_jsonl(f)
        stem = name.replace("eval_", "").replace("dev_", "")[:-6]
        problems += check(rows, name, BALANCE.get(stem, "none"))
        if stem == "vqa_choice":
            bad = [(r["id"], v2_vqa_choice.problems(r, vocabs)) for r in rows if v2_vqa_choice.problems(r, vocabs)]
            if bad:
                problems.append("%s: %d bad multiple-choice samples, e.g. %s" % (name, len(bad), bad[:2]))
        manifest["files"][name] = file_stats(f)
    # no picture shared between training and dev / eval / align_val / old eval files, under any id
    held = set().union(*({pic(s["image_id"]) for s in read_jsonl(f)} for f in held_files))
    held |= {pic(s["image_id"]) for s in read_jsonl(path(root, ALIGN_VAL))}
    for f in [path(root, x) for x in os.listdir(root) if x.startswith(("val", "pope")) and x.endswith(".jsonl")]:
        held |= {pic(s["image_id"]) for s in read_jsonl(f)}
    for f in train_files:
        n = len({pic(s["image_id"]) for s in read_jsonl(f)} & held)
        if n:
            problems.append("%s: %d pictures also in dev / eval / align_val" % (os.path.basename(f), n))
    v1_pics = {pic(i) for i in v1_pictures}
    # the final test sets must also be new to every v1 model (dev is only for v2 runs' selection / calibration)
    for f in [x for x in held_files if os.path.basename(x).startswith("eval_")]:
        n = len({pic(s["image_id"]) for s in read_jsonl(f)} & v1_pics)
        if n:
            problems.append("%s: %d pictures used by v1 training" % (os.path.basename(f), n))
    # text-only baselines: each eval / dev file against all training rows
    train_rows = [s for f in train_files for s in read_jsonl(f)]
    for f in held_files:
        name = os.path.basename(f)
        rows = read_jsonl(f)
        stem = name.replace("eval_", "").replace("dev_", "")[:-6]
        same = path(out, stem + ".jsonl")
        manifest["text_baselines"][name] = {
            "all_files": text_baseline(train_rows, rows, "all_files"),
            "review_2026_10_02": text_baseline(read_jsonl(same), rows, "review") if os.path.exists(same) and same in train_files else None}
    manifest["problems"] = problems
    with open(path(out, "MANIFEST.json"), "w") as fh:
        json.dump(manifest, fh, indent=1, ensure_ascii=False)
    for name, st in manifest["files"].items():
        print("%-28s %6d q %6d img  %s" % (name, st["questions"], st["images"], st["sha256"][:12]))
    print("text-only baselines (all):", {k: {r: (v[r] or {}).get("all") for r in v} for k, v in manifest["text_baselines"].items()})
    if problems:
        print("PROBLEMS:\n  " + "\n  ".join(problems))
        sys.exit(1)
    print("all checks passed; manifest:", path(out, "MANIFEST.json"))
    # final names: our held-out splits test_*, external benchmarks bench_* (scripts/rename_eval_sets.py,
    # which also renames the MANIFEST keys); built as eval_* above because the steps match files by stem
    import sys
    import tempfile
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    import rename_eval_sets
    with tempfile.TemporaryDirectory() as empty:
        rename_eval_sets.main(["--root", root, "--runs", empty, "--repo", empty, "--apply"])


if __name__ == "__main__":
    main()
