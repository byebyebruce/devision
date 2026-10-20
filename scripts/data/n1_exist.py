"""data-n1, existence part: "is there a <thing>?" from COCO train2014 boxes with no minimum size for the "yes"
questions (docs/experiments/2026-10-20-r16-cuda-plan.md, r17).

    uv run python scripts/data/n1_exist.py --root data --per-category 400

data-v2's builder (v2_coco_exist.build: yes = no inside every category, hard co-occurring negatives, negatives
dropped when a caption mentions the thing) with min_pos_area 0 instead of 1%: every annotated category of a picture
can be a "yes", as in POPE. r16 learned "no" with near certainty for small things that are there (COCO >= 0.5% and
GQA >= 1.5% were the only existence "yes" questions it saw): 67% of its POPE errors had p > 0.99. Ids are
"n1-exist:..." so they never collide with data-v2's. Writes <root>/n1/exist_train.jsonl and EXIST_MANIFEST.json
(questions per area bucket of the asked thing's largest box).
"""
import argparse
import glob
import json
import os
import random
import zipfile
from collections import Counter, defaultdict


def main(argv=None) -> None:
    from convert import same_images
    from fetch import coco_annotations, vg_to_coco
    from prepare import _write_jsonl, read_jsonl
    from v2_coco_exist import build

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    p.add_argument("--per-category", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    out_dir = os.path.join(a.root, "n1")
    with zipfile.ZipFile(coco_annotations(a.root)) as z:
        inst = json.load(z.open("annotations/instances_train2014.json"))
        caps_json = json.load(z.open("annotations/captions_train2014.json"))
    names = {c["id"]: c["name"] for c in inst["categories"]}
    anns = defaultdict(list)
    for x in inst["annotations"]:
        anns[x["image_id"]].append(x)
    captions = defaultdict(list)
    for x in caps_json["annotations"]:
        captions[x["image_id"]].append(x["caption"])
    evals = [f for f in glob.glob(os.path.join(a.root, "**", "*.jsonl"), recursive=True)
             if os.path.basename(f).startswith(("val", "pope", "eval", "dev", "test_", "bench_", "vqav2_yesno",
                                                "aokvqa", "scienceqa"))
             and os.sep + "raw" + os.sep not in f and not f.startswith(out_dir)]
    held_out = same_images({s["image_id"] for f in evals for s in read_jsonl(f)}, vg_to_coco(a.root))
    rows = build(inst["images"], anns, names, captions, "train2014", a.per_category, held_out,
                 random.Random(a.seed), min_pos_area=0.0)
    area = {im["id"]: im["width"] * im["height"] for im in inst["images"]}
    by_name = {v: k for k, v in names.items()}
    buckets = Counter()
    for r in rows:
        r["id"] = r["id"].replace("v2-exist:", "n1-exist:", 1)
        if r["gold"]["q"]["probabilities"]["true"] == 1.0:
            iid = int(r["image_id"].split(":")[1])
            big = max(x["area"] for x in anns[iid] if x["category_id"] == by_name[r["category"]]) / area[iid]
            buckets["<0.5%" if big < 0.005 else "0.5-1%" if big < 0.01 else ">=1%"] += 1
    rows = [r for r in rows if os.path.exists(os.path.join(a.root, r["image"]))]
    os.makedirs(out_dir, exist_ok=True)
    info = {"questions": _write_jsonl(os.path.join(out_dir, "exist_train.jsonl"), rows),
            "yes": sum(r["gold"]["q"]["probabilities"]["true"] == 1.0 for r in rows),
            "yes_by_largest_box_area": dict(buckets), "categories": len({r["category"] for r in rows}),
            "eval_files_held_out": len(evals)}
    with open(os.path.join(out_dir, "EXIST_MANIFEST.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
