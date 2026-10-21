"""data-n2, block E: every usable Visual7W telling question (plan docs/experiments/2026-10-21-v04-plan.md).

    uv run python scripts/data/n2/v7w.py --root data [--per-picture 6]

data-n1 replayed only data-v5's 7,499 Visual7W questions (a per-type quota of the 69,817 train questions). Here: the
train and val splits (the test split stays out: data/v5/test_v7w came from it), the answer plus Visual7W's three
human-written wrong answers as a four-option choice (data-v5's convert_v7w, options shuffled), at most
`per_picture` questions per picture, pictures already on disk only (gqa/images or COCO train2014), held-out
pictures dropped by n2/held.py (ids, all of COCO val2014, dHash of every evaluation picture). Ids stay
"v5-v7w:<qa_id>" so the mix can drop the ones data-n1 already has. Writes <root>/n2/v7w.jsonl and v7w.MANIFEST.json
(counts at every step, per type, the option-only baselines).
"""
import argparse
import json
import os
import random
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))


def main(argv=None) -> None:
    import held as held_mod
    from fetch import coco_annotations, vg_to_coco
    from v5_build import cap_per_picture, coco_path, coco_train2014_ids, convert_v7w, option_baselines, picture_key
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    p.add_argument("--per-picture", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    rng = random.Random(a.seed)
    vg2coco = vg_to_coco(a.root)
    train2014 = coco_train2014_ids(coco_annotations(a.root))
    h = held_mod.load(a.root)
    steps = Counter()
    rows = []
    for im in json.load(open(os.path.join(a.root, "raw", "visual7w", "dataset_v7w_telling.json")))["images"]:
        if im["split"] not in ("train", "val"):
            continue
        vid = "vg:%d" % im["image_id"]
        coco = vg2coco.get(vid)
        image = coco_path(int(coco[5:]), train2014) if coco else "gqa/images/%d.jpg" % im["image_id"]
        steps["questions read"] += len(im["qa_pairs"])
        if not os.path.exists(os.path.join(a.root, image)):
            steps["picture not on disk"] += len(im["qa_pairs"])
            continue
        if h.is_held(vid, image) or (coco and h.id_held(coco, image)):
            steps["held-out picture"] += len(im["qa_pairs"])
            continue
        for qa in im["qa_pairs"]:
            s = convert_v7w(qa, vid, image, rng)
            if s is None:
                steps["repeated options"] += 1
                continue
            s["official_split"] = im["split"]
            rows.append(s)
    steps["converted"] = len(rows)
    key = lambda r: picture_key(r["image_id"], vg2coco)  # noqa: E731
    rows = cap_per_picture(rows, a.per_picture, key, rng)
    steps["after per-picture cap"] = len(rows)
    out = os.path.join(a.root, "n2", "v7w.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    dev = [json.loads(l) for l in open(os.path.join(a.root, "v5", "dev_v7w.jsonl"))]
    m = {"built_by": "scripts/data/n2/v7w.py", "steps": dict(steps), "questions": len(rows),
         "pictures": len({key(r) for r in rows}), "per_type": dict(Counter(r["category"] for r in rows)),
         "per_split": dict(Counter(r["official_split"] for r in rows)),
         "option_only_baselines_on_v5_dev": option_baselines(rows, dev)}
    with open(out.replace(".jsonl", ".MANIFEST.json"), "w") as f:
        json.dump(m, f, indent=1)
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
