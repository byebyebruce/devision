"""Round 13 bench (evaluation only): SugarCrepe attribute and relation subsets. Licence MIT. Plan 1.5.

    uv run python scripts/data/r13/sources/sugarcrepe.py --root data [--force]

Source: github.com/RAIVNLab/sugar-crepe data/{replace_att,swap_att,replace_rel}.json (COCO val2017 pictures with a
true caption and a hard negative caption). COCO 2017 ids are COCO 2014 ids; annotations_trainval2014 tells
which 2014 split a picture is in (for the subsets used here every picture is in val2014, which training never
uses; the MANIFEST counts both splits). Round-13 training sources also drop any picture of this bench.
Each item -> choice "Which caption describes the image?" between the two captions, options in a seeded random
order (the true caption is first upstream). `kind` = subset (replace_att / swap_att / replace_rel).
Writes data/r13/bench/sugarcrepe_att.jsonl (replace_att + swap_att) and sugarcrepe_rel.jsonl (replace_rel). This is
caption matching, not a grounding accuracy.
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from base import already_built, base_args, coco2014_split, download, fetch_coco, finish_bench, raw_dir  # noqa: E402

URL = "https://raw.githubusercontent.com/RAIVNLab/sugar-crepe/main/data/%s.json"
SETS = {"sugarcrepe_att": ("replace_att", "swap_att"), "sugarcrepe_rel": ("replace_rel",)}
QUESTION = "Which caption describes the image?"


def item(subset: str, key: str, d: dict, split: str, image_id: int, rng: random.Random) -> dict:
    pos, neg = " ".join(d["caption"].split()), " ".join(d["negative_caption"].split())
    opts = [pos, neg]
    rng.shuffle(opts)
    return {"id": "sugarcrepe:%s:%s" % (subset, key), "source": "sugarcrepe", "kind": subset,
            "image_id": "coco:%d" % image_id, "image": "coco/%s/COCO_%s_%012d.jpg" % (split, split, image_id),
            "questions": {"q": {"type": "choice", "instructions": QUESTION, "criteria": {o: None for o in opts}}},
            "gold": {"q": {"probabilities": {o: float(o == pos) for o in opts}}}, "answer_key": pos,
            "group": "coco:%d" % image_id, "coco2017_file": d["filename"]}


def main(argv=None):
    p = base_args(__doc__)
    a = p.parse_args(argv)
    split_of = coco2014_split(a.root)
    for name, subsets in SETS.items():
        if already_built(a.root, "r13/bench/%s.jsonl" % name, a.force):
            continue
        rng = random.Random(a.seed)
        rows, missing = [], 0
        for sub in subsets:
            data = json.load(open(download(URL % sub, os.path.join(raw_dir(a.root, "sugarcrepe"), sub + ".json"))))
            for key, d in data.items():
                iid = int(d["filename"].split(".")[0])
                if iid not in split_of:
                    missing += 1
                    continue
                if d["caption"].strip() == d["negative_caption"].strip():
                    continue
                rows.append(item(sub, key, d, split_of[iid], iid, rng))
        got = fetch_coco(a.root, [(r["image"].split("/")[1], int(r["image_id"][5:])) for r in rows])
        have = set(got.values())
        lost = [r for r in rows if r["image"] not in have]
        rows = [r for r in rows if r["image"] in have]
        finish_bench(a.root, name, rows, {"licence": "MIT", "subsets": list(subsets),
                                          "not_in_coco_2014": missing, "picture_missing": len(lost),
                                          "coco2014_split": {s: sum(1 for r in rows if "/%s/" % s in r["image"])
                                                             for s in ("train2014", "val2014")},
                                          "note": "caption matching; true caption first upstream, shuffled here"})


if __name__ == "__main__":
    main()
