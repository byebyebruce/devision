"""Above / below questions between two objects, from GQA (Visual Genome) scene graphs (kept out of the repo):

    uv run python scripts/data/v2_vg_relation.py --root data

The COCO generator (v2_coco_spatial.py) yields only ~1,500 balanced above/below questions: most COCO
category pairs are almost always one way round (a person above a dining table), and balancing per pair
drops them. Scene graphs name far more kinds of things, so more pairs occur both ways. Same rules as the
COCO relations: a name that occurs once in the picture, each box >= 1% of the picture, a vertical gap of
>= 5% of the picture height between the boxes, and per unordered name pair as many "above" as "below"
answers (pairs seen only one way are dropped), so the names carry no prior. Background stuff (sky,
ground, wall, ...), body parts, clothing, plural names and rare names are left out. Scene graphs label
only some objects, so "the bowl" may be one of several bowls: on COCO pictures, a name that is a COCO
category must also have exactly one COCO instance. Pictures of every evaluation / dev set are skipped under
both their COCO and Visual Genome ids; only pictures on disk are used (COCO train2014 files for the half
of Visual Genome that is COCO, GQA images otherwise). Writes data/v2/vg_relation_tb.jsonl.
"""
import argparse
import json
import os
import random
import zipfile
from collections import Counter, defaultdict
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Set, Tuple

MIN_AREA = 0.01
MIN_GAP = 0.05
MIN_NAME_PICTURES = 200      # a name must be in at least this many scene graphs
PER_PICTURE = 2              # questions per picture, so a few busy pictures do not dominate
STUFF = {"sky", "ground", "wall", "floor", "ceiling", "grass", "road", "street", "water", "field", "sand",
         "snow", "dirt", "pavement", "sidewalk", "background", "air", "cloud", "clouds", "ocean", "sea",
         "shadow", "reflection", "light", "lights", "part", "side", "edge", "line", "lines", "spot", "spots",
         "area", "picture", "photo", "image", "letter", "letters", "logo", "writing", "word", "words", "number",
         "hair", "head", "face", "eye", "eyes", "nose", "mouth", "ear", "ears", "hand", "hands", "arm", "arms",
         "leg", "legs", "foot", "feet", "finger", "fingers", "tree", "trees", "leaves", "leaf", "branch",
         "branches", "building", "buildings", "window", "windows", "fence", "pole", "hill", "hills",
         "mountain", "mountains", "trunk", "tail", "mane", "fur", "shirt", "pants", "jacket", "shorts", "hat",
         "shoe", "shoes", "jeans", "coat", "dress", "sleeve", "collar", "glasses", "top", "bottom", "front",
         "back", "this", "that", "it", "they", "item", "object", "thing", "stripe", "stripes", "design",
         "pattern", "surface", "frame", "rock", "rocks", "plant", "plants", "bush", "bushes", "weeds", "park",
         "uniform", "jersey", "skirt", "sweater", "vest", "helmet", "cap", "glove", "gloves", "belt", "sock",
         "socks", "boot", "boots", "beard", "neck", "wrist", "knee", "lady", "man", "woman", "person", "boy",
         "girl", "child", "kid", "guy", "player", "people", "men", "women", "crowd", "food", "counter", "tile",
         "tiles", "paper", "wire", "railing", "board", "post", "roof"}
SINGULAR_S = {"glass", "bus", "grass", "dress", "class", "cross", "mattress", "compass", "canvas", "tennis"}


def plural(name: str) -> bool:
    return name.endswith("s") and name not in SINGULAR_S


def named_objects(scene: dict) -> Dict[str, dict]:
    """Name -> its box, for names that occur once in the picture and are big enough."""
    area = float(scene["width"]) * float(scene["height"])
    count = Counter(o["name"] for o in scene["objects"].values())
    return {o["name"]: o for o in scene["objects"].values()
            if count[o["name"]] == 1 and o["w"] * o["h"] / area >= MIN_AREA}


def candidates(scenes: Dict[str, dict], names: Set[str], coco_counts=None) -> Iterable[Tuple[tuple, str, str, Tuple[str, str]]]:
    """((name1, name2), answer for name1, picture id, (name1, name2)) for every clear above/below pair.
    `coco_counts(vg_id)` -> {COCO category name: instances} for COCO pictures, else None."""
    for vg_id, scene in scenes.items():
        H = float(scene["height"])
        counts = coco_counts(vg_id) if coco_counts else None
        objs = {n: o for n, o in named_objects(scene).items()
                if n in names and (counts is None or counts.get(n, 1) == 1)}
        for a, b in combinations(sorted(objs), 2):
            oa, ob = objs[a], objs[b]
            if ob["y"] - (oa["y"] + oa["h"]) > MIN_GAP * H:
                yield (a, b), "above", vg_id, (a, b)
            elif oa["y"] - (ob["y"] + ob["h"]) > MIN_GAP * H:
                yield (a, b), "below", vg_id, (a, b)


def balance(cands: List[tuple], cap: int, rng: random.Random) -> List[tuple]:
    """Per name pair, as many "above" as "below" (at most `cap` each); one-way pairs are dropped."""
    groups: Dict[tuple, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for c in cands:
        groups[c[0]][c[1]].append(c)
    out = []
    for g in sorted(groups):
        if len(groups[g]) < 2:
            continue
        k = min(cap, *(len(v) for v in groups[g].values()))
        for v in groups[g].values():
            rng.shuffle(v)
            out += v[:k]
    return out


def to_sample(c: tuple, image_id: str, image: str, rng: random.Random) -> dict:
    (n1, n2), answer, vg_id, _ = c
    subj, obj, ans = n1, n2, answer
    if rng.random() < 0.5:  # ask it the other way round
        subj, obj, ans = n2, n1, "below" if answer == "above" else "above"
    if rng.random() < 0.5:
        opts = ["above", "below"] if rng.random() < 0.5 else ["below", "above"]
        q = {"type": "choice", "instructions": "Is the %s %s or %s the %s?" % (subj, opts[0], opts[1], obj),
             "criteria": {o: None for o in opts}}
        probs = {o: float(o == ans) for o in opts}
    else:
        asked = rng.choice(["above", "below"])
        q = {"type": "noul", "instructions": "Is the %s %s the %s?" % (subj, asked, obj)}
        probs = {"false": float(asked != ans), "true": float(asked == ans)}
    return {"id": "v2-vgrel:%s:tb:%s-%s" % (vg_id, n1, n2), "source": "vg-relation", "kind": "relation",
            "category": "%s / %s" % (n1, n2), "group": "tb:%s/%s" % (n1, n2), "answer_key": answer, "axis": "tb",
            "image_id": "vg:%s" % vg_id, "image": image, "questions": {"q": q}, "gold": {"q": {"probabilities": probs}}}


def build(scenes: Dict[str, dict], where, held_out: Set[str], cap: int, rng: random.Random,
          min_name_pictures: int = MIN_NAME_PICTURES, coco_counts=None) -> List[dict]:
    """`where(vg_id)` -> (image_id used for identity, relative image path) or None when not on disk."""
    usable = {k: s for k, s in scenes.items()
              if "vg:%s" % k not in held_out and where(k) and where(k)[0] not in held_out}
    seen = Counter(n for s in usable.values() for n in {o["name"] for o in s["objects"].values()})
    names = {n for n, c in seen.items() if c >= min_name_pictures and n not in STUFF and not plural(n)}
    cands = list(candidates(usable, names, coco_counts))
    rng.shuffle(cands)
    per_pic: Counter = Counter()
    spread = []
    for c in cands:            # a few per picture, before balancing so the balance holds afterwards
        if per_pic[c[2]] < PER_PICTURE:
            per_pic[c[2]] += 1
            spread.append(c)
    picked = balance(spread, cap, rng)
    rng.shuffle(picked)
    out = []
    for c in picked:
        _, path = where(c[2])
        out.append(to_sample(c, "vg:%s" % c[2], path, rng))
    return out


def main(argv=None) -> None:
    import glob

    from convert import same_images
    from fetch import coco_annotations, vg_to_coco
    from prepare import _write_jsonl, read_jsonl

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="data")
    p.add_argument("--cap", type=int, default=40, help="per name pair and answer")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    mapping = vg_to_coco(a.root)
    evals = [f for f in glob.glob(os.path.join(a.root, "*.jsonl")) + glob.glob(os.path.join(a.root, "v2", "*.jsonl"))
             + glob.glob(os.path.join(a.root, "lv_bench", "*.jsonl"))
             if os.path.basename(f).startswith(("val", "pope", "eval", "dev", "test_", "bench_", "vqav2_yesno", "aokvqa", "scienceqa"))]
    held_out = same_images({s["image_id"] for f in evals for s in read_jsonl(f)}, mapping)
    coco_train = {int(f[-16:-4]) for f in os.listdir(os.path.join(a.root, "coco", "train2014")) if f.endswith(".jpg")}
    gqa_local = {f[:-4] for f in os.listdir(os.path.join(a.root, "gqa", "images")) if f.endswith(".jpg")}

    def where(vg_id: str) -> Optional[Tuple[str, str]]:
        coco = mapping.get("vg:%s" % vg_id)
        if coco:   # COCO picture: only train2014 (val2014 holds the COCO evaluation sets)
            n = int(coco.split(":")[1])
            return (coco, "coco/train2014/COCO_train2014_%012d.jpg" % n) if n in coco_train else None
        return ("vg:%s" % vg_id, "gqa/images/%s.jpg" % vg_id) if vg_id in gqa_local else None

    with zipfile.ZipFile(coco_annotations(a.root)) as z:
        inst = json.load(z.open("annotations/instances_train2014.json"))
    cat_name = {c["id"]: c["name"] for c in inst["categories"]}
    per_image: Dict[int, Counter] = defaultdict(Counter)
    for x in inst["annotations"]:
        per_image[x["image_id"]][cat_name[x["category_id"]]] += 1

    def coco_counts(vg_id: str) -> Optional[Dict[str, int]]:
        coco = mapping.get("vg:%s" % vg_id)
        if not coco:
            return None
        counts = per_image.get(int(coco.split(":")[1]), Counter())
        return {n: counts.get(n, 0) or 1 for n in cat_name.values()}   # a category COCO did not box: trust the graph

    from fetch import GQA_SCENE_GRAPHS_URL, ensure_file
    with zipfile.ZipFile(ensure_file(GQA_SCENE_GRAPHS_URL, os.path.join(a.root, "raw", "gqa", "sceneGraphs.zip"))) as z:
        scenes = json.load(z.open("train_sceneGraphs.json"))
    rows = build(scenes, where, held_out, a.cap, random.Random(a.seed), coco_counts=coco_counts)
    path = os.path.join(a.root, "v2", "vg_relation_tb.jsonl")
    n = _write_jsonl(path, rows)
    print("%s: %d questions, %d pictures, %d name pairs, answers %s" % (
        path, n, len({r["image_id"] for r in rows}), len({r["group"] for r in rows}),
        dict(Counter(r["answer_key"] for r in rows))))


if __name__ == "__main__":
    main()
