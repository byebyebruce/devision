"""data-n2, block B: spatial relations as caption choices and yes / no questions from GQA (Visual Genome) scene graphs
(plan docs/experiments/2026-10-21-v04-plan.md).

    uv run python scripts/data/n2/spatial.py --root data [--per-relation 12000]

v0.3 scores 0.287 on What'sUp (chance 0.25): "which caption describes the image" with on / under / left / right or in
front of / behind / left / right. Our earlier spatial data asked left / right / above / below as yes / no questions
from boxes only. Here the relations come from the scene graphs' human annotations (on, under, in front of, behind)
plus left / right checked on the boxes, and every distractor must be false by the boxes:

- set "vertical": options on / under / to the left of / to the right of;
- set "depth": options in front of / behind / to the left of / to the right of;
- gold on / under / in front of / behind needs the subject's centre inside the object's horizontal extent (so
  left / right are false); in the vertical set "on" also needs the subject's centre above the object's centre and
  "under" below it;
- gold left / right needs a horizontal gap of at least `MIN_GAP` of the width and vertically overlapping boxes (so
  on / under are false); in the depth set also box bottoms within 5% of the height (about the same depth, so in
  front of / behind are false);
- both names occur once in the picture (named_objects of v2_vg_relation: unique, >= 1% of the picture, no
  background stuff / body parts / clothing / plural names), the relation is not contradicted by the graph (no
  opposite relation between the same two objects);
- as many questions per gold relation within a set (at most `per_relation`), at most 3 per picture;
- 70% caption choice (four options, shuffled, varied wording), 30% yes / no on the gold or one false distractor
  (half each).

Pictures: COCO train2014 for the half of Visual Genome that is COCO, GQA images otherwise; held out by n2/held.py.
Writes <root>/n2/spatial.jsonl and spatial.MANIFEST.json.
"""
import argparse
import json
import os
import random
import sys
import zipfile
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

REL = {"on": "on", "on top of": "on", "sitting on": "on", "standing on": "on", "lying on": "on",
       "under": "under", "below": "under", "beneath": "under", "underneath": "under",
       "in front of": "in front of", "behind": "behind",
       "to the left of": "to the left of", "to the right of": "to the right of"}
OPPOSITE = {"on": "under", "under": "on", "in front of": "behind", "behind": "in front of",
            "to the left of": "to the right of", "to the right of": "to the left of"}
SETS = {"vertical": ["on", "under", "to the left of", "to the right of"],
        "depth": ["in front of", "behind", "to the left of", "to the right of"]}
MIN_GAP = 0.05
PER_PICTURE = 3
WORDING = ["Which caption describes the image?", "Which sentence matches the picture?",
           "Which description of the image is correct?", "Which of these is true in the photo?"]


def article(name: str) -> str:
    return ("an " if name[0] in "aeiou" else "a ") + name


def caption(s: str, rel: str, o: str) -> str:
    """"A cup on a table", "An apple to the left of a bowl"."""
    first = article(s)
    return "%s %s %s" % (first[0].upper() + first[1:], rel, article(o))


def cx(b: dict) -> float:
    return b["x"] + b["w"] / 2


def cy(b: dict) -> float:
    return b["y"] + b["h"] / 2


def valid(gold: str, which: str, s: dict, o: dict, W: float, H: float) -> bool:
    """Whether the boxes make `gold` the only true option of set `which`."""
    inside_x = o["x"] <= cx(s) <= o["x"] + o["w"]
    if gold in ("on", "under", "in front of", "behind"):
        if not inside_x:
            return False
        if gold == "on":
            return cy(s) < cy(o)
        if gold == "under":
            return cy(s) > cy(o)
        return True
    gap = o["x"] - (s["x"] + s["w"]) if gold == "to the left of" else s["x"] - (o["x"] + o["w"])
    v_overlap = min(s["y"] + s["h"], o["y"] + o["h"]) - max(s["y"], o["y"]) > 0
    if gap < MIN_GAP * W or not v_overlap:
        return False
    if which == "depth":
        return abs((s["y"] + s["h"]) - (o["y"] + o["h"])) <= 0.05 * H
    return True


def relations(scene: dict, objs: Dict[str, dict]) -> List[Tuple[str, str, str]]:
    """(subject name, relation, object name) among the unique named objects, without contradictions."""
    by_id = {oid: o for oid, o in scene["objects"].items()}
    name_of = {id(o): n for n, o in objs.items()}
    found = set()
    for oid, o in by_id.items():
        if id(o) not in name_of:
            continue
        for r in o.get("relations", []):
            rel = REL.get(r["name"])
            t = by_id.get(r["object"])
            if rel and t is not None and id(t) in name_of and t is not o:
                found.add((name_of[id(o)], rel, name_of[id(t)]))
    return sorted((s, r, t) for s, r, t in found if (s, OPPOSITE[r], t) not in found and (t, r, s) not in found)


def to_sample(vg_id: str, image_id: str, image: str, which: str, s: str, gold: str, o: str,
              rng: random.Random) -> dict:
    opts = SETS[which][:]
    rng.shuffle(opts)
    base = {"id": "n2-spatial:%s:%s:%s:%s" % (vg_id, s, gold.replace(" ", "_"), o), "source": "n2-spatial",
            "kind": "spatial-%s" % which, "category": gold, "group": "%s/%s" % (s, o), "image_id": image_id,
            "image": image}
    if rng.random() < 0.7:
        caps = [caption(s, r, o) for r in opts]
        base.update(questions={"q": {"type": "choice", "instructions": rng.choice(WORDING),
                                     "criteria": {c: None for c in caps}}},
                    gold={"q": {"probabilities": {c: float(r == gold) for c, r in zip(caps, opts)}}},
                    answer_key=caption(s, gold, o))
    else:
        asked = gold if rng.random() < 0.5 else rng.choice([r for r in SETS[which] if r != gold])
        base["id"] += ":yn"
        base.update(questions={"q": {"type": "noul", "instructions": "Is the %s %s the %s?" % (s, asked, o)}},
                    gold={"q": {"probabilities": {"true": float(asked == gold), "false": float(asked != gold)}}},
                    answer_key="true" if asked == gold else "false")
    return base


def main(argv=None) -> None:
    import held as held_mod
    from fetch import vg_to_coco
    from v2_vg_relation import MIN_NAME_PICTURES, STUFF, named_objects, plural
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    p.add_argument("--per-relation", type=int, default=12000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    rng = random.Random(a.seed)
    mapping = vg_to_coco(a.root)
    h = held_mod.load(a.root)
    coco_train = {int(f[-16:-4]) for f in os.listdir(os.path.join(a.root, "coco", "train2014")) if f.endswith(".jpg")}
    gqa_local = {f[:-4] for f in os.listdir(os.path.join(a.root, "gqa", "images")) if f.endswith(".jpg")}

    def where(vg_id: str) -> Optional[Tuple[str, str]]:
        coco = mapping.get("vg:%s" % vg_id)
        if coco:
            n = int(coco.split(":")[1])
            return (coco, "coco/train2014/COCO_train2014_%012d.jpg" % n) if n in coco_train else None
        return ("vg:%s" % vg_id, "gqa/images/%s.jpg" % vg_id) if vg_id in gqa_local else None

    with zipfile.ZipFile(os.path.join(a.root, "raw", "gqa", "sceneGraphs.zip")) as z:
        scenes = json.load(z.open("train_sceneGraphs.json"))
    steps = Counter(scenes=len(scenes))
    usable = {}
    for k, sc in scenes.items():
        w = where(k)
        if w is None:
            steps["picture not on disk"] += 1
        elif h.id_held("vg:%s" % k, w[1]) or h.id_held(w[0], w[1]):
            steps["held-out picture (id)"] += 1
        else:
            usable[k] = sc
    seen = Counter(n for s in usable.values() for n in {o["name"] for o in s["objects"].values()})
    names = {n for n, c in seen.items() if c >= MIN_NAME_PICTURES and n not in STUFF and not plural(n)}
    cands = defaultdict(list)   # (set, gold) -> [(vg_id, s, o)]
    for k, sc in usable.items():
        W, H = float(sc["width"]), float(sc["height"])
        objs = {n: o for n, o in named_objects(sc).items() if n in names}
        for s, rel, o in relations(sc, objs):
            for which, opts in SETS.items():
                if rel in opts and valid(rel, which, objs[s], objs[o], W, H):
                    cands[(which, rel)].append((k, s, o))
    steps.update({"candidates %s/%s" % key: len(v) for key, v in cands.items()})
    per_pic: Counter = Counter()
    rows = []
    for which, opts in SETS.items():
        k_rel = min(a.per_relation, *(len(cands[(which, r)]) for r in opts))
        for rel in opts:
            c = cands[(which, rel)][:]
            rng.shuffle(c)
            took = 0
            for vg_id, s, o in c:
                if took == k_rel:
                    break
                if per_pic[vg_id] >= PER_PICTURE:
                    continue
                image_id, image = where(vg_id)  # type: ignore[misc]
                if h.is_held(image_id, image):
                    steps["held-out picture (dHash)"] += 1
                    continue
                per_pic[vg_id] += 1
                rows.append(to_sample(vg_id, image_id, image, which, s, rel, o, rng))
                took += 1
    rng.shuffle(rows)
    out = os.path.join(a.root, "n2", "spatial.jsonl")
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    m = {"built_by": "scripts/data/n2/spatial.py", "steps": dict(steps), "questions": len(rows),
         "pictures": len({r["image_id"] for r in rows}),
         "per_set_gold": dict(Counter("%s/%s" % (r["kind"], r["category"]) for r in rows)),
         "types": dict(Counter(r["questions"]["q"]["type"] for r in rows)),
         "noul_gold": dict(Counter(r["answer_key"] for r in rows if r["questions"]["q"]["type"] == "noul"))}
    with open(out.replace(".jsonl", ".MANIFEST.json"), "w") as f:
        json.dump(m, f, indent=1)
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
