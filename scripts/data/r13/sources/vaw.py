"""Round 13 source: VAW (Visual Attributes in the Wild; positive / negative / unknown attributes of object
instances on Visual Genome pictures). Licence: Adobe research licence (non-commercial; this project is research
only, plan 2026-10-08). Plan 1.4.

    uv run python scripts/data/r13/sources/vaw.py --root data [--limit 20000] [--force]

Source: github.com/adobe-research/vaw_dataset data/train_part1.json + train_part2.json (216,790 instances;
each has `positive_attributes` and `negative_attributes`; anything else is unknown).

The instance must be the only thing of its name in the picture, checked three ways:
  1. GQA's cleaned Visual Genome scene graph (train split only: GQA val pictures are where round 13's sealed
     test_bind_gqa comes from) has exactly one object with the instance's name;
  2. that object's box and the VAW instance box overlap (IoU >= 0.5), so both annotations point at the same
     thing (when they do not, another one of that name exists or the name means something else);
  3. when the picture is a COCO picture and the name maps to a COCO category, COCO boxes exactly one
     (non-crowd) instance of it.
Pictures without a GQA train scene graph are dropped. Group / plural names ("trees", "people", "clouds") are not
used (only pluralia tantum such as "pants", "jeans"); nor sky / grass / ground (the colour is a prior). The
instance box must cover >= 1.5% of the picture.

Questions (category = colour or material; only basic values, see COLORS / MATERIALS):
  - choice "What color is the <name>?" / "What material is the <name> made of?": correct = a positive value of
    the instance; distractors = the instance's **explicit negative** values of the same category (never an
    unknown value, never another positive, never a value confusable with the answer: gray / silver, brown /
    tan, ...), values that another object of the same picture has (VAW positives or GQA attributes) first;
    1-3 distractors;
  - noul "Is the <name> <value>?": yes from a positive value, no from an explicit negative value (dropped when
    the negative is confusable with one of the instance's positives); yes = no per category.
An instance without a value of a category is never asked about it; a negative is about that instance only, never
"the picture has no <value> <name>". At most one question per category of an instance and 3 per picture.
Another VAW instance of the same name elsewhere in the picture (box IoU < 0.5; overlapping ones are the same object
annotated twice) also makes the name not unique. Every sample has
`boxes` {"q": [[x, y, w, h]]} (the instance, pixels of the picture file) and `image_size`.
"""
import json
import os
import random
import sys
import zipfile
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from base import (Identity, already_built, base_args, choice, coco_instances, contact_sheet, describe,  # noqa: E402
                  download, fetch_vg, finish, noul, raw_dir, spot_dir, vg_image_data)
from tallyqa_complex import COCO_WORDS  # noqa: E402

SOURCE = "vaw"
URL = "https://raw.githubusercontent.com/adobe-research/vaw_dataset/main/data/"
COLORS = ["white", "black", "green", "blue", "brown", "red", "gray", "yellow", "silver", "orange", "pink", "tan",
          "purple", "beige", "golden"]
MATERIALS = {"wooden": "wooden", "metal": "metal", "plastic": "plastic", "glass": "glass", "brick": "brick",
             "concrete": "concrete", "stone": "stone", "leather": "leather", "ceramic": "ceramic", "paper": "paper",
             "marble": "marble", "wicker": "wicker", "rubber": "rubber", "steel": "metal",
             "stainless steel": "metal", "iron": "metal", "aluminum": "metal", "porcelain": "ceramic",
             "cardboard": "cardboard", "denim": "denim"}
CONFUSABLE = [{"gray", "silver"}, {"brown", "tan"}, {"tan", "beige"}, {"beige", "white"}, {"yellow", "golden"},
              {"golden", "brown"}, {"golden", "orange"}, {"orange", "red"}, {"pink", "red"}, {"purple", "pink"},
              {"silver", "white"}, {"concrete", "stone"}, {"stone", "marble"}, {"stone", "brick"},
              {"paper", "cardboard"}, {"leather", "rubber"}, {"plastic", "rubber"}, {"ceramic", "glass"},
              {"ceramic", "marble"}, {"denim", "leather"}]
PLURALIA = {"pants", "jeans", "shorts", "glasses", "sunglasses", "goggles", "scissors", "trousers", "leggings",
            "headphones", "skis"}
SKIP_NAMES = {"sky", "grass", "ground", "background", "air", "water"}
MIN_AREA = 0.015


def confusable(a: str, b: str) -> bool:
    return any(a in g and b in g for g in CONFUSABLE)


def values(attrs, cat: str) -> List[str]:
    """The instance's values of category `cat` (basic colour names / material families), in order, unique."""
    if cat == "color":
        return list(dict.fromkeys(a for a in attrs if a in COLORS))
    return list(dict.fromkeys(MATERIALS[a] for a in attrs if a in MATERIALS))


def usable_name(name: str) -> bool:
    n = name.lower().strip()
    if not n or n in SKIP_NAMES or not all(c.isalpha() or c == " " for c in n):
        return False
    last = n.split()[-1]
    if last in PLURALIA:
        return True
    return not (last.endswith("s") and not last.endswith(("ss", "us", "is")))


def be(name: str) -> str:
    return "are" if name.split()[-1] in PLURALIA else "is"


def iou(a, b) -> float:
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def unique_instance(inst: dict, graph: Optional[dict], coco_counts: Optional[Counter]) -> Optional[str]:
    """None when the instance is the only one of its name (see the module doc), else the reason it is not."""
    if graph is None:
        return "picture has no GQA train scene graph"
    name = inst["object_name"]
    same = [o for o in graph["objects"].values() if o["name"] == name]
    if len(same) != 1:
        return "name not unique in the scene graph (%s)" % ("0" if not same else ">1")
    o = same[0]
    if iou(inst["instance_bbox"], [o["x"], o["y"], o["w"], o["h"]]) < 0.5:
        return "VAW box and scene-graph box disagree (IoU < 0.5)"
    cat = COCO_WORDS.get(name)
    if coco_counts is not None and cat is not None and coco_counts.get(cat, 0) != 1:
        return "COCO boxes a number other than 1 of the name's category"
    return None


def questions(inst: dict, others: Dict[str, Set[str]], rng: random.Random) -> List[dict]:
    """Candidate question specs for one instance: [{"cat", "type", "answer", "options"|None, "value"}]."""
    out = []
    pos_all, neg_all = inst["positive_attributes"], inst["negative_attributes"]
    for cat in ("color", "material"):
        pos, neg = values(pos_all, cat), values(neg_all, cat)
        neg = [v for v in neg if v not in pos]
        if pos and neg:
            gold = pos[0] if len(pos) == 1 else rng.choice(pos)
            ds = [v for v in neg if not confusable(v, gold) and not any(confusable(v, p) for p in pos)]
            if ds:
                ds.sort(key=lambda v: (v not in others[cat], rng.random()))
                out.append({"cat": cat, "type": "choice", "answer": gold, "options": [gold] + ds[:3]})
        for v in pos:
            out.append({"cat": cat, "type": "noul", "answer": True, "value": v})
        for v in neg:
            if not any(confusable(v, p) for p in pos):
                out.append({"cat": cat, "type": "noul", "answer": False, "value": v})
    return out


def text(spec: dict, name: str) -> str:
    if spec["type"] == "choice":
        return ("What color %s the %s?" % (be(name), name) if spec["cat"] == "color"
                else "What material %s the %s made of?" % (be(name), name))
    return "%s the %s %s?" % (be(name).capitalize(), name, spec["value"])


def main(argv=None):
    p = base_args(__doc__)
    p.add_argument("--limit", type=int, default=20000)
    p.add_argument("--per-picture", type=int, default=3)
    p.add_argument("--workers", type=int, default=16)
    a = p.parse_args(argv)
    if already_built(a.root, "r13/%s/train.jsonl" % SOURCE, a.force):
        return
    rng = random.Random(a.seed)
    raw = raw_dir(a.root, SOURCE)
    insts = []
    for part in ("train_part1.json", "train_part2.json"):
        insts += json.load(open(download(URL + part, os.path.join(raw, part))))
    from fetch import GQA_SCENE_GRAPHS_URL, ensure_file  # pyright: ignore[reportMissingImports]
    z = zipfile.ZipFile(ensure_file(GQA_SCENE_GRAPHS_URL, os.path.join(a.root, "raw", "gqa", "sceneGraphs.zip")))
    graphs = json.load(z.open("train_sceneGraphs.json"))
    gqa_val = set(json.load(z.open("val_sceneGraphs.json")))
    ident = Identity(a.root)
    coco = coco_instances(a.root)["train2014"]
    vgdata = vg_image_data(a.root)
    steps, dropped = Counter(rows_read=len(insts)), Counter()

    by_image: Dict[str, List[dict]] = defaultdict(list)
    for x in insts:
        by_image[x["image_id"]].append(x)
    steps["pictures_read"] = len(by_image)
    cands: Dict[str, List[tuple]] = {}
    for img, xs in by_image.items():
        iid = "vg:%s" % img
        if img in gqa_val:
            dropped["instances on GQA val pictures (sealed test_bind_gqa pool)"] += len(xs)
            continue
        if ident.held(iid):
            dropped["instances on held-out pictures"] += len(xs)
            continue
        g = graphs.get(img)
        c = ident.canonical(iid)
        counts = None
        if c.startswith("coco:") and int(c[5:]) in coco["images"]:
            counts = Counter(n for n, _, crowd in coco["boxes"].get(int(c[5:]), []) if not crowd)
        others: Dict[str, Set[str]] = {"color": set(), "material": set()}
        for x in xs:
            for cat in others:
                others[cat].update(values(x["positive_attributes"], cat))
        if g:
            for o in g["objects"].values():
                for cat in others:
                    others[cat].update(values(o.get("attributes", []), cat))
        meta = vgdata.get(int(img))
        area = (meta["width"] * meta["height"]) if meta else None
        out = []
        for x in xs:
            if not usable_name(x["object_name"]):
                dropped["name is a group / plural / background word"] += 1
                continue
            if any(y is not x and y["object_name"] == x["object_name"]
                   and iou(y["instance_bbox"], x["instance_bbox"]) < 0.5 for y in xs):
                dropped["another VAW instance of the name elsewhere in the picture"] += 1
                continue
            why = unique_instance(x, g, counts)
            if why:
                dropped[why] += 1
                continue
            b = x["instance_bbox"]
            if not area or b[2] * b[3] < MIN_AREA * area:
                dropped["instance covers < 1.5% of the picture"] += 1
                continue
            specs = questions(x, {k: v - set(values(x["positive_attributes"], k)) for k, v in others.items()}, rng)
            if not specs:
                dropped["no usable colour / material value (positive with explicit negatives)"] += 1
                continue
            out.append((x, specs))
        if out:
            cands[img] = out
    steps["unique_instances_with_questions"] = sum(len(v) for v in cands.values())
    steps["pictures_with_candidates"] = len(cands)

    # one question per instance, at most --per-picture per picture. Questions that need an explicit negative
    # (choice, "no") are the scarce ones: an instance that has them asks one of them (choice or "no" at even odds);
    # an instance with positives only is a "yes" candidate, later cut to as many as there are "no" per category.
    order = sorted(cands, key=lambda i: _h(i))
    picked = []
    for img in order:
        xs = cands[img]
        rng.shuffle(xs)
        xs.sort(key=lambda t: not any(sp["type"] == "choice" or not sp["answer"] for sp in t[1]))
        n_pic = 0
        for x, all_specs in xs:
            for cat in ("color", "material"):        # one question per category of an instance
                specs = [sp for sp in all_specs if sp["cat"] == cat]
                if not specs or n_pic >= a.per_picture:
                    continue
                scarce = [sp for sp in specs if sp["type"] == "choice" or not sp["answer"]]
                ch = [sp for sp in scarce if sp["type"] == "choice"]
                no = [sp for sp in scarce if sp["type"] == "noul"]
                if ch and (not no or rng.random() < 0.5):
                    picked.append((img, x, rng.choice(ch)))
                elif no:
                    picked.append((img, x, rng.choice(no)))
                else:
                    picked.append((img, x, rng.choice(specs)))
                n_pic += 1
    steps["questions_picked"] = len(picked)
    # yes = no per category
    by: Dict[tuple, list] = defaultdict(list)
    for t in picked:
        s = t[2]
        by[(s["type"], s["cat"], s.get("answer") if s["type"] == "noul" else "")].append(t)
    final = []
    for cat in ("color", "material"):
        final += by[("choice", cat, "")]
        y, n = by[("noul", cat, True)], by[("noul", cat, False)]
        k = min(len(y), len(n))
        final += y[:k] + n[:k]
        dropped["noul yes / no balance (%s)" % cat] = len(y) + len(n) - 2 * k
    final.sort(key=lambda t: _h(t[1]["instance_id"]))
    final = final[: int(a.limit * 1.1)]

    got = fetch_vg(a.root, [int(t[0]) for t in final], a.workers)
    rows = []
    for img, x, s in final:
        rel = got.get(int(img))
        if rel is None:
            dropped["picture could not be fetched"] += 1
            continue
        if len(rows) >= a.limit:
            break
        meta = vgdata[int(img)]
        size = _file_size(os.path.join(a.root, rel))
        sx, sy = (size[0] / meta["width"], size[1] / meta["height"]) if size else (1, 1)
        b = x["instance_bbox"]
        box = [round(b[0] * sx, 1), round(b[1] * sy, 1), round(b[2] * sx, 1), round(b[3] * sy, 1)]
        iid = "vg:%s" % img
        sid = "r13-vaw:%s:%s" % (x["instance_id"], s["cat"])
        q = text(s, x["object_name"])
        common = dict(group=ident.canonical(iid), category=s["cat"], vaw_instance=x["instance_id"],
                      boxes={"q": [box]}, image_size=list(size) if size else [meta["width"], meta["height"]])
        if s["type"] == "choice":
            r = choice(sid, SOURCE, "vaw_%s" % s["cat"], iid, rel, q, s["options"], s["answer"], rng, **common)
        else:
            r = noul(sid, SOURCE, "vaw_%s_yesno" % s["cat"], iid, rel, q, s["answer"], **common)
            r["asked_value"] = s["value"]
        rows.append(r)
    near = ident.near_held({r["image"] for r in rows}, SOURCE)
    dropped["within dHash 2 of a held-out photo"] = len(near)
    rows = [r for r in rows if r["image"] not in near]
    steps["questions"] = len(rows)
    sheets = spot(a.root, rows)
    finish(a.root, SOURCE, rows, dict(steps), dict(dropped), {
        "licence": "VAW: Adobe Research licence, non-commercial (research only)",
        "identity_backend": ident.backend, "limit": a.limit, "limit_reached": len(rows) >= a.limit,
        "seen_in_project_history": sum(1 for r in rows if ident.seen(r["image_id"])), "spot_sheets": sheets},
        report=REPORT.format(n=len(rows), steps=dict(steps), dropped=dict(dropped), sheets=sheets, spot=SPOT))


REPORT = """# VAW (round 13)

Built by `scripts/data/r13/sources/vaw.py`; counts in MANIFEST.json.

- Kept {n} questions. Steps: {steps}
- Dropped per rule: {dropped}
- Spot check sheets (red = the instance box): {sheets}
- Spot check (looked at 30): {spot}
"""
SPOT = ("25 right; 2 wrong (VAW's explicit negative contradicts the picture: 'Is the trailer metal?' -> no for a "
        "truck trailer, 'Is the dresser white?' -> no for a white dresser); 3 doubtful (the pot is not visible in "
        "its box; 'skateboard wooden?' / 'snowboard wooden?' -> no for decks that may be wood). Some material "
        "negatives are trivially true ('Is the dog stone?'). The uniqueness rules held in all 30 (one thing of the "
        "name, box on it).")


def _h(x) -> str:
    import hashlib
    return hashlib.sha1(("vaw%s" % x).encode()).hexdigest()


def _file_size(path: str):
    from PIL import Image
    try:
        with Image.open(path) as im:
            return im.size
    except OSError:
        return None


def spot(root: str, rows: List[dict]) -> List[str]:
    rng = random.Random(1)
    pick = rng.sample(rows, min(30, len(rows)))
    paths = []
    for k in range(0, len(pick), 15):
        path = os.path.join(spot_dir(), "%s_%d.png" % (SOURCE, k // 15 + 1))
        contact_sheet(root, [{"image": r["image"], "text": describe(r), "boxes": r["boxes"]["q"]}
                             for r in pick[k:k + 15]], path, cols=5)
        paths.append(path)
    return paths


if __name__ == "__main__":
    main()
