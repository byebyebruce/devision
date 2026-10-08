"""Round 13: GQA real-photo binding questions with annotation-backed labels (plan
docs/experiments/2026-10-08-r13-binding-plan.md 1.3 and 1.5; review R13-04).

    uv run python scripts/data/r13/gqa_bind.py --root data

GQA's balanced questions are generated automatically from its scene graphs, so a "no" can come from an object or
attribute the graph simply does not list. Only questions whose answer the graph backs explicitly are kept:

- attr_choose (chooseAttr, materialChoose: "Are the shorts black or green?"): the right option is an annotated
  attribute of the target; the other option is not annotated on the target, and is either annotated on another
  object of the same picture (same category) or the antonym of the right option in an exclusive pair (ANTONYMS:
  small / large, open / closed, ...). Colours and materials have no antonyms (an object can be two colours).
- attr_verify (verifyAttr*, materialVerify*: "Is the knife silver?"): "yes" when the asked value is annotated on
  the target; "no" only when the target carries a different annotated value of the same category (colour,
  material, ...) and the asked value is not annotated on it. A "no" is never inferred from a missing annotation.
  Because colours and materials co-occur on one object, a colour "no" also needs exactly one annotated colour on
  the target and an asked colour of another family (COLOUR_FAMILIES; hair colours are one family), and material
  "no"s are not used at all.
- Only categories with a concrete visual meaning (ATTR_CATEGORIES: colour, material, pose, pattern, shape,
  activity, and length of hair / beard); size, cleanliness, age, ... were often ambiguous in the spot check.
  No colour questions about large regions (REGIONS: grass, sky, field, water, ...), which are often several
  colours; no question about something worn (WORN: shirt, trousers, hair, ...) when several people are in the
  picture (COCO person count, else scene-graph people).
- same (twoSame*, twoDifferent*, twoSameMaterial*): both objects carry exactly one annotated value of the
  category, and GQA's answer agrees with comparing them.
- spatial (positionChoose, positionVerify*: "Is the cup on the left or right?"): the answer must agree with the
  target's box (centre in the outer 40% on that side); a "no" is the box being clearly on the other side.
- Every object the program refers to (target and anchors) must exist, cover >= 1.5% of the picture (as v2_gqa),
  and be the only object of its name in the scene graph; when the name is a COCO category (COCO_NAMES) and the
  picture is a COCO picture, COCO must box exactly one instance of it (v2_vg_relation's rule). Attribute filters
  in the program ("the brown bottle") must be annotated on their object; negated filters are dropped.
- No counting questions (scene graphs miss objects) and no existence questions (a "no" would come from absence).

Every sample has `kind`, `gqa_qid`, `gqa_type` (GQA's detailed type), `group` (canonical picture identity, for
group-adjacent shuffling), `left_right` (the text mentions left / right), `competing` (another object of the
picture carries a competing value of the asked category: the salient-distractor case binding is about), and
`boxes` {"q": [[x, y, w, h], ...]} for the target object(s) in pixels of the picture file, with `image_size`
[width, height] (omitted when the file's size does not match the scene graph's aspect ratio).

Splits, by canonical picture identity (scripts/data/r13/identity.py), no picture in two of them:
- dev_bind_gqa: GQA train pictures never seen in project history (identity and dHash), >= 800 pictures;
- test_bind_gqa: GQA val pictures never seen in project history, >= 800 pictures;
- train: other GQA train pictures (seen or not) on disk, minus every held-out picture of the project
  (ext/common.HeldOut, dHash near held-out photos, all of COCO val2014), up to --train questions balanced
  over kinds, yes / no and answers.
Pictures for dev / test that are not on disk are downloaded one by one (Visual Genome / COCO URLs, resumable).
Writes data/r13/gqa/{train,dev_bind_gqa,test_bind_gqa}.jsonl, MANIFEST.json and REPORT.md (counts per rule,
kind, answer and split; overlaps; truly new question ids) and contact sheets runs/r13/gqa_spot_*.png.
"""
import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_SCRIPTS = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(DATA_SCRIPTS))
sys.path.insert(0, DATA_SCRIPTS)
sys.path.insert(0, HERE)

MIN_AREA = 0.015
SIDE = 0.4                    # spatial: box centre within the outer 40% of the picture on the answer's side
PER_PICTURE = 6
OBJECT_REF = re.compile(r"\((\d+)\)")
LEFT_RIGHT = re.compile(r"\b(left|right)\b")

KINDS = {}
for _t in ("chooseAttr", "materialChoose"):
    KINDS[_t] = "attr_choose"
for _t in ("verifyAttr", "verifyAttrC", "verifyAttrK", "verifyAttrKC", "verifyAttrThis", "verifyAttrCThis",
           "materialVerify", "materialVerifyC"):
    KINDS[_t] = "attr_verify"
for _t in ("twoSame", "twoSameC", "twoDifferent", "twoDifferentC", "twoSameMaterial", "twoSameMaterialC"):
    KINDS[_t] = "same"
for _t in ("positionChoose", "positionVerify", "positionVerifyC"):
    KINDS[_t] = "spatial"
GLOBAL_CATEGORIES = {"weather", "place", "location", "hposition", "vposition"}
# attribute categories with a concrete visual meaning; size, height, cleanliness, state, age ... are relative or
# subjective ("Is the window large or small?") and were often ambiguous in the first spot check
ATTR_CATEGORIES = {"color", "material", "pose", "length", "pattern", "shape", "activity", "sportActivity"}
# a "no" needs the asked value to be excluded by the annotated one; colours and materials co-occur on one object
# (a black-and-white sweater, a glass mirror in a metal frame), so: material "no" is never used, colour "no" only
# when the target has exactly one annotated colour, the asked colour is not of the same family, and the target is
# not a large region that is often several colours (a field of grass and dirt)
COLOUR_FAMILIES = [{"white", "silver", "gray", "grey", "beige", "cream", "cream colored", "light gray", "light blue",
                    "clear", "light brown"},
                   {"black", "dark", "dark brown", "gray", "grey", "dark gray", "dark blue", "navy", "silver"},
                   {"brown", "tan", "beige", "dark brown", "light brown", "orange", "khaki", "wood", "wooden", "gold",
                    "maroon", "cream colored"},
                   {"blue", "light blue", "dark blue", "navy", "teal", "turquoise", "purple"},
                   {"green", "dark green", "light green", "olive", "teal", "turquoise", "lime"},
                   {"red", "pink", "maroon", "orange", "dark red", "purple", "magenta"},
                   {"yellow", "gold", "orange", "beige", "tan", "blond"},
                   {"purple", "pink", "violet", "magenta"}]
REGIONS = {"field", "ground", "grass", "sky", "wall", "floor", "road", "street", "water", "ocean", "sea", "sand",
           "snow", "dirt", "pavement", "sidewalk", "background", "hill", "hillside", "mountain", "mountains", "beach",
           "forest", "trees", "bushes", "leaves", "clouds", "lawn", "area", "scene", "landscape"}
# things people wear or have: "the trousers" in a crowd of riders is one labelled pair among many unlabelled
WORN = {"shirt", "t-shirt", "jacket", "coat", "pants", "trousers", "jeans", "shorts", "skirt", "dress", "sweater",
        "hoodie", "sweatshirt", "vest", "hat", "cap", "helmet", "hair", "shoe", "shoes", "sneakers", "boots", "boot",
        "glove", "gloves", "uniform", "jersey", "top", "tank top", "outfit", "suit", "tie", "scarf", "socks", "sock",
        "glasses", "sunglasses", "goggles", "belt", "backpack", "bag", "purse", "beard", "apron", "wetsuit", "hand",
        "arm", "face", "head", "leg", "legs", "costume", "blouse", "bikini", "sleeve", "collar", "headband", "watch"}


HAIR_COLOURS = {"brown", "red", "blond", "blonde", "orange", "auburn", "dark brown", "light brown", "gold", "tan",
                "yellow", "ginger", "strawberry blond"}
LENGTH_TARGETS = {"hair", "beard"}   # "long / short" of a train, fence or patch of dirt was ambiguous in the spot check


def colour_family_overlap(a: str, b: str, hair: bool = False) -> bool:
    if hair and a in HAIR_COLOURS and b in HAIR_COLOURS:
        return True
    return any(a in f and b in f for f in COLOUR_FAMILIES)

# exclusive pairs: the target annotated with one makes the other false (for attr_choose's second option only)
ANTONYMS = [("small", "large"), ("small", "big"), ("little", "big"), ("tiny", "large"), ("tiny", "big"),
            ("huge", "small"), ("giant", "small"), ("long", "short"), ("tall", "short"), ("young", "old"),
            ("thick", "thin"), ("wide", "narrow"), ("open", "closed"), ("full", "empty"), ("on", "off"),
            ("wet", "dry"), ("clean", "dirty"), ("standing", "sitting"), ("standing", "lying"),
            ("sitting", "lying"), ("raw", "cooked"), ("empty", "full"), ("curly", "straight"),
            ("happy", "sad"), ("new", "old"), ("light", "heavy"), ("smooth", "rough"), ("hard", "soft"),
            ("bright", "dark"), ("cloudy", "clear"), ("calm", "rough"), ("sunny", "cloudy"), ("up", "down"),
            ("closed", "open"), ("lit", "unlit"), ("adult", "young")]
EXCLUSIVE: Dict[str, Set[str]] = defaultdict(set)
for _a, _b in ANTONYMS:
    EXCLUSIVE[_a].add(_b)
    EXCLUSIVE[_b].add(_a)

# scene-graph name -> COCO category, for the instance-count check on COCO pictures
COCO_NAMES = {n: "person" for n in ("man", "woman", "person", "boy", "girl", "child", "kid", "lady", "guy",
                                     "player", "skier", "surfer", "skateboarder", "snowboarder", "batter",
                                     "catcher", "pitcher", "umpire", "rider", "tennis player", "baby", "chef",
                                     "biker", "spectator", "officer", "soldier", "worker", "cowboy", "student")}
COCO_NAMES.update({"bike": "bicycle", "motorbike": "motorcycle", "plane": "airplane", "jet": "airplane",
                   "television": "tv", "sofa": "couch", "table": "dining table", "phone": "cell phone",
                   "cellphone": "cell phone", "mobile phone": "cell phone", "laptop computer": "laptop",
                   "doughnut": "donut", "hydrant": "fire hydrant", "puppy": "dog", "kitten": "cat", "calf": "cow",
                   "lamb": "sheep", "pony": "horse", "racket": "tennis racket", "bat": "baseball bat", "ski": "skis",
                   "luggage": "suitcase", "purse": "handbag", "fridge": "refrigerator", "remote control": "remote",
                   "computer mouse": "mouse", "glass": "wine glass", "ball": "sports ball", "teddy": "teddy bear",
                   "plant": "potted plant", "traffic signal": "traffic light", "meter": "parking meter",
                   "couch": "couch", "desk": "dining table"})
COCO_CATEGORIES = ("person bicycle car motorcycle airplane bus train truck boat bench bird cat dog horse sheep cow "
                   "elephant bear zebra giraffe backpack umbrella handbag tie suitcase frisbee skis snowboard kite "
                   "skateboard surfboard bottle cup fork knife spoon bowl banana apple sandwich orange broccoli "
                   "carrot pizza donut cake chair couch bed toilet tv laptop mouse remote keyboard microwave oven "
                   "toaster sink refrigerator book clock vase scissors toothbrush").split()
for _c in COCO_CATEGORIES:
    COCO_NAMES.setdefault(_c, _c)
for _c in ("traffic light", "fire hydrant", "stop sign", "parking meter", "sports ball", "baseball bat",
           "baseball glove", "tennis racket", "wine glass", "hot dog", "potted plant", "dining table", "cell phone",
           "teddy bear", "hair drier"):
    COCO_NAMES.setdefault(_c, _c)

Graph = dict
CocoCounts = Optional[Dict[str, int]]


# ---------------------------------------------------------------- rules (pure, tested on hand-made graphs)

def category_values(records: Iterable[dict]) -> Dict[str, Set[str]]:
    """Attribute category -> values, read from GQA's programs (filter / verify / choose steps)."""
    out: Dict[str, Set[str]] = defaultdict(set)
    for rec in records:
        for st in rec.get("semantic") or []:
            op = st.get("operation", "").split(" ", 1)
            if len(op) == 2 and op[0] in ("filter", "verify", "choose") and not op[1].startswith("not("):
                for v in st.get("argument", "").split("|"):
                    if v.strip() and not OBJECT_REF.search(v):
                        out[op[1].strip()].add(v.strip())
    return dict(out)


def norm_name(name: str) -> str:
    n = name.lower().strip()
    if n.endswith("es") and n[:-2] + "es" == n and n[:-2].endswith(("ss", "sh", "ch", "x")):
        return n[:-2]
    if n.endswith("s") and not n.endswith("ss") and len(n) > 3:
        return n[:-1]
    return n


def step_objects(program: Sequence[dict]) -> List[Optional[str]]:
    """The object id each step is about: its own "(id)", else its first dependency's."""
    objs: List[Optional[str]] = []
    for st in program:
        m = OBJECT_REF.findall(st.get("argument", ""))
        if m:
            objs.append(m[-1])
        else:
            deps = st.get("dependencies") or []
            objs.append(objs[deps[0]] if deps and deps[0] < len(objs) else None)
    return objs


def name_count(graph: Graph, name: str) -> int:
    n = norm_name(name)
    return sum(1 for o in graph["objects"].values() if norm_name(o["name"]) == n)


def check_objects(rec: dict, graph: Graph, coco_counts: CocoCounts) -> Optional[str]:
    """None when every referenced object exists, is big enough and is the only one of its name; else a reason."""
    program = rec.get("semantic") or []
    ids = {m for st in program for m in OBJECT_REF.findall(st.get("argument", ""))}
    if not ids:
        return "no_object"
    area = float(graph["width"] * graph["height"])
    for oid in sorted(ids):
        o = graph["objects"].get(oid)
        if o is None:
            return "object_missing"
        if o["w"] * o["h"] / area < MIN_AREA:
            return "too_small"
        if name_count(graph, o["name"]) != 1:
            return "name_not_unique"
        if o["name"].lower() in WORN or norm_name(o["name"]) in WORN:
            people = (coco_counts.get("person", 0) if coco_counts is not None else
                      sum(1 for x in graph["objects"].values() if COCO_NAMES.get(x["name"].lower()) == "person"))
            if people > 1:
                return "worn_by_one_of_several_people"
        if coco_counts is not None:
            cat = COCO_NAMES.get(o["name"].lower()) or COCO_NAMES.get(norm_name(o["name"]))
            if cat is not None and (coco_counts.get(cat, 0) or 1) != 1:   # COCO did not box it: trust the graph
                return "coco_not_one"
    objs = step_objects(program)
    for st, oid in zip(program, objs):
        op = st.get("operation", "")
        if "not(" in op or "not(" in st.get("argument", ""):
            return "negated_filter"
        if op.startswith("filter "):
            if oid is None or st.get("argument", "").strip() not in graph["objects"][oid].get("attributes", []):
                return "filter_not_annotated"
    return None


def final_targets(rec: dict) -> Tuple[str, str, List[Optional[str]]]:
    """(operation word, category, target object ids) of the program's last step."""
    program = rec["semantic"]
    objs = step_objects(program)
    last = program[-1]
    op = last["operation"].split(" ", 1)
    deps = last.get("dependencies") or []
    return op[0], (op[1].strip() if len(op) > 1 else ""), [objs[d] for d in deps if d < len(objs)]


def _attrs(graph: Graph, oid: str) -> Set[str]:
    return set(graph["objects"][oid].get("attributes") or [])


def _others_with(graph: Graph, oid: str, values: Set[str]) -> List[str]:
    return [k for k, o in graph["objects"].items() if k != oid and values & set(o.get("attributes") or [])]


def judge(rec: dict, graph: Optional[Graph], cats: Dict[str, Set[str]], coco_counts: CocoCounts = None):
    """(verdict, reason, info). verdict: {"type": "noul", "truth": bool} or {"type": "choice", "options": [...],
    "answer": str}; None when the question is dropped (reason says why). info: targets, competitors, kind."""
    kind = KINDS.get(rec.get("types", {}).get("detailed", ""))
    if kind is None:
        return None, "type_not_used", {}
    if graph is None:
        return None, "no_scene_graph", {}
    bad = check_objects(rec, graph, coco_counts)
    if bad:
        return None, bad, {}
    op, cat, targets = final_targets(rec)
    if not targets or any(t is None for t in targets):
        return None, "no_target", {}
    answer = str(rec["answer"]).strip()
    info = {"kind": kind, "targets": targets, "category": cat, "competitors": []}
    if kind == "spatial":
        return _spatial(rec, graph, op, cat, targets[0], answer, info)
    if cat in GLOBAL_CATEGORIES or cat not in cats or cat not in ATTR_CATEGORIES:
        return None, "category_not_used", {}
    values = cats[cat]
    if kind == "same":
        if len(targets) != 2 or op not in ("same", "different"):
            return None, "not_two_objects", {}
        v1, v2 = _attrs(graph, targets[0]) & values, _attrs(graph, targets[1]) & values
        if len(v1) != 1 or len(v2) != 1:
            return None, "same_value_not_single", {}
        truth = (v1 == v2) if op == "same" else (v1 != v2)
        if answer not in ("yes", "no") or truth != (answer == "yes"):
            return None, "answer_disagrees", {}
        return {"type": "noul", "truth": truth}, "kept", info
    t = targets[0]
    A = _attrs(graph, t)
    tname = norm_name(graph["objects"][t]["name"])
    if cat == "length" and tname not in LENGTH_TARGETS:
        return None, "length_not_hair", {}
    if cat == "color" and (tname in REGIONS or graph["objects"][t]["name"].lower() in REGIONS):
        return None, "colour_of_region", {}   # grass, sky, field, water: often several colours (spot checks 1-3)
    if kind == "attr_verify":
        asked = rec["semantic"][-1].get("argument", "").strip()
        if op != "verify" or not asked:
            return None, "not_verify", {}
        if answer == "yes":
            if asked not in A:
                return None, "yes_not_annotated", {}
            info["competitors"] = _others_with(graph, t, values - A)
            return {"type": "noul", "truth": True}, "kept", info
        if answer != "no":
            return None, "answer_not_yes_no", {}
        if asked in A:
            return None, "no_but_annotated", {}
        if not (A & values) - {asked}:
            return None, "no_from_missing", {}
        if cat == "material":
            return None, "no_material", {}
        if cat == "color":
            own = A & values
            if len(own) != 1:
                return None, "no_colour_target_multicolour", {}
            if colour_family_overlap(asked, next(iter(own)), hair=tname == "hair"):
                return None, "no_colour_same_family", {}
        info["competitors"] = _others_with(graph, t, {asked})
        return {"type": "noul", "truth": False}, "kept", info
    # attr_choose
    options = [o.strip() for o in rec["semantic"][-1].get("argument", "").split("|")]
    if op != "choose" or len(options) != 2 or len(set(options)) != 2 or answer not in options:
        return None, "options_unclear", {}
    other = options[1] if options[0] == answer else options[0]
    if answer not in A:
        return None, "correct_not_annotated", {}
    if other in A:
        return None, "both_annotated", {}
    if answer not in values or other not in values:
        return None, "option_not_in_category", {}
    competitors = _others_with(graph, t, {other})
    if not competitors and other not in EXCLUSIVE.get(answer, set()):
        return None, "other_unbacked", {}
    info["competitors"] = competitors
    return {"type": "choice", "options": options, "answer": answer}, "kept", info


def _spatial(rec, graph, op, cat, t, answer, info):
    o = graph["objects"][t]
    if cat == "hposition":
        c, size, lo, hi = o["x"] + o["w"] / 2.0, float(graph["width"]), "left", "right"
    elif cat == "vposition":
        c, size, lo, hi = o["y"] + o["h"] / 2.0, float(graph["height"]), "top", "bottom"
    else:
        return None, "category_not_used", {}
    side = lo if c < SIDE * size else (hi if c > (1 - SIDE) * size else None)
    if side is None:
        return None, "box_not_clearly_on_a_side", {}
    if op == "choose":
        options = [x.strip() for x in rec["semantic"][-1].get("argument", "").split("|")]
        if sorted(options) != sorted([lo, hi]) or answer != side:
            return None, "box_disagrees", {}
        return {"type": "choice", "options": options, "answer": answer}, "kept", info
    if op == "verify":
        asked = rec["semantic"][-1].get("argument", "").strip()
        if asked not in (lo, hi) or answer not in ("yes", "no") or (asked == side) != (answer == "yes"):
            return None, "box_disagrees", {}
        return {"type": "noul", "truth": asked == side}, "kept", info
    return None, "not_verify", {}


def to_sample(qid: str, rec: dict, verdict: dict, info: dict, image_id: str, image: str, group: str,
              boxes: Optional[List[List[float]]], image_size: Optional[Tuple[int, int]]) -> dict:
    text = rec["question"]
    if verdict["type"] == "noul":
        q = {"type": "noul", "instructions": text}
        probs = {"false": float(not verdict["truth"]), "true": float(verdict["truth"])}
        key = "yes" if verdict["truth"] else "no"
    else:
        options = list(verdict["options"])
        random.Random(qid).shuffle(options)
        q = {"type": "choice", "instructions": text, "criteria": {o: None for o in options}}
        probs = {o: float(o == verdict["answer"]) for o in options}
        key = verdict["answer"]
    s = {"id": "gqa-bind:%s" % qid, "source": "gqa-bind", "kind": info["kind"], "gqa_qid": qid,
         "gqa_type": rec["types"]["detailed"], "category": info["category"], "answer_key": key,
         "left_right": bool(LEFT_RIGHT.search(text.lower())), "competing": bool(info["competitors"]),
         "gqa_objects": {"targets": list(info["targets"]), "competitors": list(info["competitors"])},
         "group": group, "image_id": image_id, "image": image,
         "questions": {"q": q}, "gold": {"q": {"probabilities": probs}}}
    if boxes and image_size:
        s["boxes"] = {"q": boxes}
        s["image_size"] = list(image_size)
    return s


def scaled_boxes(graph: Graph, oids: Sequence[str], file_size: Optional[Tuple[int, int]]):
    """Target boxes in the picture file's pixels, or None when the file's aspect ratio differs from the graph's."""
    if not file_size:
        return None
    W, H = file_size
    sx, sy = W / float(graph["width"]), H / float(graph["height"])
    if abs(sx - sy) > 0.02 * max(sx, sy):
        return None
    out = []
    for oid in oids:
        o = graph["objects"][oid]
        out.append([round(o["x"] * sx, 1), round(o["y"] * sy, 1), round(o["w"] * sx, 1), round(o["h"] * sy, 1)])
    return out


def candidate_pools(pic_split: Dict[str, str], seen: Callable[[str], bool], rng: random.Random) -> Dict[str, List[str]]:
    """pic_split: canonical picture -> "train" | "val" | "both" (GQA split(s) its questions come from). Returns the
    dev candidates (unseen GQA train pictures) and test candidates (unseen GQA val pictures), shuffled. Pictures
    in both GQA splits (one COCO picture, two Visual Genome ids) are never candidates."""
    out = {}
    for name, split in (("dev", "train"), ("test", "val")):
        pool = sorted(p for p, s in pic_split.items() if s == split and not seen(p))
        rng.shuffle(pool)
        out[name] = pool
    return out


def train_pictures(pic_split: Dict[str, str], dev: Set[str], test: Set[str]) -> Set[str]:
    """GQA train pictures (including those also in GQA val) that are neither dev nor test pictures."""
    return {p for p, s in pic_split.items() if s in ("train", "both") and p not in dev and p not in test}


def intersections(sets: "Mapping[str, Iterable[str]]") -> Dict[str, int]:
    pics = {n: set(v) for n, v in sets.items()}
    names = sorted(pics)
    return {"%s & %s" % (x, y): len(pics[x] & pics[y]) for i, x in enumerate(names) for y in names[i + 1:]}


def balance(rows: List[dict], quota: Dict[str, int], rng: random.Random) -> List[dict]:
    """Per kind: at most quota[kind] rows; noul rows as many yes as no; no choice answer above twice the mean."""
    out = []
    for kind, n in quota.items():
        rs = [r for r in rows if r["kind"] == kind]
        rng.shuffle(rs)
        yes = [r for r in rs if r["questions"]["q"]["type"] == "noul" and r["answer_key"] == "yes"]
        no = [r for r in rs if r["questions"]["q"]["type"] == "noul" and r["answer_key"] == "no"]
        ch = [r for r in rs if r["questions"]["q"]["type"] == "choice"]
        k = min(len(yes), len(no))
        by: Dict[str, List[dict]] = defaultdict(list)
        for r in ch:
            by[r["answer_key"]].append(r)
        cap = max(1, int(2 * len(ch) / len(by))) if by else 0
        ch = [r for v in by.values() for r in v[:cap]]
        rng.shuffle(ch)
        pools, picked, i = [yes[:k], no[:k], ch], [], 0
        while len(picked) < n and any(i < len(p) for p in pools):
            picked += [p[i] for p in pools if i < len(p)]
            i += 1
        out += picked[:n]
    return out


def cap_per_picture(rows: List[dict], cap: int, rng: random.Random) -> List[dict]:
    by: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by[r["group"]].append(r)
    out = []
    for k in sorted(by):
        v = by[k]
        rng.shuffle(v)
        out += v[:cap]
    return out


def equal_yes_no(rows: List[dict], rng: random.Random) -> List[dict]:
    """Per kind, drop the surplus of the commoner noul answer (choice rows untouched)."""
    keep = []
    for kind in sorted({r["kind"] for r in rows}):
        rs = [r for r in rows if r["kind"] == kind]
        yes = [r for r in rs if r["questions"]["q"]["type"] == "noul" and r["answer_key"] == "yes"]
        no = [r for r in rs if r["questions"]["q"]["type"] == "noul" and r["answer_key"] == "no"]
        rng.shuffle(yes)
        rng.shuffle(no)
        k = min(len(yes), len(no))
        keep += yes[:k] + no[:k] + [r for r in rs if r["questions"]["q"]["type"] == "choice"]
    return keep


# ---------------------------------------------------------------- build

def fetch_pictures(root: str, jobs: Sequence[tuple], label: str) -> Set[str]:
    """jobs: (key, relative path, picture kind, url or None). Downloads the missing ones (resumable: a file on disk is
    skipped, a partial one is written to .part first). Returns the keys that are still not on disk."""
    import urllib.request
    from fetch import run_with_progress

    def one(j) -> str:
        path = os.path.join(root, j[1])
        if os.path.exists(path):
            return ""
        if not j[3]:
            return j[0]
        try:
            with urllib.request.urlopen(j[3], timeout=30) as resp:
                data = resp.read()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path + ".part", "wb") as f:
                f.write(data)
            os.replace(path + ".part", path)
            return ""
        except OSError:
            return j[0]
    todo = [j for j in jobs if not os.path.exists(os.path.join(root, j[1]))]
    return set(run_with_progress(label, todo, one)) if todo else set()


def load_records(root: str, split: str) -> List[dict]:
    import glob
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from fetch import GQA_REPO, _parquet  # noqa: F401
    path = "%s_balanced_instructions/%s-00000-of-00001.parquet" % (split, split)
    t = _parquet(root, GQA_REPO, path).select(["id", "imageId", "question", "answer", "types", "semantic"])
    return t.to_pylist()


def coco_instance_counts(root: str):
    """COCO 2014 image id -> ({category: instances}, (width, height), split)."""
    import zipfile
    from fetch import coco_annotations
    out: Dict[int, list] = {}
    with zipfile.ZipFile(coco_annotations(root)) as z:
        for split in ("train2014", "val2014"):
            inst = json.load(z.open("annotations/instances_%s.json" % split))
            names = {c["id"]: c["name"] for c in inst["categories"]}
            for im in inst["images"]:
                out[im["id"]] = [Counter(), (im["width"], im["height"]), split]
            for a in inst["annotations"]:
                out[a["image_id"]][0][names[a["category_id"]]] += 1
    return out


def file_size(path: str) -> Optional[Tuple[int, int]]:
    from PIL import Image
    try:
        with Image.open(path) as im:
            return im.size
    except OSError:
        return None


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    p.add_argument("--train", type=int, default=40000, help="training questions (at most)")
    p.add_argument("--dev-pictures", type=int, default=1000)
    p.add_argument("--test-pictures", type=int, default=1000)
    p.add_argument("--min-pictures", type=int, default=800, help="stop when dev or test has fewer pictures")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-fetch-train", dest="fetch_train", action="store_false",
                   help="do not download Visual Genome pictures for training questions about attributes / same")
    p.add_argument("--spot", type=int, default=30, help="questions per stratum on the contact sheets")
    a = p.parse_args(argv)
    t0 = time.time()
    import zipfile

    import identity as ident_mod
    from fetch import GQA_SCENE_GRAPHS_URL, ensure_file, vg_to_coco, _download_json_zip, VG_IMAGE_DATA_URL
    root = a.root
    out_dir = os.path.join(root, "r13", "gqa")
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(a.seed)
    ident = ident_mod.load(root)
    v2c = vg_to_coco(root)
    vg_url = {"%d" % r["image_id"]: r["url"] for r in _download_json_zip(VG_IMAGE_DATA_URL, os.path.join(root, "raw", "vg"))}
    with zipfile.ZipFile(ensure_file(GQA_SCENE_GRAPHS_URL, os.path.join(root, "raw", "gqa", "sceneGraphs.zip"))) as z:
        graphs = {s: json.load(z.open("%s_sceneGraphs.json" % s)) for s in ("train", "val")}
    print("scene graphs: train %d, val %d (%.0f s)" % (len(graphs["train"]), len(graphs["val"]), time.time() - t0), flush=True)
    records = {s: [r for r in load_records(root, s) if r["types"]["detailed"] in KINDS] for s in ("train", "val")}
    print("questions of the used types: train %d, val %d" % (len(records["train"]), len(records["val"])), flush=True)
    cats = category_values(r for s in records.values() for r in s)
    coco = coco_instance_counts(root)
    print("COCO 2014 instances loaded (%.0f s)" % (time.time() - t0), flush=True)
    history_qids = set(json.load(open(os.path.join(root, "r13", "identity.json")))["meta"]["gqa_question_ids"])

    # ---- judge every question
    tallies = {s: Counter() for s in ("train", "val")}
    tally_kind = {s: defaultdict(Counter) for s in ("train", "val")}
    kept: Dict[str, List[tuple]] = {"train": [], "val": []}
    for split in ("train", "val"):
        for rec in records[split]:
            vg = str(rec["imageId"])
            coco_id = v2c.get("vg:" + vg)
            counts = coco[int(coco_id[5:])][0] if coco_id and int(coco_id[5:]) in coco else None
            verdict, reason, info = judge(rec, graphs[split].get(vg), cats, counts)
            tallies[split][reason] += 1
            tally_kind[split][KINDS[rec["types"]["detailed"]]][reason] += 1
            if verdict is not None:
                kept[split].append((rec, verdict, info))
    print("kept: train %d, val %d (%.0f s)" % (len(kept["train"]), len(kept["val"]), time.time() - t0), flush=True)

    # ---- where each picture is, and its canonical identity

    def where(vg: str) -> Tuple[str, str, Optional[str]]:
        """(image path, picture kind, download url)."""
        coco_id = v2c.get("vg:" + vg)
        if coco_id:
            n = int(coco_id[5:])
            split = coco[n][2] if n in coco else "train2014"
            name = "COCO_%s_%012d.jpg" % (split, n)
            return "coco/%s/%s" % (split, name), "coco-" + split, "http://images.cocodataset.org/%s/%s" % (split, name)
        return "gqa/images/%s.jpg" % vg, "vg", vg_url.get(vg)

    sys.path.insert(0, os.path.join(DATA_SCRIPTS, "ext"))
    from common import HeldOut  # ext/common.py
    held = HeldOut(root)
    pic_split: Dict[str, str] = {}
    pic_vg: Dict[str, str] = {}
    for split in ("train", "val"):
        for rec, _, _ in kept[split]:
            c = ident.canonical("vg:%s" % rec["imageId"])
            # a COCO picture with two Visual Genome ids can sit in both GQA splits: train only, never dev / test
            pic_split[c] = "both" if pic_split.get(c, split) != split else split
            if split == "train" or c not in pic_vg:
                pic_vg[c] = str(rec["imageId"])
    # dev / test candidates: unseen by identity; then on disk (downloaded) and not within dHash 2 of an untraced
    # history photo. Take extra candidates to cover failed downloads and near-duplicates.
    report: dict = {"rules": {s: dict(tallies[s].most_common()) for s in tallies},
                    "rules_by_kind": {s: {k: dict(v.most_common()) for k, v in tally_kind[s].items()} for s in tally_kind}}
    pools = candidate_pools(pic_split, ident.seen, rng)
    cand = {"train": pools["dev"], "val": pools["test"]}
    report["unseen_candidates"] = {s: len(v) for s, v in cand.items()}
    report["pictures_in_both_gqa_splits"] = sum(1 for sp in pic_split.values() if sp == "both")
    want = {"train": a.dev_pictures, "val": a.test_pictures}
    chosen: Dict[str, List[str]] = {}
    near_dropped: Dict[str, int] = {}
    failed_dl: Dict[str, int] = {}
    from fetch import run_with_progress
    for s in ("train", "val"):
        pool = cand[s][:int(want[s] * 1.3) + 50]
        jobs = [(c,) + where(pic_vg[c]) for c in pool]
        failed = fetch_pictures(root, jobs, "%s candidate pictures" % s)
        failed_dl[s] = len(failed)
        ok = [j for j in jobs if j[0] not in failed and os.path.exists(os.path.join(root, j[1]))]
        hashes = ident_mod.dhash_files(root, [j[1] for j in ok])
        near = {j[0] for j in ok if j[1] not in hashes or ident.near_seen(hashes[j[1]])}
        near_dropped[s] = len(near)
        chosen[s] = [j[0] for j in ok if j[0] not in near][:want[s]]
    dev_p, test_p = set(chosen["train"]), set(chosen["val"])
    report["dev_test_pictures"] = {"dev_candidates_dhash_near_seen": near_dropped["train"],
                                   "test_candidates_dhash_near_seen": near_dropped["val"],
                                   "download_failures": failed_dl}

    # ---- training pictures: GQA train, not dev, not held out anywhere, not COCO val2014, not near held. Usable:
    # COCO train2014 files and the GQA image shards the project already has (v2_gqa.TRAIN_SHARDS), plus pictures
    # fetched here because they carry a non-spatial question (attribute / same), so the set does not depend on
    # what earlier runs happened to download.
    from v2_gqa import TRAIN_SHARDS
    import pyarrow.parquet as pq
    from fetch import GQA_REPO
    from huggingface_hub import hf_hub_download
    shard_ids = set()
    for sh in TRAIN_SHARDS:
        f = hf_hub_download(GQA_REPO, "train_balanced_images/" + sh, repo_type="dataset",
                            cache_dir=os.path.join(root, "raw", "hf"))
        shard_ids |= set(pq.read_table(f, columns=["id"]).column("id").to_pylist())
    attr_pics = {ident.canonical("vg:%s" % rec["imageId"]) for rec, _, info in kept["train"] if info["kind"] != "spatial"}
    train_jobs = {}
    held_drop = Counter()
    to_fetch = []
    for c in sorted(train_pictures(pic_split, dev_p, test_p)):
        rel, pk, url = where(pic_vg[c])
        if pk == "coco-val2014":
            held_drop["coco_val2014"] += 1
            continue
        if held.is_held(c) or held.is_held("vg:" + pic_vg[c]):
            held_drop["held_out_identity"] += 1
            continue
        if pk == "vg" and pic_vg[c] not in shard_ids:
            if c not in attr_pics or not a.fetch_train:
                held_drop["not_in_local_shards_spatial_only"] += 1
                continue
            to_fetch.append((c, rel, pk, url))
        train_jobs[c] = rel
    failed = fetch_pictures(root, to_fetch, "train pictures (attribute / same questions)")
    held_drop["download_failed"] = len(failed)
    report["train_pictures_fetched"] = len(to_fetch) - len(failed)
    for c in failed:
        train_jobs.pop(c, None)
        if pk == "coco-val2014":
            held_drop["coco_val2014"] += 1
            continue
        if held.is_held(c) or held.is_held("vg:" + pic_vg[c]):
            held_drop["held_out_identity"] += 1
            continue
        train_jobs[c] = rel
    near_train = held.near_held(list(train_jobs.values()), "r13_gqa_train")
    held_drop["near_held_dhash"] = sum(1 for r in train_jobs.values() if r in near_train)
    train_p = {c for c, r in train_jobs.items() if r not in near_train}
    report["train_pictures_dropped"] = dict(held_drop)

    # ---- samples
    sizes: Dict[str, Optional[Tuple[int, int]]] = {}
    box_dropped = Counter()

    def rows_for(split: str, pics: Set[str]) -> List[dict]:
        out = []
        for rec, verdict, info in kept[split]:
            vg = str(rec["imageId"])
            c = ident.canonical("vg:" + vg)
            if c not in pics:
                continue
            rel, _, _ = where(vg)
            if rel not in sizes:
                sizes[rel] = file_size(os.path.join(root, rel))
            g = graphs[split][vg]
            boxes = scaled_boxes(g, info["targets"], sizes[rel])
            if boxes is None:
                box_dropped[split] += 1
            out.append(to_sample(rec["id"], rec, verdict, info, "vg:" + vg, rel, c, boxes, sizes[rel]))
        return out

    dev = equal_yes_no(cap_per_picture(rows_for("train", dev_p), PER_PICTURE, rng), rng)
    test = equal_yes_no(cap_per_picture(rows_for("val", test_p), PER_PICTURE, rng), rng)
    pool = cap_per_picture(rows_for("train", train_p), PER_PICTURE, rng)
    kinds = sorted({r["kind"] for r in pool})
    avail = {k: _balanced_size(pool, k) for k in kinds}
    quota = _quotas(avail, a.train)
    train = balance(pool, quota, rng)
    rng.shuffle(train)
    for name, rows in (("dev_bind_gqa", dev), ("test_bind_gqa", test)):
        n_pics = len({r["group"] for r in rows})
        if n_pics < a.min_pictures:
            raise SystemExit("%s: only %d unseen pictures (need %d); not relaxing the rules" % (name, n_pics, a.min_pictures))
    from v2_common import write
    sets = {"train": train, "dev_bind_gqa": dev, "test_bind_gqa": test}
    for name, rows in sets.items():
        rows.sort(key=lambda r: (r["group"], r["id"])) if name != "train" else None
        write(os.path.join(out_dir, name + ".jsonl"), rows)
    report.update(summary(sets, ident, history_qids, root, held))
    report["train_quota"] = quota
    report["train_available_balanced"] = avail
    report["boxes_omitted_aspect_mismatch"] = dict(box_dropped)
    report["seconds"] = round(time.time() - t0)
    json.dump(report, open(os.path.join(out_dir, "MANIFEST.json"), "w"), indent=1)
    sheets = spot_sheets(root, sets, graphs, a.spot, random.Random(a.seed))
    write_report(os.path.join(out_dir, "REPORT.md"), report, sheets)
    print(json.dumps({k: report[k] for k in ("counts", "pictures", "cross_split_identity", "history_overlap")}, indent=1))


def _balanced_size(pool: List[dict], kind: str) -> int:
    rs = [r for r in pool if r["kind"] == kind]
    yes = sum(1 for r in rs if r["questions"]["q"]["type"] == "noul" and r["answer_key"] == "yes")
    no = sum(1 for r in rs if r["questions"]["q"]["type"] == "noul" and r["answer_key"] == "no")
    ch = [r["answer_key"] for r in rs if r["questions"]["q"]["type"] == "choice"]
    by = Counter(ch)
    cap = max(1, int(2 * len(ch) / len(by))) if by else 0
    return 2 * min(yes, no) + sum(min(v, cap) for v in by.values())


def _quotas(avail: Dict[str, int], total: int) -> Dict[str, int]:
    """Equal shares per kind; a kind that cannot fill its share gives the rest to the others."""
    quota = {k: 0 for k in avail}
    left, open_ = total, set(avail)
    while left > 0 and open_:
        share = max(1, left // len(open_))
        for k in sorted(open_):
            take = min(share, avail[k] - quota[k], left)
            quota[k] += take
            left -= take
        open_ = {k for k in open_ if quota[k] < avail[k]}
    return quota


def summary(sets: Dict[str, List[dict]], ident, history_qids: Set[str], root: str, held) -> dict:
    out: dict = {"counts": {}, "pictures": {}, "by_kind_answer": {}, "by_type": {}, "truly_new": {}, "boxes": {}}
    for name, rows in sets.items():
        out["counts"][name] = len(rows)
        out["pictures"][name] = len({r["group"] for r in rows})
        ka: Dict[str, Counter] = defaultdict(Counter)
        for r in rows:
            ka[r["kind"]][r["answer_key"] if r["questions"]["q"]["type"] == "noul" else "choice"] += 1
        out["by_kind_answer"][name] = {k: dict(v) for k, v in sorted(ka.items())}
        out["by_type"][name] = dict(Counter(r["gqa_type"] for r in rows).most_common())
        old = sum(1 for r in rows if r["gqa_qid"] in history_qids)
        out["truly_new"][name] = {"new": len(rows) - old, "in_earlier_training_files": old}
        out["boxes"][name] = sum(1 for r in rows if "boxes" in r)
        out.setdefault("competing", {})[name] = sum(1 for r in rows if r["competing"])
        out.setdefault("left_right", {})[name] = sum(1 for r in rows if r["left_right"])
    pics = {n: {r["group"] for r in rows} for n, rows in sets.items()}
    names = sorted(pics)
    out["cross_split_identity"] = intersections(pics)
    if any(out["cross_split_identity"].values()):
        raise SystemExit("a picture is in two splits: %s" % out["cross_split_identity"])
    out["history_overlap"] = {n: sum(1 for p in pics[n] if ident.seen(p)) for n in names}
    out["held_out_overlap"] = {n: sum(1 for p in pics[n] if held.is_held(p)) for n in names}
    for n, rows in sets.items():
        for r in rows:
            q = r["questions"]["q"]
            assert set(r["gold"]["q"]["probabilities"]) == (set(q["criteria"]) if q["type"] == "choice" else {"false", "true"})
    return out


# ---------------------------------------------------------------- spot check

def strata(rows: List[dict]) -> Dict[str, List[dict]]:
    return {"multi_instance_attr": [r for r in rows if r["kind"] in ("attr_choose", "attr_verify") and r["competing"]],
            "negative": [r for r in rows if r["kind"] == "attr_verify" and r["answer_key"] == "no"],
            "choose": [r for r in rows if r["kind"] == "attr_choose"]}


def spot_sheets(root: str, sets: Dict[str, List[dict]], graphs, n: int, rng: random.Random) -> Dict[str, List[str]]:
    """Contact sheets of n questions per stratum (from all splits), 10 per sheet: picture with the target box(es)
    in red and the objects carrying the competing value in cyan; question and gold under it."""
    from PIL import Image, ImageDraw, ImageFont
    out_dir = os.path.join(REPO, "runs", "r13")
    os.makedirs(out_dir, exist_ok=True)
    allrows = [r for rows in sets.values() for r in rows]
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    sheets: Dict[str, List[str]] = {}
    for name, rows in strata(allrows).items():
        pick = rng.sample(rows, min(n, len(rows)))
        paths = []
        for k in range(0, len(pick), 10):
            tiles = []
            for r in pick[k:k + 10]:
                vg = r["image_id"][3:]
                g = graphs["train"].get(vg) or graphs["val"].get(vg)
                im = Image.open(os.path.join(root, r["image"])).convert("RGB")
                sx, sy = im.size[0] / float(g["width"]), im.size[1] / float(g["height"])
                d = ImageDraw.Draw(im)
                q = r["questions"]["q"]
                for colour, key in (((0, 255, 255), "competitors"), ((255, 0, 0), "targets")):
                    for oid in r["gqa_objects"][key]:
                        o = g["objects"][oid]
                        d.rectangle([o["x"] * sx, o["y"] * sy, (o["x"] + o["w"]) * sx, (o["y"] + o["h"]) * sy],
                                    outline=colour, width=4)
                im.thumbnail((420, 320))
                tile = Image.new("RGB", (430, 420), "white")
                tile.paste(im, (5, 5))
                gold = r["answer_key"]
                opts = " | ".join(q["criteria"]) if q["type"] == "choice" else "yes/no"
                text = "%s [%s] %s\n%s\nopts: %s   GOLD: %s" % (r["gqa_qid"], r["kind"], r["gqa_type"],
                                                               _wrap(q["instructions"], 52), opts, gold)
                ImageDraw.Draw(tile).multiline_text((5, 330), text, fill="black", font=font)
                tiles.append(tile)
            sheet = Image.new("RGB", (430 * 5, 420 * ((len(tiles) + 4) // 5)), "white")
            for i, t in enumerate(tiles):
                sheet.paste(t, ((i % 5) * 430, (i // 5) * 420))
            path = os.path.join(out_dir, "gqa_spot_%s_%d.png" % (name, k // 10 + 1))
            sheet.save(path)
            paths.append(os.path.relpath(path, REPO))
        sheets[name] = paths
        json.dump([{"gqa_qid": r["gqa_qid"], "id": r["id"], "question": r["questions"]["q"]["instructions"],
                    "gold": r["answer_key"], "image": r["image"]} for r in pick],
                  open(os.path.join(out_dir, "gqa_spot_%s.json" % name), "w"), indent=1)
    return sheets


def _wrap(text: str, width: int) -> str:
    import textwrap
    return "\n".join(textwrap.wrap(text, width)[:2])


def write_report(path: str, rep: dict, sheets: Dict[str, List[str]]) -> None:
    L = ["# GQA binding questions (round 13)", "",
         "Built by `scripts/data/r13/gqa_bind.py` (rules in its docstring; plan 1.3, review R13-04). "
         "Machine-readable numbers in `MANIFEST.json`.", ""]
    L += ["## Sets", "", "| set | questions | pictures | truly new | in earlier training files | with boxes | competing value | left/right words |",
          "|---|---|---|---|---|---|---|---|"]
    for n in rep["counts"]:
        L.append("| %s | %d | %d | %d | %d | %d | %d | %d |" % (n, rep["counts"][n], rep["pictures"][n],
                 rep["truly_new"][n]["new"], rep["truly_new"][n]["in_earlier_training_files"], rep["boxes"][n],
                 rep["competing"][n], rep["left_right"][n]))
    L += ["", "Cross-split picture identity intersections (must be 0): %s." % ", ".join(
        "%s = %d" % kv for kv in rep["cross_split_identity"].items()),
          "Pictures of each set in the project's training history (`identity.seen`; must be 0 for dev / test): %s."
          % ", ".join("%s = %d" % kv for kv in rep["history_overlap"].items()),
          "Pictures in a project held-out set (`ext/common.HeldOut`; informational for dev / test): %s."
          % ", ".join("%s = %d" % kv for kv in rep["held_out_overlap"].items()), "",
          "Unseen candidate pictures with >= 1 kept question: GQA train %(train)d, GQA val %(val)d." % rep["unseen_candidates"],
          "dev / test candidates dropped as within dHash 2 of an untraced history photo: %d / %d; download failures %s."
          % (rep["dev_test_pictures"]["dev_candidates_dhash_near_seen"], rep["dev_test_pictures"]["test_candidates_dhash_near_seen"],
             rep["dev_test_pictures"]["download_failures"]),
          "Training pictures dropped: %s." % rep["train_pictures_dropped"],
          "Boxes omitted (picture file's aspect ratio differs from the scene graph's): %s." % rep["boxes_omitted_aspect_mismatch"],
          "", "## Per kind and answer", ""]
    for n, ka in rep["by_kind_answer"].items():
        L.append("- %s: %s" % (n, "; ".join("%s %s" % (k, dict(v)) for k, v in ka.items())))
    L += ["", "Training quota per kind (equal shares, capped by the balanced pool): %s; balanced pool: %s."
          % (rep["train_quota"], rep["train_available_balanced"]), "", "## GQA detailed types", ""]
    for n, bt in rep["by_type"].items():
        L.append("- %s: %s" % (n, bt))
    L += ["", "## Filter tallies (every question of the used types, per GQA split)", ""]
    for s, t in rep["rules"].items():
        L.append("- %s: %s" % (s, t))
    L += ["", "Per kind:", ""]
    for s, bk in rep["rules_by_kind"].items():
        for k, t in bk.items():
            L.append("- %s / %s: %s" % (s, k, t))
    L += ["", "## Spot check", "", "Contact sheets (red = target box): %s." % "; ".join(
        "%s: %s" % (k, ", ".join(v)) for k, v in sheets.items()), "", "SPOT_CHECK_RESULTS: to be written here by the person who looks at the sheets (a rebuild resets it)", ""]
    open(path, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
