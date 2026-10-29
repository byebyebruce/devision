"""Winoground-style caption matching with hard negatives from COCO train2014 captions -> data/v05/board/caption/.

    uv run --no-sync python scripts/data/v05_board_caption.py --root data

One 2-way question per (image, true caption, generated negative): "Which caption describes this image?" with the
true COCO caption and a negative made from it by rules (no LLM, no parser; SugarCrepe / ARO style):

  swap_obj   two nouns of the same number around an asymmetric relation word swap places
             ("a man riding a horse" -> "a horse riding a man"); symmetric links (and, next to, with, near, ...)
             between them -> no candidate.
  swap_att   two "<attribute> <noun>" pairs with different attributes of the same group (colour, age, size, ...)
             and different nouns swap attributes ("a red car near a white truck" -> "a white car near a red
             truck"); an attribute word used twice in the caption -> no candidate.
  replace_rel a spatial relation becomes its opposite (above <-> below, in front of <-> behind, left <-> right, ...);
             skipped when the caption already has both directions.
  replace_att a colour becomes another colour not in the caption, or an attribute its antonym (young <-> old, ...).
  replace_obj a COCO object noun becomes another category of the same super-category that is NOT annotated in the
             image (instances_train2014), so the negative is false for the picture, not only for the caption.

Every negative differs from the caption, is not another caption of the same image, and keeps a/an agreement.
Per image at most PER_IMAGE questions (kinds rotated, rarest kind first); each kind is capped at --cap-kind.
Images: every board image hit (data/v05/board_exclusion.json) and every project held-out picture (pub/held.PubHeld:
our own eval / monitoring sets, COCO = VG identity) are dropped. The gold is A or B by a seeded hash so positions
balance. dev = ~4% of images (hash), never in train.
"""
import argparse
import json
import os
import re
import sys
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Sequence, Set, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from v05_board_common import BoardIndex, choice_sample, image_hashes, manifest, place_gold, seeded, write_split  # noqa: E402

OUT = os.path.join("v05", "board", "caption")
QUESTION = "Which caption describes this image?"
PER_IMAGE = 2

COLORS = ["red", "blue", "green", "yellow", "white", "black", "brown", "orange", "pink", "purple", "gray", "grey",
          "silver"]
ANTONYM = {"young": "old", "old": "young", "big": "small", "small": "big", "large": "small", "little": "big",
           "tall": "short", "short": "tall", "empty": "full", "full": "empty", "open": "closed", "closed": "open",
           "wet": "dry", "dry": "wet", "dark": "bright", "bright": "dark", "clean": "dirty", "dirty": "clean",
           "new": "old", "long": "short", "wooden": "metal", "metal": "wooden"}
ATTRS = set(COLORS) | set(ANTONYM)
ATTR_GROUP = {**{c: "colour" for c in COLORS}, **{w: "age" for w in ("young", "old", "new")},
              **{w: "size" for w in ("big", "small", "large", "little", "tall", "short", "long")},
              **{w: "state" for w in ("open", "closed", "empty", "full", "wet", "dry", "clean", "dirty")},
              **{w: "light" for w in ("dark", "bright")}, **{w: "material" for w in ("wooden", "metal")}}
RELATIONS = [("in front of", "behind"), ("behind", "in front of"), ("above", "below"), ("below", "above"),
             ("underneath", "on top of"), ("on top of", "underneath"), ("beneath", "above"), ("under", "above"),
             ("left", "right"), ("right", "left"), ("inside", "outside"), ("outside", "inside"),
             ("into", "out of"), ("out of", "into"), ("towards", "away from"), ("toward", "away from"),
             ("away from", "towards")]   # up / down ("walking down the beach") is rarely visible
ASYM = re.compile(r"\b(riding|rides|ride|holding|holds|hold|chasing|chases|chase|eating|eats|carrying|carries|"
                  r"pulling|pulls|pushing|pushes|wearing|wears|feeding|feeds|watching|watches|looking at|"
                  r"on top of|on|under|underneath|beneath|above|below|behind|in front of|inside|in|over|"
                  r"sitting on|standing on|laying on|lying on|sitting in|standing in|leaning on|leaning against|"
                  r"hanging from|attached to|kissing|hugging|petting|biting|following|leading|towing|throwing|"
                  r"catching|hitting|kicking|walking|resting on|parked in front of|filled with|covered in|"
                  r"covered with|full of|topped with)\b")
SYM = re.compile(r"\b(and|next to|beside|besides|near|with|by|along|alongside|or|together|between|among|"
                 r"across from|opposite|while|as|each other|both)\b|,")

# caption noun -> (COCO category, plural?)
_N = {
    "person": ["man", "woman", "person", "boy", "girl", "child", "kid", "guy", "lady", "player", "skier", "surfer",
               "skateboarder", "snowboarder", "baby", "toddler", "people", "men", "women", "boys", "girls",
               "children", "kids", "guys", "ladies", "players", "skiers", "surfers"],
    "bicycle": ["bicycle", "bike", "bicycles", "bikes"], "car": ["car", "cars"],
    "motorcycle": ["motorcycle", "motorbike", "motorcycles", "motorbikes"],
    "airplane": ["airplane", "plane", "jet", "airplanes", "planes", "jets"], "bus": ["bus", "buses"],
    "train": ["train", "trains"], "truck": ["truck", "trucks"], "boat": ["boat", "boats"],
    "bird": ["bird", "birds"], "cat": ["cat", "cats", "kitten"], "dog": ["dog", "dogs", "puppy"],
    "horse": ["horse", "horses"], "sheep": ["sheep"], "cow": ["cow", "cows"], "elephant": ["elephant", "elephants"],
    "bear": ["bear", "bears"], "zebra": ["zebra", "zebras"], "giraffe": ["giraffe", "giraffes"],
    "umbrella": ["umbrella", "umbrellas"], "frisbee": ["frisbee"], "kite": ["kite", "kites"],
    "skateboard": ["skateboard"], "surfboard": ["surfboard", "surfboards"], "bench": ["bench", "benches"],
    "bottle": ["bottle", "bottles"], "cup": ["cup", "cups", "mug"], "bowl": ["bowl", "bowls"],
    "banana": ["banana", "bananas"], "apple": ["apple", "apples"], "sandwich": ["sandwich", "sandwiches"],
    "orange": ["oranges"], "broccoli": ["broccoli"], "carrot": ["carrot", "carrots"], "pizza": ["pizza", "pizzas"],
    "donut": ["donut", "donuts", "doughnut", "doughnuts"], "cake": ["cake", "cakes"], "chair": ["chair", "chairs"],
    "couch": ["couch", "sofa", "couches"], "bed": ["bed", "beds"], "dining table": ["table", "tables"],
    "toilet": ["toilet", "toilets"], "tv": ["tv", "television"], "laptop": ["laptop", "laptops"],
    "cell phone": ["phone", "cellphone", "phones"], "book": ["book", "books"], "clock": ["clock", "clocks"],
    "vase": ["vase", "vases"], "teddy bear": ["teddy"], "sink": ["sink", "sinks"],
    "refrigerator": ["refrigerator", "fridge"], "oven": ["oven"], "suitcase": ["suitcase", "suitcases"],
    "fire hydrant": ["hydrant"], "traffic light": ["light"], "keyboard": ["keyboard"], "mouse": ["mouse"],
    "knife": ["knife", "knives"], "fork": ["fork", "forks"], "spoon": ["spoon", "spoons"],
}
NOUN: Dict[str, Tuple[str, bool]] = {}
for _cat, _ws in _N.items():
    for _w in _ws:
        NOUN[_w] = (_cat, _w in {"people", "men", "women", "children"} or (_w.endswith("s") and _w not in {
            "bus", "glass", "grass", "dress", "sheep", "skis"} and not _w.endswith("ss")) or _w == "knives")
PERSON_SING = ["man", "woman", "boy", "girl", "child"]
# replace_obj vocabulary: category -> singular caption word (for the replacement)
CAT_WORD = {"bicycle": "bicycle", "car": "car", "motorcycle": "motorcycle", "airplane": "airplane", "bus": "bus",
            "train": "train", "truck": "truck", "boat": "boat", "bird": "bird", "cat": "cat", "dog": "dog",
            "horse": "horse", "sheep": "sheep", "cow": "cow", "elephant": "elephant", "bear": "bear",
            "zebra": "zebra", "giraffe": "giraffe", "umbrella": "umbrella", "frisbee": "frisbee", "kite": "kite",
            "skateboard": "skateboard", "surfboard": "surfboard", "bench": "bench", "bottle": "bottle", "cup": "cup",
            "bowl": "bowl", "banana": "banana", "apple": "apple", "sandwich": "sandwich", "broccoli": "broccoli",
            "carrot": "carrot", "pizza": "pizza", "donut": "donut", "cake": "cake", "chair": "chair",
            "couch": "couch", "bed": "bed", "dining table": "table", "toilet": "toilet", "tv": "television",
            "laptop": "laptop", "cell phone": "phone", "book": "book", "clock": "clock", "vase": "vase",
            "sink": "sink", "refrigerator": "refrigerator", "oven": "oven", "suitcase": "suitcase",
            "knife": "knife", "fork": "fork", "spoon": "spoon"}
SUPER = {"vehicle": ["bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"],
         "animal": ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"],
         "food": ["banana", "apple", "sandwich", "broccoli", "carrot", "pizza", "donut", "cake"],
         "furniture": ["chair", "couch", "bed", "dining table", "toilet"],
         "kitchen": ["bottle", "cup", "bowl", "knife", "fork", "spoon"],
         "electronic": ["tv", "laptop", "cell phone"], "sports": ["frisbee", "kite", "skateboard", "surfboard"],
         "appliance": ["sink", "refrigerator", "oven"], "indoor": ["book", "clock", "vase"]}
SUPER_OF = {c: s for s, cs in SUPER.items() for c in cs}
KINDS = ("swap_obj", "swap_att", "replace_rel", "replace_att", "replace_obj")


# ----------------------------------------------------------------------------- text rules

def tokens(caption: str) -> List[str]:
    c = caption.strip().rstrip(".").strip().lower()
    c = re.sub(r"[^a-z0-9' ,\-]", " ", c)
    c = re.sub(r",", " , ", c)
    return c.split()


def detok(toks: Sequence[str]) -> str:
    s = " ".join(toks).replace(" ,", ",")
    return fix_articles(s)


def fix_articles(s: str) -> str:
    def rep(m):
        art, word = m.group(1), m.group(2)
        an = word[0] in "aeiou" and not word.startswith(("uni", "use", "one", "eu"))
        want = ("An" if art[0] == "A" else "an") if an else ("A" if art[0] == "A" else "a")
        return want + " " + word
    return re.sub(r"\b([Aa]n?) ([a-z]+)", rep, s)


def swap_obj(toks: List[str]) -> Optional[List[str]]:
    idx = [i for i, t in enumerate(toks) if t in NOUN]
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            i, j = idx[a], idx[b]
            n1, n2 = toks[i], toks[j]
            if n1 == n2 or NOUN[n1][0] == NOUN[n2][0] or NOUN[n1][1] != NOUN[n2][1]:
                continue
            between = " ".join(toks[i + 1:j])
            if not between or j - i > 6 or SYM.search(between) or not ASYM.search(between):
                continue
            if any(t in NOUN for t in toks[i + 1:j]):
                continue
            out = list(toks)
            out[i], out[j] = n2, n1
            return out
    return None


def swap_att(toks: List[str]) -> Optional[List[str]]:
    pairs = [i for i in range(len(toks) - 1) if toks[i] in ATTRS and toks[i + 1] in NOUN]
    for a in range(len(pairs)):
        for b in range(a + 1, len(pairs)):
            i, j = pairs[a], pairs[b]
            if toks[i] == toks[j] or toks[i + 1] == toks[j + 1]:
                continue
            if ATTR_GROUP[toks[i]] != ATTR_GROUP[toks[j]] or toks.count(toks[i]) > 1 or toks.count(toks[j]) > 1:
                continue
            out = list(toks)
            out[i], out[j] = toks[j], toks[i]
            return out
    return None


def replace_rel(toks: List[str]) -> Optional[List[str]]:
    s = " " + " ".join(toks) + " "
    for old, new in RELATIONS:
        if old in ("left", "right") and not re.search(r"\b(to the|on the|at the) %s\b|\b%s (of|side)\b|"
                                                                  r"\b(looking|walking|going|climbing|flying) %s\b"
                                                                  % (old, old, old), s):
            continue
        if " %s " % old in s:
            if " %s " % new in s:
                return None   # both directions already present: the swap may describe the image too
            return (s.replace(" %s " % old, " %s " % new, 1)).split()
    return None


def replace_att(toks: List[str], key: str) -> Optional[List[str]]:
    for i, t in enumerate(toks):
        if i + 1 >= len(toks) or toks[i + 1] not in NOUN:
            continue
        if t in COLORS:
            pool = [c for c in COLORS if c not in toks and c not in ("grey", "gray", "silver")]
            if not pool:
                return None
            out = list(toks)
            out[i] = seeded(key + ":col").choice(pool)
            return out
        if t in ANTONYM and ANTONYM[t] not in toks:
            out = list(toks)
            out[i] = ANTONYM[t]
            return out
    return None


def replace_obj(toks: List[str], present: Set[str], key: str) -> Optional[List[str]]:
    for i, t in enumerate(toks):
        if t not in NOUN or NOUN[t][1]:
            continue
        cat = NOUN[t][0]
        sup = SUPER_OF.get(cat)
        if not sup:
            continue
        pool = [c for c in SUPER[sup] if c != cat and c not in present and CAT_WORD[c] not in toks]
        if not pool:
            continue
        out = list(toks)
        out[i] = CAT_WORD[seeded(key + ":obj").choice(sorted(pool))]
        return out
    return None


def negatives(caption: str, present: Set[str], key: str) -> Dict[str, str]:
    """kind -> negative caption (only the kinds whose rule fires)."""
    toks = tokens(caption)
    if len(toks) < 4 or len(toks) > 30:
        return {}
    base = detok(toks)
    got = {"swap_obj": swap_obj(toks), "swap_att": swap_att(toks), "replace_rel": replace_rel(toks),
           "replace_att": replace_att(toks, key), "replace_obj": replace_obj(toks, present, key)}
    out = {}
    for k, v in got.items():
        if v is None:
            continue
        neg = detok(v)
        if neg.casefold() != base.casefold():
            out[k] = neg
    return out


def caption_text(caption: str) -> str:
    return detok(tokens(caption))


# ----------------------------------------------------------------------------- build

def build_questions(captions: Dict[int, List[str]], present: Dict[int, Set[str]], cap_kind: int,
                    per_image: int = PER_IMAGE) -> Tuple[List[Tuple[int, str, str, str]], Counter]:
    """[(image id, kind, true caption, negative)] with at most per_image per image, kinds capped at cap_kind."""
    cands: Dict[int, Dict[str, List[Tuple[str, str]]]] = {}
    stats: Counter = Counter()
    for iid in sorted(captions):
        own = {caption_text(c).casefold() for c in captions[iid]}
        by_kind: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        for n, c in enumerate(captions[iid]):
            for k, neg in negatives(c, present.get(iid, set()), "%d:%d" % (iid, n)).items():
                if neg.casefold() in own:
                    stats["neg_is_other_caption"] += 1
                    continue
                by_kind[k].append((caption_text(c), neg))
        if by_kind:
            cands[iid] = by_kind
    avail = Counter(k for bk in cands.values() for k in bk)
    stats.update({"images_with_candidates": len(cands), **{"images_with_" + k: v for k, v in avail.items()}})
    used: Counter = Counter()
    out = []
    for iid in sorted(cands, key=lambda i: seeded("img:%d" % i).random()):
        bk = cands[iid]
        took = 0
        for k in sorted(bk, key=lambda k: (avail[k], k)):   # rarest kind first
            if took >= per_image or used[k] >= cap_kind:
                continue
            true, neg = bk[k][int(seeded("%d:%s" % (iid, k)).random() * len(bk[k]))]
            out.append((iid, k, true, neg))
            used[k] += 1
            took += 1
    stats.update({"kind_" + k: v for k, v in used.items()})
    return out, stats


def hash_path(p: str):
    try:
        with open(p, "rb") as f:
            return image_hashes(f.read())
    except Exception as e:
        return {"error": str(e)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data")
    ap.add_argument("--cap-kind", type=int, default=30000)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    t0 = time.time()
    root = a.root
    board = BoardIndex.load(os.path.join(root, "v05", "board_exclusion.json"))
    sys.path[:0] = [os.path.join(HERE, "ext"), os.path.join(HERE, "pub")]
    from held import PubHeld  # pyright: ignore[reportMissingImports]
    held = PubHeld(root)
    z = zipfile.ZipFile(os.path.join(root, "raw", "coco", "annotations_trainval2014.zip"))
    caps = json.loads(z.read("annotations/captions_train2014.json"))
    inst = json.loads(z.read("annotations/instances_train2014.json"))
    cats = {c["id"]: c["name"] for c in inst["categories"]}
    present: Dict[int, Set[str]] = defaultdict(set)
    for an in inst["annotations"]:
        present[an["image_id"]].add(cats[an["category_id"]])
    captions: Dict[int, List[str]] = defaultdict(list)
    for c in caps["annotations"]:
        captions[c["image_id"]].append(c["caption"])
    drops: Counter = Counter()
    keep = {}
    for iid in sorted(captions):
        if held.is_held("coco:%d" % iid):
            drops["project_held_out"] += 1
            continue
        if board.hit_id("coco:%d" % iid):
            drops["board_source_id"] += 1
            continue
        keep[iid] = captions[iid]
    rel = {iid: "coco/train2014/COCO_train2014_%012d.jpg" % iid for iid in keep}
    with ProcessPoolExecutor(a.workers) as ex:
        hs = list(ex.map(hash_path, [os.path.join(root, rel[i]) for i in sorted(keep)], chunksize=256))
    hits = []
    for iid, h in zip(sorted(keep), hs):
        if "error" in h:
            drops["unreadable_image"] += 1
            del keep[iid]
            continue
        got = board.hit_hashes(h)
        if got:
            hits.append({"image": rel[iid], "kind": got[0], "board": got[1]})
            drops["board_image_" + got[0]] += 1
            del keep[iid]
    qs, stats = build_questions(keep, present, a.cap_kind)

    def rows():
        for n, (iid, kind, true, neg) in enumerate(qs):
            slot = int(seeded("slot:%d:%s" % (iid, kind)).random() * 2)
            opts = place_gold(true, [neg], slot)
            yield choice_sample("board-caption:%d:%s" % (iid, kind), "board-caption", "coco:%d" % iid, rel[iid],
                                QUESTION, opts, true, kind=kind)

    out = os.path.join(root, OUT)
    counts = write_split(out, rows())
    manifest(out, {"family": "caption", "board": "Winoground (text subtask)",
                   "source": "COCO captions train2014 + instances_train2014 (data/raw/coco/annotations_trainval2014.zip)",
                   "images_considered": len(captions), "drops": dict(drops), "board_exclusion_hits": hits,
                   "rule_stats": dict(stats), "split_counts": counts, "cap_kind": a.cap_kind,
                   "seconds": round(time.time() - t0)})
    print("caption: %s, drops %s, %.0f s" % (counts, dict(drops), time.time() - t0))


if __name__ == "__main__":
    main()
