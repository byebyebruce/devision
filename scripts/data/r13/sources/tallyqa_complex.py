"""Round 13 source: TallyQA complex counting questions (counting with a qualifier: "How many people are wearing
a red shirt?"). Licence Apache-2.0. Plan 1.4.

    uv run python scripts/data/r13/sources/tallyqa_complex.py --root data [--limit 15000] [--force]

Source: the original TallyQA release (github.com/manoja328/TallyQA_dataset, tallyqa.zip: train.json, test.json),
not the Cauldron copy (no picture ids, no data_source). **The released train.json has no `issimple` field** (only
test.json has it), so simple / complex is decided here with a rule that follows the authors' classifier
(tallyqacode/simpcomp.py: a question is simple when it is "<COCO name> <image template>" or its words carry no
modifier, preposition, auxiliary or object): after "How many", a question is simple when what remains, once an
image template ("are there", "are in the picture", "can you see", ...) is removed, is a single word or a COCO
category name; anything else is complex. AMT questions (data_source "amt") are complex by construction. The
rule's agreement with test.json's `issimple` is reported in the MANIFEST.

Kept: complex questions with answer 0..10 on pictures that are not held out (every COCO val2014 picture and
every picture of a project evaluation / monitoring / calibration set, by COCO / VG identity).
data_source "generate" (50,000 zero-count questions copied from other pictures) is kept only when the question
names a COCO category that has no instance in the picture's COCO annotations, so the 0 is verified; otherwise
it is dropped (an unannotated thing is not an absent thing).

Question -> choice "How many ...?" among the gold count and distractors. The first distractor is the count
**without the qualifier** when it can be derived: the answer of a simple TallyQA question on the same picture
whose noun phrase opens the complex question ("How many people are there?" for "How many people are wearing
hats?"), else the picture's COCO instance count of the named category (crowd boxes not counted); it is used
only when larger than the gold count. The other distractors are neighbouring counts (within 3, >= 0).
At most --per-picture questions per picture; AMT questions first; answers flattened (no count above twice the
mean). Pictures: COCO train2014 from data/coco (fetched when missing), Visual Genome from data/gqa/images, the
COCO copy of the same picture, or fetched from Visual Genome's server to data/r13/vg/.
"""
import json
import os
import random
import re
import sys
import zipfile
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from base import (Identity, base_args, choice, coco_instances, coco_rel, contact_sheet, describe, download,  # noqa: E402
                  fetch_coco, fetch_vg, finish, already_built, raw_dir, spot_dir)

SOURCE = "tallyqa_complex"
URL = "https://raw.githubusercontent.com/manoja328/TallyQA_dataset/master/tallyqa.zip"

TEMPLATES = ("are there in the picture", "are there in the image", "are there in this picture",
             "are there in this image", "are there in the photo", "are there in this photo",
             "are in the picture", "are in this picture", "are in the photo", "are in this photo",
             "are in the image", "are in this image", "are in the scene", "are in this scene",
             "are in the photograph", "are in this photograph", "in the picture", "in this picture",
             "in the photo", "in this photo", "in the image", "in this image",
             "are shown", "are pictured", "are visible", "are present", "can be seen", "can you see",
             "do you see", "are pictured here", "are shown here", "are seen here", "are seen", "are here",
             "are in photo", "are in picture", "in picture", "in photo", "are photographed", "appear",
             "are in this shot", "are in the shot", "are there", "is there", "there are", "there")
COCO_NAMES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
              "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
              "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
              "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
              "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
              "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
              "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
              "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
              "scissors", "teddy bear", "hair drier", "toothbrush"]
# words that name a COCO category (plural / synonyms) -> category
COCO_WORDS = {**{n: n for n in COCO_NAMES}, **{n + "s": n for n in COCO_NAMES if not n.endswith("s")},
              "people": "person", "persons": "person", "man": "person", "men": "person", "woman": "person",
              "women": "person", "child": "person", "children": "person", "kids": "person", "kid": "person",
              "boy": "person", "boys": "person", "girl": "person", "girls": "person", "players": "person",
              "player": "person", "bike": "bicycle", "bikes": "bicycle", "motorcycles": "motorcycle",
              "motorbike": "motorcycle", "motorbikes": "motorcycle", "plane": "airplane", "planes": "airplane",
              "airplanes": "airplane", "buses": "bus", "trucks": "truck", "sheep": "sheep", "knives": "knife",
              "benches": "bench", "couches": "couch", "sofa": "couch", "sofas": "couch", "doughnut": "donut",
              "doughnuts": "donut", "sandwiches": "sandwich", "glasses": "wine glass", "mice": "mouse",
              "skis": "skis", "tvs": "tv", "television": "tv", "televisions": "tv", "phone": "cell phone",
              "phones": "cell phone", "laptops": "laptop", "oranges": "orange", "carrots": "carrot",
              "giraffes": "giraffe", "zebras": "zebra", "cows": "cow", "horses": "horse", "dogs": "dog",
              "cats": "cat", "birds": "bird", "elephants": "elephant", "bears": "bear", "umbrellas": "umbrella",
              "kites": "kite", "bottles": "bottle", "cups": "cup", "bowls": "bowl", "chairs": "chair",
              "books": "book", "clocks": "clock", "vases": "vase", "pizzas": "pizza", "bananas": "banana",
              "apples": "apple", "beds": "bed", "toilets": "toilet", "sinks": "sink", "boats": "boat",
              "trains": "train", "cars": "car", "ties": "tie", "suitcases": "suitcase"}
COCO_PHRASES = sorted(set(COCO_NAMES) | {n + "s" for n in COCO_NAMES if " " in n}, key=len, reverse=True)


def remainder(question: str) -> Optional[str]:
    """What follows "How many" with an image template at the end removed (lower case), or None when the
    question does not start with "How many"."""
    s = " ".join(question.strip().rstrip("?.").split()).lower()
    if not s.startswith("how many "):
        return None
    rest = s[len("how many "):]
    for t in TEMPLATES:
        if rest.endswith(" " + t):
            return rest[: -len(t) - 1].strip()
    return rest


def is_simple(question: str) -> Optional[bool]:
    """TallyQA's simple / complex split (see the module doc); None when not a "How many" question."""
    rest = remainder(question)
    if rest is None or not rest:
        return None
    return len(rest.split()) == 1 or rest in COCO_PHRASES


def coco_category(question: str) -> Optional[str]:
    """The COCO category the question counts: the first word (or two-word name) after "How many" that names one."""
    rest = remainder(question) or ""
    words = rest.split()
    for i in range(min(len(words), 4)):
        two = " ".join(words[i:i + 2])
        for w in (two, words[i]):
            if w in COCO_WORDS:
                return COCO_WORDS[w]
    return None


def unqualified(question: str, simple: Dict[str, int]) -> Optional[int]:
    """The answer of a simple question on the same picture whose noun phrase opens `question` (`simple` maps
    noun phrase -> count)."""
    rest = remainder(question) or ""
    words = rest.split()
    for phrase in sorted(simple, key=lambda p: -len(p.split())):
        n = len(phrase.split())
        if words[:n] == phrase.split() or (n == 1 and len(words) > 1 and words[1] == phrase):
            return simple[phrase]     # "people ..." or "<adjective> people ..."
    return None


def options(gold: int, unq: Optional[int], rng: random.Random) -> List[str]:
    k = rng.choice([3, 3, 4, 4, 5]) if unq is not None else rng.choice([2, 3, 3, 4, 4, 5])
    out = [str(gold)] + ([str(unq)] if unq is not None else [])
    pool = [m for m in range(max(0, gold - 3), gold + 4) if m != gold and m != unq]
    rng.shuffle(pool)
    return out + [str(m) for m in pool[: k - len(out)]]


def picture_of(rec: dict) -> Tuple[str, str]:
    """(namespaced image id, split) of a TallyQA record: ("vg:N", "vg") or ("coco:N", "train2014"|"val2014")."""
    folder, name = rec["image"].split("/", 1)
    if folder.startswith("VG_100K"):
        return "vg:%d" % int(name.split(".")[0]), "vg"
    return "coco:%d" % int(re.search(r"(\d+)\.jpg$", name).group(1)), folder


def main(argv=None):
    p = base_args(__doc__)
    p.add_argument("--limit", type=int, default=15000)
    p.add_argument("--per-picture", type=int, default=2)
    p.add_argument("--workers", type=int, default=16)
    a = p.parse_args(argv)
    if already_built(a.root, "r13/%s/train.jsonl" % SOURCE, a.force):
        return
    rng = random.Random(a.seed)
    z = zipfile.ZipFile(download(URL, os.path.join(raw_dir(a.root, SOURCE), "tallyqa.zip")))
    train = json.load(z.open("train.json"))
    test = json.load(z.open("test.json"))
    agree = Counter((t["issimple"], is_simple(t["question"])) for t in test)
    steps, dropped = Counter(rows_read=len(train)), Counter()
    ident = Identity(a.root)
    inst = coco_instances(a.root)

    simple_by: Dict[str, Dict[str, int]] = defaultdict(dict)
    conflict = set()
    for r in train:
        if r["data_source"] != "amt" and is_simple(r["question"]):
            iid, _ = picture_of(r)
            key = remainder(r["question"])
            if key in simple_by[iid] and simple_by[iid][key] != r["answer"]:
                conflict.add((iid, key))
            simple_by[iid][key] = r["answer"]
    for iid, key in conflict:
        del simple_by[iid][key]

    cands = []
    for r in train:
        s = is_simple(r["question"])
        if s is None:
            dropped["not a 'How many' question"] += 1
            continue
        if s and r["data_source"] != "amt":
            dropped["simple question"] += 1
            continue
        steps["complex"] += 1
        iid, split = picture_of(r)
        if r["answer"] > 10:
            dropped["answer > 10"] += 1
            continue
        if ident.held(iid):
            dropped["held-out picture (COCO val2014 / evaluation sets)"] += 1
            continue
        cat = coco_category(r["question"])
        coco_id = ident.canonical(iid)
        counts = None
        if coco_id.startswith("coco:"):
            cid = int(coco_id[5:])
            if cid in inst["train2014"]["images"]:
                counts = Counter(n for n, _, crowd in inst["train2014"]["boxes"].get(cid, []) if not crowd)
        if r["data_source"] == "generate":
            if counts is None or cat is None or counts.get(cat, 0) != 0 or r["answer"] != 0:
                dropped["generate: zero count not verified by COCO annotations"] += 1
                continue
        unq, how = unqualified(r["question"], simple_by.get(iid, {})), "simple_question"
        if unq is None and counts is not None and cat is not None and r["data_source"] != "generate":
            unq, how = counts.get(cat, 0), "coco_instances"
        if unq is not None and unq < r["answer"]:
            dropped["(distractor only) unqualified count below the gold count"] += 1
            unq = None
        if unq is not None and (unq == r["answer"] or unq > 20):
            unq = None
        cands.append((r, iid, split, unq, how if unq is not None else "neighbours"))
    steps["candidates"] = len(cands)

    # AMT first, then a fixed shuffle; per picture cap; flatten answers
    rng.shuffle(cands)
    cands.sort(key=lambda c: c[0]["data_source"] != "amt")
    per = Counter()
    chosen = []
    for c in cands:
        if per[c[1]] < a.per_picture:
            per[c[1]] += 1
            chosen.append(c)
    dropped["over the per-picture cap"] = len(cands) - len(chosen)
    by_ans: Dict[int, list] = defaultdict(list)
    for c in chosen:
        by_ans[c[0]["answer"]].append(c)
    cap = int(2 * len(chosen) / max(1, len(by_ans)))
    flat = [c for v in by_ans.values() for c in v[:cap]]
    dropped["answer flattening (no count above 2x the mean)"] = len(chosen) - len(flat)
    flat.sort(key=lambda c: (c[0]["data_source"] != "amt", hash_key(c[0]["question_id"])))
    pool = flat[: int(a.limit * 1.25)]

    # pictures
    vg_ids, coco_items, rel_of = [], [], {}
    v2c = ident.v2c
    for r, iid, split, _, _ in pool:
        if split == "vg":
            c = v2c.get(iid)
            if c and os.path.exists(os.path.join(a.root, coco_rel("train2014", int(c[5:])))):
                rel_of[iid] = coco_rel("train2014", int(c[5:]))
            else:
                vg_ids.append(int(iid[3:]))
        else:
            coco_items.append((split, int(iid[5:])))
    got_vg = fetch_vg(a.root, vg_ids, a.workers)
    got_coco = fetch_coco(a.root, coco_items, a.workers)
    for i, rel in got_vg.items():
        rel_of["vg:%d" % i] = rel
    for (split, i), rel in got_coco.items():
        rel_of["coco:%d" % i] = rel
    rows = []
    for r, iid, split, unq, how in pool:
        if iid not in rel_of:
            dropped["picture could not be fetched"] += 1
            continue
        if len(rows) >= a.limit:
            break
        rows.append(choice("r13-tallyqa:%d" % r["question_id"], SOURCE, "count_complex", iid, rel_of[iid],
                           " ".join(r["question"].split()), options(r["answer"], unq, rng), str(r["answer"]), rng,
                           group=ident.canonical(iid), data_source=r["data_source"], distractor_from=how,
                           tallyqa_question_id=r["question_id"]))
    steps["questions"] = len(rows)
    near = ident.near_held({r["image"] for r in rows}, SOURCE)
    rows = [r for r in rows if r["image"] not in near]
    dropped["within dHash 2 of a held-out photo"] = len(near)
    steps["after_near_duplicate_check"] = len(rows)
    sheet = spot(a.root, rows)
    finish(a.root, SOURCE, rows, dict(steps), dict(dropped), {
        "licence": "Apache-2.0", "identity_backend": ident.backend,
        "rule_vs_test_issimple": {"%s->%s" % k: v for k, v in agree.items()},
        "data_source": dict(Counter(r["data_source"] for r in rows)),
        "distractor_from": dict(Counter(r["distractor_from"] for r in rows)),
        "count_answers": dict(sorted(Counter(r["answer_key"] for r in rows).items(), key=lambda kv: int(kv[0]))),
        "limit": a.limit, "limit_reached": len(rows) >= a.limit, "spot_sheet": sheet},
        report=report(rows, steps, dropped, agree, sheet))


# verdict from looking at the first build's sheet (seed 1 sample of 30; ids next to the sheet)
SPOT = ("24 / 30 look right; 5 cannot be confirmed at sheet size (stop signs 7, people clearly visible 4, streams "
        "of smoke 3, boats on the water 4, people giving peace signs 1); 1 looks wrong ('people wearing all black' "
        "= 1 for a skier in a red jacket). The 'generate' zero-count questions kept (COCO-verified) all looked "
        "right (4 / 4). VQA wording typos are kept ('stops signs').")


def hash_key(x) -> str:
    import hashlib
    return hashlib.sha1(("%s%s" % (SOURCE, x)).encode()).hexdigest()


def spot(root: str, rows: List[dict]) -> str:
    rng = random.Random(1)
    pick = rng.sample(rows, min(30, len(rows)))
    path = os.path.join(spot_dir(), "%s_1.png" % SOURCE)
    contact_sheet(root, [{"image": r["image"], "text": "[%s] %s" % (r["data_source"], describe(r))} for r in pick],
                  path, cols=6)
    json.dump([r["id"] for r in pick], open(path + ".ids.json", "w"))
    return path


def report(rows, steps, dropped, agree, sheet) -> str:
    return """# TallyQA complex (round 13)

Built by `scripts/data/r13/sources/tallyqa_complex.py`; counts in MANIFEST.json.

- Rows read: {read}; complex by the rule (+ AMT): {cx}; kept: {kept}.
- Dropped per rule: {dropped}
- train.json has no `issimple` field; the rule against test.json's `issimple` (true->rule): {agree}
- Spot check: {sheet} (30 questions, looked at): {spot}
""".format(spot=SPOT, read=steps["rows_read"], cx=steps["complex"], kept=len(rows), dropped=dict(dropped),
           agree={"%s->%s" % k: v for k, v in agree.items()}, sheet=sheet)


if __name__ == "__main__":
    main()
