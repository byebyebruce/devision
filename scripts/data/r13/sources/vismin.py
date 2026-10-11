"""Round 13 source: VisMin training set (a COCO photo and minimally edited copies: one object, attribute, count or
spatial relation changed, with a caption for each). Licence CC BY-SA 4.0. Plan 1.4.

    uv run python scripts/data/r13/sources/vismin.py --root data [--pairs 15000] [--force]

Source: mair-lab/vismin (Hugging Face, data/train/*.arrow, 39 GB, downloaded once into data/raw/hf). The rows are
flat: an original has an empty `source_image_id`; an edited row names its original in `source_image_id` and has
`category` (object / attribute / counting / relation) and `edit_instruction` [before, after]. Rows are not pairs:
families are rebuilt by `source_image_id` and a pair is (original, one edited copy).

Pair -> two noul questions with the same wording "Is it true that <caption>?" on both pictures: true on the
picture the caption belongs to, false on the other. Whose caption is asked alternates by a hash of the pair (so
"the edited picture is the true one" is not a shortcut). Rules:
  - a COCO original (object / attribute families) must not be held out (every COCO val2014 picture -- VisMin
    uses COCO 2017 train, which contains val2014 pictures -- and every project evaluation / monitoring /
    calibration picture); the edited copy inherits the original's identity (image_id
    "coco:<id>:vismin-<edit id>", group "coco:<id>"). Most counting / relation families start from a generated
    picture (hex id): image_id "vismin:<id>", group "vismin:<original id>";
  - object / attribute pairs: `before` must appear in the original's caption and `after` in the edited caption
    (normalised words), so the two captions differ by the edit; counting / relation edits are written as counts
    or relations ("2 giraffes", "left of") the captions word differently, so for them only "captions differ" is
    checked and the spot check decides;
  - both pictures decode; at most 2 pairs per original; at most --pairs pairs, smaller categories taken whole first.
Spot check (plan 1.4): 30 pairs per category on contact sheets (runs/r13/spot/vismin_<category>_*.png, original
left, edited right, the asked caption under them), judged by looking; a category with more than 15% wrong pairs
is dropped (SPOT below records the verdicts; DROP the dropped categories).
Each sample keeps `vismin_category`, `vismin_pair`, `edit_instruction` and `edit_boxes` (the edited region,
[x, y, w, h] pixels of the picture file; not a target box for the asked caption, so not under `boxes`).
"""
import glob
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from base import (Identity, already_built, base_args, contact_sheet, finish, hf_files, noul, save_bytes,  # noqa: E402
                  spot_dir)

SOURCE = "vismin"
REPO = "mair-lab/vismin"
CATEGORIES = ("object", "attribute", "counting", "relation")
MAX_WRONG = 0.15
# spot-check verdicts (pairs looked at, pairs wrong) per category, filled in after looking at the sheets
# Looked at 2026-10-08 (sheets runs/r13/spot/vismin_<category>_<n>.png). Wrong = the asked caption does not hold
# on its own picture, or also holds on the other one.
#   object 2 / 30: "suitcase in front of a refrigerator / wardrobe" (neither visible, pictures alike), "on a bed /
#     on the floor" (the edit still shows a bed);
#   attribute 2 / 30: "woman in the navy / in the navy uniform" (both true on both), "lush green field / dry
#     grassland" (the original is already dry);
#   relation 4 / 30: "dog to the left of a cat" (the original shows it on the right), "basketball above a frisbee"
#     (reversed), "horse bridle above a water bucket" (reversed), "a soda can is holding a hot dog" (broken caption);
#   counting 5 / 10: "five / three small toys", "six / five wooden lids" (the two pictures look the same), "seven /
#     four balloons", "seven / five basketballs", "four / one weird hats" (counts off in at least one picture).
#     Five wrong already exceeds 15% of 30, so the other 20 were not needed: the category is dropped.
SPOT: Dict[str, tuple] = {"object": (30, 2), "attribute": (30, 2), "relation": (30, 4), "counting": (10, 5)}
DROP: set = {c for c, (n, bad) in SPOT.items() if bad / n > MAX_WRONG}


def norm(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())


def caption_question(caption: str) -> str:
    c = " ".join(caption.split()).strip().rstrip(".").strip()
    if len(c) > 1 and c[0].isupper() and not c[1].isupper():
        c = c[0].lower() + c[1:]
    return "Is it true that %s?" % c


def text_ok(cat: str, before: str, after: str, orig_caption: str, edit_caption: str) -> bool:
    if norm(orig_caption) == norm(edit_caption):
        return False
    if cat in ("object", "attribute"):
        return norm(before) in norm(orig_caption) and norm(after) in norm(edit_caption)
    return True


def read_rows(root: str):
    """All rows without picture bytes, plus (shard path, row index) per image id."""
    import pyarrow as pa
    files = hf_files(root, REPO, "data/train/*.arrow")
    rows, where = [], {}
    for f in files:
        with pa.memory_map(f) as src:
            t = pa.ipc.open_stream(src).read_all()
        for i, r in enumerate(t.drop_columns(["image"]).to_pylist()):
            rows.append(r)
            where[r["image_id"]] = (f, i)
    return rows, where


def extract(root: str, where: Dict[str, tuple], ids: List[str]) -> Dict[str, str]:
    """image id -> picture path under root (data/r13/vismin/images/<id>.<ext>), extracted once."""
    import pyarrow as pa
    by_file: Dict[str, list] = defaultdict(list)
    out = {}
    for i in ids:
        for ext in (".jpg", ".png"):
            rel = "r13/vismin/images/%s%s" % (i, ext)
            if os.path.exists(os.path.join(root, rel)):
                out[i] = rel
                break
        else:
            by_file[where[i][0]].append(i)
    for f, need in sorted(by_file.items()):
        with pa.memory_map(f) as src:
            t = pa.ipc.open_stream(src).read_all()
        col = t.column("image")
        for i in need:
            data = col[where[i][1]].as_py()["bytes"]
            ext = ".png" if data[:8] == b"\x89PNG\r\n\x1a\n" else ".jpg"
            rel = "r13/vismin/images/%s%s" % (i, ext)
            save_bytes(os.path.join(root, rel), data)
            out[i] = rel
    return out


def pixel_boxes(raw: str, size) -> List[List[float]]:
    try:
        bs = json.loads(raw) if raw else []
    except ValueError:
        return []
    if bs and isinstance(bs[0], (int, float)):      # a single box written flat
        bs = [bs]
    w, h = size
    return [[round(b[0] * w, 1), round(b[1] * h, 1), round((b[2] - b[0]) * w, 1), round((b[3] - b[1]) * h, 1)]
            for b in bs if isinstance(b, list) and len(b) == 4]


def main(argv=None):
    p = base_args(__doc__)
    p.add_argument("--pairs", type=int, default=15000)
    p.add_argument("--per-original", type=int, default=2)
    a = p.parse_args(argv)
    if already_built(a.root, "r13/%s/train.jsonl" % SOURCE, a.force):
        return
    rows, where = read_rows(a.root)
    steps, dropped = Counter(rows_read=len(rows)), Counter()
    orig = {r["image_id"]: r for r in rows if not r["source_image_id"]}
    edits = [r for r in rows if r["source_image_id"]]
    steps["originals"], steps["edited"] = len(orig), len(edits)
    steps["edited_by_category"] = dict(Counter(r["category"] for r in edits))
    ident = Identity(a.root)
    cands = []
    for e in edits:
        o = orig.get(e["source_image_id"])
        if o is None:
            dropped["original not in the dataset"] += 1
            continue
        if e["category"] not in CATEGORIES:
            dropped["unknown category"] += 1
            continue
        # object / attribute edits start from COCO photos; most counting / relation families start from a
        # generated picture (hex id), which has no photo identity to check
        oid = "coco:%d" % int(o["image_id"]) if o["image_id"].isdigit() else "vismin:%s" % o["image_id"]
        if ident.held(oid):
            dropped["held-out original (COCO val2014 / evaluation sets), with its edits"] += 1
            continue
        try:
            before, after = json.loads(e["edit_instruction"])
        except (ValueError, TypeError):
            dropped["edit instruction unreadable"] += 1
            continue
        if not text_ok(e["category"], before, after, o["caption"], e["caption"]):
            dropped["captions do not differ by the edit (%s)" % e["category"]] += 1
            continue
        cands.append((o, e, oid, before, after))
    steps["candidate_pairs"] = len(cands)
    steps["candidate_pairs_by_category"] = dict(Counter(c[1]["category"] for c in cands))
    # categories interleaved (one pair of each in turn, in a fixed hash order), so --pairs keeps them balanced
    by_cat: Dict[str, list] = defaultdict(list)
    for c in sorted(cands, key=lambda c: _h(c[1]["image_id"])):
        if c[1]["category"] in DROP:
            dropped["category dropped by the spot check (%s)" % c[1]["category"]] += 1
        else:
            by_cat[c[1]["category"]].append(c)
    per = Counter()
    pairs = []
    queues = [list(v) for _, v in sorted(by_cat.items())]
    while queues and len(pairs) < a.pairs:
        for q in queues:
            while q:
                c = q.pop(0)
                if per[c[2]] < a.per_original:
                    per[c[2]] += 1
                    pairs.append(c)
                    break
                dropped["over 2 pairs per original"] += 1
            if len(pairs) >= a.pairs:
                break
        queues = [q for q in queues if q]
    # spot sample: 30 per category from the candidates (dropped categories included, so the verdict is visible)
    spot_pairs = {}
    for cat in CATEGORIES:
        pool = [c for c in cands if c[1]["category"] == cat]
        spot_pairs[cat] = random.Random(1).sample(pool, min(30, len(pool)))
    need = {x for c in pairs for x in (c[0]["image_id"], c[1]["image_id"])}
    need |= {x for v in spot_pairs.values() for c in v for x in (c[0]["image_id"], c[1]["image_id"])}
    rel = extract(a.root, where, sorted(need))
    sizes = {i: _size(os.path.join(a.root, r)) for i, r in rel.items()}
    out = []
    for o, e, oid, before, after in pairs:
        if not sizes.get(o["image_id"]) or not sizes.get(e["image_id"]):
            dropped["picture does not decode"] += 1
            continue
        ask_orig = int(_h("ask" + e["image_id"])[:2], 16) % 2 == 0
        cap = o["caption"] if ask_orig else e["caption"]
        q = caption_question(cap)
        eid = "%s:vismin-%s" % (oid, e["image_id"])
        common = dict(group=ident.canonical(oid), vismin_category=e["category"], vismin_pair=e["image_id"],
                      edit_instruction=[before, after], asked_caption_of="original" if ask_orig else "edited")
        for which, iid, img_id in (("original", oid, o["image_id"]), ("edited", eid, e["image_id"])):
            truth = (which == "original") == ask_orig
            r = noul("r13-vismin:%s:%s" % (e["image_id"], which), SOURCE, "vismin_%s" % e["category"], iid,
                     rel[img_id], q, truth, **common)
            r["edit_boxes"] = pixel_boxes(e["bounding_boxes"], sizes[img_id])
            r["picture_role"] = which
            out.append(r)
    near = ident.near_held({r["image"] for r in out}, SOURCE)
    bad = {r["vismin_pair"] for r in out if r["image"] in near}
    dropped["pair with a picture within dHash 2 of a held-out photo / bench picture"] = len(bad)
    out = [r for r in out if r["vismin_pair"] not in bad]
    steps["pairs"] = len(out) // 2
    steps["questions"] = len(out)
    steps["pairs_by_category"] = dict(Counter(r["vismin_category"] for r in out if r["picture_role"] == "edited"))
    sheets = spot(a.root, spot_pairs, rel)
    finish(a.root, SOURCE, out, dict(steps), dict(dropped), {
        "licence": "CC BY-SA 4.0", "identity_backend": ident.backend, "spot_results": {
            k: {"looked": v[0], "wrong": v[1], "share_wrong": round(v[1] / max(1, v[0]), 3)} for k, v in SPOT.items()},
        "dropped_categories": sorted(DROP), "pairs_limit": a.pairs, "spot_sheets": sheets},
        report=REPORT.format(steps=dict(steps), dropped=dict(dropped), sheets=sheets, spot=SPOT, drop=sorted(DROP)))


REPORT = """# VisMin (round 13)

Built by `scripts/data/r13/sources/vismin.py`; counts in MANIFEST.json.

- Steps: {steps}
- Dropped per rule: {dropped}
- Spot check (pairs looked at, wrong) per category: {spot}; dropped categories (> 15% wrong): {drop}
- Sheets (original left, edited right; the caption is the one asked, T/F = its truth on each picture): {sheets}
"""


def _h(x) -> str:
    return hashlib.sha1(("%s%s" % (SOURCE, x)).encode()).hexdigest()


def _size(path: str):
    from PIL import Image
    try:
        with Image.open(path) as im:
            im.load()
            return im.size
    except Exception:
        return None


def spot(root: str, spot_pairs: Dict[str, list], rel: Dict[str, str]) -> List[str]:
    """Sheets of 10 pairs each (original, edited side by side)."""
    paths = []
    for cat, pairs in spot_pairs.items():
        for k in range(0, len(pairs), 10):
            items = []
            for o, e, _, before, after in pairs[k:k + 10]:
                items.append({"image": rel[o["image_id"]], "text": "ORIG %s | caption: %s" % (
                    json.dumps([before, after]), o["caption"])})
                items.append({"image": rel[e["image_id"]], "text": "EDIT | caption: %s" % e["caption"]})
            path = os.path.join(spot_dir(), "%s_%s_%d.png" % (SOURCE, cat, k // 10 + 1))
            contact_sheet(root, items, path, cols=4, cell=280)
            paths.append(path)
    return paths


if __name__ == "__main__":
    main()
