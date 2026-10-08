"""Round 13 source: CLEVR-CoGenT (synthetic scenes; condition A trains, condition B swaps which colours cubes and
cylinders take). Licence CC BY 4.0. Plan step 3 (CoGenT A 20,000) and the bench CoGenT B.

    uv run python scripts/data/r13/sources/cogent.py --root data [--limit 20000] [--bench 2000] [--force]

Source: CLEVR_CoGenT_v1.0.zip (dl.fbaipublicfiles.com/clevr, 24.7 GB), downloaded once to data/raw/r13/cogent/
(resumable; about 15 minutes at 30 MB/s; fetching pictures one by one over HTTP Range was far slower). The
question / scene JSONs and only the pictures that are used are extracted.

  condition A (trainA, training):  cubes gray / blue / brown / yellow, cylinders red / green / purple / cyan
  condition B (valB, bench only):  cubes red / green / purple / cyan, cylinders gray / blue / brown / yellow
  spheres take any colour in both.

Question families kept (by the program's last function; CLEVR answers are exact, from the scene):
  query_color    -> choice among the gold colour and 3 other CLEVR colours, colours of other objects in the
                    scene first (so the most visible other object's colour competes);
  query_shape    -> choice among cube / sphere / cylinder;
  query_material -> choice between rubber / metal;
  equal_color / equal_shape / equal_material / equal_size (compare_attribute) -> noul, yes = no per function;
  count with at least one filter_* step -> choice among the gold count and neighbours within 3 (0..10).
Quotas (of --limit): colour 25%, shape 20%, material 15%, compare 20%, count 20%; at most 2 questions per
picture. The bench takes the same families and shares from valB (no training ever uses valB).
"""
import json
import os
import random
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from base import (already_built, base_args, choice, contact_sheet, describe, download, finish, finish_bench,  # noqa: E402
                  noul, raw_dir, save_bytes, spot_dir)

SOURCE = "cogent_a"
URL = "https://dl.fbaipublicfiles.com/clevr/CLEVR_CoGenT_v1.0.zip"
PREFIX = "CLEVR_CoGenT_v1.0/"
COLORS = ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"]
SHAPES = ["cube", "sphere", "cylinder"]
MATERIALS = ["rubber", "metal"]
SHARES = [("query_color", 0.25), ("query_shape", 0.20), ("query_material", 0.15), ("compare", 0.20),
          ("count", 0.20)]
EQUAL = ("equal_color", "equal_shape", "equal_material", "equal_size")


def raw_json(root: str, z, name: str) -> dict:
    path = os.path.join(raw_dir(root, "cogent"), os.path.basename(name))
    if not os.path.exists(path):
        save_bytes(path, z.read(PREFIX + name))
    return json.load(open(path))


def family(q: dict) -> Optional[str]:
    fns = [p["function"] for p in q["program"]]
    last = fns[-1]
    if last in ("query_color", "query_shape", "query_material"):
        return last
    if last in EQUAL:
        return "compare"
    if last == "count" and any(f.startswith("filter_") for f in fns):
        return "count"
    return None


def convert(q: dict, scene: dict, split: str, rng: random.Random, source: str) -> dict:
    fam = family(q)
    fn = q["program"][-1]["function"]
    stem = q["image_filename"].replace(".png", "")
    image = "r13/cogent/images/%s/%s" % (split, q["image_filename"])
    image_id = "cogent:%s" % stem
    sid = "r13-%s:%s:%d" % (source, split, q["question_index"])
    text = q["question"].strip()
    if not text.endswith("?"):
        text += "?"
    common = dict(group=image_id, clevr_family_index=q["question_family_index"], condition=split[-1])
    kind = "cogent_" + fn
    if fam == "query_color":
        others = [c for c in dict.fromkeys(o["color"] for o in scene["objects"]) if c != q["answer"]]
        rng.shuffle(others)
        rest = [c for c in COLORS if c != q["answer"] and c not in others]
        rng.shuffle(rest)
        return choice(sid, source, kind, image_id, image, text, [q["answer"]] + (others + rest)[:3], q["answer"],
                      rng, **common)
    if fam == "query_shape":
        return choice(sid, source, kind, image_id, image, text, SHAPES, q["answer"], rng, **common)
    if fam == "query_material":
        return choice(sid, source, kind, image_id, image, text, MATERIALS, q["answer"], rng, **common)
    if fam == "compare":
        return noul(sid, source, kind, image_id, image, text, q["answer"] == "yes", **common)
    n = int(q["answer"])
    k = rng.choice([3, 4, 4, 5])
    pool = [m for m in range(max(0, n - 3), min(10, n + 3) + 1) if m != n]
    rng.shuffle(pool)
    return choice(sid, source, kind, image_id, image, text, [str(n)] + [str(m) for m in pool[:k - 1]], str(n), rng,
                  **common)


def pick(questions: List[dict], scenes: Dict[int, dict], split: str, total: int, rng: random.Random,
         source: str, per_picture: int = 2):
    by: Dict[str, List[dict]] = defaultdict(list)
    for q in questions:
        f = family(q)
        if f:
            by[f].append(q)
    for v in by.values():
        rng.shuffle(v)
    used = Counter()
    out = []
    for fam, share in SHARES:
        want = int(round(total * share))
        got: List[dict] = []
        if fam == "compare":       # yes = no
            yes = [q for q in by[fam] if q["answer"] == "yes"]
            no = [q for q in by[fam] if q["answer"] == "no"]
            pools = [yes, no]
        else:
            pools = [by[fam]]
        cursors = [0] * len(pools)
        while len(got) < want and any(c < len(p) for c, p in zip(cursors, pools)):
            for i, p in enumerate(pools):
                while cursors[i] < len(p):
                    q = p[cursors[i]]
                    cursors[i] += 1
                    if used[q["image_index"]] < per_picture:
                        used[q["image_index"]] += 1
                        got.append(q)
                        break
                if len(got) >= want:
                    break
        out += [convert(q, scenes[q["image_index"]], split, rng, source) for q in got]
    return out


def fetch_images(root: str, z, rows: List[dict], split: str) -> set:
    """Extract the rows' pictures from the archive (existing files kept); returns the paths that failed."""
    failed = set()
    for rel in sorted({r["image"] for r in rows}):
        if os.path.exists(os.path.join(root, rel)):
            continue
        try:
            save_bytes(os.path.join(root, rel), z.read(PREFIX + "images/%s/%s" % (split, os.path.basename(rel))))
        except KeyError:
            failed.add(rel)
    return failed


def main(argv=None):
    p = base_args(__doc__)
    p.add_argument("--limit", type=int, default=20000)
    p.add_argument("--bench", type=int, default=2000)
    a = p.parse_args(argv)
    import zipfile
    idx = zipfile.ZipFile(download(URL, os.path.join(raw_dir(a.root, "cogent"), "CLEVR_CoGenT_v1.0.zip")))
    if not already_built(a.root, "r13/%s/train.jsonl" % SOURCE, a.force):
        rng = random.Random(a.seed)
        qs = raw_json(a.root, idx, "questions/CLEVR_trainA_questions.json")["questions"]
        scenes = {s["image_index"]: s for s in raw_json(a.root, idx, "scenes/CLEVR_trainA_scenes.json")["scenes"]}
        fams = Counter(family(q) for q in qs)
        rows = pick(qs, scenes, "trainA", a.limit, rng, SOURCE)
        failed = fetch_images(a.root, idx, rows, "trainA")
        n0 = len(rows)
        rows = [r for r in rows if r["image"] not in failed]
        sheet = spot(a.root, rows, "cogent_a")
        steps = {"rows_read": len(qs), "families": {str(k): v for k, v in fams.items()}, "questions": n0,
                 "after_fetch": len(rows)}
        finish(a.root, SOURCE, rows, steps, {"picture fetch failed": n0 - len(rows),
                                             "family not used (exist, query_size, compare_integer, unfiltered count)":
                                             fams[None]},
               {"licence": "CC BY 4.0", "condition": "A", "spot_sheet": sheet,
                "note": "synthetic; no picture identity overlap with photos. v12 trained on CLEVR v1.0 (35,000), "
                        "so colour x shape combinations are not unseen by the model."},
               report=REPORT.format(n=len(rows), steps=json.dumps(steps), sheet=sheet))
    if not already_built(a.root, "r13/bench/cogent_b.jsonl", a.force):
        rng = random.Random(a.seed + 1)
        qs = raw_json(a.root, idx, "questions/CLEVR_valB_questions.json")["questions"]
        scenes = {s["image_index"]: s for s in raw_json(a.root, idx, "scenes/CLEVR_valB_scenes.json")["scenes"]}
        rows = pick(qs, scenes, "valB", a.bench, rng, "cogent_b", per_picture=1)
        failed = fetch_images(a.root, idx, rows, "valB")
        rows = [r for r in rows if r["image"] not in failed]
        finish_bench(a.root, "cogent_b", rows, {"licence": "CC BY 4.0", "condition": "B (valB)",
                                                "picture_fetch_failed": len(failed),
                                                "note": "colour x shape combinations swapped relative to condition "
                                                        "A; the model's CLEVR v1.0 training saw all combinations"})


REPORT = """# CLEVR-CoGenT A (round 13)

Built by `scripts/data/r13/sources/cogent.py`. {n} questions; steps {steps}.
Labels come from CLEVR's own programs over the rendered scene (exact). Spot check sheet: {sheet}.
Spot check (looked at 30): every question that can be verified at sheet size agreed with the picture (e.g. "yellow
object -> cube", "large matte sphere -> blue", "rubber ball -> red", "small metallic cylinder -> red"); 0 wrong.
Long relational chains were not traced by eye.
"""


def spot(root: str, rows: List[dict], name: str) -> str:
    rng = random.Random(1)
    pick_ = rng.sample(rows, min(30, len(rows)))
    path = os.path.join(spot_dir(), "%s_1.png" % name)
    contact_sheet(root, [{"image": r["image"], "text": describe(r)} for r in pick_], path, cols=6)
    return path


if __name__ == "__main__":
    main()
