"""exp40 decision mixes (critic EXP40-PREP-01; docs/experiments/2026-10-18-exp40-pair-plan.md). Two arms trained
independently from exp37-L's end, the same budget as candidate B (26,384 questions); only the real-layout block
differs from B, and only in its second exposure's picture:

  - B's other blocks unchanged: r14b replay 12,000, exp34 semantics 2,380, exp25 whole-family panels 1,920 picture
    + 484 text (the rows of data/v04b/train_mix.jsonl that are not real-layout, as they are);
  - real layout: of B's 600 temperature and 600 force training families, 300 of each by random.Random(40) (never
    by a model output), every family with its two value views and all their questions, options and golds, asked
    TWICE -- arm O: both exposures on the original picture; arm M: the first on the original picture, the second
    on exp37's L rendering of the same scene (shared layout, digit ink about 16 px at 256). Still 9,600
    questions: half as many families as B, each exposed twice (not more data).
  - every real-layout question becomes a one-question row with a normalised qid (field: mass / speed / distance,
    or cmp for the ScienceQA-style question), group = family; so a unit (group, qid) holds the A / B (Pair 1 / 2)
    reading questions of one field x the two views x the two exposures (8), or the comparison question x two
    views x two exposures (4). Text, options and gold are those of B's rows, unchanged.

The two arms' files are identical row by row apart from the second exposure's image / image_id (checked here), so
their sample ids, question order, golds and batch plan are the same. L pictures: exp37's where they exist; the
missing ones re-drawn by scripts/exp37/render.py from the same training scenes (real_layout.families()) into
data/exp40/images/L -- the numbers in each L picture are checked against the row's reading golds. Also writes
data/exp40/rl_tune_L.jsonl (the real-layout tune decision questions on exp37's L tune pictures; report only) and
data/exp40/MANIFEST.json (sources, choices, per-block counts, sha256).

    uv run python scripts/exp40/build.py --root data
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter
from typing import Dict, List

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "data"))
sys.path.insert(0, os.path.join(HERE, "..", "exp37"))
import real_layout as rl  # type: ignore  # noqa: E402

ARMS = ("O", "M")


def sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def field_of(ins: str) -> str:
    """The normalised qid: the field a reading question asks, or cmp for the ScienceQA-style question."""
    return "mass" if "mass" in ins else "speed" if "speed" in ins else "distance" if "distance" in ins else "cmp"


def object_of(ins: str) -> str:
    m = re.search(r"(sample [AB]|Pair [12])", ins)
    return m.group(1) if m else ""


def gold_of(row: Dict, qid: str) -> str:
    return next(o for o, p in row["gold"][qid]["probabilities"].items() if p == 1.0)


def choose(rows: List[Dict], seed: int = 40, per_form: int = 300) -> List[str]:
    fams = {"temp": sorted({r["family"] for r in rows if r["kind"] == "temp"}),
            "force": sorted({r["family"] for r in rows if r["kind"] == "force"})}
    rng = random.Random(seed)
    return sorted(f for form in ("temp", "force") for f in rng.sample(fams[form], per_form))


def l_path(root: str, fam: str, view: str) -> str:
    rel = "exp37/images/L/%s_%s.png" % (fam, view)
    return rel if os.path.exists(os.path.join(root, rel)) else "exp40/images/L/%s_%s.png" % (fam, view)


def render_missing(root: str, fams: List[str]) -> Dict[str, Dict]:
    """Draw the L pictures exp37 did not draw, from the same training scenes, with exp37's renderer; their number
    boxes / texts as exp37's pictures.json records them."""
    import render as r37  # type: ignore
    want = {f for f in fams if not os.path.exists(os.path.join(root, "exp37/images/L/%s_v0.png" % f))
            or not os.path.exists(os.path.join(root, "exp37/images/L/%s_v1.png" % f))}
    out = os.path.join(root, "exp40", "images", "L")
    os.makedirs(out, exist_ok=True)
    meta: Dict[str, Dict] = {}
    if not want:
        return meta
    last = max(int(f[2:]) for f in want)
    for i, split, form, s, twin in rl.families(6000, 300, 300):
        fam = "rl%05d" % i
        if fam in want:
            assert split == "train", fam
            for k, scene in enumerate((s, twin)):
                img, numbers = r37.render(form, scene, "L")
                img.save(os.path.join(out, "%s_v%d.png" % (fam, k)))
                meta["%s_v%d" % (fam, k)] = {"family": fam, "view": k, "form": form, "split": split, "numbers": numbers,
                                            "size": list(img.size)}
        if i >= last:
            break
    assert {m["family"] for m in meta.values()} == want
    return meta


def numbers_match(row: Dict, numbers: List[Dict]) -> bool:
    """The L picture shows the values the row's reading questions take as gold (field, object, number)."""
    drawn = {(n["field"], n["object"].lower(), n["text"]) for n in numbers}
    for qid, q in row["questions"].items():
        f = field_of(q["instructions"])
        if f != "cmp":
            g = gold_of(row, qid).split(" ")[0]
            if (f, object_of(q["instructions"]).lower(), g) not in drawn:
                return False
    return True


def split_rows(row: Dict, exposure: int, image: str, image_id: str) -> List[Dict]:
    fam, view = row["family"], row["id"].split(":")[2]
    out = []
    for qid, q in row["questions"].items():
        f = field_of(q["instructions"])
        keep = {k: v for k, v in row.items() if k not in ("id", "questions", "gold", "image", "image_id")}
        out.append(dict(keep, id="exp40:%s:%s:%s:e%d" % (fam, view, qid, exposure), image=image, image_id=image_id,
                        view=view, exposure=exposure, orig_id=row["id"], orig_qid=qid,
                        questions={f: q}, gold={f: row["gold"][qid]}))
    return out


def build(root: str, seed: int = 40) -> Dict:
    b = os.path.join(root, "v04b", "train_mix.jsonl")
    rows = [json.loads(x) for x in open(b)]
    real = [r for r in rows if r.get("source", "").startswith("real-layout")]
    other = [r for r in rows if not r.get("source", "").startswith("real-layout")]
    fams = choose(real, seed)
    chosen = set(fams)
    pick = sorted((r for r in real if r["family"] in chosen), key=lambda r: r["id"])
    assert len(pick) == 2 * len(fams) and Counter(r["family"] for r in pick) == Counter({f: 2 for f in fams})
    new_meta = render_missing(root, fams)
    l37 = json.load(open(os.path.join(root, "exp37", "pictures.json")))
    checked = 0
    for r in pick:
        key = "%s_%s" % (r["family"], r["id"].split(":")[2])
        meta = new_meta.get(key) or l37[key]
        if meta.get("split", "train") != "train" or not numbers_match(r, meta["numbers"]):
            raise SystemExit("L picture %s does not show the row's values" % key)
        checked += 1
    blocks: Dict[str, List[Dict]] = {}
    for arm in ARMS:
        rl_rows = []
        for r in pick:
            view = r["id"].split(":")[2]
            rl_rows += split_rows(r, 0, r["image"], r["image_id"])
            if arm == "O":
                rl_rows += split_rows(r, 1, r["image"], r["image_id"])
            else:
                rl_rows += split_rows(r, 1, l_path(root, r["family"], view), "exp40-L:%s:%s" % (r["family"], view))
        mix = other + rl_rows
        random.Random(seed).shuffle(mix)
        blocks[arm] = mix
    strip = lambda r: {k: v for k, v in r.items() if k not in ("image", "image_id")}  # noqa: E731
    assert [strip(x) for x in blocks["O"]] == [strip(x) for x in blocks["M"]]
    differ = sum(x["image"] != y["image"] for x, y in zip(blocks["O"], blocks["M"]))
    out = os.path.join(root, "exp40")
    os.makedirs(out, exist_ok=True)
    m: Dict = {"seed": seed, "sources": {"v04b_train_mix": sha(b), "exp37_pictures": sha(os.path.join(root, "exp37", "pictures.json"))},
               "families": fams, "families_by_form": dict(Counter(r["kind"] for r in pick if r["id"].endswith(":v0"))),
               "l_pictures": {"from_exp37": sum(not k in new_meta for k in ("%s_%s" % (r["family"], r["id"].split(":")[2]) for r in pick)),
                              "redrawn": len(new_meta), "numbers_checked_against_golds": checked},
               "rows_differing_between_arms": differ, "arms": {}}
    json.dump(new_meta, open(os.path.join(out, "pictures_redrawn.json"), "w"))
    for arm, mix in blocks.items():
        path = os.path.join(out, "train_%s.jsonl" % arm)
        with open(path, "w") as fh:
            for r in mix:
                fh.write(json.dumps(r) + "\n")
        q = Counter()
        for r in mix:
            src = r.get("source", "")
            blk = ("real_layout" if src.startswith("real-layout") else "exp25_panels:text" if src == "exp21-syn-text"
                   else "exp25_panels:picture" if src == "exp21-syn" else "semantics" if src.startswith("exp34") else "r14b_replay")
            q[blk] += len(r["questions"])
        m["arms"][arm] = {"rows": len(mix), "questions": sum(q.values()), "questions_by_block": dict(sorted(q.items())),
                          "real_layout_images": dict(Counter(r["image"].split("/")[0] for r in mix if r.get("source", "").startswith("real-layout"))),
                          "sha256": sha(path)}
    tune = os.path.join(root, "real_layout", "questions_tune.jsonl")
    dst = os.path.join(out, "rl_tune_L.jsonl")
    with open(dst, "w") as fh:
        for r in map(json.loads, open(tune)):
            fam, view = r["family"], r["id"].split(":")[2]
            rel = "exp37/images/L/%s_%s.png" % (fam, view)
            if not os.path.exists(os.path.join(root, rel)) or not numbers_match(r, l37["%s_%s" % (fam, view)]["numbers"]):
                raise SystemExit("tune L picture %s missing or wrong" % rel)
            fh.write(json.dumps(dict(r, image=rel, image_id="exp40-L:%s:%s" % (fam, view))) + "\n")
    m["rl_tune_L"] = {"source_sha256": sha(tune), "sha256": sha(dst)}
    json.dump(m, open(os.path.join(out, "MANIFEST.json"), "w"), indent=1)
    return m


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    m = build(p.parse_args().root)
    print(json.dumps({k: v for k, v in m.items() if k != "families"}, indent=1))
