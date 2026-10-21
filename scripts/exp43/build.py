"""exp43 inputs (critic EXP43-PREP-01..02; docs/experiments/2026-10-18-exp43-reference-plan.md; ORACLE diagnosis
conditions, not product preprocessing). For the one fixed endpoint exp42-C, six new question files beside the
reused name-question records (exp42-C's fit_O / fit_L / rl_tune / rl_tune_L at temperature 1):

  - spatial (fit_O, fit_L, tune_O, tune_L): the same picture and question, only the object's name replaced by
    where it is drawn -- "sample A" -> "the sample on the left / right", "Pair 1" -> "the pair on the left / right"
    (TEMPLATES); the side comes from the generator's own layout flag (a_left / p1_left, real_layout.families()
    replayed read-only) and is cross-checked against the object boxes of exp37's picture metadata; numbers, units,
    options and their order, gold, state_text and pixels unchanged;
  - transcript (fit, tune; the same for O and L): the original name question on the blank picture
    (exp21/images/blank.png) with state_text + both objects' full values (temperature: mass and speed of A and B;
    force: both distances with the unit), the objects in a per-family hash order (the same in both views), one
    transcript for every question of a picture; never only the asked object, never "the answer is".

Only the reading questions (mass / speed / distance); fit = exp42's 144 units (576 questions per picture kind);
tune = the 600 real-layout tune pictures (1,800). Each row keeps the original id / qid for pairing ("orig_id",
"orig_qid") and the asked object and its side ("objects", "sides"). Writes data/exp43/*.jsonl, MANIFEST.json.

    uv run python scripts/exp43/build.py --root data
"""
import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from typing import Dict, List, Tuple

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "data"))
import real_layout as rl  # type: ignore  # noqa: E402

TEMPLATES = {  # the three reading templates of the tune / fit questions, with the object replaced by its side
    "What is the mass of each particle in sample %s?": "What is the mass of each particle in the sample on the %s?",
    "What is the average particle speed in sample %s?": "What is the average particle speed in the sample on the %s?",
    "What is the distance between the magnets in Pair %s?": "What is the distance between the magnets in the pair on the %s?",
}
BLANK = "exp21/images/blank.png"
FIELDS = ("mass", "speed", "distance")
CHECKED: set = set()      # pictures whose sides were cross-checked against exp37 / exp40 box metadata


def sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def load(path: str) -> List[Dict]:
    return [json.loads(x) for x in open(path)]


def field_of(ins: str) -> str:
    return "mass" if "mass" in ins else "speed" if "speed" in ins else "distance" if "distance" in ins else ""


def split_question(ins: str) -> Tuple[str, str]:
    """(template, object letter / number) of a reading question."""
    for t in TEMPLATES:
        head, tail = t.split("%s")
        if ins.startswith(head) and ins.endswith(tail):
            return t, ins[len(head):len(ins) - len(tail)]
    raise SystemExit("not a reading template: %r" % ins)


def scenes(fams: set) -> Dict[Tuple[str, str], Tuple[str, Dict]]:
    """(family, view) -> (form, scene), replaying the generator read-only."""
    out = {}
    for i, split, form, s, twin in rl.families(6000, 300, 300):
        fam = "rl%05d" % i
        if fam in fams:
            out[(fam, "v0")], out[(fam, "v1")] = (form, s), (form, twin)
    return out


def side(form: str, scene: Dict, obj: str) -> str:
    """'left' / 'right' where the generator draws object A / B (temperature) or Pair 1 / 2 (force)."""
    first_left = scene.get("a_left", True) if form == "temp" else scene.get("p1_left", True)
    first = obj in ("A", "1")
    return "left" if first == first_left else "right"


def transcript(form: str, scene: Dict, family: str) -> str:
    flip = int(hashlib.sha256(family.encode()).hexdigest(), 16) % 2 == 1      # B / Pair 2 first in about half
    if form == "temp":
        parts = ["Sample %s: mass of each particle %d u, average particle speed %s m/s." % (
            n, scene[n]["mass"], rl.fmt(scene[n]["speed"])) for n in (("B", "A") if flip else ("A", "B"))]
    else:
        parts = ["Pair %d: the magnets are %s %s apart." % (p + 1, rl.fmt(scene["d"][p]), scene["unit"])
                 for p in ((1, 0) if flip else (0, 1))]
    return " ".join(parts)


def drawn_values(form: str, scene: Dict) -> Dict[Tuple[str, str], str]:
    """(field, object) -> the gold string the reading question takes."""
    if form == "temp":
        return {(f, n): ("%d u" % scene[n]["mass"] if f == "mass" else "%s m/s" % rl.fmt(scene[n]["speed"]))
                for n in ("A", "B") for f in ("mass", "speed")}
    return {("distance", str(p + 1)): "%s %s" % (rl.fmt(scene["d"][p]), scene["unit"]) for p in range(2)}


def box_side(meta: Dict, obj: str) -> str:
    """The side of the object's number box in exp37's metadata (the layout keeps the original sides)."""
    name = ("Sample " + obj) if obj in "AB" else ("Pair " + obj)
    xs = [(n["box"][0] + n["box"][2]) / 2 for n in meta["numbers"] if n["object"] == name]
    return "left" if sum(xs) / len(xs) < meta["size"][0] / 2 else "right"


def gold_of(g: Dict) -> str:
    return next(o for o, p in g["probabilities"].items() if p == 1.0)


def convert(rows: List[Dict], sc, meta, tag: str, kind: str) -> List[Dict]:
    """kind 'spatial' or 'transcript'; rows: the source rows (one or several questions each)."""
    out = []
    for r in rows:
        fam = r["family"]
        view = r.get("view") or r["id"].split(":")[2]
        form, scene = sc[(fam, view)]
        values = drawn_values(form, scene)
        qs, gs, objs, sides = {}, {}, {}, {}
        for q, v in r["questions"].items():
            f = field_of(v["instructions"])
            if not f:
                continue
            t, obj = split_question(v["instructions"])
            if values[(f, obj)] != gold_of(r["gold"][q]):
                raise SystemExit("%s %s: drawn value %s, gold %s" % (r["id"], q, values[(f, obj)], gold_of(r["gold"][q])))
            sd = side(form, scene, obj)
            m = meta.get("%s_%s" % (fam, view))
            CHECKED.add((fam, view)) if m is not None else None
            if m is not None and box_side(m, obj) != sd:
                raise SystemExit("%s %s: generator side %s, exp37 box side %s" % (r["id"], q, sd, box_side(m, obj)))
            qs[q] = dict(v, instructions=TEMPLATES[t] % sd) if kind == "spatial" else v
            gs[q], objs[q], sides[q] = r["gold"][q], ("sample " + obj) if obj in "AB" else ("Pair " + obj), sd
        if not qs:
            continue
        row = {k: v for k, v in r.items() if k not in ("questions", "gold", "id", "image", "image_id", "state_text")}
        row.update(id="exp43:%s:%s:%s" % (tag, kind, r["id"]), orig_id=r["id"], questions=qs, gold=gs, objects=objs,
                   sides=sides, view=view, condition=kind, oracle=True)
        if kind == "spatial":
            row.update(image=r["image"], image_id=r["image_id"], state_text=r["state_text"])
        else:
            row.update(image=BLANK, image_id="exp43:blank:%s:%s" % (fam, view),
                       state_text=r["state_text"] + "\n" + transcript(form, scene, fam))
        out.append(row)
    return out


def build(root: str) -> Dict:
    src = {"fit_O": os.path.join(root, "exp42", "fit_O.jsonl"), "fit_L": os.path.join(root, "exp42", "fit_L.jsonl"),
           "tune_O": os.path.join(root, "real_layout", "questions_tune.jsonl"), "tune_L": os.path.join(root, "exp40", "rl_tune_L.jsonl")}
    rows = {k: load(v) for k, v in src.items()}
    sc = scenes({r["family"] for rs in rows.values() for r in rs})
    meta = json.load(open(os.path.join(root, "exp37", "pictures.json")))
    meta.update(json.load(open(os.path.join(root, "exp40", "pictures_redrawn.json"))))
    files = {"fit_spatial_O": convert(rows["fit_O"], sc, meta, "fit", "spatial"),
             "fit_spatial_L": convert(rows["fit_L"], sc, meta, "fitL", "spatial"),
             "tune_spatial_O": convert(rows["tune_O"], sc, meta, "tune", "spatial"),
             "tune_spatial_L": convert(rows["tune_L"], sc, meta, "tuneL", "spatial"),
             "fit_transcript": convert(rows["fit_O"], sc, meta, "fit", "transcript"),
             "tune_transcript": convert(rows["tune_O"], sc, meta, "tune", "transcript")}
    out = os.path.join(root, "exp43")
    os.makedirs(out, exist_ok=True)
    m: Dict = {"sources": {k: sha(v) for k, v in src.items()}, "templates": TEMPLATES, "blank": BLANK, "files": {}}
    for name, rs in files.items():
        path = os.path.join(out, name + ".jsonl")
        with open(path, "w") as fh:
            for r in rs:
                fh.write(json.dumps(r) + "\n")
        sides = Counter((r["kind"], r["sides"][q], o.split()[-1]) for r in rs for q, o in r["objects"].items())
        m["files"][name] = {"rows": len(rs), "questions": sum(len(r["questions"]) for r in rs), "sha256": sha(path),
                            "by_field": dict(Counter(field_of(v["instructions"]) for r in rs for v in r["questions"].values())),
                            "first_object_side": {"%s:%s" % (k, s): n for (k, s, o), n in sorted(sides.items()) if o in ("A", "1")}}
    m["questions"] = sum(f["questions"] for f in m["files"].values())
    m["pictures"] = len({(r["family"], r["view"]) for rs in files.values() for r in rs})
    m["sides_cross_checked_against_box_metadata"] = len(CHECKED)
    json.dump(m, open(os.path.join(out, "MANIFEST.json"), "w"), indent=1)
    return m


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    print(json.dumps(build(p.parse_args().root), indent=1))
