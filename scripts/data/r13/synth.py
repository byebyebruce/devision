"""Round 13 counterfactual binding scenes (docs/experiments/2026-10-08-r13-binding-plan.md, revision 2, 1.2).

    uv run python scripts/data/r13/synth.py --root data --train 1200 --dev 1200 --test 1200   # -> data/r13/synth-v2/

A scene family is one base scene with 2 kinds of shapes (shape + colour + size per kind, 1-4 of each, 3-6
objects), laid out ONCE, and three minimal edits of it, all in one `group`:

  base       the scene
  recolour   kind 0's instances get a new colour; nothing else changes (same boxes, same pixels elsewhere)
  recolour2  the same for kind 1
  count      one instance of one kind removed, or one added in free space; every other instance stays put

The whole family is asked ONE fixed question set (same wording and option order on every picture), about EVERY
kind: colour (the family's four colours as options), count (every count any variant needs among the options),
and three "Is there a <colour> <shape>?" statements per kind -- own (the kind's base colour: true, false after
its recolour), trap (the other kind's base colour: false while that colour is on the other kind), new (the
kind's recoloured colour: false, true after its recolour) -- plus "Are there more Xs than Ys?". So a target edit
flips answers, a distractor edit must not, and asking about the other kind on the same picture changes them.

Per question `meta` records the task (colour, count, exists_true / exists_trap / exists_absent from the gold on
that picture, compare), the salience stratum (target kind's total area / the other kind's: >= 1.5
"target_salient", <= 1/1.5 "target_minor", else "close"; families drawn so the three are about equal), the two
name-blind answers (read off the most / least salient kind) and whether the referent is present. `boxes` holds
the asked kind's boxes ([x, y, w, h] in picture pixels) for the grounding loss ONLY when the whole referent is in
the picture: a false "Is there a yellow square?" gets none (critic R13-I03).

Backgrounds: half plain colours, half COCO train2014 photos (all train2014 pictures were seen by caption
alignment, so they are not claimed unseen); train, dev and test draw backgrounds from disjoint photo pools.
Colour x shape combinations in HELD_OUT never appear in train or dev scenes; every test family uses one.
"""
import argparse
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw

SHAPES = ["circle", "square", "triangle", "diamond"]
COLORS = {"red": (210, 50, 45), "green": (45, 150, 70), "blue": (45, 90, 210), "yellow": (235, 200, 30),
          "purple": (130, 60, 170), "orange": (240, 130, 30)}
HELD_OUT = {("purple", "triangle"), ("orange", "circle"), ("yellow", "diamond"), ("blue", "square")}
NUMBERS = ["zero", "one", "two", "three", "four", "five", "six"]
PLAIN = [(244, 232, 220), (236, 238, 230), (240, 240, 245), (250, 248, 240), (225, 230, 235), (60, 60, 64)]
W, H = 640, 420
SPLIT_SEED = {"train": 11, "dev": 22, "test": 33}
# v1 (data/r13/synth/, 2026-10-08, the first v12 baseline) re-placed every object in each variant (critic R13-I01);
# v2 keeps one layout per family and edits only the named instances.
VERSION = "v2"


def plural(shape: str) -> str:
    return shape + "s"


def draw(d: ImageDraw.ImageDraw, shape: str, box: Tuple[int, int, int, int], fill, outline=None):
    x0, y0, x1, y1 = box
    if shape == "circle":
        d.ellipse(box, fill=fill, outline=outline, width=3)
    elif shape == "square":
        d.rectangle(box, fill=fill, outline=outline, width=3)
    elif shape == "triangle":
        d.polygon([((x0 + x1) / 2, y0), (x1, y1), (x0, y1)], fill=fill, outline=outline)
    else:
        d.polygon([((x0 + x1) / 2, y0), (x1, (y0 + y1) / 2), ((x0 + x1) / 2, y1), (x0, (y0 + y1) / 2)],
                  fill=fill, outline=outline)


def place(rng: random.Random, sizes: List[int]) -> Optional[List[Tuple[int, int, int, int]]]:
    """Non-overlapping boxes for objects of the given sizes, or None."""
    boxes: List[Tuple[int, int, int, int]] = []
    for size in sizes:
        for _ in range(300):
            x, y = rng.randint(8, W - 8 - size), rng.randint(8, H - 8 - size)
            b = (x, y, x + size, y + size)
            if all(b[2] + 10 < o[0] or o[2] + 10 < b[0] or b[3] + 10 < o[1] or o[3] + 10 < b[1] for o in boxes):
                boxes.append(b)
                break
        else:
            return None
    return boxes


def base_scene(rng: random.Random, split: str) -> Dict[str, Any]:
    """Two kinds with shape, colour, count and size. Two thirds of the families have a clear salience gap (one
    kind's total area 2-4 times the other's: one "target_salient" and one "target_minor" question per picture),
    one third are "close" (ratio 0.8-1.25), so the three strata come out about equal."""
    while True:
        shapes = rng.sample(SHAPES, 2)
        colours = rng.sample(list(COLORS), 2)
        combos = set(zip(colours, shapes))
        if (split == "test") != bool(combos & HELD_OUT):
            continue
        counts = [rng.randint(1, 4), rng.randint(1, 4)]
        if not 3 <= sum(counts) <= 6:
            continue
        ratio = rng.uniform(2, 4) if rng.random() < 2 / 3 else rng.uniform(0.8, 1.25)
        if rng.random() < 0.5:
            ratio = 1 / ratio
        size0 = rng.randint(40, 95)
        size1 = round(size0 * (counts[0] / (counts[1] * ratio)) ** 0.5)   # area1 / area0 = 1 / ratio
        if not 30 <= size1 <= 120:
            continue
        return {"kinds": [{"shape": sh, "colour": c, "count": n, "size": sz}
                          for sh, c, n, sz in zip(shapes, colours, counts, (size0, size1))]}


def layout(kinds: List[Dict[str, Any]], rng: random.Random) -> Optional[List[Dict[str, Any]]]:
    """Instances of the base scene, placed once: [{"kind": i, "box": (x0, y0, x1, y1)}]."""
    order = [i for i, k in enumerate(kinds) for _ in range(k["count"])]
    boxes = place(rng, [kinds[i]["size"] for i in order])
    return None if boxes is None else [{"kind": i, "box": b} for i, b in zip(order, boxes)]


def render(kinds, instances, background) -> Tuple[Image.Image, List[List[List[int]]]]:
    """Draw the instances with their kind's current colour; boxes per kind as [x, y, w, h]."""
    if isinstance(background, str):
        with Image.open(background) as photo:
            im = photo.convert("RGB").resize((W, H))
    else:
        im = Image.new("RGB", (W, H), background)
    d = ImageDraw.Draw(im)
    per_kind: List[List[List[int]]] = [[] for _ in kinds]
    for inst in instances:
        k, b = kinds[inst["kind"]], inst["box"]
        draw(d, k["shape"], b, COLORS[k["colour"]], outline=(255, 255, 255) if isinstance(background, str) else None)
        per_kind[inst["kind"]].append([b[0], b[1], b[2] - b[0], b[3] - b[1]])
    return im, per_kind


def area(k: Dict[str, Any]) -> float:
    return k["count"] * k["size"] ** 2


def stratum(kinds, i) -> str:
    other = max(area(k) for j, k in enumerate(kinds) if j != i)
    r = area(kinds[i]) / other
    return "target_salient" if r >= 1.5 else "target_minor" if r <= 1 / 1.5 else "close"


def variants(base: Dict[str, Any], instances, rng: random.Random, split: str):
    """[(name, kinds, instances)]: base; recolour / recolour2 change one kind's colour only (same instances, so
    every pixel outside that kind's shapes is unchanged); count removes one instance of one kind, or adds one in
    free space, leaving every other instance where it was. None if the count edit cannot be placed."""
    kinds = base["kinds"]
    out = [("base", kinds, instances)]
    used = {k["colour"] for k in kinds}
    new_colours = []
    for v, idx in (("recolour", 0), ("recolour2", 1)):
        shape = kinds[idx]["shape"]
        choices = [c for c in COLORS if c not in used and c not in new_colours
                   and (split == "test" or (c, shape) not in HELD_OUT)]
        c = rng.choice(choices)
        new_colours.append(c)
        new = [dict(k) for k in kinds]
        new[idx]["colour"] = c
        out.append((v, new, instances))
    j = rng.randrange(len(kinds))
    new = [dict(k) for k in kinds]
    remove = kinds[j]["count"] > 1 and (rng.random() < 0.5 or sum(k["count"] for k in kinds) >= 6)
    if remove:
        last = max(i for i, inst in enumerate(instances) if inst["kind"] == j)
        moved = instances[:last] + instances[last + 1:]
        new[j]["count"] -= 1
    else:
        size = kinds[j]["size"]
        for _ in range(300):
            x, y = rng.randint(8, W - 8 - size), rng.randint(8, H - 8 - size)
            b = (x, y, x + size, y + size)
            if all(b[2] + 10 < o["box"][0] or o["box"][2] + 10 < b[0] or b[3] + 10 < o["box"][1]
                   or o["box"][3] + 10 < b[1] for o in instances):
                break
        else:
            return None
        moved = instances + [{"kind": j, "box": b}]
        new[j]["count"] += 1
    out.append(("count", new, moved))
    return out


def family_questions(vs, rng: random.Random):
    """One fixed question set for the whole family: the same instructions and option order on every picture, so
    answers can be compared variant by variant. Options cover every value any variant needs."""
    kinds0 = vs[0][1]
    colours = list(dict.fromkeys(k["colour"] for _, ks, _ in vs for k in ks))   # the family's colours, <= 4
    for c in COLORS:
        if len(colours) >= 4:
            break
        if c not in colours:
            colours.append(c)
    qs = {}
    for i, k in enumerate(kinds0):
        sh = k["shape"]
        other = kinds0[1 - i]
        opts = list(colours[:4])
        rng.shuffle(opts)
        qs["colour_%s" % sh] = {"type": "choice", "instructions": "What color are the %s?" % plural(sh),
                                "criteria": {o: None for o in opts}}
        counts = sorted({ks[i]["count"] for _, ks, _ in vs} | {ks[1 - i]["count"] for _, ks, _ in vs})
        n = k["count"]
        for x in sorted(range(1, 7), key=lambda x: (abs(x - n), x)):
            if len(counts) >= 4:
                break
            if x not in counts:
                counts.append(x)
        words = [NUMBERS[x] for x in sorted(counts)[:4]]
        rng.shuffle(words)
        qs["count_%s" % sh] = {"type": "choice", "instructions": "How many %s are there?" % plural(sh),
                               "criteria": {w: None for w in words}}
        recoloured = [ks[i]["colour"] for _, ks, _ in vs if ks[i]["colour"] != k["colour"]]
        qs["own_%s" % sh] = {"type": "noul", "instructions": "Is there a %s %s?" % (k["colour"], sh)}
        qs["trap_%s" % sh] = {"type": "noul", "instructions": "Is there a %s %s?" % (other["colour"], sh)}
        if recoloured:
            qs["new_%s" % sh] = {"type": "noul", "instructions": "Is there a %s %s?" % (recoloured[0], sh)}
    a, b = kinds0
    qs["more_%s_%s" % (a["shape"], b["shape"])] = {
        "type": "noul", "instructions": "Are there more %s than %s?" % (plural(a["shape"]), plural(b["shape"]))}
    return qs


def answers(qs, kinds, boxes):
    """(gold, meta, boxes) of the family's questions on one picture. A box set is given only when the question's
    whole referent is in the picture (a "Is there a yellow square?" with no yellow square gets none)."""
    gold, meta, qboxes = {}, {}, {}
    order = sorted(range(len(kinds)), key=lambda j: area(kinds[j]))
    least, most = kinds[order[0]], kinds[order[-1]]
    by_shape = {k["shape"]: i for i, k in enumerate(kinds)}
    for qid, q in qs.items():
        task, sh = qid.split("_", 1)
        if task == "more":
            a, b = kinds
            truth = a["count"] > b["count"]
            gold[qid] = {"probabilities": {"false": float(not truth), "true": float(truth)}}
            meta[qid] = {"task": "compare", "kind": 0, "stratum": stratum(kinds, 0),
                         "salient": "true" if area(a) >= area(b) else "false", "referent_present": True}
            qboxes[qid] = boxes[0] + boxes[1]
            continue
        i = by_shape[sh]
        k, st = kinds[i], stratum(kinds, i)
        if task == "colour":
            gold[qid] = {"probabilities": {o: float(o == k["colour"]) for o in q["criteria"]}}
            meta[qid] = {"task": "colour", "kind": i, "stratum": st, "salient": most["colour"], "minor": least["colour"]}
            qboxes[qid] = boxes[i]
        elif task == "count":
            gold[qid] = {"probabilities": {w: float(w == NUMBERS[k["count"]]) for w in q["criteria"]}}
            meta[qid] = {"task": "count", "kind": i, "stratum": st, "salient": NUMBERS[most["count"]],
                         "minor": NUMBERS[least["count"]]}
            qboxes[qid] = boxes[i]
        else:   # own / trap / new: "Is there a <colour> <shape>?"
            colour = q["instructions"].split()[3]
            truth = colour == k["colour"]
            on_other = colour == kinds[1 - i]["colour"]
            gold[qid] = {"probabilities": {"false": float(not truth), "true": float(truth)}}
            meta[qid] = {"task": "exists_true" if truth else "exists_trap" if on_other else "exists_absent",
                         "kind": i, "stratum": st, "statement": task}
            if truth:
                qboxes[qid] = boxes[i]
        meta[qid]["referent_present"] = qid in qboxes
    return gold, meta, qboxes


def backgrounds(root: str, split: str, n: int) -> List[str]:
    """Disjoint COCO train2014 photo pools per split (deterministic)."""
    folder = os.path.join(root, "coco", "train2014")
    names = sorted(os.listdir(folder)) if os.path.isdir(folder) else []
    if not names:
        return []
    third = len(names) // 3
    pool = {"train": names[:third], "dev": names[third:2 * third], "test": names[2 * third:]}[split]
    rng = random.Random(SPLIT_SEED[split])
    return [os.path.join(folder, x) for x in rng.sample(pool, min(n, len(pool)))]


def build_split(root: str, split: str, families: int, version: str = VERSION, photo_share: float = 0.5,
                seed: Optional[int] = None) -> Dict[str, Any]:
    """`photo_share`: share of families on a photo background (0.5 for v2; exp13-long's training part uses 0.75 with
    its own seed, written to synth-<version> so the v2 files stay as they are)."""
    rng = random.Random(SPLIT_SEED[split] if seed is None else seed)
    photos = backgrounds(root, split, families)
    out_dir = os.path.join(root, "r13", "synth-" + version)
    img_dir = os.path.join(out_dir, "images", split)
    os.makedirs(img_dir, exist_ok=True)
    rows, strata = [], {}
    f = 0
    while f < families:
        base = base_scene(rng, split)
        on_photo = f % 2 == 1 if photo_share == 0.5 else rng.random() < photo_share   # 0.5: exactly as v2 was built
        background = photos[f % len(photos)] if photos and on_photo else rng.choice(PLAIN)
        instances = layout(base["kinds"], rng)
        vs = variants(base, instances, rng, split) if instances else None
        if not vs:
            continue
        qs = family_questions(vs, rng)
        group = "r13synth-%s:%s:%05d" % (version, split, f)
        for name, kinds, insts in vs:
            im, boxes = render(kinds, insts, background)
            rel = os.path.join("r13", "synth-" + version, "images", split, "%05d_%s.png" % (f, name))
            im.save(os.path.join(root, rel))
            gold, meta, qboxes = answers(qs, kinds, boxes)
            for m in meta.values():
                strata[m["stratum"]] = strata.get(m["stratum"], 0) + 1
            sid = "%s_%s" % (group, name)
            rows.append({"id": sid, "source": "r13-synth", "kind": "binding", "image_id": sid, "group": group,
                         "variant": name, "image": rel, "width": W, "height": H,
                         "background": "photo" if isinstance(background, str) else "plain",
                         "scene": kinds, "instances": [{"kind": x["kind"], "box": list(x["box"])} for x in insts],
                         "questions": qs, "gold": gold, "meta": meta, "boxes": qboxes})
        f += 1
    name = {"train": "train", "dev": "dev_bind_cf", "test": "test_bind_cf"}[split]
    with open(os.path.join(out_dir, name + ".jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    tasks: Dict[str, int] = {}
    for r in rows:
        for m in r["meta"].values():
            tasks[m["task"]] = tasks.get(m["task"], 0) + 1
    return {"split": split, "file": name + ".jsonl", "families": families, "pictures": len(rows),
            "questions": sum(len(r["questions"]) for r in rows), "strata": strata, "tasks": tasks,
            "grounded_questions": sum(len(r["boxes"]) for r in rows)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    p.add_argument("--train", type=int, default=1400, help="scene families")
    p.add_argument("--dev", type=int, default=300)
    p.add_argument("--test", type=int, default=300)
    p.add_argument("--photo75-train", type=int, default=0, help="only build a training split with 75%% photo "
                   "backgrounds (seed 44) into synth-v2p75/, for exp13-long")
    a = p.parse_args(argv)
    if a.photo75_train:
        info = build_split(a.root, "train", a.photo75_train, version="v2p75", photo_share=0.75, seed=44)
        print(json.dumps(info))
        return
    manifest = {"version": VERSION, "held_out_combinations": sorted(map(list, HELD_OUT)), "splits": []}
    for split, n in (("train", a.train), ("dev", a.dev), ("test", a.test)):
        info = build_split(a.root, split, n)
        manifest["splits"].append(info)
        print(json.dumps(info), flush=True)
    with open(os.path.join(a.root, "r13", "synth-" + VERSION, "MANIFEST.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)


if __name__ == "__main__":
    main()
