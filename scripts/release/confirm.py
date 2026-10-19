"""The synthetic confirmation (docs/experiments/2026-10-19-v04-reader-evidence-prep03.md 5.3; critic V04-K-CONFIRM-PREP-01).
CPU only, loads no model. 120 pictures drawn by an INDEPENDENT renderer (its own canvas, containers, particles, magnet
shapes, fonts Verdana / Georgia / Trebuchet, layouts -- never real_layout's temp_render / force_render; real_layout draws
in Arial), on facts from the accepted definitions (real_layout's value ranges, M x V^2 for temperature, the magnet rules
and four wordings of force_items, the dense magnet pool and the case partition of rl_calib):

  - temperature 60: same mass / same speed / both differ / equal, 15 each; in the first three the hotter sample is A or B
    within one of each other; per picture the comparison question and the four readings (mass / speed of A / B);
  - magnets 60: distance / size / equal, 20 each (size: one small magnet, distances equal -- the reader reads distances
    only, kept in the main denominator and described apart); per picture the comparison question (four wordings, gold
    flipped as defined) and the two distance readings;
  - every picture's canonical case on the "confirm" side of rl_calib's partition, used by no data file, not in rl_calib,
    not a transcribed scene, unique in the set; the picture shows only the drawn objects, labels, values and units -- no
    hint of the answer;
  - a fixed wrong picture per picture (a derangement inside its domain) for the mismatched control.

The formal entry (generate) refuses unless the final combination is frozen: a complete IDENTITY.json (check_eval freeze:
files / reading / code) of this very checkpoint, its frozen code a real runtime record that is this implementation now,
its reading bundle with the K rule and the LIVE-K lineage (rebind's release source) -- all checked before anything is
selected or drawn. Its seed is taken from the frozen identity's bytes; together with the exclusion record (data files and
rl_calib) of the run it fixes the set. This is a process rule -- the set is generated once, under one permission, from one
frozen input -- not a cryptographic guarantee that nobody could compute it earlier. Output must be new. preview draws at most 4 + 4 pictures with an explicit seed
and "preview:" ids for checking the renderer -- never the confirmation.

Request contract (models run only with separate permission): candidate true / fixed-wrong / reversed options 120 each;
v0.3 true / the same wrong 120 each. Primary (`outcome`, against the formal set's 120 ids from its confirm.jsonl -- each
once in all three record sets, explicit bools, else no verdict): one comparison per picture, equally weighted, picture
resampling seed 0 x 2,000: candidate - v0.3 accuracy >= +0.10 with lower bound > 0, and the candidate's own true - wrong
accuracy lower bound > 0 (its own picture gain -- not 68:114's increment). Readings clustered by picture, per domain /
class reported only.

    uv run python scripts/release/confirm.py preview --out /tmp/confirm-preview --seed 7
    uv run python scripts/release/confirm.py generate --identity <eval>/IDENTITY.json --checkpoint <frozen candidate> --out data/v04_k_confirm
"""
import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "scripts", "data"))
import real_layout as rl  # type: ignore  # noqa: E402

FONTS = {"body": "/System/Library/Fonts/Supplemental/Verdana.ttf", "label": "/System/Library/Fonts/Supplemental/Georgia Bold.ttf",
         "value": "/System/Library/Fonts/Supplemental/Trebuchet MS.ttf"}
TEMP_CLASSES = {"same mass": 15, "same speed": 15, "both differ": 15, "equal": 15}
FORCE_CLASSES = {"distance": 20, "size": 20, "equal": 20}
REQUESTS = {"candidate": {"true": 120, "mismatched": 120, "reversed": 120}, "v0.3": {"true": 120, "mismatched": 120}}
PREVIEW_MAX = 4
GAIN_LINE, RESAMPLES = 0.10, 2000


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


rlc = _load("confirm_rl_calib", "scripts/release/rl_calib.py")


def sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ scenes (facts from the accepted definitions)

def temp_scene(rng: random.Random, case: str, hotter: Optional[str]) -> Dict:
    """A temperature scene of `case` in real_layout's value ranges; `hotter` "A" / "B" for the first three classes."""
    while True:
        ma, mb = rng.randint(4, 46), rng.randint(4, 46)
        va, vb = rng.randrange(300, 2000, 10), rng.randrange(300, 2000, 10)
        if case == "same mass":
            mb = ma
        elif case == "same speed":
            vb = va
        elif case == "equal":
            mb, vb = ma, va
        ka, kb = ma * va * va, mb * vb * vb
        if case == "equal":
            break
        if ka == kb or (case == "both differ" and (ma == mb or va == vb)):
            continue
        if (ka > kb) != (hotter == "A"):
            ma, mb, va, vb = mb, ma, vb, va
        break
    return {"case": case, "A": {"mass": ma, "speed": va}, "B": {"mass": mb, "speed": vb}, "a_left": rng.random() < 0.7,
            "count": rng.randint(6, 14), "palette": rng.randrange(4), "seed": rng.random(), "width": rng.randint(600, 680)}


def force_scene(rng: random.Random, case: str) -> Dict:
    """A magnet scene of `case` from rl_calib's dense pool (real_layout's units, ranges and precision)."""
    s = rlc.force_scene_dense(rng, case)
    s.update(style=rng.randrange(3), width=rng.randint(620, 720))
    return s


def items_of(form: str, s: Dict, k: int) -> List[Dict]:
    """The comparison and reading questions with gold, by real_layout's own fact rules (temp_items / force_items)."""
    with rlc.dense_values():
        _, items = (rl.temp_items if form == "temp" else rl.force_items)(s, k, [s])
    return items


# ------------------------------------------------------------------ the independent renderer

PALETTES = [("#3b7dd8", "#e2574c"), ("#2e9e6a", "#9b59b6"), ("#d98e04", "#1f8a9e"), ("#c2185b", "#5d7d2a")]


def render_temp(s: Dict) -> Image.Image:
    w, h = s["width"], 430
    img = Image.new("RGB", (w, h), "#f4f4f0")
    d = ImageDraw.Draw(img)
    lab, body = ImageFont.truetype(FONTS["label"], 19), ImageFont.truetype(FONTS["body"], 15)
    rng = random.Random(s["seed"])
    for slot, name in enumerate(("A", "B") if s["a_left"] else ("B", "A")):
        x0 = 40 + slot * (w // 2)
        cw = w // 2 - 80
        d.rounded_rectangle((x0, 40, x0 + cw, 250), radius=26, outline="#4a4a4a", width=3, fill="white")
        d.rectangle((x0 + cw // 4, 28, x0 + 3 * cw // 4, 42), fill="#8a8a8a")              # a lid, not a jar neck
        colour = PALETTES[s["palette"]][slot]
        r = 5 + s[name]["mass"] // 6
        for _ in range(s["count"]):
            cx, cy = rng.randint(x0 + 20 + r, x0 + cw - 20 - r), rng.randint(60 + r, 230 - r)
            d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=colour, outline="#222222")
        d.text((x0 + cw / 2, 275), "Sample %s" % name, fill="#111111", font=lab, anchor="mm")
        d.text((x0 + cw / 2, 315), "Mass of each particle: %d u" % s[name]["mass"], fill="#111111", font=body, anchor="mm")
        d.text((x0 + cw / 2, 345), "Average particle speed: %s m/s" % rl.fmt(s[name]["speed"]), fill="#111111", font=body, anchor="mm")
    return img


def render_force(s: Dict) -> Image.Image:
    w, h = s["width"], 210
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    lab, val, pole = ImageFont.truetype(FONTS["label"], 18), ImageFont.truetype(FONTS["value"], 15), ImageFont.truetype(FONTS["body"], 13)
    pw = (w - 40) // 2
    top = max(DENSE_MAX[u] for u in DENSE_MAX)
    c1, c2 = s["colours"]
    for slot in range(2):
        p = slot if s.get("p1_left", True) else 1 - slot
        bx = 15 + slot * (pw + 10)
        d.rounded_rectangle((bx, 30, bx + pw, h - 8), radius=14, outline="#9aa0a6", width=2)
        d.text((bx + pw / 2, 15), "Pair %d" % (p + 1), fill="#111111", font=lab, anchor="mm")
        big = 56
        sizes = [big if s["big"][p][j] else int(big * 0.55) for j in range(2)]
        gap = int(20 + (pw - 40 - 2 * big * 2) * (s["d"][p] / DENSE_MAX[s["unit"]]) * 0.9)
        widths = [z * 2 if s["style"] != 1 else z for z in sizes]
        x = bx + (pw - widths[0] - gap - widths[1]) // 2
        cy = 95
        xs = [x, x + widths[0] + gap]
        for j in range(2):
            x0, z = xs[j], sizes[j]
            if s["style"] == 1:                                 # round magnets
                d.pieslice((x0, cy - z // 2, x0 + z, cy + z // 2), 90, 270, fill=c1)
                d.pieslice((x0, cy - z // 2, x0 + z, cy + z // 2), 270, 450, fill=c2)
            else:                                               # rounded bars
                d.rounded_rectangle((x0, cy - z // 2, x0 + z, cy + z // 2), radius=6, fill=c1)
                d.rounded_rectangle((x0 + z, cy - z // 2, x0 + 2 * z, cy + z // 2), radius=6, fill=c2)
            halves = (x0 + widths[j] / 4, x0 + 3 * widths[j] / 4)
            for hx, letter in zip(halves, s["poles"][p][j]):
                d.text((hx, cy), letter, fill="white", font=pole, anchor="mm")
        ax0, ax1, ty = xs[0] + widths[0], xs[1], 160
        d.line((ax0, ty, ax1, ty), fill="#333333", width=2)
        for ax in (ax0, ax1):
            d.polygon([(ax, ty), (ax + (6 if ax == ax0 else -6), ty - 4), (ax + (6 if ax == ax0 else -6), ty + 4)], fill="#333333")
        d.text(((ax0 + ax1) / 2, ty + 18), "%s %s" % (rl.fmt(s["d"][p]), s["unit"]), fill="#111111", font=val, anchor="mm")
    return img


DENSE_MAX = {u: v[-1] for u, v in rlc.DENSE.items()}


# ------------------------------------------------------------------ isolation and selection

def excluded(root: str) -> Set[tuple]:
    """Cases the confirmation may not use: every data file's (rl_calib.used_cases) and rl_calib's own (its MANIFEST, or
    re-drawn deterministically when it is not built yet)."""
    import ast
    used = rlc.used_cases(root)
    man = os.path.join(root, "rl_calib", "MANIFEST.json")
    if os.path.exists(man):
        used |= {ast.literal_eval(c) for c in json.load(open(man))["cases"]}
    else:
        st: Dict = {}
        list(rlc.families(set(used), stats=st))
        used |= {ast.literal_eval(c) for c in st["cases"]}
    return used


def select(seed: int, used: Set[tuple], temp_classes: Dict[str, int] = TEMP_CLASSES, force_classes: Dict[str, int] = FORCE_CLASSES) -> List[Tuple[str, Dict]]:
    """(form, scene) of the set: per class its count, hotter A / B alternating in the first three temperature classes,
    each case unused, on the confirm side, not transcribed, unique."""
    rng = random.Random(seed)
    banned = rl.transcribed_keys()
    mine: Set[tuple] = set()
    out: List[Tuple[str, Dict]] = []
    plan = [("temp", c, ("A", "B")[i % 2] if c != "equal" else None) for c, n in temp_classes.items() for i in range(n)] + \
           [("force", c, None) for c, n in force_classes.items() for _ in range(n)]
    for form, case, hotter in plan:
        for _ in range(20000):
            s = temp_scene(rng, case, hotter) if form == "temp" else force_scene(rng, case)
            key = rl.case_key(form, s)
            if key in used or key in mine or rl.scene_key(form, s) in banned or rlc.partition(key) != "confirm":
                continue
            mine.add(key)
            out.append((form, s))
            break
        else:
            raise SystemExit("cannot fill %s %s without relaxing the isolation" % (form, case))
    return out


def rows_of(selected: List[Tuple[str, Dict]], prefix: str, outdir: str, root_rel: str) -> List[Dict]:
    rows = []
    for k, (form, s) in enumerate(selected):
        iid = "%s:%s%03d" % (prefix, form, k)
        rel = os.path.join(root_rel, "images", "%s%03d.png" % (form, k))
        (render_temp(s) if form == "temp" else render_force(s)).save(os.path.join(outdir, "images", "%s%03d.png" % (form, k)))
        items = items_of(form, s, k)
        hint = rl.TEMP_HINT if form == "temp" else rl.FORCE_HINT + (", but some of them are different sizes." if s["case"] == "size" else ".")
        rows.append({"id": iid, "source": "v04-confirm-" + form, "kind": form, "axis": s["case"], "image_id": iid, "image": rel,
                     "case": repr(rl.case_key(form, s)), "state_text": hint,
                     "questions": {"q%d" % j: {"type": "choice", "instructions": it["q"], "criteria": {o: None for o in it["options"]}} for j, it in enumerate(items)},
                     "gold": {"q%d" % j: {"probabilities": {o: float(o == it["gold"]) for o in it["options"]}} for j, it in enumerate(items)}})
    return rows


def mismatched(rows: List[Dict], seed: int) -> List[Dict]:
    """Each picture's questions on another picture of its domain (a fixed derangement)."""
    out = []
    rng = random.Random(seed + 1)
    for form in ("temp", "force"):
        own = [r for r in rows if r["kind"] == form]
        order = list(range(len(own)))
        while True:
            rng.shuffle(order)
            if all(i != j for i, j in enumerate(order)) or len(own) < 2:
                break
        out += [dict(r, id=r["id"] + ":mism", image=own[j]["image"], image_id=own[j]["image_id"]) for r, j in zip(own, order)]
    return out


def manifest(rows: List[Dict], used: Set[tuple], extra: Dict) -> Dict:
    from collections import Counter
    temp = [r for r in rows if r["kind"] == "temp"]
    gold_side = Counter((r["axis"], max(r["gold"]["q0"]["probabilities"], key=r["gold"]["q0"]["probabilities"].get)) for r in temp)
    return {"pictures": len(rows), "classes": {"%s|%s" % k: v for k, v in Counter((r["kind"], r["axis"]) for r in rows).most_common()},
            "temperature_answers": {"%s|%s" % k: v for k, v in sorted(gold_side.items())},
            "questions": sum(len(r["questions"]) for r in rows), "isolation": {"excluded_cases": len(used),
            "overlap": sum(eval(r["case"]) in used for r in rows), "confirm_side": all(rlc.partition(eval(r["case"])) == "confirm" for r in rows),
            "unique_cases": len({r["case"] for r in rows}) == len(rows)},
            "renderer": {"code": sha(__file__), "fonts": {k: os.path.basename(v) for k, v in FONTS.items()}}, "requests": REQUESTS, **extra}


# ------------------------------------------------------------------ entries

def preview(out: str, seed: int, n_temp: int = 2, n_force: int = 2, root: str = "data") -> Dict:
    """A few pictures with an explicit seed and "preview:" ids for checking the renderer -- never the confirmation."""
    if n_temp > PREVIEW_MAX or n_force > PREVIEW_MAX:
        raise SystemExit("a preview draws at most %d pictures per domain" % PREVIEW_MAX)
    os.makedirs(os.path.join(out, "images"), exist_ok=False)
    used = excluded(root)
    tc = {c: 1 for c in list(TEMP_CLASSES)[:n_temp]}
    fc = {c: 1 for c in list(FORCE_CLASSES)[:n_force]}
    rows = rows_of(select(seed, used, tc, fc), "preview", out, ".")
    man = manifest(rows, used, {"preview": True, "note": "renderer check only; not the confirmation"})
    open(os.path.join(out, "preview.jsonl"), "w").write("".join(json.dumps(r) + "\n" for r in rows))
    json.dump(man, open(os.path.join(out, "MANIFEST.json"), "w"), indent=1)
    return man


def freeze_problems(identity: str, checkpoint: str, pins: Optional[Dict[str, str]] = None) -> List[str]:
    """The final combination is frozen: a complete IDENTITY.json of this checkpoint (files, reading, code), the checkpoint
    still hashing to it, its reading bundle and LIVE-K lineage the frozen ones (check_eval.reading_identity, rebind's
    release source), the K rule present."""
    if not os.path.exists(identity):
        return ["no frozen identity %s" % identity]
    ident = json.load(open(identity))
    missing = [k for k in ("checkpoint", "files", "reading", "code") if not ident.get(k)]
    if missing:
        return ["%s lacks %s: not a complete frozen identity" % (identity, missing)]
    out = code_problems(ident["code"])
    ce = _load("confirm_check_eval", "scripts/release/check_eval.py")
    setattr(ce, "SOURCE_PINS", pins)
    named = ident["checkpoint"]
    if os.path.realpath(named if os.path.isabs(named) else os.path.join(ROOT, named)) != os.path.realpath(checkpoint):
        out.append("%s is frozen for %s, not %s" % (identity, named, checkpoint))
    now = ce.tree_hashes(checkpoint)
    changed = sorted(k for k in set(now) | set(ident["files"]) if now.get(k) != ident["files"].get(k))
    if changed:
        out.append("checkpoint differs from the frozen identity: %s" % changed[:5])
    try:
        if ce.reading_identity(checkpoint) != ident["reading"]:
            out.append("reading bundle / lineage are not the frozen ones")
    except SystemExit as e:
        out.append(str(e))
    rj = os.path.join(checkpoint, "reading", "reading.json")
    if not os.path.exists(rj) or "k_rule" not in json.load(open(rj)):
        out.append("the frozen combination has no K rule")
    return out


def code_problems(code: Any) -> List[str]:
    """The frozen code reference (check_eval freeze: evaluate.runtime_code of a fresh probe process) is a real one and
    is this implementation now (critic V04-K-CONFIRM-R1-01): {"modules": {name: {"path", "sha256"}}, "version", "python"},
    the formal evaluation's required modules present (devision.train.evaluate.REQUIRED_MODULES: devision, infer, image,
    model, reading, train.evaluate, train.cli), every module's content equal to this repository's file of the same module
    (route_calib.runtime_problems: by name and content; paths only traced)."""
    if not isinstance(code, dict) or not isinstance(code.get("modules"), dict) or not code["modules"] \
            or not all(isinstance(v, dict) and isinstance(v.get("sha256"), str) and len(v["sha256"]) == 64 for v in code["modules"].values()) \
            or not code.get("version") or not code.get("python"):
        return ["the frozen code is not a runtime code record (modules with sha256, version, python)"]
    from devision.train.evaluate import REQUIRED_MODULES                # the formal evaluation's seven (critic R2)
    rc = _load("confirm_route_calib", "scripts/release/route_calib.py")
    out = ["frozen code: module %s: required but not loaded" % n for n in REQUIRED_MODULES if n not in code["modules"]]
    return out + ["frozen code: %s" % x for x in rc.runtime_problems(code, ROOT)]


def generate(out: str, identity: str, checkpoint: str, root: str = "data", pins: Optional[Dict[str, str]] = None) -> Dict:
    """The formal set: only after the freeze checks, into a new directory, seed from the frozen identity's bytes."""
    bad = freeze_problems(identity, checkpoint, pins)
    if bad:
        raise SystemExit("not frozen -- no confirmation is generated:\n  " + "\n  ".join(bad))
    if os.path.exists(out):
        raise SystemExit("%s exists (fresh only)" % out)
    seed = int(sha(identity)[:16], 16)
    used = excluded(root)
    os.makedirs(os.path.join(out, "images"))
    rel = os.path.relpath(out, root)
    rows = rows_of(select(seed, used), "v04-confirm", out, rel)
    mism = mismatched(rows, seed)
    open(os.path.join(out, "confirm.jsonl"), "w").write("".join(json.dumps(r) + "\n" for r in rows))
    open(os.path.join(out, "confirm_mism.jsonl"), "w").write("".join(json.dumps(r) + "\n" for r in mism))
    man = manifest(rows, used, {"preview": False, "identity": {"path": os.path.abspath(identity), "sha256": sha(identity)},
                                "seed": "int(sha256(IDENTITY.json)[:16], 16)", "mismatched": len(mism),
                                "files": {f: sha(os.path.join(out, f)) for f in ("confirm.jsonl", "confirm_mism.jsonl")}})
    json.dump(man, open(os.path.join(out, "MANIFEST.json"), "w"), indent=1)
    return man


# ------------------------------------------------------------------ the pre-written outcome

PICTURES = 120


def picture_ids(confirm_jsonl: str) -> List[str]:
    """The formal set's picture ids, from its confirm.jsonl (one comparison question per picture)."""
    return [json.loads(x)["id"] for x in open(confirm_jsonl) if x.strip()]


def outcome(cand_true: Dict[str, bool], cand_wrong: Dict[str, bool], v03_true: Dict[str, bool], expected: List[str]) -> Dict:
    """Per picture id, whether the comparison question was right (candidate true / fixed wrong, v0.3 true), against the
    formal set's ids (`expected`, from its manifest's confirm.jsonl): exactly PICTURES (120) pictures, no override, each once in every record
    set, every value an explicit bool -- otherwise no verdict (critic V04-K-CONFIRM-R1-02). The pictures equally weighted;
    95 % percentile interval of 2,000 picture resamples, seed 0."""
    ids = sorted(expected)
    if len(ids) != PICTURES or len(set(ids)) != PICTURES:
        return {"branch": "no verdict: the formal set does not hold %d distinct pictures" % PICTURES}
    for name, recs in (("candidate true", cand_true), ("candidate wrong", cand_wrong), ("v0.3 true", v03_true)):
        if set(recs) != set(ids):
            return {"branch": "no verdict: %s does not cover exactly the formal pictures (missing %d, foreign %d)"
                              % (name, len(set(ids) - set(recs)), len(set(recs) - set(ids)))}
        if not all(type(v) is bool for v in recs.values()):
            return {"branch": "no verdict: %s holds a value that is not true / false" % name}
    rng = random.Random(0)

    def stat(f):
        vals = sorted(sum(f(ids[rng.randrange(len(ids))]) for _ in ids) / len(ids) for _ in range(RESAMPLES))
        return {"value": sum(f(i) for i in ids) / len(ids), "ci95": [vals[int(0.025 * RESAMPLES)], vals[int(0.975 * RESAMPLES) - 1]]}
    diff = stat(lambda i: float(cand_true[i]) - float(v03_true[i]))
    gain = stat(lambda i: float(cand_true[i]) - float(cand_wrong[i]))
    ok = diff["value"] >= GAIN_LINE and diff["ci95"][0] > 0 and gain["ci95"][0] > 0
    return {"candidate_minus_v03": diff, "candidate_own_gain": gain,
            "branch": "synthetic confirmation passed" if ok else "synthetic confirmation not passed: the current combination is not approved for release"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("preview")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, required=True)
    g = sub.add_parser("generate")
    for k in ("identity", "checkpoint", "out"):
        g.add_argument("--" + k, required=True)
    a = ap.parse_args(argv)
    man = preview(a.out, a.seed) if a.cmd == "preview" else generate(a.out, a.identity, a.checkpoint)
    print(json.dumps({k: v for k, v in man.items() if k != "renderer"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
