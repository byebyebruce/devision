"""exp28 diagnostic question sets (docs/experiments/2026-10-17-exp28-diagnose-plan.md), no training data:

  - control30.jsonl: exp22's dev16 (exp21 dev families at 16 px) restricted to families d0000-d0014, both value
    views -- 30 pictures, 90 questions, chosen by family id before looking at any prediction (positive control of
    the exported model against exp26's own step-5000 monitor records);
  - fields.jsonl: direct readings on the 24 transcribed ScienceQA pictures (scripts/exp21/sqa_transcribed.json):
    temperature -- mass and speed of sample A and B (4 per picture); force -- the distance in Pair 1 and Pair 2
    (2 per picture); both objects of a field share one candidate set (both real values + a third above, below or
    between them, rotating; critic EXP28-R1-01), the display order rotating; the original picture and hint;
  - fields_blank.jsonl: the same 72 questions on the blank picture (only the image changes);
  - force_rule.jsonl: the 12 transcribed force questions as in exp21's sqa_text (blank picture, hint + transcribed
    values), plus one general rule sentence that names no value or answer;
  - equal_wording.jsonl: every temperature (41) and concentration (45) dev question in 2 wordings of the "same"
    option (original / a rewrite with the same meaning) x 3 cyclic option orders = 516 questions, same picture,
    question and other options, the gold mapped with the option.

    uv run python scripts/exp28/build.py --root data
"""
import argparse
import hashlib
import json
import os
import re
from typing import Dict, List

HERE = os.path.dirname(__file__)
RULE = ("Rule: in each pair, the magnetic force between the magnets is stronger when the magnets are closer "
        "together, and stronger when the magnets are larger.")
REWRITE = {"neither; the samples have the same temperature": "the two samples have equal temperatures",
           "neither; their concentrations are the same": "the two solutions have equal concentrations"}
CONTROL_FAMILIES = 15


def read(path: str) -> List[Dict]:
    return [json.loads(x) for x in open(path)]


def fmt_num(v: float) -> str:
    """1,400 / 42.8 / 0.25: integers with thousands commas, decimals as short as they are."""
    if abs(v - round(v)) < 1e-9:
        return "{:,}".format(int(round(v)))
    return ("%.3f" % v).rstrip("0").rstrip(".")


def parse_temp(text: str) -> Dict[str, Dict[str, float]]:
    out = {}
    for s, mass, speed in re.findall(r"Sample ([AB]): mass of each particle ([\d,]+) u, average particle speed ([\d,]+) m/s", text):
        out[s] = {"mass": float(mass.replace(",", "")), "speed": float(speed.replace(",", ""))}
    if set(out) != {"A", "B"}:
        raise ValueError("cannot parse %r" % text)
    return out


def parse_force(text: str) -> Dict[str, Dict]:
    out = {}
    for pair, body in re.findall(r"Pair ([12]): ([^.]*(?:\.\d+[^.]*)?)\.(?:\s|$)", text):
        m = re.search(r"([\d.]+) (mm|cm|in) apart", body)
        if not m:
            raise ValueError("no distance in %r" % body)
        out[pair] = {"distance": float(m.group(1)), "unit": m.group(2)}
    if set(out) != {"1", "2"}:
        raise ValueError("cannot parse %r" % text)
    return out


STEP = {"mass": 4.0, "speed": 100.0}   # distances scale instead (x0.5 / x1.5 / x2)


def _up(v: float, field: str, n: int = 1) -> float:
    return v + n * STEP[field] if field in STEP else v * (1.5 if n == 1 else 2.0)


def _down(v: float, field: str, n: int = 1) -> float:
    return v - n * STEP[field] if field in STEP else v * (0.5 if n == 1 else 0.25)


def _between(lo: float, hi: float, field: str):
    """A value strictly between lo and hi in the field's precision (integers for mass, tens for speed, up to 3
    decimals for distance), or None when there is none."""
    if field == "mass":
        m = round((lo + hi) / 2)
    elif field == "speed":
        m = round((lo + hi) / 20) * 10
    else:
        m = round((lo + hi) / 2, 3)
    return m if lo < m < hi else None


def candidates(a: float, b: float, field: str, k: int) -> List[float]:
    """One candidate set for both objects of a field (critic EXP28-R1-01): the two real values plus a third
    placed above both, below both or between them as k rotates; with equal values, two distractors putting the
    shared value lowest / in the middle / highest. Falls back to the next placement when one does not exist."""
    if a == b:
        v = a
        plans = [[_up(v, field), _up(v, field, 2)], [_down(v, field), _up(v, field)], [_down(v, field), _down(v, field, 2)]]
        for j in range(3):
            d = plans[(k + j) % 3]
            if all(x > 0 for x in d):
                return sorted({v, *d})
        raise ValueError("no distractors for %s" % v)
    lo, hi = min(a, b), max(a, b)
    for j in range(3):
        place = (k + j) % 3
        third = _up(hi, field) if place == 0 else _down(lo, field) if place == 1 else _between(lo, hi, field)
        if third is not None and third > 0 and third not in (a, b):
            return sorted({a, b, third})
    raise ValueError("no third value for %s / %s" % (a, b))


def options(cands: List[float], right: float, slot: int, unit: str) -> Dict[str, float]:
    """The shared candidate set shown in a rotating order: the right value at display slot `slot % 3`."""
    others = [x for x in cands if x != right]
    order = list(others)
    order.insert(slot % 3, right)
    return {"%s %s" % (fmt_num(x), unit): float(x == right) for x in order}


def fields(root: str, notes: Dict[str, str]) -> List[Dict]:
    temp = {r["id"]: r for r in read(os.path.join(root, "exp21", "dev_sqa_temp.jsonl"))}
    force = {r["id"]: r for r in read(os.path.join(root, "exp21", "dev_sqa_force.jsonl"))}
    rows, slot, k = [], 0, 0
    for sid in sorted(x for x in notes if x != "_note"):
        if sid in temp:
            src, vals = temp[sid], parse_temp(notes[sid])
            asks = [("mass", "u", "What is the mass of each particle in sample %s?"),
                    ("speed", "m/s", "What is the average particle speed in sample %s?")]
            for field, unit, q in asks:
                cands = candidates(vals["A"][field], vals["B"][field], field, k)
                k += 1
                for s in ("A", "B"):
                    rows.append(row(src, "%s:%s:%s" % (sid, field, s), q % s,
                                    options(cands, vals[s][field], slot, unit), "temp-" + field))
                    slot += 1
        elif sid in force:
            src, vals = force[sid], parse_force(notes[sid])
            cands = candidates(vals["1"]["distance"], vals["2"]["distance"], "distance", k)
            k += 1
            for p in ("1", "2"):
                rows.append(row(src, "%s:distance:%s" % (sid, p), "What is the distance between the magnets in Pair %s?" % p,
                                options(cands, vals[p]["distance"], slot, vals[p]["unit"]), "force-distance"))
                slot += 1
        else:
            raise ValueError("%s is in neither temperature nor force" % sid)
    return rows


def blank(rows: List[Dict]) -> List[Dict]:
    """The same field questions on the blank picture: only the image changes; image_id stays the original
    picture so the two conditions pair by picture."""
    return [dict(r, id=r["id"] + ":blank", image="exp21/images/blank.png", source=r["source"] + "-blank") for r in rows]


def rank_baseline(rows: List[Dict]) -> Dict:
    """Accuracy of always choosing the lowest / middle / highest number, overall and per kind."""
    def value(opt):
        return float(opt.split()[0].replace(",", ""))
    out = {}
    for kind in ("all", "temp", "temp-mass", "temp-speed", "force-distance"):
        sel = [r for r in rows if kind == "all" or r["kind"].startswith(kind)]
        hits = [0, 0, 0]
        for r in sel:
            g = r["gold"]["q"]["probabilities"]
            order = sorted(g, key=value)
            hits[order.index(next(o for o, p in g.items() if p == 1.0))] += 1
        out[kind] = {"n": len(sel), "lowest": hits[0] / len(sel), "middle": hits[1] / len(sel), "highest": hits[2] / len(sel)}
    return out


def row(src: Dict, sid: str, question: str, probs: Dict[str, float], kind: str) -> Dict:
    return {"id": "exp28-field:" + sid, "source": "exp28-" + kind, "kind": kind, "image_id": src["image_id"],
            "image": src["image"], "state_text": src.get("state_text", ""),
            "questions": {"q": {"type": "choice", "instructions": question, "criteria": {k: None for k in probs}}},
            "gold": {"q": {"probabilities": probs}}}


def force_rule(root: str) -> List[Dict]:
    out = []
    for r in read(os.path.join(root, "exp21", "sqa_text.jsonl")):
        if "magnet" in r["questions"]["q"]["instructions"]:
            out.append(dict(r, id="exp28-rule:" + r["id"], source="exp28-force-rule",
                            state_text=r["state_text"] + "\n" + RULE))
    return out


def equal_wording(root: str) -> List[Dict]:
    out = []
    for path in (os.path.join(root, "exp21", "dev_sqa_temp.jsonl"), os.path.join(root, "v10", "dev_sqa_conc.jsonl")):
        for r in read(path):
            q, gold = r["questions"]["q"], r["gold"]["q"]["probabilities"]
            keys = list(q["criteria"])
            same = [k for k in keys if k in REWRITE]
            if len(same) != 1:
                raise ValueError("%s: no single 'same' option" % r["id"])
            for w in (0, 1):
                name = {k: (REWRITE[k] if w and k in REWRITE else k) for k in keys}
                for o in range(3):
                    order = keys[o:] + keys[:o]
                    out.append(dict(r, id="%s:w%d:o%d" % (r["id"], w, o), source=r["source"] + ":w%d" % w,
                                    questions={"q": dict(q, criteria={name[k]: None for k in order})},
                                    gold={"q": {"probabilities": {name[k]: gold.get(k, 0.0) for k in order}}}))
    return out


def control30(root: str) -> List[Dict]:
    keep = {"d%04d" % i for i in range(CONTROL_FAMILIES)}
    return [s for s in read(os.path.join(root, "exp22", "dev16.jsonl")) if s["family"] in keep]


def build(root: str) -> Dict:
    out = os.path.join(root, "exp28")
    os.makedirs(out, exist_ok=True)
    notes = json.load(open(os.path.join(HERE, "..", "exp21", "sqa_transcribed.json")))
    f = fields(root, notes)
    files = {"control30": control30(root), "fields": f, "fields_blank": blank(f), "force_rule": force_rule(root),
             "equal_wording": equal_wording(root)}
    manifest = {}
    for name, rows in files.items():
        path = os.path.join(out, name + ".jsonl")
        with open(path, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        manifest[name] = {"rows": len(rows), "questions": sum(len(r["questions"]) for r in rows),
                          "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}
    manifest["fields_rank_baseline"] = rank_baseline(f)
    json.dump(manifest, open(os.path.join(out, "MANIFEST.json"), "w"), indent=1)
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    print(json.dumps(build(p.parse_args().root), indent=1))
