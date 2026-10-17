"""exp28 diagnostic question sets (docs/experiments/2026-10-17-exp28-diagnose-plan.md), no training data:

  - control30.jsonl: exp22's dev16 (exp21 dev families at 16 px) restricted to families d0000-d0014, both value
    views -- 30 pictures, 90 questions, chosen by family id before looking at any prediction (positive control of
    the exported model against exp26's own step-5000 monitor records);
  - fields.jsonl: direct readings on the 24 transcribed ScienceQA pictures (scripts/exp21/sqa_transcribed.json):
    temperature -- mass and speed of sample A and B (4 per picture); force -- the distance in Pair 1 and Pair 2
    (2 per picture); three options: the right value, the other object's value of the same field (when it differs)
    and a near value; the original picture and hint, the right option's slot rotating;
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


def near(v: float, field: str) -> List[float]:
    """Two near values of the same kind, never equal to v."""
    if field == "mass":
        return [v + 2, v - 2]
    if field == "speed":
        return [v + 100, v - 100]
    return [v * 2, v / 2]


def options(right: float, other: float, field: str, slot: int, unit: str) -> Dict[str, float]:
    cands = [right] + ([other] if other != right else []) + [x for x in near(right, field) if x != other]
    cands = cands[:3]
    order = cands[1:]
    order.insert(slot % 3, right)
    return {"%s %s" % (fmt_num(x), unit): float(x == right) for x in order}


def fields(root: str, notes: Dict[str, str]) -> List[Dict]:
    temp = {r["id"]: r for r in read(os.path.join(root, "exp21", "dev_sqa_temp.jsonl"))}
    force = {r["id"]: r for r in read(os.path.join(root, "exp21", "dev_sqa_force.jsonl"))}
    rows, slot = [], 0
    for sid in sorted(k for k in notes if k != "_note"):
        if sid in temp:
            src, vals = temp[sid], parse_temp(notes[sid])
            asks = [("mass", "u", "What is the mass of each particle in sample %s?"),
                    ("speed", "m/s", "What is the average particle speed in sample %s?")]
            for field, unit, q in asks:
                for s, o in (("A", "B"), ("B", "A")):
                    probs = options(vals[s][field], vals[o][field], field, slot, unit)
                    slot += 1
                    rows.append(row(src, "%s:%s:%s" % (sid, field, s), q % s, probs, "temp-" + field))
        elif sid in force:
            src, vals = force[sid], parse_force(notes[sid])
            for p, o in (("1", "2"), ("2", "1")):
                unit = vals[p]["unit"]
                probs = options(vals[p]["distance"], vals[o]["distance"], "distance", slot, unit)
                slot += 1
                rows.append(row(src, "%s:distance:%s" % (sid, p),
                                "What is the distance between the magnets in Pair %s?" % p, probs, "force-distance"))
        else:
            raise ValueError("%s is in neither temperature nor force" % sid)
    return rows


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
    files = {"control30": control30(root), "fields": fields(root, notes), "force_rule": force_rule(root),
             "equal_wording": equal_wording(root)}
    manifest = {}
    for name, rows in files.items():
        path = os.path.join(out, name + ".jsonl")
        with open(path, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        manifest[name] = {"rows": len(rows), "questions": sum(len(r["questions"]) for r in rows),
                          "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}
    json.dump(manifest, open(os.path.join(out, "MANIFEST.json"), "w"), indent=1)
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    print(json.dumps(build(p.parse_args().root), indent=1))
