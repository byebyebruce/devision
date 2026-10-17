"""exp34 decision-semantics questions (docs/experiments/2026-10-17-exp34-decision-semantics-plan.md; critic EXP34-
DESIGN-01..03, V04-NIGHT-02; candidate data for a later decision stage, not used yet). On a blank picture, the values
given as text, each canonical case (one family) asked in two views that differ only in which object holds which
value (same wording, same option order):

  - force: "numbers" -- the two distances alone, less / greater / equal; "pairs" -- in which pair the magnets are
    closer, and the ScienceQA question with "closer -> stronger" stated, in a strong wording (greater / stronger)
    and a weak one (smaller / weaker);
  - temperature: which sample has the higher average kinetic energy (M * V^2), and the ScienceQA question in a
    strong (higher temperature) and a weak (lower temperature) wording.

About a quarter of the families are equal ("the same" is right and stays right in both views). Every question of
a family uses group = family and a fixed qid, so rlcd.plan_batches keeps a question's two views in one micro batch
(units of 2 <= 8). Cases never drawn: exp31's canonical cases (unit, unordered distance pair), the 24 transcribed
diagnostic cases, and every canonical case of real_layout tune / confirm. Train / tune split by canonical case.

    uv run python scripts/exp34/build.py --root data --force 170 --temp 170
"""
import argparse
import hashlib
import json
import os
import random
import sys
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
import real_layout as rl  # type: ignore  # noqa: E402

HINT_F = rl.FORCE_HINT + "."
HINT_T = rl.TEMP_HINT
SAME_F = "The magnitude of the magnetic force is the same in both pairs"


def fmt(v) -> str:
    return rl.fmt(v)


def excluded(root: str) -> set:
    """Canonical keys never drawn: exp31's distance cases, the transcribed cases, real_layout tune / confirm."""
    out = set()
    for line in open(os.path.join(root, "exp31", "text_force.jsonl")):
        r = json.loads(line)
        if r["case"].startswith("distance"):
            import re
            nums = re.findall(r"([\d.]+) (in|cm|mm) apart", r["state_text"])
            if len(nums) == 2:
                out.add(("force", nums[0][1], tuple(sorted(float(n) for n, _ in nums))))
    for k in rl.transcribed_keys():
        if k[0] == "force":
            out.add(("force", k[3], tuple(sorted((float(k[1]), float(k[2]))))))
        else:
            out.add(("temp", tuple(sorted(((k[1], k[2]), (k[3], k[4]))))))
    for split in ("tune", "confirm"):
        for line in open(os.path.join(root, "real_layout", "questions_%s.jsonl" % split)):
            r = json.loads(line)
            key = eval(r["case"])  # repr of real_layout.case_key, written by real_layout.build
            if key[0] == "force":
                out.add(("force", key[1], tuple(sorted(d for d, _ in key[2]))))
            else:
                out.add(("temp", tuple(sorted(tuple(map(float, o)) for o in key[2]))))
    return out


def row(fam: str, view: int, kind: str, text: str, questions: Dict[str, Dict], gold: Dict[str, str]) -> Dict:
    return {"id": "exp34:%s:%s:v%d" % (fam, kind, view), "source": "exp34-" + kind, "kind": kind, "family": fam,
            "group": fam, "image_id": "exp34:blank", "image": "exp21/images/blank.png", "state_text": text,
            "questions": {q: {"type": "choice", "instructions": v["q"], "criteria": {o: None for o in v["options"]}}
                          for q, v in questions.items()},
            "gold": {q: {"probabilities": {o: float(o == gold[q]) for o in questions[q]["options"]}} for q in questions}}


def force_family(fam: str, unit: str, d: List[float], rng: random.Random) -> List[Dict]:
    perm = {k: rng.sample(range(3), 3) for k in ("num", "closer", "strong", "weak")}
    strong_w, weak_w = rng.choice(["greater", "stronger"]), rng.choice(["smaller", "weaker"])
    rows = []
    for v in (0, 1):
        a, b = (d[1], d[0]) if v else (d[0], d[1])
        x, y = "%s %s" % (fmt(a), unit), "%s %s" % (fmt(b), unit)
        first = "same" if a == b else "1" if a < b else "2"            # which object is smaller / closer
        def ordered(opts, k):
            return [opts[i] for i in perm[k]]
        num = ["%s is less than %s" % (x, y), "%s is greater than %s" % (x, y), "%s is equal to %s" % (x, y)]
        rows.append(row(fam, v, "numbers", "Compare the two distances: %s and %s." % (x, y),
                        {"num": {"q": "Which statement is true?", "options": ordered(num, "num")}},
                        {"num": num[{"1": 0, "2": 1, "same": 2}[first]]}))
        closer = ["Pair 1", "Pair 2", "neither; the distances are the same"]
        def force_opts(word):
            if word in ("greater", "smaller"):
                return ["The magnitude of the magnetic force is %s in Pair 1" % word,
                        "The magnitude of the magnetic force is %s in Pair 2" % word, SAME_F]
            return ["The magnetic force is %s in Pair 1" % word, "The magnetic force is %s in Pair 2" % word,
                    "The strength of the magnetic force is the same in both pairs"]
        so, wo = force_opts(strong_w), force_opts(weak_w)
        idx = {"1": 0, "2": 1, "same": 2}
        flip = {"1": "2", "2": "1", "same": "same"}
        text = (HINT_F + "\nPair 1: the magnets are %s apart. Pair 2: the magnets are %s apart.\n" % (x, y)
                + "Rule: magnets that are closer together have a stronger magnetic force between them.")
        rows.append(row(fam, v, "pairs", text,
                        {"closer": {"q": "In which pair are the magnets closer together?", "options": ordered(closer, "closer")},
                         "strong": {"q": rl.FORCE_Q, "options": ordered(so, "strong")},
                         "weak": {"q": rl.FORCE_Q, "options": ordered(wo, "weak")}},
                        {"closer": closer[idx[first]], "strong": so[idx[first]], "weak": wo[idx[flip[first]]]}))
    return rows


def temp_family(fam: str, objs: List[tuple], rng: random.Random) -> List[Dict]:
    perm = {k: rng.sample(range(3), 3) for k in ("ke", "strong", "weak")}
    rows = []
    for v in (0, 1):
        (ma, va), (mb, vb) = (objs[1], objs[0]) if v else (objs[0], objs[1])
        ka, kb = ma * va * va, mb * vb * vb
        hi = "same" if ka == kb else "A" if ka > kb else "B"
        text = HINT_T + "\nSample A: mass of each particle %d u, average particle speed %s m/s. Sample B: mass of each particle %d u, average particle speed %s m/s." % (
            ma, fmt(va), mb, fmt(vb)) + "\nRule: the sample whose particles have the higher average kinetic energy has the higher temperature."
        opts = ["sample A", "sample B", rl.TEMP_SAME]
        ke = ["sample A", "sample B", "neither; the average kinetic energies are the same"]
        idx = {"A": 0, "B": 1, "same": 2}
        flip = {"A": "B", "B": "A", "same": "same"}
        rows.append(row(fam, v, "temp", text,
                        {"ke": {"q": "Which sample's particles have the higher average kinetic energy?", "options": [ke[i] for i in perm["ke"]]},
                         "strong": {"q": rl.TEMP_Q, "options": [opts[i] for i in perm["strong"]]},
                         "weak": {"q": "Compare the average kinetic energies of the particles in each sample. Which sample has the lower temperature?",
                                  "options": [opts[i] for i in perm["weak"]]}},
                        {"ke": ke[idx[hi]], "strong": opts[idx[hi]], "weak": opts[idx[flip[hi]]]}))
    return rows


def split_of(key: tuple, tune_share: float = 0.1) -> str:
    u = int(hashlib.sha256(repr(key).encode()).hexdigest()[:12], 16) / 16 ** 12
    return "tune" if u < tune_share else "train"


def is_same(option: str) -> bool:
    return any(w in option for w in ("same", "equal", "neither"))


def build(root: str, n_force: int = 170, n_temp: int = 170, seed: int = 34) -> dict:
    """n_force / n_temp training families of each form (tune families come on top, about 10% by canonical case)."""
    rng = random.Random(seed)
    banned = excluded(root)
    used: set = set()
    files: Dict[str, List[Dict]] = {"train": [], "tune": []}
    have = {("train", "force"): 0, ("train", "temp"): 0}
    k = 0
    while have[("train", "force")] < n_force or have[("train", "temp")] < n_temp:
        form = "force" if have[("train", "force")] < n_force and (k % 2 == 0 or have[("train", "temp")] >= n_temp) else "temp"
        k += 1
        equal = rng.random() < 0.25
        if form == "force":
            unit = rng.choice(list(rl.FORCE_VALUES))
            vals = rl.FORCE_VALUES[unit]
            a = rng.choice(vals)
            b = a if equal else rng.choice([x for x in vals if x != a])
            key = ("force", unit, tuple(sorted((float(a), float(b)))))
        else:
            ma, va = rng.randint(4, 46), rng.randrange(300, 2000, 10)
            mb, vb = (ma, va) if equal else (rng.randint(4, 46), rng.randrange(300, 2000, 10))
            if not equal and ma * va * va == mb * vb * vb:
                continue
            key = ("temp", tuple(sorted(((float(ma), float(va)), (float(mb), float(vb))))))
        if key in banned or key in used:
            continue
        used.add(key)
        split = split_of(key)
        fam = "%s%04d" % (form[0], len(used))
        if form == "force":
            files[split] += force_family(fam, unit, [a, b], rng)
        else:
            files[split] += temp_family(fam, [(ma, va), (mb, vb)], rng)
        if split == "train":
            have[("train", form)] += 1
    out = os.path.join(root, "exp34")
    os.makedirs(out, exist_ok=True)
    m: dict = {"seed": seed, "excluded_cases": len(banned)}
    for name, data in files.items():
        path = os.path.join(out, "semantics_%s.jsonl" % name)
        with open(path, "w") as fh:
            for r in data:
                fh.write(json.dumps(r) + "\n")
        golds = [o for r in data for g in r["gold"].values() for o, p in g["probabilities"].items() if p == 1.0]
        m[name] = {"rows": len(data), "questions": len(golds), "families": len({r["family"] for r in data}),
                   "by_kind": {k: sum(r["kind"] == k for r in data) for k in ("numbers", "pairs", "temp")},
                   "equal_share": sum(map(is_same, golds)) / max(1, len(golds)),
                   "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}
    json.dump(m, open(os.path.join(out, "MANIFEST.json"), "w"), indent=1)
    return m


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    p.add_argument("--force", type=int, default=170)
    p.add_argument("--temp", type=int, default=170)
    a = p.parse_args()
    print(json.dumps(build(a.root, a.force, a.temp), indent=1))
