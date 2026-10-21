"""Text-given rule questions for the two ScienceQA forms deVision misses most (docs/experiments/2026-10-17-
exp28-diagnose-plan.md; candidate training data for v0.4, not yet used): the values a ScienceQA picture shows are
given as text on a blank picture, with the question and options in ScienceQA's wording, so the model can learn the
rule that turns the values into the answer.

  - temperature: two gas samples, "mass of each particle M u, average particle speed V m/s"; the sample with the
    higher average kinetic energy (M * V^2) has the higher temperature, equal -> "neither";
  - force: two pairs of magnets, distances (and, sometimes, "one large magnet and one small magnet" vs "two large
    magnets"); closer -> stronger, larger magnets -> stronger, only one factor differs; six answer wordings
    (greater / smaller, stronger / weaker, magnitude / strength the same).

Values are random (seeded); no picture or value of the ScienceQA dev / test questions is used. The cases are
balanced: temperature same mass / same speed / both differ (kinetic energy decides) / equal; force distance /
size / equal.

    uv run python scripts/data/rules_text.py --root data --n 6000 --out data/rules/text_rules.jsonl
"""
import argparse
import json
import os
import random
from typing import Dict

TEMP_HINT = ("The diagrams below show two pure samples of gas in identical closed, rigid containers. Each colored "
             "ball represents one gas particle. Both samples have the same number of particles.")
TEMP_Q = "Compare the average kinetic energies of the particles in each sample. Which sample has the higher temperature?"
TEMP_SAME = "neither; the samples have the same temperature"
FORCE_HINT = ("The images below show two pairs of magnets. The magnets in different pairs do not affect each other. "
              "All the magnets shown are made of the same material")
FORCE_Q = ("Think about the magnetic force between the magnets in each pair. Which of the following statements is "
           "true?")
UNITS = [("in", [0.5, 0.75, 1, 1.25, 1.5, 2, 2.25, 3, 3.5, 4, 4.75, 5]), ("cm", [0.1, 0.25, 1, 1.25, 1.6, 2, 3, 4.5]),
         ("mm", [8.4, 13.2, 16.9, 24.7, 28.9, 42.8, 58.7])]


def fmt(v: float) -> str:
    return "{:,}".format(int(v)) if float(v).is_integer() else ("%g" % v)


def temperature(rng: random.Random) -> Dict:
    case = rng.choice(["same mass", "same speed", "both differ", "equal"])
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
        if case == "equal" or (ka != kb and (case != "both differ" or (ma != mb and va != vb))):
            break
    gold = "sample A" if ka > kb else "sample B" if kb > ka else TEMP_SAME
    text = "Sample A: mass of each particle %d u, average particle speed %s m/s. Sample B: mass of each particle %d u, average particle speed %s m/s." % (
        ma, fmt(va), mb, fmt(vb))
    opts = ["sample A", "sample B", TEMP_SAME]
    rng.shuffle(opts)
    return {"case": "temperature: " + case, "state_text": TEMP_HINT + "\n" + text, "question": TEMP_Q, "options": opts, "gold": gold}


def force(rng: random.Random) -> Dict:
    case = rng.choice(["distance", "size", "equal"])
    unit, values = rng.choice(UNITS)
    d1 = d2 = rng.choice(values)
    s1 = s2 = "two large magnets"
    if case == "distance":
        while d2 == d1:
            d2 = rng.choice(values)
        stronger = 1 if d1 < d2 else 2
    elif case == "size":
        small = "one large magnet and one small magnet"
        if rng.random() < 0.5:
            s1 = small
            stronger = 2
        else:
            s2 = small
            stronger = 1
    else:
        stronger = 0
    hint = FORCE_HINT + (", but some of them are different sizes." if case == "size" else ".")
    text = "Pair 1: %s, %s %s apart. Pair 2: %s, %s %s apart." % (s1, fmt(d1), unit, s2, fmt(d2), unit)
    style = rng.choice(["magnitude-greater", "magnitude-smaller", "strength"])
    end = "." if rng.random() < 0.5 else ""
    if style == "strength":
        word = rng.choice(["stronger", "weaker"])
        opts = ["The magnetic force is %s in Pair 1%s" % (word, end), "The magnetic force is %s in Pair 2%s" % (word, end),
                "The strength of the magnetic force is the same in both pairs%s" % end]
        right = (1 if stronger == 1 else 2) if word == "stronger" else (2 if stronger == 1 else 1)
    else:
        word = style.split("-")[1]
        opts = ["The magnitude of the magnetic force is %s in Pair 1%s" % (word, end),
                "The magnitude of the magnetic force is %s in Pair 2%s" % (word, end),
                "The magnitude of the magnetic force is the same in both pairs%s" % end]
        right = (1 if stronger == 1 else 2) if word == "greater" else (2 if stronger == 1 else 1)
    gold = opts[2] if stronger == 0 else opts[right - 1]
    rng.shuffle(opts)
    return {"case": "force: " + case, "state_text": hint + "\n" + text, "question": FORCE_Q, "options": opts, "gold": gold}


def sample(i: int, item: Dict, image: str) -> Dict:
    return {"id": "rules-text:%06d" % i, "source": "rules-text", "kind": item["case"].split(":")[0], "axis": item["case"],
            "image_id": "rules-text:blank", "image": image, "state_text": item["state_text"],
            "questions": {"q": {"type": "choice", "instructions": item["question"], "criteria": {o: None for o in item["options"]}}},
            "gold": {"q": {"probabilities": {o: float(o == item["gold"]) for o in item["options"]}}}}


def build(n: int, seed: int = 30, image: str = "exp21/images/blank.png"):
    rng = random.Random(seed)
    return [sample(i, (temperature if i % 2 == 0 else force)(rng), image) for i in range(n)]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    p.add_argument("--n", type=int, default=6000)
    p.add_argument("--out", default="data/rules/text_rules.jsonl")
    a = p.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    rows = build(a.n)
    with open(a.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print("wrote %d to %s" % (len(rows), a.out))
