"""exp31 minimal text diagnosis for the force questions (critic EXP28-RESULT-02 item 3; evaluation only): new,
balanced number cases on a blank picture, asked level by level about the same pair of distances:

  (a) which distance is smaller;
  (b) in which pair are the magnets closer together;
  (c) with the rule "closer -> stronger" stated, the ScienceQA question, in a strong wording (greater / stronger)
      and a weak wording (smaller / weaker);
  (d) equal distances, only the magnet sizes differ, with the rule "larger -> stronger", strong and weak wording.

Every family is also asked with Pair 1 and Pair 2 swapped. Integers and decimals in in / cm / mm. No ScienceQA
picture, value or the 12 transcribed questions is used. Option order is shuffled per question (seeded) and the
right option's slot recorded.

    uv run python scripts/exp31/build.py --root data
"""
import argparse
import hashlib
import json
import os
import random

HINT = ("The images below show two pairs of magnets. The magnets in different pairs do not affect each other. "
        "All the magnets shown are made of the same material")
RULE_DIST = "Rule: magnets that are closer together have a stronger magnetic force between them."
RULE_SIZE = "Rule: larger magnets have a stronger magnetic force between them."
Q = "Think about the magnetic force between the magnets in each pair. Which of the following statements is true?"
VALUES = {"in": [1, 2, 3, 4, 5, 0.5, 1.25, 2.25, 4.75], "cm": [1, 2, 3, 4, 0.1, 0.25, 1.25, 1.6], "mm": [8.4, 13.2, 16.9, 24.7, 42.8, 58.7]}


def fmt(v):
    return "%g" % v


def force_options(word: str):
    if word in ("greater", "smaller"):
        return ["The magnitude of the magnetic force is %s in Pair 1" % word,
                "The magnitude of the magnetic force is %s in Pair 2" % word,
                "The magnitude of the magnetic force is the same in both pairs"]
    return ["The magnetic force is %s in Pair 1" % word, "The magnetic force is %s in Pair 2" % word,
            "The strength of the magnetic force is the same in both pairs"]


def q(fid, level, swap, extra, text, question, options, gold, rng):
    order = list(options)
    rng.shuffle(order)
    return {"id": "exp31:%s:%s:s%d%s" % (fid, level, swap, extra), "source": "exp31-" + level, "kind": level,
            "family": fid, "image_id": "exp31:%s" % fid, "image": "exp21/images/blank.png", "state_text": text,
            "questions": {"q": {"type": "choice", "instructions": question, "criteria": {o: None for o in order}}},
            "gold": {"q": {"probabilities": {o: float(o == gold) for o in order}}}, "slot": order.index(gold)}


def build(root: str, n_dist: int = 60, n_size: int = 30, seed: int = 31) -> dict:
    rng = random.Random(seed)
    rows = []
    for i in range(n_dist):
        unit = rng.choice(list(VALUES))
        d = rng.sample(VALUES[unit], 2)
        fid = "d%03d" % i
        for swap in (0, 1):
            d1, d2 = (d[1], d[0]) if swap else (d[0], d[1])
            text = HINT + ".\nPair 1: the magnets are %s %s apart. Pair 2: the magnets are %s %s apart." % (fmt(d1), unit, fmt(d2), unit)
            closer = 1 if d1 < d2 else 2
            rows.append(q(fid, "a-smaller-number", swap, "", text, "Which distance is smaller?",
                          ["the distance in Pair 1", "the distance in Pair 2", "the distances are equal"],
                          "the distance in Pair %d" % closer, rng))
            rows.append(q(fid, "b-closer-pair", swap, "", text, "In which pair are the magnets closer together?",
                          ["Pair 1", "Pair 2", "neither; the distances are the same"], "Pair %d" % closer, rng))
            for word in (rng.choice(["greater", "stronger"]), rng.choice(["smaller", "weaker"])):
                opts = force_options(word)
                strong = word in ("greater", "stronger")
                gold = opts[(closer if strong else 3 - closer) - 1]
                rows.append(q(fid, "c-rule-distance", swap, ":" + word, text + "\n" + RULE_DIST, Q, opts, gold, rng))
    for i in range(n_size):
        unit = rng.choice(list(VALUES))
        dist = rng.choice(VALUES[unit])
        fid = "s%03d" % i
        for swap in (0, 1):
            sizes = ["two large magnets", "one large magnet and one small magnet"]
            if swap:
                sizes = sizes[::-1]
            text = (HINT + ", but some of them are different sizes.\nPair 1: %s, %s %s apart. Pair 2: %s, %s %s apart." %
                    (sizes[0], fmt(dist), unit, sizes[1], fmt(dist), unit))
            stronger = 1 if sizes[0] == "two large magnets" else 2
            for word in (rng.choice(["greater", "stronger"]), rng.choice(["smaller", "weaker"])):
                opts = force_options(word)
                strong = word in ("greater", "stronger")
                gold = opts[(stronger if strong else 3 - stronger) - 1]
                rows.append(q(fid, "d-rule-size", swap, ":" + word, text + "\n" + RULE_SIZE, Q, opts, gold, rng))
    out = os.path.join(root, "exp31")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "text_force.jsonl")
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    m = {"rows": len(rows), "families": len({r["family"] for r in rows}),
         "by_level": {lv: sum(r["kind"] == lv for r in rows) for lv in sorted({r["kind"] for r in rows})},
         "slots": [sum(r["slot"] == s for r in rows) for s in range(3)],
         "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}
    json.dump(m, open(os.path.join(out, "MANIFEST.json"), "w"), indent=1)
    return m


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    print(json.dumps(build(p.parse_args().root), indent=1))
