"""exp31 minimal text diagnosis for the force questions (critic EXP28-RESULT-02 item 3, EXP31-R1-01..03; evaluation
only): new number cases on a blank picture, asked level by level about the same case:

  (a) the two numbers alone, no magnets or pairs: which relation holds (less / greater / equal);
  (b) in which pair are the magnets closer together;
  (c) with the rule "closer -> stronger" stated, the ScienceQA question, in a strong wording (greater / stronger)
      and a weak wording (smaller / weaker);
  (d) equal distances, the magnet sizes differ or not, with the rule "larger -> stronger", strong and weak wording.

A family is one canonical case -- (unit, unordered distance pair) for a-c, (unit, distance, sizes differ or not)
for d -- drawn without replacement, so no case appears twice. Some families have equal distances (a-c) or equal
sizes (d), so "equal / the same" is the right answer for part of the questions. Every family is asked in both pair
orders with the same wording and the same option arrangement: only the values (sizes) swap between Pair 1 and 2.
Each row records which object each option names ("1", "2", "same"). The code does not read or copy any ScienceQA
picture or transcribed case.

    uv run python scripts/exp31/build.py --root data
"""
import argparse
import hashlib
import itertools
import json
import os
import random

HINT = ("The images below show two pairs of magnets. The magnets in different pairs do not affect each other. "
        "All the magnets shown are made of the same material")
RULE_DIST = "Rule: magnets that are closer together have a stronger magnetic force between them."
RULE_SIZE = "Rule: larger magnets have a stronger magnetic force between them."
Q = "Think about the magnetic force between the magnets in each pair. Which of the following statements is true?"
VALUES = {"in": [1, 2, 3, 4, 5, 0.5, 1.25, 2.25, 4.75], "cm": [1, 2, 3, 4, 0.1, 0.25, 1.25, 1.6], "mm": [8.4, 13.2, 16.9, 24.7, 42.8, 58.7]}
LARGE, SMALL = "two large magnets", "one large magnet and one small magnet"


def fmt(v):
    return "%g" % v


def force_options(word: str):
    """[names Pair 1, names Pair 2, same]."""
    if word in ("greater", "smaller"):
        return ["The magnitude of the magnetic force is %s in Pair 1" % word,
                "The magnitude of the magnetic force is %s in Pair 2" % word,
                "The magnitude of the magnetic force is the same in both pairs"]
    return ["The magnetic force is %s in Pair 1" % word, "The magnetic force is %s in Pair 2" % word,
            "The strength of the magnetic force is the same in both pairs"]


def row(fid, case, level, swap, extra, text, question, options, gold_cat, perm):
    """options = [cat "1", cat "2", cat "same"]; perm is the family's display order, shared by both pair orders."""
    cats = ["1", "2", "same"]
    order = [options[i] for i in perm]
    gold = options[cats.index(gold_cat)]
    return {"id": "exp31:%s:%s:s%d%s" % (fid, level, swap, extra), "source": "exp31-" + level, "kind": level,
            "family": fid, "case": case, "image_id": "exp31:%s" % fid, "image": "exp21/images/blank.png",
            "state_text": text,
            "questions": {"q": {"type": "choice", "instructions": question, "criteria": {o: None for o in order}}},
            "gold": {"q": {"probabilities": {o: float(o == gold) for o in order}}}, "slot": order.index(gold),
            "gold_cat": gold_cat, "option_cats": {options[i]: cats[i] for i in range(3)}}


def weak_flip(cat: str, strong: bool) -> str:
    return cat if strong or cat == "same" else {"1": "2", "2": "1"}[cat]


def distance_family(fid, unit, d, rng):
    perms = {k: rng.sample(range(3), 3) for k in ("a", "b", "c-strong", "c-weak")}
    words = (rng.choice(["greater", "stronger"]), rng.choice(["smaller", "weaker"]))
    case = "distance-equal" if d[0] == d[1] else "distance"
    rows = []
    for swap in (0, 1):
        d1, d2 = (d[1], d[0]) if swap else (d[0], d[1])
        closer = "same" if d1 == d2 else "1" if d1 < d2 else "2"
        x, y = "%s %s" % (fmt(d1), unit), "%s %s" % (fmt(d2), unit)
        rows.append(row(fid, case, "a-compare-numbers", swap, "", "Compare the two numbers: %s and %s." % (x, y),
                        "Which statement is true?", ["%s is less than %s" % (x, y), "%s is greater than %s" % (x, y),
                                                     "%s is equal to %s" % (x, y)], closer, perms["a"]))
        text = HINT + ".\nPair 1: the magnets are %s apart. Pair 2: the magnets are %s apart." % (x, y)
        rows.append(row(fid, case, "b-closer-pair", swap, "", text, "In which pair are the magnets closer together?",
                        ["Pair 1", "Pair 2", "neither; the distances are the same"], closer, perms["b"]))
        for word, key in zip(words, ("c-strong", "c-weak")):
            strong = key == "c-strong"
            rows.append(row(fid, case, "c-rule-distance", swap, ":" + word, text + "\n" + RULE_DIST, Q,
                            force_options(word), weak_flip(closer, strong), perms[key]))
    return rows


def size_family(fid, unit, dist, differ, rng):
    perms = {k: rng.sample(range(3), 3) for k in ("d-strong", "d-weak")}
    words = (rng.choice(["greater", "stronger"]), rng.choice(["smaller", "weaker"]))
    rows = []
    for swap in (0, 1):
        sizes = [LARGE, SMALL] if differ else [LARGE, LARGE]
        if swap:
            sizes = sizes[::-1]
        hint = HINT + (", but some of them are different sizes." if differ else ".")
        text = hint + "\nPair 1: %s, %s %s apart. Pair 2: %s, %s %s apart." % (sizes[0], fmt(dist), unit, sizes[1], fmt(dist), unit)
        stronger = "same" if sizes[0] == sizes[1] else "1" if sizes[0] == LARGE else "2"
        for word, key in zip(words, ("d-strong", "d-weak")):
            rows.append(row(fid, "size" if differ else "size-equal", "d-rule-size", swap, ":" + word,
                            text + "\n" + RULE_SIZE, Q, force_options(word), weak_flip(stronger, key == "d-strong"), perms[key]))
    return rows


def build(root: str, n_dist: int = 45, n_dist_equal: int = 15, n_size: int = 22, n_size_equal: int = 8,
          seed: int = 31) -> dict:
    rng = random.Random(seed)
    pairs = [(u, p) for u in VALUES for p in itertools.combinations(VALUES[u], 2)]
    singles = [(u, v) for u in VALUES for v in VALUES[u]]
    rows = []
    for i, (u, p) in enumerate(rng.sample(pairs, n_dist)):
        rows += distance_family("d%03d" % i, u, list(rng.sample(p, 2)), rng)
    for i, (u, v) in enumerate(rng.sample(singles, n_dist_equal)):
        rows += distance_family("e%03d" % i, u, [v, v], rng)
    for i, (u, v) in enumerate(rng.sample(singles, n_size)):
        rows += size_family("s%03d" % i, u, v, True, rng)
    for i, (u, v) in enumerate(rng.sample(singles, n_size_equal)):
        rows += size_family("t%03d" % i, u, v, False, rng)
    out = os.path.join(root, "exp31")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "text_force.jsonl")
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    levels = sorted({r["kind"] for r in rows})
    m = {"rows": len(rows), "families": len({r["family"] for r in rows}),
         "by_level": {lv: sum(r["kind"] == lv for r in rows) for lv in levels},
         "by_case": {c: len({r["family"] for r in rows if r["case"] == c}) for c in sorted({r["case"] for r in rows})},
         "gold_cat": {lv: {c: sum(r["kind"] == lv and r["gold_cat"] == c for r in rows) for c in ("1", "2", "same")} for lv in levels},
         "slots": [sum(r["slot"] == s for r in rows) for s in range(3)],
         "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}
    json.dump(m, open(os.path.join(out, "MANIFEST.json"), "w"), indent=1)
    return m


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    print(json.dumps(build(p.parse_args().root), indent=1))
