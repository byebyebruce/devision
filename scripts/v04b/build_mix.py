"""Candidate B decision mix for v0.4 (critic V04-NIGHT-02 / 03; draft, NOT approved or run): the exp32 B mix with
the exp34 decision-semantics questions added, every block's exposure written down. Built from files already made:

  - r14b replay (v0.3's own mix, old decision tasks): exp32 B's 12,000-question selection, reused as is;
  - real-layout reading + ScienceQA-form questions of 1,200 training families on the ORIGINAL pictures (both
    views; a family's views share a micro batch through "group"): exp32 B's selection, reused as is;
  - exp25 synthetic panels, re-drawn as WHOLE families with explicit (group, qid) (critic V04-B-PREP-05): exp32's
    2,401 were mostly isolated single views (639 views of 598 families, 557 of them single); here the same budget
    is 320 families x both views x readA / readB / cmp + 242 text families x 2 views (exposure only; it does not
    show that dev16 comes back -- exp32 had its panels and dev16 stayed at 0.366);
  - exp34 decision semantics (data/exp34/semantics_train.jsonl, 2,380 questions, 340 canonical families, two views
    per question in one micro batch).

Shuffled with a fixed seed; nothing selected by any model output. Written to data/v04b/train_mix.jsonl with a
MANIFEST of the sources, their sha256 and the per-block question counts.

    uv run python scripts/v04b/build_mix.py --root data
"""
import argparse
import hashlib
import json
import os
import random
from collections import Counter


def sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def block_of(row: dict) -> str:
    src = row.get("source", "")
    if src.startswith("real-layout"):
        return "real_layout"
    if src.startswith("exp21-syn"):          # exp25's synthetic panels carry exp21's source names
        return "exp25_panels"
    if src.startswith("exp34"):
        return "semantics"
    return "r14b_replay"


def synthetic_families(root: str, n_image_fam: int = 320, n_text_fam: int = 242, seed: int = 40):
    """exp25's synthetic panels as WHOLE families (critic V04-B-PREP-05): both views of every chosen family, each row
    with group = its family, so each (group, qid) holds the family's two views (readA / readB / cmp on the pictures,
    cmp on the text-only rows). Same budget as exp32's 2,401 questions: 320 families x 2 views x 3 = 1,920 picture
    questions and 242 x 2 = 484 text questions. Chosen by a fixed seed, never by a model output."""
    rows = [json.loads(x) for x in open(os.path.join(root, "exp25", "train.jsonl"))]
    rng = random.Random(seed)
    out = []
    for src, n, tag in (("exp21-syn", n_image_fam, "syn-img"), ("exp21-syn-text", n_text_fam, "syn-txt")):
        fams: dict = {}
        for r in rows:
            if r["source"] == src:
                fams.setdefault(r["family"], []).append(r)
        whole = sorted(f for f, v in fams.items() if len(v) == 2)
        for f in rng.sample(whole, n):
            out += [dict(r, group="%s:%s" % (tag, f)) for r in sorted(fams[f], key=lambda r: r["id"])]
    return out


def gold_of(row: dict, qid: str) -> str:
    return next(o for o, p in row["gold"][qid]["probabilities"].items() if p == 1.0)


def build(root: str, seed: int = 40) -> dict:
    b32 = os.path.join(root, "exp32", "train_mix.jsonl")
    sem = os.path.join(root, "exp34", "semantics_train.jsonl")
    old = [json.loads(x) for x in open(b32)]
    kept = [r for r in old if not r.get("source", "").startswith("exp21-syn")]     # exp32's isolated single views out
    syn = synthetic_families(root, seed=seed)
    rows = kept + syn + [json.loads(x) for x in open(sem)]
    random.Random(seed).shuffle(rows)
    out = os.path.join(root, "v04b")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "train_mix.jsonl")
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    q = Counter()
    for r in rows:
        q[block_of(r) + (":text" if r.get("source") == "exp21-syn-text" else ":picture" if r.get("source") == "exp21-syn" else "")] += len(r["questions"])
    img = [r for r in syn if r["source"] == "exp21-syn"]
    a_ne_b = sum(gold_of(r, "readA") != gold_of(r, "readB") for r in img)
    m = {"seed": seed, "sources": {"exp32_train_mix": sha(b32), "exp34_semantics_train": sha(sem),
                                    "exp25_train": sha(os.path.join(root, "exp25", "train.jsonl"))},
         "rows": len(rows), "questions": sum(q.values()), "questions_by_block": dict(sorted(q.items())),
         "synthetic": {"picture_families_whole": len({r["family"] for r in img}), "picture_rows": len(img),
                       "picture_rows_A_ne_B": a_ne_b, "text_families_whole": len({r["family"] for r in syn if r["source"] != "exp21-syn"}),
                       "replaces": "exp32's 2,401 exp25 questions (1,917 picture questions on 639 views of 598 families, "
                                   "557 single-view; 484 text) -- same budget, whole families, explicit groups"},
         "sha256": sha(path)}
    json.dump(m, open(os.path.join(out, "MANIFEST.json"), "w"), indent=1)
    return m


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    print(json.dumps(build(p.parse_args().root), indent=1))
