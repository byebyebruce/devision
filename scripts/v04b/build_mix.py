"""Candidate B decision mix for v0.4 (critic V04-NIGHT-02 / 03; draft, NOT approved or run): the exp32 B mix with
the exp34 decision-semantics questions added, every block's exposure written down. Built from files already made:

  - r14b replay (v0.3's own mix, old decision tasks): exp32 B's 12,000-question selection, reused as is;
  - real-layout reading + ScienceQA-form questions of 1,200 training families on the ORIGINAL pictures (both
    views; a family's views share a micro batch through "group"): exp32 B's selection, reused as is;
  - exp25 synthetic panel questions: exp32 B's 2,401, all kept (kept exposure only; it does not by itself show
    that dev16 comes back -- exp32 had them and dev16 stayed at 0.366);
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


def build(root: str, seed: int = 40) -> dict:
    b32 = os.path.join(root, "exp32", "train_mix.jsonl")
    sem = os.path.join(root, "exp34", "semantics_train.jsonl")
    rows = [json.loads(x) for x in open(b32)] + [json.loads(x) for x in open(sem)]
    random.Random(seed).shuffle(rows)
    out = os.path.join(root, "v04b")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "train_mix.jsonl")
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    q = Counter()
    for r in rows:
        q[block_of(r)] += len(r["questions"])
    m = {"seed": seed, "sources": {"exp32_train_mix": sha(b32), "exp34_semantics_train": sha(sem)},
         "rows": len(rows), "questions": sum(q.values()), "questions_by_block": dict(sorted(q.items())),
         "steps_at_micro_batch_8": "about %d (plan_batches packs groups; exact count from the run)" % (sum(q.values()) // 8),
         "sha256": sha(path)}
    json.dump(m, open(os.path.join(out, "MANIFEST.json"), "w"), indent=1)
    return m


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    print(json.dumps(build(p.parse_args().root), indent=1))
