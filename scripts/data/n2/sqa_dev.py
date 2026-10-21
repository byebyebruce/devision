"""v0.4: a ScienceQA development set from its official test split (plan docs/experiments/2026-10-21-v04-plan.md,
stage 1: error analysis and monitoring without touching a test set).

    HF_ENDPOINT=https://hf-mirror.com uv run python scripts/data/n2/sqa_dev.py --root data

bench_lv_scienceqa is ScienceQA's validation split (2,097 questions with a picture), and the training sources read
only its train split, so the test split (2,017 with a picture) was never used by the project. ScienceQA reuses the
same drawn pictures across splits, so each question is flagged:
- `train_same_picture`: identical pixels to a train picture (the picture is not new to a model trained on train);
- `train_same_question`: also the same question text and the same correct answer text (effectively a seen question).
Reports should split by these flags. Each row keeps ScienceQA's subject / category / skill (`kind` = category) so
errors can be grouped by skill. Writes <root>/n2/dev_sqa.jsonl and dev_sqa.MANIFEST.json.
"""
import argparse
import glob
import json
import os
import random
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


def main(argv=None) -> None:
    from v3_build import make_sample, pixel_hash_bytes, read_parquets, save_image
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    snap = os.path.join(a.root, "raw", "hf", "datasets--derek-thomas--ScienceQA", "snapshots", "*", "data")
    train_pix, train_q = set(), set()
    for r in read_parquets(os.path.join(snap, "train-*.parquet")):
        if r["image"] is None:
            continue
        h = pixel_hash_bytes(r["image"]["bytes"])
        train_pix.add(h)
        train_q.add((h, r["question"].strip(), list(r["choices"])[int(r["answer"])].strip()))
    rng = random.Random(a.seed)
    rows, drop = [], Counter()
    test_files = glob.glob(os.path.join(snap, "test-*.parquet"))
    if not test_files:
        raise SystemExit("ScienceQA test split missing: download derek-thomas/ScienceQA data/test-*.parquet first")
    for i, r in enumerate(read_parquets(os.path.join(snap, "test-*.parquet"))):
        if r["image"] is None:
            continue
        data = r["image"]["bytes"]
        h = pixel_hash_bytes(data)
        pic = "sqa-test:" + h[:16]
        rel = "n2/images/sqa_test/%s.png" % h[:16]
        s = make_sample("scienceqa", "test-%d" % i, pic, rel, r["question"], list(r["choices"]), int(r["answer"]),
                        rng, r.get("hint") or "",
                        {"subject": r.get("subject"), "category": r.get("category"), "skill": r.get("skill"),
                         "topic": r.get("topic")})
        if s is None:
            drop["unusable options"] += 1
            continue
        s["id"] = "n2-sqa:test-%d" % i
        s["source"], s["kind"] = "scienceqa-test", str(r.get("category"))
        s["train_same_picture"] = h in train_pix
        s["train_same_question"] = (h, r["question"].strip(), list(r["choices"])[int(r["answer"])].strip()) in train_q
        save_image(a.root, rel, data)
        rows.append(s)
    out = os.path.join(a.root, "n2", "dev_sqa.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        for s in rows:
            f.write(json.dumps(s) + "\n")
    m = {"built_by": "scripts/data/n2/sqa_dev.py", "questions": len(rows), "dropped": dict(drop),
         "train_same_picture": sum(s["train_same_picture"] for s in rows),
         "train_same_question": sum(s["train_same_question"] for s in rows),
         "per_subject": dict(Counter(s["subject"] for s in rows)),
         "per_category": dict(Counter(s["kind"] for s in rows).most_common())}
    with open(out.replace(".jsonl", ".MANIFEST.json"), "w") as f:
        json.dump(m, f, indent=1)
    print(json.dumps({k: m[k] for k in ("questions", "dropped", "train_same_picture", "train_same_question",
                                        "per_subject")}, indent=1))


if __name__ == "__main__":
    main()
