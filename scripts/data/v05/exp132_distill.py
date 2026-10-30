"""exp132 training mix: the exp129 training file with v0.4's own answers mixed into the gold of the old natural-photo
sources (offline distillation toward v0.4; docs/experiments/2026-10-29-exp132-v4-offline-distill.md).

Every fine-tune from v0.4 so far lost 0.3-1.0 on POPE against v0.4 (r22, exp125-exp129), and replaying more
yes/no / existence rows (exp129) did not stop it. Here the rows stay the same; only the targets of the old sources
change to (1 - w) * gold + w * v0.4, so the student is pulled toward v0.4's function where it was already good.

  subset: write the old-source rows of --base to <out>/teacher_input.jsonl (evaluate v0.4 on it with devision-eval)
  mix:    read the v0.4 details and write <out>/train.jsonl + MANIFEST.json
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

OLD = ("vqav2", "coco-exist", "gqa", "n2-spatial", "coco-size", "coco-count-val", "v7w", "tallyqa", "coco-position",
       "coco-relation", "aokvqa", "vsr", "vg-relation")


def load(p: Path) -> list[dict]:
    return [json.loads(line) for line in open(p)]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["subset", "mix"])
    ap.add_argument("--root", default="data")
    ap.add_argument("--base", default="v05/exp129r/train.jsonl")
    ap.add_argument("--out", default="v05/exp132")
    ap.add_argument("--teacher", default="", help="devision-eval details of v0.4 on teacher_input.jsonl (mix)")
    ap.add_argument("--weight", type=float, default=0.5, help="share of the v0.4 answer in the new target")
    a = ap.parse_args()
    root = Path(a.root)
    out = root / a.out
    out.mkdir(parents=True, exist_ok=True)
    base = load(root / a.base)

    if a.step == "subset":
        rows = [r for r in base if r.get("source") in OLD]
        with open(out / "teacher_input.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print("teacher_input: %d rows, %d questions" % (len(rows), sum(len(r["questions"]) for r in rows)))
        return

    teacher = {}
    for d in load(Path(a.teacher)):
        if d.get("control", "none") == "none":
            # ids repeat with different option sets (n2-spatial), so the options are part of the key
            teacher[(d["sample_id"], d["qid"], tuple(sorted(d["options"])))] = d["probabilities"]
    changed, moved, src = 0, 0, Counter()
    with open(out / "train.jsonl", "w") as f:
        for r in base:
            if r.get("source") in OLD:
                for qid in r["questions"]:
                    q = r["questions"][qid]
                    opts = ["false", "true"] if q["type"] == "noul" else list(q["criteria"])
                    t = teacher[(r["id"], qid, tuple(sorted(opts)))]   # KeyError = the teacher pass is incomplete; never train on half of it
                    g = r["gold"][qid]["probabilities"]
                    gs, ts = sum(g.values()), sum(t.values())
                    new = {k: round((1 - a.weight) * g.get(k, 0.0) / gs + a.weight * t.get(k, 0.0) / ts, 6)
                           for k in {**g, **t}}
                    if max(g, key=g.__getitem__) != max(t, key=t.__getitem__):
                        moved += 1
                    r["gold"][qid]["probabilities"] = new
                    changed += 1
                    src[r["source"]] += 1
            f.write(json.dumps(r) + "\n")
    man = {"base": a.base, "base_sha256": sha(root / a.base), "teacher": a.teacher, "weight": a.weight,
           "old_sources": list(OLD), "rows": len(base), "questions_changed": changed,
           "teacher_disagrees_with_gold": moved, "changed_by_source": dict(src.most_common()),
           "train_sha256": sha(out / "train.jsonl")}
    json.dump(man, open(out / "MANIFEST.json", "w"), indent=1)
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
