"""exp115 training mix: exp81 (chart + v0.4 replay) + exp114's ScreenQA UI rows + extra COCO counting.

Drops the science rows exp114 added (no significant ScienceQA gain) and adds counting
questions to undo the CountBench regression seen in exp114 (count share 4.4% -> ~10%).
"""
import argparse
import json
import random
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--count", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=115)
    a = ap.parse_args()
    root = Path(a.root)
    base = [json.loads(l) for l in open(root / "v05/exp81/train.jsonl")]
    ui = [r for r in map(json.loads, open(root / "v05/exp114/train.jsonl")) if r["source"] == "screenqa"]
    seen = {r["id"] for r in base}
    pool = [r for r in map(json.loads, open(root / "n2/count_train.jsonl")) if r["id"] not in seen]
    rng = random.Random(a.seed)
    count = rng.sample(pool, a.count)
    rows = base + ui + count
    rng.shuffle(rows)
    out = root / "v05/exp115"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "train.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    manifest = {"built_by": "scripts/data/v05/exp115_mix.py", "seed": a.seed, "rows": len(rows),
                "exp81": len(base), "screenqa": len(ui), "coco_count_extra": len(count)}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    print(manifest)


if __name__ == "__main__":
    main()
