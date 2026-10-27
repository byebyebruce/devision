"""exp117 training mix: exp116 (exp115 + GUI next-action + egocentric) + exp114's science rows (TQA / AI2D / ScienceQA).

exp115 dropped exp114's science rows and ScienceQA regressed significantly (-1.72 vs v0.4); exp114 kept them
and ScienceQA held (+0.57). exp117 puts them back on top of the exp116 mix, so the only change from exp116
is the science rows. Dev / test sets stay the frozen copies in data/v05/exp116/.
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

SCIENCE = {"tqa", "ai2d", "scienceqa"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--seed", type=int, default=117)
    a = ap.parse_args()
    root = Path(a.root)
    base = [json.loads(l) for l in open(root / "v05/exp116/train.jsonl")]
    seen = {r["id"] for r in base}
    sci = [r for r in map(json.loads, open(root / "v05/exp114/train.jsonl"))
           if r["source"] in SCIENCE and r["id"] not in seen]
    rows = base + sci
    random.Random(a.seed).shuffle(rows)
    out = root / "v05/exp117"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "train.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    manifest = {"built_by": "scripts/data/v05/exp117_mix.py", "seed": a.seed, "rows": len(rows),
                "exp116": len(base), "science_from_exp114": dict(Counter(r["source"] for r in sci))}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    print(manifest)


if __name__ == "__main__":
    main()
