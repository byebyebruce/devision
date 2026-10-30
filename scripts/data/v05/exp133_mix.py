"""exp133 training mix: one file for the tiles 0 / tiles 2 pair (native-resolution tiles, spec 原生分辨率切块).

A per-source subset of the exp129 training file (data/v05/exp129r/train.jsonl, eval pictures already excluded),
weighted toward the sources whose pictures are large and carry small text (Mind2Web, ScreenQA, ChartQA, AITW,
diagrams), plus all FUNSD form-binding rows and a replay of the small-photo sources that guard POPE / yes-no / size.
Both arms train on exactly this file in the same order (same seed), so the difference is the tiles alone.
"""
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

# source -> rows taken from the exp129 file (0 = all)
QUOTA = {
    "mind2web": 0, "screenqa": 0, "chartqa": 12000, "aitw-general": 6000, "aitw-shopping": 4000,
    "epic-visor": 6000, "ai2d": 2000, "tqa": 2000, "scienceqa": 1500, "iconqa": 500,
    "vqav2": 3000, "coco-exist": 2000, "gqa": 2000, "n2-spatial": 2000, "coco-size": 1000, "v7w": 500,
    "coco-count-val": 500, "tallyqa": 300,
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--seed", type=int, default=133)
    ap.add_argument("--base", default="v05/exp129r/train.jsonl")
    ap.add_argument("--funsd", default="v05/funsd/derived/train.jsonl")
    ap.add_argument("--out", default="v05/exp133")
    a = ap.parse_args()
    root = Path(a.root)
    rng = random.Random(a.seed)
    out = root / a.out
    out.mkdir(parents=True, exist_ok=True)

    by: dict[str, list[dict]] = {}
    for line in open(root / a.base):
        r = json.loads(line)
        by.setdefault(r["source"], []).append(r)
    rows = []
    for source, n in QUOTA.items():
        pick = by[source][:]
        rng.shuffle(pick)
        rows += pick[:n] if n else pick
    rows += [json.loads(line) for line in open(root / a.funsd)]
    rng.shuffle(rows)
    with open(out / "train.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    manifest = {"built_by": "scripts/data/v05/exp133_mix.py", "seed": a.seed,
                "base": a.base, "base_sha256": sha(root / a.base), "funsd": a.funsd, "quota": QUOTA,
                "train": {"rows": len(rows), "sha256": sha(out / "train.jsonl"),
                          "sources": dict(Counter(r["source"] for r in rows))}}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest["train"], indent=1))


if __name__ == "__main__":
    main()
