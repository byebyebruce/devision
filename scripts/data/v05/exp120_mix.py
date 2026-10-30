"""exp120 training mix: exp119 + more egocentric + depth-ordering replay.

exp117/exp118 both lost CV-Bench 3D (depth / distance, -1.4 / -2.0 vs their bases) after the GUI / ego rows came in,
while the v0.4-era depth questions (n2-spatial "in front of" / "behind", VG relations on GQA/COCO photos, held-out
pictures already removed by scripts/data/n2/spatial.py) are only 297 rows in the mix. So here, on top of exp119:
- + --depth n2-spatial depth rows whose category is "in front of" / "behind" (not already in exp119);
- + --ego more EPIC-VISOR rows (uniform over the three kinds, not already in exp119; same train participants).
Eval sets are unchanged from exp119 (no new files besides train.jsonl).
"""
import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def load(p: Path) -> list[dict]:
    return [json.loads(line) for line in open(p)]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--seed", type=int, default=120)
    ap.add_argument("--ego", type=int, default=12000)
    ap.add_argument("--depth", type=int, default=4000)
    a = ap.parse_args()
    root = Path(a.root)
    rng = random.Random(a.seed)
    out = root / "v05/exp120"
    out.mkdir(parents=True, exist_ok=True)

    base = load(root / "v05/exp119/train.jsonl")
    seen = {r["id"] for r in base}

    depth = [r for r in load(root / "n2/spatial.jsonl")
             if r["kind"] == "spatial-depth" and r["category"] in ("in front of", "behind") and r["id"] not in seen]
    depth = rng.sample(depth, min(a.depth, len(depth)))

    by_kind: dict[str, list[dict]] = defaultdict(list)
    for r in load(root / "v05/ego/train.jsonl"):
        if r["id"] not in seen:
            by_kind[r["kind"]].append(r)
    ego = []
    for k in sorted(by_kind):
        ego += rng.sample(by_kind[k], min(len(by_kind[k]), a.ego // len(by_kind)))

    rows = base + depth + ego
    rng.shuffle(rows)
    with open(out / "train.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    manifest = {"built_by": "scripts/data/v05/exp120_mix.py", "seed": a.seed,
                "train_from": "v05/exp119/train.jsonl", "train_from_sha256": sha(root / "v05/exp119/train.jsonl"),
                "added": {"n2-spatial depth": dict(Counter(r["category"] for r in depth)),
                          "epic-visor": dict(Counter(r["kind"] for r in ego))},
                "ego_manifest_sha256": sha(root / "v05/ego/MANIFEST.json"),
                "train": {"rows": len(rows), "sha256": sha(out / "train.jsonl")}}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
