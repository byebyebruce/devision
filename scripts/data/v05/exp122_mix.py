"""exp122 training mix: exp121 + the rest of n2-spatial, so relation words are balanced again.

exp120 added 4,000 n2-spatial depth rows whose gold is always "in front of" / "behind" while "to the left of" /
"to the right of" only ever appear as distractors. What'sUp left_of / right_of fell from ~50% to 1.5% (the model now
never answers left/right), What'sUp -12.9 vs exp119. Here, on top of exp121 (unchanged), every n2-spatial row not
already in the mix is added: spatial-depth left/right/front/behind and spatial-vertical on/under/left/right, so the
mix carries the source's balanced category distribution (held-out pictures already removed by scripts/data/n2/spatial.py).
Eval sets are unchanged from exp121.
"""
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


def load(p: Path) -> list[dict]:
    return [json.loads(line) for line in open(p)]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--seed", type=int, default=122)
    a = ap.parse_args()
    root = Path(a.root)
    rng = random.Random(a.seed)
    out = root / "v05/exp122"
    out.mkdir(parents=True, exist_ok=True)

    base = load(root / "v05/exp121/train.jsonl")
    seen = {r["id"] for r in base}
    spatial = [r for r in load(root / "n2/spatial.jsonl") if r["id"] not in seen]

    rows = base + spatial
    rng.shuffle(rows)
    with open(out / "train.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    in_mix = Counter(f'{r["kind"]}/{r["category"]}' for r in rows if r["source"] == "n2-spatial")
    manifest = {"built_by": "scripts/data/v05/exp122_mix.py", "seed": a.seed,
                "train_from": "v05/exp121/train.jsonl", "train_from_sha256": sha(root / "v05/exp121/train.jsonl"),
                "added": {"n2-spatial": dict(Counter(f'{r["kind"]}/{r["category"]}' for r in spatial))},
                "n2_spatial_in_mix": dict(in_mix),
                "train": {"rows": len(rows), "sha256": sha(out / "train.jsonl")}}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
