"""exp128 training mix: r22 (exp124) training file + coco-size replay from the v0.4 (r21-soft) training mix.

r22 held 31 of 32 sets against v0.4 but test_size fell -2.59 [-4.10, -1.18]. v0.4 trained on 4,331 coco-size rows
(data/n2/soft_mix.jsonl); every v05 mix since exp116 keeps only 129 (3%, while the other v0.4 sources kept ~35%).
Here the exp124 file is kept unchanged and all coco-size rows of soft_mix not already in it are added (soft_mix
already excludes eval pictures). exp128 vs r22 (same seed 22) isolates the replay.
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
    ap.add_argument("--seed", type=int, default=128)
    ap.add_argument("--base", default="v05/exp124/train.jsonl")
    ap.add_argument("--size", type=int, default=0, help="rows to add (0 = all available)")
    ap.add_argument("--out", default="v05/exp128")
    ap.add_argument("--extra", action="append", default=[], metavar="SOURCE[/KIND]=N",
                    help="also replay N soft_mix rows of this source (and kind) not in the base (0 = all); repeatable")
    a = ap.parse_args()
    root = Path(a.root)
    rng = random.Random(a.seed)
    out = root / a.out
    out.mkdir(parents=True, exist_ok=True)

    base = load(root / a.base)
    seen = {r["id"] for r in base}
    soft = [r for r in load(root / "n2/soft_mix.jsonl") if r["id"] not in seen]
    add = [r for r in soft if r["source"] == "coco-size"]
    rng.shuffle(add)
    if a.size:
        add = add[: a.size]
    for spec in a.extra:
        key, n = spec.rsplit("=", 1)
        source, _, kind = key.partition("/")
        pick = [r for r in soft if r["source"] == source and (not kind or r.get("kind") == kind)]
        rng.shuffle(pick)
        add += pick[: int(n)] if int(n) else pick

    rows = base + add
    rng.shuffle(rows)
    with open(out / "train.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    manifest = {"built_by": "scripts/data/v05/exp128_mix.py", "seed": a.seed,
                "train_from": a.base, "train_from_sha256": sha(root / a.base),
                "replay_from": "n2/soft_mix.jsonl",
                "extra": a.extra,
                "added": dict(Counter(r["source"] for r in add)),
                "train": {"rows": len(rows), "sha256": sha(out / "train.jsonl"),
                          "sources": dict(Counter(r["source"] for r in rows))}}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: v for k, v in manifest.items() if k != "train"} | {"rows": len(rows)}, indent=1))


if __name__ == "__main__":
    main()
