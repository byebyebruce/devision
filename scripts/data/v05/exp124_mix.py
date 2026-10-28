"""exp124 training mix: exp122 + existence yes/no replay from the v0.4 (r21-soft) training mix.

exp122 (GUI 33% + balanced n2-spatial) passed its What'sUp / CV-Bench criteria, but against v0.4 POPE fell
-0.71 [-1.36, -0.06] (lv POPE identical) and lv VQAv2 yes/no -0.74 (CI touches 0). v0.4 trained on 40,000
coco-exist rows (data/n2/soft_mix.jsonl); the exp122 mix keeps only 3,076. Here exp122's training file is kept
unchanged and 6,000 coco-exist (half true / half false) plus 3,000 VQAv2 noul rows are added from soft_mix rows not
already in the mix (soft_mix already excludes eval pictures). exp124 vs exp122 isolates the replay.
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
    ap.add_argument("--seed", type=int, default=124)
    ap.add_argument("--exist", type=int, default=6000)
    ap.add_argument("--vqav2", type=int, default=3000)
    a = ap.parse_args()
    root = Path(a.root)
    rng = random.Random(a.seed)
    out = root / "v05/exp124"
    out.mkdir(parents=True, exist_ok=True)

    base = load(root / "v05/exp122/train.jsonl")
    seen = {r["id"] for r in base}
    pool = [r for r in load(root / "n2/soft_mix.jsonl") if r["id"] not in seen]
    rng.shuffle(pool)
    exist = [r for r in pool if r["source"] == "coco-exist"]
    add = [r for r in exist if r["answer_key"] == "true"][: a.exist // 2]
    add += [r for r in exist if r["answer_key"] == "false"][: a.exist // 2]
    add += [r for r in pool if r["source"] == "vqav2"
            and all(q["type"] == "noul" for q in r["questions"].values())][: a.vqav2]

    rows = base + add
    rng.shuffle(rows)
    with open(out / "train.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    manifest = {"built_by": "scripts/data/v05/exp124_mix.py", "seed": a.seed,
                "train_from": "v05/exp122/train.jsonl", "train_from_sha256": sha(root / "v05/exp122/train.jsonl"),
                "replay_from": "n2/soft_mix.jsonl",
                "added": dict(Counter(f'{r["source"]}/{r.get("answer_key", "-")}' for r in add)),
                "train": {"rows": len(rows), "sha256": sha(out / "train.jsonl")}}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
