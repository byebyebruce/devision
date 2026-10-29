"""exp137 training mix: the exp132 training file (r23's data) plus the board-gap families of data/v05/board/
(docs/experiments/2026-10-29-exp137-board-mix.md, docs/research/board-gap-data.md).

  caption      Winoground-style "which caption describes this image" with one-word hard negatives (COCO train2014),
               subsampled to --caption rows, at most --per-image rows per picture
  kie          receipt / form field value and "is the field filled in" (CORD + FUNSD train), all rows
  infographic  InfographicVQA train with same-image distractors, all rows

The board dev splits (held out by image inside each family) are copied to <out>/dev_board_<family>.jsonl, the caption
one subsampled to --caption-dev rows, for the per-family check. Output: <out>/train.jsonl + MANIFEST.json.
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


def by_image_sample(rows: list[dict], n: int, per_image: int, rng: random.Random) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[r["image"]].append(r)
    keys = sorted(groups)
    rng.shuffle(keys)
    out: list[dict] = []
    for k in keys:
        g = groups[k][:]
        rng.shuffle(g)
        out.extend(g[:per_image])
        if len(out) >= n:
            break
    return out[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--base", default="v05/exp132/train.jsonl")
    ap.add_argument("--board", default="v05/board")
    ap.add_argument("--out", default="v05/exp137")
    ap.add_argument("--caption", type=int, default=20000)
    ap.add_argument("--caption-dev", type=int, default=1000)
    ap.add_argument("--per-image", type=int, default=1)
    ap.add_argument("--seed", type=int, default=137)
    a = ap.parse_args()
    root, rng = Path(a.root), random.Random(a.seed)
    out = root / a.out
    out.mkdir(parents=True, exist_ok=True)
    board = root / a.board

    base = load(root / a.base)
    add = {"caption": by_image_sample(load(board / "caption/train.jsonl"), a.caption, a.per_image, rng),
           "kie": load(board / "kie/train.jsonl"),
           "infographic": load(board / "infographic/train.jsonl")}
    dev = {"caption": by_image_sample(load(board / "caption/dev.jsonl"), a.caption_dev, 1, rng),
           "kie": load(board / "kie/dev.jsonl"),
           "infographic": load(board / "infographic/dev.jsonl")}

    train_images = {r["image"] for rows in add.values() for r in rows}
    for fam, rows in dev.items():
        leak = {r["image"] for r in rows} & train_images
        if leak:
            raise SystemExit("dev_board_%s shares %d images with the training mix" % (fam, len(leak)))
        with open(out / ("dev_board_%s.jsonl" % fam), "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    ids = {r["id"] for r in base}
    rows = base[:]
    for fam, extra in add.items():
        dup = [r["id"] for r in extra if r["id"] in ids]
        if dup:
            raise SystemExit("%s: %d ids already in the base file" % (fam, len(dup)))
        rows.extend(extra)
    rng.shuffle(rows)
    with open(out / "train.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest = {
        "base": {"path": a.base, "rows": len(base), "sha256": sha(root / a.base)},
        "board_manifests": {fam: sha(board / fam / "MANIFEST.json") for fam in add},
        "added": {fam: {"rows": len(x), "images": len({r["image"] for r in x}),
                        "kinds": Counter(r.get("kind", "").split(":")[0] for r in x)} for fam, x in add.items()},
        "dev": {fam: {"rows": len(x), "images": len({r["image"] for r in x})} for fam, x in dev.items()},
        "rows": len(rows), "seed": a.seed, "caption_per_image": a.per_image,
        "sources": Counter(r["source"] for r in rows),
        "train_sha256": sha(out / "train.jsonl"),
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    print(json.dumps({k: manifest[k] for k in ("added", "dev", "rows")}, indent=1))


if __name__ == "__main__":
    main()
