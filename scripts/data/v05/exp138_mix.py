"""exp138 training mix: the exp132 training file (r23's data) filtered by the board exclusion index, plus the second
batch of board-gap families of data/v05/board/ (docs/experiments/2026-10-29-exp138-board-mix2.md,
docs/research/board-gap-data.md). Independent of exp137: each batch is judged against r23 on its own.

  figure   ArxivQA figure questions (CharXiv), --figure rows, at most --per-image per picture
  depth    which of two marked objects / points is closer (CV-Bench 3D, BLINK depth), --depth rows
  som      set-of-marks Mind2Web: which marked element to act on, all rows
  drive    DriveLM front camera: which description matches the boxed object, --drive rows
  hateful  Hateful Memes train: is this meme hateful, all rows

Every row of the base file whose picture is a Decision Index board image (sha256 / pixels / near-duplicate hash) or
whose image_id is a board source id is dropped; the count goes to the MANIFEST ("base_board_filtered"). The family
builders already excluded board images. Each family's dev split is copied to <out>/dev_board_<family>.jsonl.
"""
import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v05_board_common import BoardIndex  # noqa: E402
from v05_board_exclusion import hash_file  # noqa: E402


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


def board_filter(rows: list[dict], root: Path, index: BoardIndex, workers: int) -> tuple[list[dict], dict]:
    rels = sorted({r["image"] for r in rows})
    with ProcessPoolExecutor(workers) as ex:
        hs = list(ex.map(hash_file, [str(root / p) for p in rels], chunksize=256))
    bad, kinds = set(), Counter()
    for rel, h in zip(rels, hs):
        if "error" in h:
            raise SystemExit("cannot hash %s: %s" % (rel, h["error"]))
        got = index.hit_hashes(h)
        if got:
            bad.add(rel)
            kinds[got[0]] += 1
    keep = [r for r in rows if r["image"] not in bad and not index.hit_id(r.get("image_id", ""))]
    return keep, {"rows": len(rows) - len(keep), "images": len(bad), "hash_kinds": dict(kinds),
                  "id_rows": sum(index.hit_id(r.get("image_id", "")) for r in rows)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--base", default="v05/exp132/train.jsonl")
    ap.add_argument("--board", default="v05/board")
    ap.add_argument("--exclusion", default="v05/board_exclusion.json")
    ap.add_argument("--out", default="v05/exp138")
    ap.add_argument("--figure", type=int, default=12000)
    ap.add_argument("--depth", type=int, default=10000)
    ap.add_argument("--drive", type=int, default=6000)
    ap.add_argument("--per-image", type=int, default=2)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=138)
    a = ap.parse_args()
    root, rng = Path(a.root), random.Random(a.seed)
    out = root / a.out
    out.mkdir(parents=True, exist_ok=True)
    board = root / a.board
    index = BoardIndex.load(str(root / a.exclusion))

    base_all = load(root / a.base)
    base, filtered = board_filter(base_all, root, index, a.workers)
    print("base board filter:", filtered, flush=True)
    cap = {"figure": a.figure, "depth": a.depth, "drive": a.drive}
    add, dev = {}, {}
    for fam in ("figure", "depth", "som", "drive", "hateful"):
        rows = load(board / fam / "train.jsonl")
        add[fam] = by_image_sample(rows, cap[fam], a.per_image, rng) if fam in cap else rows
        dev[fam] = load(board / fam / "dev.jsonl")

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
        "base": {"path": a.base, "rows": len(base_all), "sha256": sha(root / a.base)},
        "base_board_filtered": filtered,
        "exclusion_sha256": sha(root / a.exclusion),
        "board_manifests": {fam: sha(board / fam / "MANIFEST.json") for fam in add},
        "added": {fam: {"rows": len(x), "images": len({r["image"] for r in x}),
                        "kinds": Counter(r.get("kind", "").split(":")[0] for r in x)} for fam, x in add.items()},
        "dev": {fam: {"rows": len(x), "images": len({r["image"] for r in x})} for fam, x in dev.items()},
        "rows": len(rows), "seed": a.seed, "per_image": a.per_image,
        "sources": Counter(r["source"] for r in rows),
        "train_sha256": sha(out / "train.jsonl"),
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    print(json.dumps({k: manifest[k] for k in ("base_board_filtered", "added", "dev", "rows")}, indent=1))


if __name__ == "__main__":
    main()
