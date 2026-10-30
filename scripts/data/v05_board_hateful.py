"""Hateful-meme yes / no questions (like the board's Moderation benchmark) from the Hateful Memes TRAIN split
-> data/v05/board/hateful/.

    uv run --no-sync python scripts/data/v05_board_hateful.py --root data

Source: neuralcatcher/hateful_memes (train.jsonl + img/, fetched by v05_board_fetch.py --only hateful; the repo lacks
~1.8k of the 8.5k train images, those rows are counted as "image_missing"). The board uses dev_seen + test_seen.
Question: "Is this meme hateful?" with options A = Yes, B = No (the board's order), gold from the train label; the
meme text is NOT given (the board does not give it either: the model has to read the picture).
Exclusion: ids of every dev / test row (dev_seen, dev_unseen, test_seen, test_unseen; via board_exclusion.json
source ids), and every image hit by the board index. Hateful Memes builds "confounders" by putting another text on
the same picture, so besides the default near rule a looser one (dHash <= LOOSE_D AND pHash <= LOOSE_P bits) also
drops an image; both counts are in MANIFEST.json. Yes / No are balanced by dropping the majority label (seeded).
dev = ~5% of images.
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from v05_board_common import BoardIndex, choice_sample, image_hashes, manifest, seeded, write_split  # noqa: E402

OUT = os.path.join("v05", "board", "hateful")
RAW = os.path.join("raw", "board", "hateful")
QUESTION = "Is this meme hateful?"
LOOSE_D, LOOSE_P = 10, 12
SAVE_MAX = 1024


def hash_file(p: str) -> Dict:
    try:
        with open(p, "rb") as f:
            return image_hashes(f.read())
    except Exception as e:
        return {"error": str(e)}


def balance(rows: List[Dict], key=lambda r: r["label"]) -> List[Dict]:
    """Equal numbers of each label (lowest seeded hash of the id kept)."""
    by: Dict[int, List[Dict]] = {}
    for r in rows:
        by.setdefault(key(r), []).append(r)
    n = min(len(v) for v in by.values()) if len(by) > 1 else 0
    out = []
    for lab in sorted(by):
        out += sorted(by[lab], key=lambda r: seeded("hm:%s" % r["id"]).random())[:n]
    return sorted(out, key=lambda r: str(r["id"]))


def sample(r: Dict, rel: str) -> Dict:
    hid = "%05d" % int(r["id"])
    return choice_sample("board-hateful:" + hid, "board-hateful", "hm:" + hid, rel, QUESTION, ["Yes", "No"],
                         "Yes" if r["label"] == 1 else "No", kind="hateful" if r["label"] == 1 else "not_hateful")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    t0 = time.time()
    root = a.root
    board = BoardIndex.load(os.path.join(root, "v05", "board_exclusion.json"))
    loose = BoardIndex.load(os.path.join(root, "v05", "board_exclusion.json"), dhash_bits=LOOSE_D, phash_bits=LOOSE_P)
    if not any(i.startswith("hm:") for i in board.ids):
        sys.exit("board_exclusion.json has no Hateful Memes ids: rebuild it after fetching data/raw/board/hateful")
    held = set()
    for split in ("dev_seen", "dev_unseen", "test_seen", "test_unseen"):
        held |= {"hm:%05d" % int(json.loads(l)["id"]) for l in open(os.path.join(root, RAW, split + ".jsonl"))}
    train = [json.loads(l) for l in open(os.path.join(root, RAW, "train.jsonl")) if l.strip()]
    drops: Counter = Counter()
    ok = []
    for r in train:
        hid = "hm:%05d" % int(r["id"])
        if hid in held or board.hit_id(hid):
            drops["dev_test_id"] += 1
        elif not os.path.exists(os.path.join(root, RAW, r["img"])):
            drops["image_missing"] += 1
        else:
            ok.append(r)
    with ProcessPoolExecutor(a.workers) as ex:
        hs = list(ex.map(hash_file, [os.path.join(root, RAW, r["img"]) for r in ok], chunksize=32))
    hits, kept = [], []
    for r, h in zip(ok, hs):
        if "error" in h:
            drops["unreadable_image"] += 1
            continue
        got = board.hit_hashes(h) or loose.hit_hashes(h)
        if got:
            kind = got[0] if board.hit_hashes(h) else "loose_near"
            hits.append({"id": r["id"], "kind": kind, "board": got[1]})
            drops["board_image_" + kind] += 1
            continue
        kept.append(r)
    labels_before = Counter(r["label"] for r in kept)
    bal = balance(kept)
    drops["label_balance"] = len(kept) - len(bal)
    from PIL import Image

    def rows():
        for r in bal:
            rel = os.path.join(OUT, "images", "%05d.jpg" % int(r["id"]))
            if not os.path.exists(os.path.join(root, rel)):
                im = Image.open(os.path.join(root, RAW, r["img"])).convert("RGB")
                s = SAVE_MAX / max(im.size)
                if s < 1:
                    im = im.resize((round(im.size[0] * s), round(im.size[1] * s)), Image.Resampling.LANCZOS)
                os.makedirs(os.path.dirname(os.path.join(root, rel)), exist_ok=True)
                im.save(os.path.join(root, rel), quality=92)
            yield sample(r, rel)

    out = os.path.join(root, OUT)
    counts = write_split(out, rows(), dev_frac=0.05)
    manifest(out, {"family": "hateful", "board": "Moderation (Hateful Memes)",
                   "source": "neuralcatcher/hateful_memes@d201c488 train.jsonl + img/ (research licence, see LICENSE.txt)",
                   "train_rows": len(train), "labels_before_balance": {str(k): v for k, v in labels_before.items()},
                   "drops": dict(drops), "board_exclusion_hits": hits, "loose_near_rule": [LOOSE_D, LOOSE_P],
                   "split_counts": counts, "seconds": round(time.time() - t0)})
    print("hateful: %s drops %s, %.0f s" % (counts, dict(drops), time.time() - t0))


if __name__ == "__main__":
    main()
