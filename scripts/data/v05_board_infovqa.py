"""Infographic / document reading 4-way choice questions (like the board's InfographicVQA) from TRAIN splits
-> data/v05/board/infographic/.

    uv run --no-sync python scripts/data/v05_board_infovqa.py --root data [--subsets infographic_vqa docvqa]

Source: The Cauldron subsets (HuggingFaceM4/the_cauldron, fetched by v05_board_fetch.py into data/raw/board/<name>/;
`infographic_vqa` = InfographicVQA train, the board uses InfographicVQA validation). Each row is one image with
1-10 (question, short answer) pairs. One question per pair: the cauldron's answer-style suffix ("\\nKeep it brief.")
is removed, answers are lower-cased and lose a trailing period (the board's options are lower-case too).
Options = the gold + 3 answers to OTHER questions about the SAME image of the same kind (has a digit or not, has % or not), never a
sub / super-string of the gold; a numeric gold with exactly 2 same-image numbers gets ONE near number in the gold's format
(+-1..3, x0.5 / x1.5 / x2; more fillers would make the gold the centre the others are derived from). Fewer than 3 distractors -> drop "few_same_image_answers". Gold letter rotated evenly.
Every image is checked against data/v05/board_exclusion.json (no source ids in the cauldron). dev = ~5% of images.
"""
import argparse
import glob
import io
import json
import os
import re
import sys
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from v05_board_common import BoardIndex, choice_sample, manifest, seeded, write_split  # noqa: E402

OUT = os.path.join("v05", "board", "infographic")
SAVE_MAX = 1600
SUFFIX = re.compile(r"\n(Keep it brief|Answer briefly|Quick response, please|Short answer required|Concise answer only|"
                    r"Be succinct|Write very briefly|Write a very short answer|Answer with a single word or phrase|A short response will do|"
                    r"Give a very brief answer|Answer in a word or phrase|Respond with few words|Be brief|"
                    r"Short answer only|Provide a short answer|Answer concisely)[^\n]*$", re.I)
NUM = re.compile(r"^-?[\d,]*\.?\d+%?$")


def clean_q(q: str) -> str:
    q = SUFFIX.sub("", q.strip()).strip()
    return q.split("\n")[0].strip() if "\n" in q else q


def clean_a(a: str) -> str:
    a = a.strip()
    if a.endswith("."):
        a = a[:-1]
    return a.strip().lower()


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9%.]", "", s.lower())


def has_digit(s: str) -> bool:
    return bool(re.search(r"\d", s))


def near_numbers(gold: str, rng) -> List[str]:
    if not NUM.match(gold):
        return []
    pct = gold.endswith("%")
    core = gold.rstrip("%").replace(",", "")
    dec = len(core.split(".")[1]) if "." in core else 0
    g = float(core)
    cands = [g + d for d in (1, -1, 2, -2, 3)] if dec == 0 and abs(g) < 30 else \
        [g * m for m in (0.5, 1.5, 2, 0.75, 1.25)]
    out = []
    for v in cands:
        if v < 0 <= g:
            continue
        s = ("{:,.%df}" % dec).format(v) if "," in gold else ("{:.%df}" % dec).format(v)
        s += "%" if pct else ""
        if norm(s) != norm(gold) and s not in out:
            out.append(s)
    rng.shuffle(out)
    return out


def image_questions(key: str, pairs: List[Tuple[str, str]], slot_counts: Counter, stats: Counter) -> List[Tuple]:
    """[(n, question, options, gold)] for one image's (question, answer) pairs."""
    qa = [(clean_q(q), clean_a(a)) for q, a in pairs]
    out = []
    seen_q = Counter(q.casefold() for q, _ in qa)
    for n, (q, gold) in enumerate(qa):
        if not q or not gold or len(gold) > 80 or seen_q[q.casefold()] > 1:
            stats["bad_or_repeated_question"] += 1
            continue
        g = norm(gold)
        pool = []
        for m, (_, a) in enumerate(qa):
            na = norm(a)
            if m == n or not na or has_digit(a) != has_digit(gold) or na == g or na in g or g in na or len(a) > 80 \
                    or ("%" in a) != ("%" in gold):
                continue
            if a not in pool:
                pool.append(a)
        rng = seeded("%s:%d" % (key, n))
        rng.shuffle(pool)
        ds = pool[:3]
        if len(ds) == 2 and has_digit(gold):   # at most ONE filler: fillers derived from the gold make it the "centre"
            for s in near_numbers(gold, rng):
                if len(ds) == 3:
                    break
                if all(norm(s) != norm(d) for d in ds):
                    ds.append(s)
                    stats["numeric_filler"] += 1
        if len(ds) < 3:
            stats["few_same_image_answers"] += 1
            continue
        slot = min(range(4), key=lambda s: (slot_counts[s], rng.random()))
        slot_counts[slot] += 1
        opts = ds[:slot] + [gold] + ds[slot:]
        if len({o.casefold() for o in opts}) != 4:
            stats["duplicate_options"] += 1
            continue
        out.append((n, q, opts, gold))
    return out


def save(data: bytes, path: str) -> None:
    from PIL import Image
    im = Image.open(io.BytesIO(data)).convert("RGB")
    s = SAVE_MAX / max(im.size)
    if s < 1:
        im = im.resize((round(im.size[0] * s), round(im.size[1] * s)), Image.Resampling.LANCZOS)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    im.save(path, quality=92)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data")
    ap.add_argument("--subsets", nargs="*", default=["infographic_vqa"])
    a = ap.parse_args(argv)
    t0 = time.time()
    import pyarrow.parquet as pq
    root = a.root
    board = BoardIndex.load(os.path.join(root, "v05", "board_exclusion.json"))
    drops: Counter = Counter()
    stats: Counter = Counter()
    hits: List[Dict] = []
    slot_counts: Counter = Counter()
    rows: List[Dict] = []
    images: Counter = Counter()
    for sub in a.subsets:
        shards = sorted(glob.glob(os.path.join(root, "raw", "board", "*", sub, "*.parquet")))
        if not shards:
            sys.exit("no shards for %s under data/raw/board/*/%s (run v05_board_fetch.py)" % (sub, sub))
        for sh in shards:
            pf = pq.ParquetFile(sh)
            for b in range(pf.num_row_groups):
                tb = pf.read_row_group(b)
                for i, r in enumerate(tb.to_pylist()):
                    key = "%s:%s:%d:%d" % (sub, os.path.basename(sh).split("-of-")[0], b, i)
                    if len(r["images"]) != 1:
                        drops["not_one_image"] += 1
                        continue
                    data = r["images"][0]["bytes"]
                    got = board.hit(data)
                    if got:
                        hits.append({"row": key, "kind": got[0], "board": got[1]})
                        drops["board_image_" + got[0]] += 1
                        continue
                    qs = image_questions(key, [(t["user"], t["assistant"]) for t in r["texts"]], slot_counts, stats)
                    if not qs:
                        drops["no_question"] += 1
                        continue
                    rel = os.path.join(OUT, "images", sub, re.sub(r"[^A-Za-z0-9_.-]", "_", key) + ".jpg")
                    if not os.path.exists(os.path.join(root, rel)):
                        save(data, os.path.join(root, rel))
                    images[sub] += 1
                    for n, q, opts, gold in qs:
                        rows.append(choice_sample("board-info:%s:%d" % (key, n), "board-" + sub.replace("_", "-"),
                                                  "cauldron-" + key, rel, q, opts, gold,
                                                  kind="numeric" if has_digit(gold) else "text"))
    out = os.path.join(root, OUT)
    counts = write_split(out, rows, dev_frac=0.05)
    manifest(out, {"family": "infographic", "board": "InfographicVQA (also CharXiv-style reading)",
                   "source": "HuggingFaceM4/the_cauldron@847a98a7 subsets %s (train)" % a.subsets,
                   "images_used": dict(images), "drops": dict(drops), "question_stats": dict(stats),
                   "board_exclusion_hits": hits, "split_counts": counts, "seconds": round(time.time() - t0)})
    print("infographic: %s images %s drops %s stats %s, %.0f s" % (counts, dict(images), dict(drops), dict(stats),
                                                                     time.time() - t0))


if __name__ == "__main__":
    main()
