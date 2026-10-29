"""Infographic / document reading 4-way choice questions (like the board's InfographicVQA) from TRAIN splits
-> data/v05/board/infographic/.

    uv run --no-sync python scripts/data/v05_board_infovqa.py --root data [--subsets infovqa_full docvqa]

Sources (fetched by v05_board_fetch.py into data/raw/board/<name>/):
  infovqa_full  lmms-lab/DocVQA InfographicVQA/train-* (4.4k train images, all accepted answers; the board uses
                InfographicVQA validation, data_split is asserted to be "train");
  docvqa        The Cauldron's DocVQA train (scanned documents; no board benchmark uses DocVQA);
  infographic_vqa  The Cauldron's 2.1k-image subset of InfographicVQA train (do not combine with infovqa_full).
One question per (question, answer): the cauldron's answer-style suffix ("\nKeep it brief.") is removed, answers are
lower-cased and lose a trailing period (the board's options are lower-case too). Options = the gold + 3 answers to
OTHER questions about the SAME image of the same kind (has a digit or not, has % or not), never equal to / a sub- or
super-string of any accepted answer; a numeric gold with exactly 2 same-image numbers gets ONE near number in the
gold's format (+-1..3, x0.5 / x1.5 / x2; more fillers would make the gold the centre the others are derived from).
Fewer than 3 distractors -> "few_same_image_answers". Gold letter rotated evenly. Every image is checked against
data/v05/board_exclusion.json (bytes, pixels, dHash + pHash; InfographicVQA image names as ids). dev = ~5% of images.
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
from typing import Dict, Iterator, List, Tuple

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


def image_questions(key: str, pairs: List[Tuple], slot_counts: Counter, stats: Counter) -> List[Tuple]:
    """[(n, question, options, gold)] for one image's (question, answer[, other accepted answers]) pairs."""
    qa = [(clean_q(p[0]), clean_a(p[1])) for p in pairs]
    accepted = [{norm(clean_a(x)) for x in (p[2] if len(p) > 2 else [])} | {norm(qa[n][1])} for n, p in enumerate(pairs)]
    out = []
    seen_q = Counter(q.casefold() for q, _ in qa)
    for n, (q, gold) in enumerate(qa):
        if not q or not gold or len(gold) > 80 or seen_q[q.casefold()] > 1:
            stats["bad_or_repeated_question"] += 1
            continue
        golds = {g for g in accepted[n] if g}
        pool = []
        for m, (_, a) in enumerate(qa):
            na = norm(a)
            if m == n or not na or has_digit(a) != has_digit(gold) or len(a) > 80 or ("%" in a) != ("%" in gold) \
                    or any(na == g or na in g or g in na for g in golds):
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


def cauldron(root: str, sub: str) -> Iterator[Tuple[str, bytes, List[Tuple], str]]:
    import pyarrow.parquet as pq
    shards = sorted(glob.glob(os.path.join(root, "raw", "board", "*", sub, "*.parquet")))
    if not shards:
        sys.exit("no shards for %s under data/raw/board/*/%s (run v05_board_fetch.py)" % (sub, sub))
    for sh in shards:
        pf = pq.ParquetFile(sh)
        for b in range(pf.num_row_groups):
            for i, r in enumerate(pf.read_row_group(b).to_pylist()):
                key = "%s:%s:%d:%d" % (sub, os.path.basename(sh).split("-of-")[0], b, i)
                if len(r["images"]) == 1:
                    yield key, r["images"][0]["bytes"], [(t["user"], t["assistant"]) for t in r["texts"]], \
                        "cauldron-" + key


def lmms_infovqa(root: str, sub: str) -> Iterator[Tuple[str, bytes, List[Tuple], str]]:
    """lmms-lab/DocVQA InfographicVQA train: one row per question, grouped by image_url (all accepted answers kept)."""
    import pyarrow.parquet as pq
    shards = sorted(glob.glob(os.path.join(root, "raw", "board", "infovqa_full", "InfographicVQA", "train-*.parquet")))
    if not shards:
        sys.exit("no InfographicVQA train shards (run v05_board_fetch.py --only infovqa_full)")
    for sh in shards:
        meta = pq.read_table(sh, columns=["questionId", "question", "answers", "image_url", "data_split"]).to_pylist()
        groups: Dict[str, List[int]] = {}
        for i, m in enumerate(meta):
            assert m["data_split"] in (None, "train"), (sh, m["data_split"])
            groups.setdefault(m["image_url"], []).append(i)
        imgs = pq.read_table(sh, columns=["image"]).column("image")
        for url, idx in groups.items():
            name = os.path.splitext(os.path.basename(url))[0][:80]
            pairs = [(meta[i]["question"], meta[i]["answers"][0], meta[i]["answers"][1:]) for i in idx if meta[i]["answers"]]
            yield "infovqa:%s" % meta[idx[0]]["questionId"], imgs[idx[0]].as_py()["bytes"], pairs, "infovqa:" + name


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
    ap.add_argument("--subsets", nargs="*", default=["infovqa_full", "docvqa"],
                    help="infovqa_full (lmms-lab InfographicVQA train, all 4.4k images) or cauldron subsets "
                         "(infographic_vqa = a 2.1k-image subset of the same train split: do not combine the two)")
    a = ap.parse_args(argv)
    t0 = time.time()
    root = a.root
    board = BoardIndex.load(os.path.join(root, "v05", "board_exclusion.json"))
    drops: Counter = Counter()
    stats: Counter = Counter()
    hits: List[Dict] = []
    slot_counts: Counter = Counter()
    rows: List[Dict] = []
    images: Counter = Counter()
    for sub in a.subsets:
        reader = lmms_infovqa if sub == "infovqa_full" else cauldron
        for key, data, pairs, image_id in reader(root, sub):
            got = board.hit(data)
            if got:
                hits.append({"row": key, "kind": got[0], "board": got[1]})
                drops["board_image_" + got[0]] += 1
                continue
            if board.hit_id(image_id):
                drops["board_source_id"] += 1
                continue
            qs = image_questions(key, pairs, slot_counts, stats)
            if not qs:
                drops["no_question"] += 1
                continue
            rel = os.path.join(OUT, "images", sub, re.sub(r"[^A-Za-z0-9_.-]", "_", key) + ".jpg")
            if not os.path.exists(os.path.join(root, rel)):
                save(data, os.path.join(root, rel))
            images[sub] += 1
            for n, q, opts, gold in qs:
                rows.append(choice_sample("board-info:%s:%d" % (key, n), "board-" + sub.replace("_", "-"), image_id,
                                          rel, q, opts, gold, kind="numeric" if has_digit(gold) else "text"))
    out = os.path.join(root, OUT)
    counts = write_split(out, rows, dev_frac=0.05)
    manifest(out, {"family": "infographic", "board": "InfographicVQA (also CharXiv-style reading)",
                   "source": {"infovqa_full": "lmms-lab/DocVQA@539088ef InfographicVQA/train-*",
                              "cauldron": "HuggingFaceM4/the_cauldron@847a98a7 (docvqa / infographic_vqa train)"},
                   "subsets": a.subsets,
                   "images_used": dict(images), "drops": dict(drops), "question_stats": dict(stats),
                   "board_exclusion_hits": hits, "split_counts": counts, "seconds": round(time.time() - t0)})
    print("infographic: %s images %s drops %s stats %s, %.0f s" % (counts, dict(images), dict(drops), dict(stats),
                                                                     time.time() - t0))


if __name__ == "__main__":
    main()
