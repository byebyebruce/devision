"""Scientific-figure multiple choice (like the board's CharXiv reasoning questions) from ArxivQA -> data/v05/board/figure/.

    tar -xzf data/raw/board/arxivqa/images.tgz -C data/raw/board/arxivqa     # once, after v05_board_fetch.py --only arxivqa
    uv run --no-sync python scripts/data/v05_board_arxivqa.py --root data

Source: MMInstruction/ArxivQA (arxivqa.jsonl + images.tgz; 100k GPT-4V-written multiple-choice questions on arXiv
paper figures, CC BY-SA 4.0 per its card). Options lose their "A." / "B)" prefixes; the label (a letter, "C) Four" or
the option text) is resolved to one option, unresolvable -> "bad_label". Kept: 2-6 distinct options, <= MAX_OPT chars
each. The gold is moved to a rotating slot (ArxivQA's own letters are skewed); the other options keep their order.
Exclusion: figures from any arXiv paper that CharXiv (val or test) drew a figure from ("arxiv:<id>" source ids in
data/v05/board_exclusion.json) and every image hit by the board index (CharXiv val board images and CharXiv test
images are both in it). Text-only cue: when the gold is the strictly longest option in more than LONGEST_MAX of the
kept questions, the excess (lowest seeded hash first) is dropped so "pick the longest" stays near chance.
dev = ~3% of papers (all figures of a paper on one side).
"""
import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from v05_board_common import BoardIndex, choice_sample, image_hashes, manifest, seeded, write_split  # noqa: E402

OUT = os.path.join("v05", "board", "figure")
RAW = os.path.join("raw", "board", "arxivqa")
MAX_OPT = 200
LONGEST_MAX = 0.35
PREFIX = re.compile(r"^\s*\(?([A-Ja-j])\s*[\.\):]\s+")


def strip_option(o: str) -> Tuple[Optional[str], str]:
    m = PREFIX.match(o)
    return (m.group(1).upper(), o[m.end():].strip()) if m else (None, o.strip())


def parse(r: Dict) -> Tuple[Optional[Tuple[List[str], int]], str]:
    """((options, gold index), "") or (None, drop reason)."""
    raw = r.get("options") or []
    if not 2 <= len(raw) <= 6:
        return None, "option_count"
    parsed = [strip_option(str(o)) for o in raw]
    opts = [t for _, t in parsed]
    if any(not t or len(t) > MAX_OPT for t in opts) or len({t.casefold() for t in opts}) != len(opts):
        return None, "bad_options"
    letters = [l for l, _ in parsed]
    lab = str(r.get("label", "")).strip()
    idx = None
    lm = PREFIX.match(lab + " ") if len(lab) <= 3 else PREFIX.match(lab)
    if re.fullmatch(r"\(?[A-Ja-j]\)?\.?", lab):
        idx = ord(lab.strip("().").upper()) - 65
    elif lm:
        idx = ord(lm.group(1).upper()) - 65
        rest = lab[lm.end():].strip()
        if rest and idx < len(opts) and opts[idx].casefold() != rest.casefold():
            hits = [k for k, t in enumerate(opts) if t.casefold() == rest.casefold()]
            idx = hits[0] if len(hits) == 1 else None
    else:
        hits = [k for k, t in enumerate(opts) if t.casefold() == lab.casefold()]
        idx = hits[0] if len(hits) == 1 else None
    if idx is not None and any(letters) and letters[idx if idx < len(letters) else 0] not in (None, chr(65 + idx)):
        idx = None   # options lettered out of order: the letter label is ambiguous
    if idx is None or not 0 <= idx < len(opts):
        return None, "bad_label"
    return (opts, idx), ""


def paper(image: str) -> str:
    return os.path.basename(image).rsplit("_", 1)[0]


def hash_path(p: str) -> Dict:
    try:
        with open(p, "rb") as f:
            return image_hashes(f.read())
    except Exception as e:
        return {"error": str(e)}


def strictly_longest(opts: List[str], i: int) -> bool:
    return all(len(opts[i]) > len(o) for k, o in enumerate(opts) if k != i)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    t0 = time.time()
    root = a.root
    board = BoardIndex.load(os.path.join(root, "v05", "board_exclusion.json"))
    if not any(i.startswith("arxiv:") for i in board.ids):
        sys.exit("board_exclusion.json has no CharXiv paper ids: fetch charxiv_test and rebuild it")
    rows = [json.loads(l) for l in open(os.path.join(root, RAW, "arxivqa.jsonl")) if l.strip()]
    drops: Counter = Counter()
    ok = []
    for r in rows:
        got, why = parse(r)
        if not got:
            drops[why] += 1
        elif board.hit_id("arxiv:" + paper(r["image"])):
            drops["charxiv_paper"] += 1
        elif not os.path.exists(os.path.join(root, RAW, r["image"])):
            drops["image_missing"] += 1
        else:
            ok.append((r, got))
    rels = sorted({r["image"] for r, _ in ok})
    with ProcessPoolExecutor(a.workers) as ex:
        hs = dict(zip(rels, ex.map(hash_path, [os.path.join(root, RAW, p) for p in rels], chunksize=64)))
    hits = {}
    for p, h in hs.items():
        if "error" in h:
            hits[p] = ("unreadable", "")
        else:
            got = board.hit_hashes(h)
            if got:
                hits[p] = got
    kept = []
    for r, got in ok:
        if r["image"] in hits:
            drops["unreadable_image" if hits[r["image"]][0] == "unreadable" else "board_image_" + hits[r["image"]][0]] += 1
        else:
            kept.append((r, got))
    # rotate the gold; then cap "gold is the strictly longest option"
    slots: Counter = Counter()
    placed = []
    for r, (opts, idx) in kept:
        gold = opts[idx]
        others = [o for k, o in enumerate(opts) if k != idx]
        n = len(opts)
        rng = seeded("arxivqa:" + r["id"])
        slot = min(range(n), key=lambda s: (slots[(n, s)], rng.random()))
        slots[(n, slot)] += 1
        new = others[:slot] + [gold] + others[slot:]
        placed.append((r, new, gold))
    longest = [p for p in placed if strictly_longest(p[1], p[1].index(p[2]))]
    cap = int(LONGEST_MAX * len(placed))
    over = set()
    if len(longest) > cap:
        # dropping d of the longest-gold rows out of N leaves (L - d) / (N - d) <= LONGEST_MAX
        d = int((len(longest) - LONGEST_MAX * len(placed)) / (1 - LONGEST_MAX)) + 1
        over = {p[0]["id"] for p in sorted(longest, key=lambda p: seeded("long:" + p[0]["id"]).random())[:d]}
    drops["longest_cue_cap"] = len(over)
    out_rows = []
    for r, opts, gold in placed:
        if r["id"] in over:
            continue
        out_rows.append(choice_sample("board-figure:" + r["id"], "board-arxivqa", "arxiv:" + paper(r["image"]),
                                      os.path.join(RAW, r["image"]), r["question"].strip(), opts, gold,
                                      kind=r["id"].split("-")[0]))
    out = os.path.join(root, OUT)
    counts = write_split(out, out_rows, dev_frac=0.03)
    manifest(out, {"family": "figure", "board": "CharXiv (reasoning), also R-Bench-M / MMMU-Pro style reading",
                   "source": "MMInstruction/ArxivQA@85a6dca0 arxivqa.jsonl + images.tgz (CC BY-SA 4.0)",
                   "rows": len(rows), "drops": dict(drops),
                   "board_exclusion_hits": [{"image": p, "kind": v[0], "board": v[1]} for p, v in hits.items()
                                            if v[0] != "unreadable"],
                   "split_counts": counts, "seconds": round(time.time() - t0)})
    print("figure: %s drops %s, %.0f s" % (counts, dict(drops), time.time() - t0))


if __name__ == "__main__":
    main()
