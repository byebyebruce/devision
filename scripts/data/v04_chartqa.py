"""The v0.4 frozen chart-reading confirmation subset from ChartQA (official test, augmented questions).

    uv run python scripts/data/v04_chartqa.py --root data

Source (fixed): github.com/vis-nlp/ChartQA at commit 044eabfc306abfe9340c5741f0093aefc5973d06, directory
"ChartQA Dataset/test": test_augmented.json (1,250 questions), png/<imgname>, tables/<imgname stem>.csv, plus the
repo LICENSE (GPL-3.0) and README. Only these small files are downloaded (raw URLs at that commit), byte for byte,
to <root>/raw/chartqa/044eabfc/, and their sha256 go into the manifest; no archive, no model, no model output.

Selection (nothing is picked by a model output; the query and the official label are kept unchanged, no hint is
added, the CSV is never part of the model input):

  1. Candidates: every question of test_augmented.json. Eligible are direct single-value reading questions only:
     a. text   the query is not a sum / difference / ratio / average / max-min / argmax / count / comparison /
               yes-no / change question and does not ask for a time or a name (REJECT patterns below), and the
               official label is a plain number (optionally with % or a currency sign);
     b. csv    the label equals exactly one value cell of the chart's CSV (not the first, label column). Equality is
               numeric after removing thousands commas, %, currency signs, '*' and spaces ("45" == "45.0",
               "45%" == "45"); nothing else (no rounding, no unit scaling);
     c. cell   the query names that cell: exactly one row label of the CSV occurs in the query as a whole word
               (labels contained in a longer matched label are ignored) and it is the cell's row; when the CSV has
               more than one value column, the words of exactly one column header occur in the query and it is
               the cell's column. Otherwise the question is ambiguous and excluded.
  2. Walk: candidates ordered by SHA256(imgname + query) ascending (UTF-8, no separator); at most one question per
     picture; stop at TARGET (80) kept questions. All checks run inside the walk and every exclusion is logged with
     its reason (EXCLUSIONS.json); questions after the 80th kept one are counted as not reached. Nothing refills.
  3. Hand checks (MANUAL below): kept items whose PNG was opened and checked by a person -- value visible on the
     chart, series / unit agree, gold matches; any item found wrong or unverifiable is listed in MANUAL_EXCLUDE and
     dropped with its reason (the walk then continues to the next candidate; this is the same walk, not another
     criterion). Every kept item records the original picture size, the scale of the 256x256 letterbox
     (devision.image.letterbox, saved to images256/) and est_text_px_256 = 12 * scale, the height a ~12 px chart
     label (typical for these Statista-style charts) has at 256; readability is recorded, never used to drop.
  4. Options: the official value plus 3 distractors from the same CSV: other rows of the same column first (in a
     fixed random order seeded by the question id), then cells of the other value columns. A value is skipped when
     it is not a number or when, formatted like the label, it equals the gold or an earlier option. Fewer than 3 ->
     excluded ("distractors"). All four are formatted like the official label: the label's number of decimals, its
     % or currency sign; the gold option is the official label string itself. Order: random.Random(question id)
     shuffle. The gold maps to exactly one option (asserted).
  5. Isolation of the picture against every picture we have used: (exact / near) 64-bit dHash and pHash of the
     grayscale picture and its mirror, excluded when one reference is within THRESHOLD (6) bits by both
     (v04_confirm.nearest / is_near), against every picture referenced by any JSONL under the data root except
     data/raw and the output directories of v04_confirm and this set (the v04_confirm scan, 751,987 pictures);
     (recolor) the same on the saturated-pixel mask, against the data pictures whose path suggests charts / plots
     (CHART_PATH); (used_name) the ChartQA picture name ("two_col_123" / "multi_col_123") occurs in any of those JSONL
     files, or a referenced picture path contains "chartqa".
  6. Outputs in <root>/v04_chartqa/: images/<imgname> (original PNG), images256/<imgname> (letterbox preview),
     chartqa.jsonl (id "v04-chartqa:<imgname>:<k>", k = index of the question among that picture's questions in
     test_augmented.json; source "chartqa-test", kind "chart-reading", image_id "chartqa:<imgname>"; extra
     official_label, csv_cell, readability), chartqa_mism.jsonl (each row's picture replaced through a fixed seed-0
     derangement over the kept pictures, v04_confirm.derangement; image_id stays the ORIGINAL picture id,
     "mismatched_from" = that original id, "shown_image_id" = the picture actually shown), MANIFEST.json,
     EXCLUSIONS.json.
"""
import argparse
import csv
import datetime
import hashlib
import io
import json
import os
import random
import re
import urllib.parse
import urllib.request
from collections import Counter
from multiprocessing import Pool
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from v04_confirm import (THRESHOLD, data_jsonls, derangement, hash_file, is_near, mask_file, mask_hashes, nearest,
                         perceptual, scan_data, sha256, write_jsonl)

COMMIT = "044eabfc306abfe9340c5741f0093aefc5973d06"
RAW_BASE = "https://raw.githubusercontent.com/vis-nlp/ChartQA/%s/" % COMMIT
TEST_DIR = "ChartQA Dataset/test/"
RAW = os.path.join("raw", "chartqa", COMMIT[:8])
OUT = "v04_chartqa"
TARGET = 80
CHART_PATH = re.compile(r"chart|plot|cauldron|dvqa|figureqa|infographic|table", re.I)
CHARTQA_NAME = re.compile(r"\b((?:two|multi)_col_\d+)\b")

REJECT = [
    ("sum", r"\b(total|sum|combined|together|altogether|in all|add up)\b"),
    ("difference", r"\b(difference|differ|gap|more than|less than|fewer than|higher than|lower than|"
                   r"greater than|smaller than|larger than|exceed|increase|decrease|change|grew|declin|rose|fell|drop)"),
    ("ratio", r"\b(ratio|times|divided|proportion of .* to)\b"),
    ("average", r"\b(average|mean|median)\b"),
    ("max_min", r"\b(highest|lowest|maximum|minimum|max|min|most|least|largest|smallest|biggest|top|peak|"
                r"best|worst|record|first|last|second|third|rank)"),
    ("comparison", r"\b(compare|compared|versus|vs|than|between|both|either|same)\b"),
    ("count", r"\bhow many (bars|colou?rs|categories|years|countries|values|data|segments|lines|sections|items)\b"),
    ("yes_no", r"^(is|are|was|were|does|do|did|has|have|had|can|could|will|would|should)\b"),
    ("label_answer", r"\b(what year|which year|in what year|what month|which month|when|which|who|whom|where|"
                     r"what country|what company|what region|what type|what kind|what brand|name)\b"),
]
REJECT_RE = [(k, re.compile(p, re.I)) for k, p in REJECT]
NUM = re.compile(r"^-?\d+(?:\.\d+)?$")
STOP = {"the", "of", "in", "and", "a", "an", "to", "for", "on", "at", "by", "per", "with", "from", "as", "or", "s",
        "value", "share", "number"}

# Hand checks of kept items (the PNG opened and read by a person): id -> value printed on the chart, verdict, note.
MANUAL: Dict[str, dict] = {
    "v04-chartqa:multi_col_101283.png:0": {"value_printed": False, "gold_matches_chart": True, "verdict": "keep",
        "note": 'no values printed; Industrial sector 2020 bar ~900 on a 25,000 axis, consistent with 944; distractors 1513-1914 are visibly taller bars'},
    "v04-chartqa:multi_col_40409.png:0": {"value_printed": False, "gold_matches_chart": True, "verdict": "keep",
        "note": 'stacked bars, no values printed; 2020 Air segment ~47.8 to ~50.3, consistent with 2.4 -- a segment thickness read from its upper and lower edges, not a printed figure (critic V04-CHART-R1-01)'},
    "v04-chartqa:multi_col_80175.png:0": {"value_printed": False, "gold_matches_chart": True, "verdict": "keep",
        "note": "grouped bars, no values printed; Asia Pacific 2020 (green) ~2,100, consistent with 2111; distractors are the other regions' 2020 bars"},
    "v04-chartqa:multi_col_60912.png:1": {"value_printed": True, "gold_matches_chart": True, "verdict": "keep",
        "note": '5.9 printed at Female 2019'},
    "v04-chartqa:multi_col_80452.png:1": {"value_printed": True, "gold_matches_chart": True, "verdict": "keep",
        "note": '21.43 printed at 2019; only the Female labels are printed, the Male line nearly overlaps'},
    "v04-chartqa:multi_col_80988.png:0": {"value_printed": True, "gold_matches_chart": True, "verdict": "keep",
        "note": '39.8 printed on the Foodservice segment of 2019'},
    "v04-chartqa:two_col_101066.png:0": {"value_printed": True, "gold_matches_chart": True, "verdict": "keep",
        "note": '1 038 000 printed at 2020'},
    "v04-chartqa:two_col_102480.png:0": {"value_printed": True, "gold_matches_chart": True, "verdict": "keep",
        "note": '5.8 printed at March 22, 2020'},
    "v04-chartqa:two_col_1993.png:0": {"value_printed": True, "gold_matches_chart": True, "verdict": "keep",
        "note": '44.7 printed at Costa Rica'},
    "v04-chartqa:two_col_2074.png:1": {"value_printed": True, "gold_matches_chart": True, "verdict": "keep",
        "note": '656 920 printed at Helsinki'},
    "v04-chartqa:two_col_41715.png:0": {"value_printed": True, "gold_matches_chart": True, "verdict": "keep",
        "note": '659 printed at Brazil'},
    "v04-chartqa:two_col_4513.png:0": {"value_printed": True, "gold_matches_chart": True, "verdict": "keep",
        "note": 'printed as 54% at Americas; official label 54 kept unchanged'},
    "v04-chartqa:two_col_20673.png:0": {"value_printed": True, "gold_matches_chart": True, "verdict": "keep",
        "note": '101.88 printed at 2016****'},
    "v04-chartqa:two_col_24074.png:1": {"value_printed": True, "gold_matches_chart": True, "verdict": "keep",
        "note": 'printed as 21.1% at HTML; official label 21.1 kept unchanged'},
    "v04-chartqa:two_col_41612.png:0": {"value_printed": True, "gold_matches_chart": True, "verdict": "keep",
        "note": '85 printed at Fall*'},
    "v04-chartqa:two_col_82323.png:0": {"value_printed": True, "gold_matches_chart": True, "verdict": "keep",
        "note": '212 printed at New Brunswick'},
    "v04-chartqa:two_col_4455.png:0": {"value_printed": False, "gold_matches_chart": True, "verdict": "keep",
        "note": "2020 point unlabelled (~1.6, consistent); the nearby printed 1.44% belongs to 2019; distractors rounded to the label's 1 decimal (33.17 -> 33.2)"},
    "v04-chartqa:two_col_63677.png:0": {"value_printed": True, "gold_matches_chart": True, "verdict": "keep",
        "note": '316 printed at Thailand'},
    "v04-chartqa:two_col_2898.png:2": {"value_printed": False, "gold_matches_chart": True, "verdict": "keep",
        "note": 'no values printed; 2100 bar ~10.9 on a 12.5 axis, consistent with 10.87; nearest distractor 9.96 is the visibly lower 2055 bar'},
    "v04-chartqa:two_col_20322.png:1": {"value_printed": True, "gold_matches_chart": True, "verdict": "keep",
        "note": '375 967.8 printed at 2011'},
    "v04-chartqa:two_col_62268.png:0": {"value_printed": True, "gold_matches_chart": True, "verdict": "keep",
        "note": '52 printed at 1999/00'},
}
# The remaining 59 kept pictures, looked at by the critic (27) and the Spec reviewer (32) on 2026-10-17 13:49
# (critic/critic.md, "ChartQA输入内容复核通过", V04-CHART-R1-01): stem -> who. Each target value is printed on the
# chart except the stems in REVIEW_NOT_PRINTED (read from bar / point position); question, axis unit and gold agree.
REVIEW_CHECKS: Dict[str, str] = {
    **{"two_col_%s" % n: "critic" for n in (
        "4072 41019 41344 41555 42220 4260 42998 43586 4651 47 5289 5541 60100 61425 62073 62141 62315 63561 80053 "
        "81813 81873 81950 82084 82215 82741 83153 83537").split()},
    **{"two_col_%s" % n: "spec-review" for n in (
        "82499 41604 3220 43770 81794 102195 102327 102453 102598 103550 103789 104396 1192 1260 1391 20064 20485 "
        "20539 20669 20883 20905 21004 21062 21855 22593 23099 23555 23724 2392 3253 40073 40350").split()},
}
REVIEW_NOT_PRINTED: Dict[str, str] = {
    "two_col_20669": "2019 fresh-fruit bar ~880 with no figure on top; distinguishable from 844.73 and the others",
    "two_col_22593": "2019 point ~23, unlabelled; the nearby 23.13 belongs to 2018, other options <= 14.31",
    "two_col_41555": "2019 bar ~264,000, clearly above the nearest option 241174",
    "two_col_82215": "2019 bar ~480, clearly above the nearest option 414.94",
    "two_col_83537": "2019 point ~77,500, unlabelled; 75505 belongs to 2018, other options <= 53715",
}


def value_printed(qid: str, img: str) -> Optional[bool]:
    """From the hand checks: the engineer's 21 (MANUAL, by question id) and the reviewers' 59 (by picture stem)."""
    if qid in MANUAL:
        return MANUAL[qid]["value_printed"]
    stem = img.rsplit(".", 1)[0]
    return (stem not in REVIEW_NOT_PRINTED) if stem in REVIEW_CHECKS else None


def checked_by(qid: str, img: str) -> Optional[str]:
    return "engineer" if qid in MANUAL else REVIEW_CHECKS.get(img.rsplit(".", 1)[0])


# Items a hand check found wrong or unverifiable: id -> reason.
MANUAL_EXCLUDE: Dict[str, str] = {}


# ----------------------------------------------------------------------------- numbers and text

def parse_num(s: str) -> Optional[float]:
    """The number in a label or CSV cell after removing thousands commas, %, currency signs, '*' and spaces."""
    t = re.sub(r"[,%$€£¥*\s]", "", str(s))
    return float(t) if NUM.match(t) else None


def label_format(label: str) -> Tuple[int, str, str]:
    """(decimals, prefix, suffix) of the official label, e.g. '$45.5%' -> (1, '$', '%')."""
    t = label.strip()
    prefix = t[0] if t[:1] in "$€£¥" else ""
    suffix = "%" if t.endswith("%") else ""
    core = re.sub(r"[,%$€£¥*\s]", "", t)
    return (len(core.split(".")[1]) if "." in core else 0), prefix, suffix


def fmt(v: float, decimals: int, prefix: str, suffix: str) -> str:
    return "%s%.*f%s" % (prefix, decimals, v, suffix)


def text_reason(query: str, label: str) -> Optional[str]:
    """Why a question is not a direct single-value reading question by its text, or None."""
    for k, r in REJECT_RE:
        if r.search(query):
            return "text_" + k
    if parse_num(label) is None:
        return "label_not_number"
    return None


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("*", "").strip().lower())


def _words(s: str) -> List[str]:
    return [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in re.findall(r"[a-z0-9]+", s.lower())]


def mentioned(label: str, query: str) -> bool:
    lab = _norm(label)
    return bool(lab) and re.search(r"(?<!\w)" + re.escape(lab) + r"(?!\w)", _norm(query)) is not None


def header_matches(header: str, query: str) -> bool:
    words = [w for w in _words(header) if w not in STOP]
    q = set(_words(query))
    return bool(words) and all(w in q for w in words)


def cell_check(query: str, label: str, table: List[List[str]]) -> Tuple[Optional[dict], str]:
    """(the CSV cell the question reads, "") or (None, reason). table[0] is the header, column 0 the row labels."""
    gold = parse_num(label)
    if len(table) < 2 or len(table[0]) < 2 or gold is None:
        return None, "csv_unusable"
    hits = [(i, j) for i in range(1, len(table)) for j in range(1, len(table[i]))
            if parse_num(table[i][j]) is not None and parse_num(table[i][j]) == gold]
    if not hits:
        return None, "gold_not_in_csv"
    if len(hits) > 1:
        return None, "gold_in_several_cells"
    i, j = hits[0]
    rows = [r for r in range(1, len(table)) if mentioned(table[r][0], query)]
    rows = [r for r in rows if not any(r != s and _norm(table[r][0]) in _norm(table[s][0]) for s in rows)]
    if len(rows) != 1 or rows[0] != i:
        return None, "row_not_named" if not rows else ("several_rows_named" if len(rows) > 1 else "other_row_named")
    if len(table[0]) > 2:
        cols = [c for c in range(1, len(table[0])) if header_matches(table[0][c], query)]
        if cols != [j]:
            return None, "column_not_named" if not cols else ("several_columns_named" if len(cols) > 1
                                                              else "other_column_named")
    return {"row": table[i][0], "column": table[0][j], "raw": table[i][j], "row_index": i, "column_index": j}, ""


def distractors(table: List[List[str]], cell: dict, label: str, qid: str) -> List[Tuple[str, dict]]:
    """Up to 3 (formatted option, source cell) from the same CSV: the same column first, then other value columns."""
    decimals, prefix, suffix = label_format(label)
    seen = {fmt(parse_num(label) or 0.0, decimals, prefix, suffix), label.strip()}
    i0, j0 = cell["row_index"], cell["column_index"]
    same = [(i, j0) for i in range(1, len(table)) if i != i0 and j0 < len(table[i])]
    random.Random("same:" + qid).shuffle(same)
    other = [(i, j) for j in range(1, len(table[0])) if j != j0 for i in range(1, len(table)) if j < len(table[i])]
    random.Random("other:" + qid).shuffle(other)
    out = []
    for where, cells in (("same_column", same), ("other_column", other)):
        for i, j in cells:
            v = parse_num(table[i][j])
            if v is None:
                continue
            o = fmt(v, decimals, prefix, suffix)
            if o in seen or parse_num(o) == parse_num(label):
                continue
            seen.add(o)
            out.append((o, {"row": table[i][0], "column": table[0][j], "raw": table[i][j], "from": where}))
            if len(out) == 3:
                return out
    return out


def make_row(imgname: str, k: int, query: str, label: str, options: Sequence[str], cell: dict) -> dict:
    qid = "v04-chartqa:%s:%d" % (imgname, k)
    order = list(options)
    random.Random(qid).shuffle(order)
    gold = label.strip()
    if sum(o == gold for o in order) != 1 or len(set(order)) != 4:
        raise ValueError("gold must map to exactly one of four options: %s %s" % (qid, order))
    return {"id": qid, "source": "chartqa-test", "kind": "chart-reading", "image_id": "chartqa:" + imgname,
            "image": "%s/images/%s" % (OUT, imgname),
            "questions": {"q": {"type": "choice", "instructions": query, "criteria": {o: None for o in order}}},
            "gold": {"q": {"probabilities": {o: float(o == gold) for o in order}}},
            "answer_key": gold, "answer_position": order.index(gold), "official_label": label,
            "csv_cell": {"row": cell["row"], "column": cell["column"], "raw": cell["raw"]}}


def mismatched(rows: List[dict], seed: int = 0) -> Tuple[List[dict], Dict[str, str]]:
    """Each row with another kept row's picture (fixed derangement); image_id stays the original picture."""
    image_of = {r["image_id"]: r["image"] for r in rows}
    move = derangement(sorted(image_of), seed)
    out = []
    for r in rows:
        m = dict(r)
        m["image"] = image_of[move[r["image_id"]]]
        m["mismatched_from"] = r["image_id"]
        m["shown_image_id"] = move[r["image_id"]]
        out.append(m)
    return out, move


# ----------------------------------------------------------------------------- downloads and references

def fetch(root: str, rel: str) -> str:
    """Download RAW_BASE + rel once to <root>/RAW/<basename-path> and return the local path."""
    local = os.path.join(root, RAW, rel.replace(TEST_DIR, ""))
    if not os.path.exists(local):
        os.makedirs(os.path.dirname(local) or ".", exist_ok=True)
        with urllib.request.urlopen(RAW_BASE + urllib.parse.quote(rel), timeout=60) as r:
            data = r.read()
        with open(local + ".part", "wb") as f:
            f.write(data)
        os.replace(local + ".part", local)
    return local


def read_table(path: str) -> List[List[str]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return [r for r in csv.reader(f) if r]


def reference_hashes(root: str, workers: int, cache: str = "") -> Tuple[Dict[str, Any], np.ndarray, np.ndarray,
                                                                    np.ndarray, np.ndarray]:
    """Perceptual hashes of every picture referenced by a data/ JSONL (only these are cached to `cache` if given),
    the mask hashes of the chart-like ones, and the ChartQA picture names / paths those JSONL files mention."""
    files = [f for f in data_jsonls(root) if os.path.normpath(os.path.join(root, OUT)) not in os.path.normpath(f)]
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        meta: Dict[str, Any] = json.loads(str(z["meta"]))
        d, p = z["d"], z["p"]
        new = [f for f in files if os.path.relpath(f, root) not in set(meta["files"])]
        if set(meta["files"]) - {os.path.relpath(f, root) for f in files}:
            raise SystemExit("reference cache %s names data files that no longer exist" % cache)
        if new:   # data files added since the cache: hash only the pictures they add
            pictures, _, _ = scan_data(root, new)
            have = set(meta["names"]) | {b[0] for b in meta["bad"]}
            add = [x for x in sorted(pictures) if os.path.relpath(x, root) not in have]
            dl, pl = list(d), list(p)
            for x in add:
                h = hash_file(x)
                if isinstance(h, str):
                    meta["bad"].append([os.path.relpath(x, root), pictures[x], h])
                else:
                    meta["names"].append(os.path.relpath(x, root))
                    meta["file_of"].append(pictures[x])
                    dl.append(h[0])
                    pl.append(h[2])
            meta["files"] = [os.path.relpath(f, root) for f in files]
            meta["referenced"] = len(meta["names"]) + len(meta["bad"])
            meta["added_after_cache"] = {"files": [os.path.relpath(f, root) for f in new], "pictures": len(add)}
            d, p = np.array(dl, dtype=np.uint64), np.array(pl, dtype=np.uint64)
    else:
        pictures, _, _ = scan_data(root, files)
        paths = sorted(pictures)
        names, dl, pl, bad = [], [], [], []
        with Pool(workers) as pool:
            for n, (path, h) in enumerate(zip(paths, pool.imap(hash_file, paths, chunksize=256)), 1):
                if isinstance(h, str):
                    bad.append([os.path.relpath(path, root), pictures[path], h])
                else:
                    names.append(path)
                    dl.append(h[0])
                    pl.append(h[2])
                if n % 50000 == 0:
                    print("  hashed %d / %d" % (n, len(paths)), flush=True)
        meta = {"files": [os.path.relpath(f, root) for f in files], "names": [os.path.relpath(x, root) for x in names],
                "file_of": [pictures[x] for x in names], "bad": bad, "referenced": len(paths)}
        d, p = np.array(dl, dtype=np.uint64), np.array(pl, dtype=np.uint64)
        if cache:
            np.savez(cache, meta=json.dumps(meta), d=d, p=p)
    chart_paths = [x for x in meta["names"] if CHART_PATH.search(x)]
    mnames, md, mp = [], [], []
    with Pool(workers) as pool:
        for x, mh in zip(chart_paths, pool.imap(mask_file, [os.path.join(root, c) for c in chart_paths], chunksize=64)):
            if mh is not None and not isinstance(mh, str):
                mnames.append(x)
                md.append(mh[0])
                mp.append(mh[2])
    used_names: Dict[str, str] = {}
    for f in files:
        with open(f) as fh:
            for line in fh:
                for m in CHARTQA_NAME.finditer(line):
                    used_names.setdefault(m.group(1), os.path.relpath(f, root))
    meta = dict(meta, chart_paths=chart_paths, mask_names=mnames, used_names=used_names,
                chartqa_paths=[x for x in meta["names"] + [b[0] for b in meta["bad"]] if "chartqa" in x.lower()])
    return meta, d, p, np.array(md, dtype=np.uint64), np.array(mp, dtype=np.uint64)


# ----------------------------------------------------------------------------- build

def main(argv=None) -> None:
    from PIL import Image

    from devision.image import letterbox

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--target", type=int, default=TARGET)
    ap.add_argument("--ref-cache", default="", help="optional .npz to save / reuse the reference hashes")
    a = ap.parse_args(argv)
    root, out = a.root, os.path.join(a.root, OUT)

    raw = {n: fetch(root, n) for n in ("LICENSE", "README.md", TEST_DIR + "test_augmented.json")}
    with open(raw[TEST_DIR + "test_augmented.json"]) as f:
        qs = json.load(f)
    index_in_image: Dict[int, int] = {}
    per_image: Counter = Counter()
    for n, q in enumerate(qs):
        index_in_image[n] = per_image[q["imgname"]]
        per_image[q["imgname"]] += 1
    order = sorted(range(len(qs)), key=lambda n: hashlib.sha256((qs[n]["imgname"] + qs[n]["query"]).encode()).hexdigest())

    meta, ref_d, ref_p, mask_d, mask_p = reference_hashes(root, a.workers, a.ref_cache)
    print("references: %d pictures hashed of %d referenced, %d chart-like masks" %
          (len(meta["names"]), meta["referenced"], len(meta["mask_names"])), flush=True)

    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    os.makedirs(os.path.join(out, "images256"), exist_ok=True)
    kept: List[dict] = []
    taken: set = set()
    iso_cache: Dict[str, Tuple[List[dict], dict]] = {}
    exclusions: List[dict] = []
    steps: Counter = Counter()
    downloaded: Dict[str, str] = {}
    for n in order:
        q = qs[n]
        img, query, label = q["imgname"], q["query"], q["label"]
        qid = "v04-chartqa:%s:%d" % (img, index_in_image[n])
        if len(kept) >= a.target:
            steps["not_reached"] += 1
            continue
        steps["walked"] += 1

        def drop(reason: str, **extra) -> None:
            exclusions.append({"id": qid, "imgname": img, "query": query, "label": label, "reason": reason, **extra})

        if img in taken:
            drop("picture_already_kept")
            continue
        r = text_reason(query, label)
        if r:
            drop(r)
            continue
        steps["text_ok"] += 1
        stem = img[:-4] if img.endswith(".png") else img
        table_rel = TEST_DIR + "tables/" + stem + ".csv"
        downloaded[table_rel] = fetch(root, table_rel)
        table = read_table(downloaded[table_rel])
        cell, r = cell_check(query, label, table)
        if cell is None:
            drop(r)
            continue
        steps["cell_ok"] += 1
        ds = distractors(table, cell, label, qid)
        if len(ds) < 3:
            drop("distractors", found=len(ds))
            continue
        steps["distractors_ok"] += 1
        png_rel = TEST_DIR + "png/" + img
        downloaded[png_rel] = fetch(root, png_rel)
        if img not in iso_cache:
            with Image.open(downloaded[png_rel]) as im:
                im.load()
                h, mh, size = perceptual(im), mask_hashes(im), im.size
            nd = nearest(h, ref_d, ref_p)
            found = []
            if is_near(nd):
                found.append({"stage": "near_data", "match": meta["names"][nd["both"][1]],
                              "file": meta["file_of"][nd["both"][1]], "distance_max_dhash_phash": nd["both"][0]})
            dist = {"data": {k: v[0] for k, v in nd.items()}, "data_match": meta["names"][nd["both"][1]]
                    if nd["both"][1] >= 0 else None, "size": list(size)}
            if mh is not None:
                nm = nearest(mh, mask_d, mask_p)
                dist["mask"] = {k: v[0] for k, v in nm.items()}
                if is_near(nm):
                    found.append({"stage": "recolor", "match": meta["mask_names"][nm["both"][1]],
                                  "distance_max_dhash_phash": nm["both"][0]})
            if stem in meta["used_names"]:
                found.append({"stage": "used_name", "file": meta["used_names"][stem]})
            iso_cache[img] = (found, dist)
        found, dist = iso_cache[img]
        if found:
            drop("isolation_" + found[0]["stage"], matches=found)
            continue
        steps["isolation_ok"] += 1
        if qid in MANUAL_EXCLUDE:
            drop("hand_check", note=MANUAL_EXCLUDE[qid])
            continue
        row = make_row(img, index_in_image[n], query, label, [label.strip()] + [o for o, _ in ds], cell)
        row["distractor_cells"] = [c for _, c in ds]
        w, hgt = dist["size"]
        scale = 256 / max(w, hgt)
        row["readability"] = {"original_size": [w, hgt], "letterbox_scale": round(scale, 4),
                              "est_text_px_256": round(12 * scale, 1),
                              "value_printed": value_printed(qid, img), "checked_by": checked_by(qid, img)}
        row["isolation_nearest"] = {"data_max_dhash_phash": dist["data"]["both"], "data_dhash": dist["data"]["dhash"],
                                    "data_phash": dist["data"]["phash"]}
        dst = os.path.join(out, "images", img)
        if not os.path.exists(dst):
            with open(downloaded[png_rel], "rb") as s, open(dst, "wb") as t:
                t.write(s.read())
        with Image.open(downloaded[png_rel]) as im:
            letterbox(im, 256).save(os.path.join(out, "images256", img))
        kept.append(row)
        taken.add(img)
        steps["kept"] += 1

    if len(kept) < 2:
        raise SystemExit("fewer than two kept items; no mismatched control")
    kept.sort(key=lambda r: r["id"])
    mism, move = mismatched(kept, 0)
    keep_names = {r["image_id"].split(":", 1)[1] for r in kept}
    for d in ("images", "images256"):
        for f in os.listdir(os.path.join(out, d)):
            if f not in keep_names:
                os.remove(os.path.join(out, d, f))
    write_jsonl(os.path.join(out, "chartqa.jsonl"), kept)
    write_jsonl(os.path.join(out, "chartqa_mism.jsonl"), mism)
    with open(os.path.join(out, "EXCLUSIONS.json"), "w") as f:
        json.dump(exclusions, f, indent=1, ensure_ascii=False)

    scales = [r["readability"]["letterbox_scale"] for r in kept]
    texts = [r["readability"]["est_text_px_256"] for r in kept]
    lic = open(raw["LICENSE"]).readline().strip()
    manifest = {
        "name": OUT,
        "role": "frozen v0.4 chart-reading confirmation subset (held out; not for training, monitoring or "
                "temperature fitting)",
        "frozen": datetime.date.today().isoformat(),
        "source": {"repo": "https://github.com/vis-nlp/ChartQA", "commit": COMMIT, "directory": TEST_DIR,
                   "questions_file": "test_augmented.json", "questions": len(qs), "pictures": len(per_image),
                   "licence": {"file": "LICENSE", "first_line": lic, "spdx": "GPL-3.0",
                               "note": "the repository is GPL-3.0; the charts themselves come from Statista / Pew / "
                                       "OWID / OECD pages with their own terms (research use only here)"},
                   "raw_dir": RAW,
                   "raw_files_read_by_this_build": sorted(os.path.relpath(p, os.path.join(root, RAW))
                                                          for p in set(raw.values()) | set(downloaded.values())),
                   "raw_sha256_all_files_in_raw_dir": {
                       os.path.relpath(os.path.join(dp, f), os.path.join(root, RAW)): sha256(os.path.join(dp, f))
                       for dp, _, fs in sorted(os.walk(os.path.join(root, RAW))) for f in sorted(fs)}},
        "selection": {"rule": "see the builder docstring: text / csv / cell eligibility, walk by "
                              "sha256(imgname + query) ascending, one question per picture, target %d" % a.target,
                      "reject_patterns": dict(REJECT), "equality": "numeric after removing thousands commas, %, "
                      "currency signs, '*', spaces; no rounding, no unit scaling",
                      "hand_checks": MANUAL, "review_checks": REVIEW_CHECKS, "review_not_printed": REVIEW_NOT_PRINTED,
                      "hand_excluded": MANUAL_EXCLUDE},
        "distractor_rule": "3 cells of the same CSV, same column first (random.Random('same:'+id) order), then other "
                           "value columns (random.Random('other:'+id)); formatted with the label's decimals and %% / "
                           "currency sign; values equal to the gold or an earlier option after formatting skipped; "
                           "option order random.Random(id).shuffle",
        "counts": {"questions": len(qs), **dict(steps)},
        "exclusion_reasons": dict(Counter(e["reason"] for e in exclusions)),
        "distractor_sources": dict(Counter(c["from"] for r in kept for c in r["distractor_cells"])),
        "isolation": {
            "threshold_bits": THRESHOLD,
            "rule": "excluded when one reference picture is within threshold bits by both dHash and pHash (each the "
                    "smaller of original / mirrored candidate), v04_confirm.nearest / is_near",
            "data_jsonl_files": len(meta["files"]),
            "data_pictures_referenced": meta["referenced"],
            "data_pictures_hashed": len(meta["names"]),
            "data_pictures_unreadable_or_missing": len(meta["bad"]),
            "unreadable_examples": meta["bad"][:20],
            "chart_like_mask_references": len(meta["mask_names"]),
            "chart_like_paths_examples": meta["chart_paths"][:20],
            "chartqa_names_in_data_jsonl": meta["used_names"],
            "chartqa_paths_in_data_jsonl": meta["chartqa_paths"],
            "skipped_dirs": ["raw", "v04_confirm", OUT],
            "checked_pictures": {k: {"nearest": v[1], "excluded_by": [x["stage"] for x in v[0]]}
                                 for k, v in sorted(iso_cache.items())},
            "kept_nearest_max_dhash_phash_histogram": dict(sorted(Counter(
                r["isolation_nearest"]["data_max_dhash_phash"] for r in kept).items())),
        },
        "readability": {"letterbox": "devision.image.letterbox(img, 256), saved to images256/",
                        "est_text_px_256": "12 px * scale (assumed typical label height in the original picture)",
                        "scale_min_median_max": [min(scales), float(np.median(scales)), max(scales)],
                        "est_text_px_256_min_median_max": [min(texts), float(np.median(texts)), max(texts)],
                        "original_sizes": dict(Counter("%dx%d" % tuple(r["readability"]["original_size"])
                                                       for r in kept).most_common(10)),
                        "note": "most pictures are 800 px wide Statista screenshots, so the letterbox scale is ~0.32 "
                                "and printed values are an estimated ~2-4 px tall at 256 (12 px assumed, not measured per "
                                "picture); every kept picture was looked at (21 engineer + 27 critic + 32 spec review): "
                                "70 print the asked value, 10 do not (read from bar / point position; multi_col_40409 "
                                "is a stacked-segment thickness)",
                        "hand_checked": sum(r["readability"]["checked_by"] is not None for r in kept),
                        "checked_by": dict(Counter(str(r["readability"]["checked_by"]) for r in kept)),
                        "value_printed": dict(Counter(str(r["readability"]["value_printed"]) for r in kept))},
        "mismatched_control": {"seed": 0, "rule": "v04_confirm.derangement over the kept image_ids (sorted); "
                               "image_id stays the original, shown_image_id is the picture shown", "map": move},
        "files": {"chartqa.jsonl": {"rows": len(kept), "sha256": sha256(os.path.join(out, "chartqa.jsonl"))},
                  "chartqa_mism.jsonl": {"rows": len(mism), "sha256": sha256(os.path.join(out, "chartqa_mism.jsonl"))},
                  "EXCLUSIONS.json": {"rows": len(exclusions), "sha256": sha256(os.path.join(out, "EXCLUSIONS.json"))}},
        "images": {f: sha256(os.path.join(out, "images", f)) for f in sorted(os.listdir(os.path.join(out, "images")))},
        "images256": {f: sha256(os.path.join(out, "images256", f))
                      for f in sorted(os.listdir(os.path.join(out, "images256")))},
    }
    with open(os.path.join(out, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: manifest[k] for k in ("counts", "exclusion_reasons", "distractor_sources", "files")},
                     indent=1))


if __name__ == "__main__":
    main()
