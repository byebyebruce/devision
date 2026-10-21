"""The v0.4 frozen confirmation set: ScienceQA official **test** pictures, isolated from every picture we have used.

    uv run python scripts/data/v04_confirm.py --root data

Selection (every row that passes; nothing is picked by any model output, no content is edited, and nothing is added
to replace an excluded row): ScienceQA test rows (derek-thomas/ScienceQA, the local snapshot under data/raw/hf) that
have a picture and whose `skill` is one of four, in three subsets (`kind`; the skill is kept as `axis`):

  chart-reading       "Use climate data to make predictions"
  graphic-organizer   "Read graphic organizers"
  object-property     "Compare properties of objects", "Identify properties of an object"

A row is converted exactly as the other ScienceQA sets are (`v3_build.make_sample`, as in v10_build.py): the hint is
`state_text`, the question is the instructions of one choice question whose options are the ScienceQA choices in a
seeded random order, and the gold is 1.0 on the choice at ScienceQA's `answer` index. The option order of a row
depends only on the seed and its row index (not on which other rows are excluded).

Isolation, applied to every candidate picture (a picture's questions stay or go together):

  1. exact      decoded pixels (`v3_build.pixel_hash`) equal a ScienceQA train or validation picture.
  2. near_sqa   near duplicate of a ScienceQA train / validation picture: 64-bit dHash and 64-bit pHash of the
                grayscale picture, computed for the candidate and its horizontal mirror; excluded when some reference
                picture is within THRESHOLD bits by dHash AND by pHash (each distance the smaller of original /
                mirrored). Either hash alone is not enough on these pictures: ScienceQA's white three-card layouts
                put unrelated pictures within 2-5 dHash bits (e.g. test row 2216 pickles / apple / lime vs train
                row 4203 paper / balloons / towel: dHash 2, pHash 10), while the real near duplicates (the same
                climate chart redrawn in other bar colours; a card picture sharing two of its three cards) are
                close by both.
  2b. recolor  the same test on a colour-agnostic picture: the mask of saturated pixels (HSV S > 60, V > 40; only
                when it covers >= 1% of the picture), against ScienceQA train / validation and every data/ picture
                whose path names ScienceQA. Needed for the climate charts: the test charts are the train /
                validation charts of the same city with the same data, redrawn in other bar colours (grayscale
                distance 7-9, mask distance 0-2).
  3. used_id    the test row is already used by id in a data file: any `...sqa...:test-<row>` id (e.g.
                "v3-sqa-test:test-12", "exp28-field:v3-sqa-test:test-12:..."), and JevBench's ScienceQA cases
                (`row_index` of derek-thomas/ScienceQA rows; their split is checked to be test by question text).
                The other questions of that picture go with it (reason used_id_sibling).
     near_data  the same perceptual test against every picture referenced by any JSONL under the data root
                (fields `image`, or `image_path` resolved against the file's directory or its parent), skipping
                only data/raw and this set's own output directory; training, alignment, monitoring, calibration,
                evaluation and benchmark files alike, synthetic ones included. Missing / unreadable pictures are
                counted and listed (examples) in the manifest.

JPEG references are decoded with Pillow's draft mode (DCT-scaled to >= 64 px, then grayscale), which changes the
9x8 / 32x32 thumbnails by a bit or two at most; the threshold absorbs that. Every candidate's nearest distances
(dHash and pHash, against ScienceQA train+validation and against data/ pictures) are written to the manifest, with
a 0..12 histogram, so the threshold choice is visible.

Outputs in <root>/v04_confirm/:
  images/<pixelhash16>.png   the decoded picture as RGB PNG (lossless: its pixel hash is the image_id)
  confirm.jsonl              id "v04-confirm:test-<row>", source "scienceqa-test", kind = subset, axis = skill,
                             image_id = "img:" + first 16 hex of the pixel hash (evaluation resamples by picture)
  confirm_mism.jsonl         the same rows, each picture replaced by another picture of the same subset through a
                             fixed derangement over that subset's pictures (seed 0; two pictures swap);
                             "mismatched_from" keeps the original image_id. The mismatched-picture control.
  EXCLUSIONS.json            every excluded candidate row: row, skill, subset, first reason, all reasons, match
  MANIFEST.json              snapshot, parquet sha256, rule, per-stage per-subset counts, threshold, coverage,
                             sha256 of both JSONL files and every picture, label source, frozen date.
"""
import argparse
import datetime
import glob
import hashlib
import io
import json
import os
import random
import re
from collections import Counter, defaultdict
from multiprocessing import Pool
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from v3_build import make_sample, pixel_hash, read_parquets

SKILLS = {"Use climate data to make predictions": "chart-reading",
          "Read graphic organizers": "graphic-organizer",
          "Compare properties of objects": "object-property",
          "Identify properties of an object": "object-property"}
SUBSETS = ("chart-reading", "graphic-organizer", "object-property")
THRESHOLD = 6
MASK_MIN_FILL = 0.01
SQA_PATH = re.compile(r"scienceqa|sqa", re.I)   # data/ pictures that are ScienceQA pictures (any split)
HIST_MAX = 12
OUT = "v04_confirm"
TEST_ID = re.compile(r"(?:sqa|scienceqa)[\w-]*:test-(\d+)")
SNAPSHOT = os.path.join("raw", "hf", "datasets--derek-thomas--ScienceQA", "snapshots", "*", "data")

Hashes = Tuple[int, int, int, int]   # dHash, mirrored dHash, pHash, mirrored pHash


# ----------------------------------------------------------------------------- perceptual hashes

def _dct_matrix(n: int = 32) -> np.ndarray:
    k, x = np.arange(n)[:, None], np.arange(n)[None, :]
    return np.cos(np.pi * (2 * x + 1) * k / (2 * n))


_DCT = _dct_matrix()


def _bits(flags: np.ndarray) -> int:
    v = 0
    for b in flags.ravel():
        v = (v << 1) | int(b)
    return v


def _dhash(g) -> int:
    from PIL import Image
    a = np.asarray(g.resize((9, 8), Image.Resampling.BILINEAR), dtype=np.int16)
    return _bits(a[:, :-1] < a[:, 1:])


def _phash(g) -> int:
    from PIL import Image
    a = np.asarray(g.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    low = (_DCT @ a @ _DCT.T)[:8, :8]
    return _bits(low > np.median(low))


def perceptual(im) -> Hashes:
    """dHash and pHash of the grayscale picture and of its horizontal mirror."""
    from PIL import Image
    g = im.convert("L")
    m = g.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return _dhash(g), _dhash(m), _phash(g), _phash(m)


def hash_file(path: str) -> Union[Hashes, str]:
    """Perceptual hashes of a picture file, or the error text when it cannot be read."""
    from PIL import Image
    try:
        with Image.open(path) as im:
            if im.format == "JPEG":
                im.draft("L", (64, 64))
            return perceptual(im)
    except Exception as e:   # noqa: BLE001 -- any unreadable file is counted, not fatal
        return "%s: %s" % (type(e).__name__, e)


def mask_hashes(im) -> Optional[Hashes]:
    """Colour-agnostic hashes: dHash / pHash (and mirrored) of the mask of saturated pixels, so a chart redrawn with
    other bar colours hashes the same. None when under MASK_MIN_FILL of the picture is saturated (a grey picture's
    empty mask says nothing)."""
    from PIL import Image
    hsv = np.asarray(im.convert("RGB").convert("HSV"))
    on = (hsv[..., 1] > 60) & (hsv[..., 2] > 40)
    if on.mean() < MASK_MIN_FILL:
        return None
    m = Image.fromarray(on.astype(np.uint8) * 255, "L")
    mm = m.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return _dhash(m), _dhash(mm), _phash(m), _phash(mm)


def mask_file(path: str) -> Union[Optional[Hashes], str]:
    from PIL import Image
    try:
        with Image.open(path) as im:
            return mask_hashes(im)
    except Exception as e:   # noqa: BLE001
        return "%s: %s" % (type(e).__name__, e)


def hash_bytes(data: bytes) -> Tuple[str, Hashes, Optional[Hashes]]:
    from PIL import Image
    with Image.open(io.BytesIO(data)) as im:
        return pixel_hash(im), perceptual(im), mask_hashes(im)


def hamming(refs: np.ndarray, h: int) -> np.ndarray:
    return np.bitwise_count(refs ^ np.uint64(h)).astype(np.int64)


def nearest(c: Hashes, ref_d: np.ndarray, ref_p: np.ndarray) -> Dict[str, Tuple[int, int]]:
    """(distance, index of the reference) of the nearest reference: per hash kind ("dhash", "phash"; each the smaller
    of the original / mirrored candidate), and "both" = the reference minimising max(dHash, pHash distance), i.e.
    the closest picture that is near by both hashes at once."""
    if len(ref_d) == 0:
        return {k: (64, -1) for k in ("dhash", "phash", "both")}
    dd = np.minimum(hamming(ref_d, c[0]), hamming(ref_d, c[1]))
    dp = np.minimum(hamming(ref_p, c[2]), hamming(ref_p, c[3]))
    out = {}
    for kind, d in (("dhash", dd), ("phash", dp), ("both", np.maximum(dd, dp))):
        j = int(np.argmin(d))
        out[kind] = (int(d[j]), j)
    return out


def is_near(near: Dict[str, Tuple[int, int]], threshold: int = THRESHOLD) -> bool:
    """Near duplicate: one reference picture within `threshold` bits by dHash and by pHash."""
    return near["both"][0] <= threshold


# ----------------------------------------------------------------------------- mismatched control

def derangement(images: Sequence[str], seed: int = 0) -> Dict[str, str]:
    """A fixed permutation of `images` that moves every picture (two pictures swap)."""
    order = sorted(images)
    if len(order) < 2:
        raise ValueError("a derangement needs at least two pictures, got %d" % len(order))
    rng = random.Random(seed)
    while True:
        perm = order[:]
        rng.shuffle(perm)
        if all(a != b for a, b in zip(order, perm)):
            return dict(zip(order, perm))


def mismatched(rows: List[dict], seed: int = 0) -> List[dict]:
    by_kind: Dict[str, List[str]] = defaultdict(list)
    image_of: Dict[str, str] = {}
    for r in rows:
        if r["image_id"] not in image_of:
            by_kind[r["kind"]].append(r["image_id"])
        image_of[r["image_id"]] = r["image"]
    move = {}
    for kind, ids in by_kind.items():
        if len(ids) >= 2:   # a subset with one picture has no within-subset control; its rows are left out
            move.update(derangement(ids, seed))
    out = []
    for r in rows:
        if r["image_id"] not in move:
            continue
        m = dict(r)
        m["image_id"] = move[r["image_id"]]
        m["image"] = image_of[m["image_id"]]
        m["mismatched_from"] = r["image_id"]
        out.append(m)
    return out


# ----------------------------------------------------------------------------- conversion

def convert(row_index: int, r: dict, picture: str, image: str, subset: str, seed: int = 0) -> Optional[dict]:
    """One ScienceQA test row in our sample format (the v3 / v10 ScienceQA conversion), or None."""
    rng = random.Random("%d:test-%d" % (seed, row_index))
    s = make_sample("sqa-test", "test-%d" % row_index, picture, image, r["question"], list(r["choices"]),
                    int(r["answer"]), rng, r.get("hint") or "",
                    {"subject": r.get("subject"), "category": r.get("category")})
    if s is None:
        return None
    s.update({"id": "v04-confirm:test-%d" % row_index, "source": "scienceqa-test", "kind": subset,
              "axis": r["skill"]})
    return s


# ----------------------------------------------------------------------------- data files

def data_jsonls(root: str) -> List[str]:
    """Every JSONL under the data root except data/raw and this set's own output directory."""
    skip = {os.path.normpath(os.path.join(root, "raw")), os.path.normpath(os.path.join(root, OUT))}
    files = []
    for d, dirs, names in os.walk(root):
        dirs[:] = sorted(x for x in dirs if os.path.normpath(os.path.join(d, x)) not in skip)
        files += [os.path.join(d, n) for n in names if n.endswith(".jsonl")]
    return sorted(files)


def _resolve(root: str, f: str, p: str) -> str:
    for base in (root, os.path.dirname(f), os.path.dirname(os.path.dirname(f))):
        q = os.path.join(base, p)
        if os.path.exists(q):
            return os.path.normpath(q)
    return os.path.normpath(os.path.join(root, p))


def scan_data(root: str, files: Iterable[str]):
    """(picture path -> first file naming it, test row -> files using it by id, JevBench ScienceQA rows -> question)."""
    pictures: Dict[str, str] = {}
    used: Dict[int, set] = defaultdict(set)
    jev: Dict[int, Tuple[str, str]] = {}
    for f in files:
        with open(f) as fh:
            for line in fh:
                if not line.strip():
                    continue
                for m in TEST_ID.finditer(line):
                    used[int(m.group(1))].add(os.path.relpath(f, root))
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(r, dict):
                    continue
                for key in ("image", "image_path"):
                    p = r.get(key)
                    if isinstance(p, str) and p and not p.startswith(("http://", "https://", "data:")):
                        pictures.setdefault(_resolve(root, f, p), os.path.relpath(f, root))
                if r.get("hf_repo") == "derek-thomas/ScienceQA" and isinstance(r.get("row_index"), int):
                    q = (r.get("state") or {}).get("question", "") if isinstance(r.get("state"), dict) else ""
                    jev[r["row_index"]] = (q, os.path.relpath(f, root))
    return pictures, used, jev


# ----------------------------------------------------------------------------- build

def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_jsonl(path: str, rows: List[dict]) -> None:
    with open(path, "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def histogram(values: Iterable[int]) -> Dict[str, int]:
    c = Counter(values)
    h = {str(d): c.get(d, 0) for d in range(HIST_MAX + 1)}
    h[">%d" % HIST_MAX] = sum(n for d, n in c.items() if d > HIST_MAX)
    return h


def main(argv=None) -> None:
    from PIL import Image

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--threshold", type=int, default=THRESHOLD)
    a = p.parse_args(argv)
    root, out = a.root, os.path.join(a.root, OUT)
    parquets = {s: sorted(glob.glob(os.path.join(root, SNAPSHOT, "%s-*.parquet" % s))) for s in ("train", "validation", "test")}
    if not all(parquets.values()):
        raise SystemExit("ScienceQA parquets missing under %s" % os.path.join(root, SNAPSHOT))

    # ScienceQA train + validation pictures: pixel hashes and perceptual hashes
    sqa_pixels: Dict[str, str] = {}
    sqa_names, sqa_d, sqa_p = [], [], []
    mask_names, mask_d, mask_p = [], [], []
    with Pool(a.workers) as pool:
        for split in ("train", "validation"):
            rows = [(i, r["image"]["bytes"]) for i, r in enumerate(read_parquets(parquets[split][0], columns=["image"]))
                    if r["image"] is not None]
            for (i, _), (px, h, mh) in zip(rows, pool.imap(hash_bytes, [b for _, b in rows], chunksize=64)):
                name = "scienceqa-%s:%d" % (split, i)
                sqa_pixels.setdefault(px, name)
                sqa_names.append(name)
                sqa_d.append(h[0])
                sqa_p.append(h[2])
                if mh is not None:
                    mask_names.append(name)
                    mask_d.append(mh[0])
                    mask_p.append(mh[2])
    sqa_d_arr, sqa_p_arr = np.array(sqa_d, dtype=np.uint64), np.array(sqa_p, dtype=np.uint64)
    print("ScienceQA train+validation pictures:", len(sqa_names), "unique pixels:", len(sqa_pixels), flush=True)

    # candidates
    test_rows = list(read_parquets(parquets["test"][0]))
    cands = []   # (row index, row, subset, pixel hash, bytes)
    for i, r in enumerate(test_rows):
        if r["image"] is not None and r.get("skill") in SKILLS:
            data = r["image"]["bytes"]
            with Image.open(io.BytesIO(data)) as im:
                cands.append((i, r, SKILLS[r["skill"]], pixel_hash(im), perceptual(im), mask_hashes(im)))
    print("candidates:", len(cands), flush=True)

    # every picture named by a data/ JSONL
    files = data_jsonls(root)
    pictures, used, jev = scan_data(root, files)
    paths = sorted(pictures)
    print("data JSONLs:", len(files), "pictures:", len(paths), flush=True)
    ref_names, ref_d, ref_p, bad = [], [], [], []
    with Pool(a.workers) as pool:
        for n, (path, h) in enumerate(zip(paths, pool.imap(hash_file, paths, chunksize=256)), 1):
            if isinstance(h, str):
                bad.append((os.path.relpath(path, root), pictures[path], h))
            else:
                ref_names.append(path)
                ref_d.append(h[0])
                ref_p.append(h[2])
            if n % 50000 == 0:
                print("  hashed %d / %d (unreadable %d)" % (n, len(paths), len(bad)), flush=True)
    ref_d_arr, ref_p_arr = np.array(ref_d, dtype=np.uint64), np.array(ref_p, dtype=np.uint64)
    sqa_paths = [q for q in ref_names if SQA_PATH.search(os.path.relpath(q, root))]
    with Pool(a.workers) as pool:
        for q, mh in zip(sqa_paths, pool.imap(mask_file, sqa_paths, chunksize=64)):
            if mh is not None and not isinstance(mh, str):
                mask_names.append(os.path.relpath(q, root))
                mask_d.append(mh[0])
                mask_p.append(mh[2])
    mask_d_arr, mask_p_arr = np.array(mask_d, dtype=np.uint64), np.array(mask_p, dtype=np.uint64)
    print("colour-agnostic references:", len(mask_names), "(ScienceQA-named data pictures: %d)" % len(sqa_paths),
          flush=True)
    jev_checked = {row: q == test_rows[row]["question"] for row, (q, _) in jev.items() if row < len(test_rows)}
    for row, (_, f) in jev.items():
        if jev_checked.get(row):
            used[row].add(f)

    # checks per picture
    stages = ("exact", "near_sqa", "recolor", "used_id", "near_data")
    by_pic: Dict[str, List[tuple]] = defaultdict(list)
    for c in cands:
        by_pic[c[3]].append(c)
    reasons: Dict[str, List[dict]] = {}
    distances, matches = {}, {}
    for px, group in by_pic.items():
        h, mh = group[0][4], group[0][5]
        found = []
        if px in sqa_pixels:
            found.append({"stage": "exact", "match": sqa_pixels[px], "distance": 0})
        ns = nearest(h, sqa_d_arr, sqa_p_arr)
        nd = nearest(h, ref_d_arr, ref_p_arr)
        distances[px] = {"sqa": {k: v[0] for k, v in ns.items()}, "data": {k: v[0] for k, v in nd.items()}}
        matches[px] = {"sqa": {k: sqa_names[v[1]] for k, v in ns.items() if v[1] >= 0},
                       "data": {k: os.path.relpath(ref_names[v[1]], root) for k, v in nd.items() if v[1] >= 0}}
        if is_near(ns, a.threshold):
            found.append({"stage": "near_sqa", "match": sqa_names[ns["both"][1]], "distance_max_dhash_phash": ns["both"][0]})
        if mh is not None:
            nm = nearest(mh, mask_d_arr, mask_p_arr)
            distances[px]["mask"] = {k: v[0] for k, v in nm.items()}
            matches[px]["mask"] = {k: mask_names[v[1]] for k, v in nm.items() if v[1] >= 0}
            if is_near(nm, a.threshold):
                found.append({"stage": "recolor", "match": mask_names[nm["both"][1]],
                              "distance_max_dhash_phash": nm["both"][0]})
        else:
            distances[px]["mask"] = {k: 64 for k in ("dhash", "phash", "both")}
        rows_used = {c[0]: sorted(used[c[0]]) for c in group if c[0] in used}
        if rows_used:
            found.append({"stage": "used_id", "rows": rows_used})
        if is_near(nd, a.threshold):
            m = ref_names[nd["both"][1]]
            found.append({"stage": "near_data", "match": os.path.relpath(m, root), "file": pictures[m],
                          "distance_max_dhash_phash": nd["both"][0]})
        reasons[px] = found

    # funnel, conversion, outputs
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    funnel = {s: Counter() for s in ("candidates",) + stages + ("unusable_options", "kept")}
    pics = {s: defaultdict(set) for s in funnel}
    exclusions, rows = [], []
    for i, r, subset, px, _, _ in cands:
        funnel["candidates"][subset] += 1
        pics["candidates"][subset].add(px)
        found = reasons[px]
        first = next((s for s in stages if any(f["stage"] == s for f in found)), None)
        if first is None:
            image_id = "img:" + px[:16]
            rel = "%s/images/%s.png" % (OUT, px[:16])
            s = convert(i, r, image_id, rel, subset, a.seed)
            if s is None:
                first = "unusable_options"
            else:
                path = os.path.join(root, rel)
                if not os.path.exists(path):
                    with Image.open(io.BytesIO(r["image"]["bytes"])) as im:
                        im.convert("RGB").save(path, format="PNG")
                rows.append(s)
                funnel["kept"][subset] += 1
                pics["kept"][subset].add(px)
                continue
        funnel[first][subset] += 1
        pics[first][subset].add(px)
        reason_list = found
        if first == "used_id" and i not in next(f["rows"] for f in found if f["stage"] == "used_id"):
            first = "used_id_sibling"
        exclusions.append({"row": i, "skill": r["skill"], "subset": subset, "image_id": "img:" + px[:16],
                           "reason": first, "all_reasons": reason_list})

    # a picture whose rows are split between kept and excluded cannot happen (all checks are per picture, except
    # unusable options); keep pictures whole anyway
    dropped_pics = {e["image_id"] for e in exclusions}
    split_pics = sorted({r["image_id"] for r in rows} & dropped_pics)
    if split_pics:
        raise SystemExit("pictures with both kept and excluded questions: %s" % split_pics)
    no_control = [k for k in SUBSETS if len(pics["kept"][k]) < 2]
    if no_control:
        print("WARNING: no within-subset mismatched control (fewer than 2 kept pictures) for", no_control, flush=True)
    rows.sort(key=lambda s: int(s["id"].rsplit("-", 1)[1]))
    mism = mismatched(rows, a.seed)
    write_jsonl(os.path.join(out, "confirm.jsonl"), rows)
    write_jsonl(os.path.join(out, "confirm_mism.jsonl"), mism)
    with open(os.path.join(out, "EXCLUSIONS.json"), "w") as f:
        json.dump(exclusions, f, indent=1, ensure_ascii=False)

    kept_px = {px for s in SUBSETS for px in pics["kept"][s]}
    for f in glob.glob(os.path.join(out, "images", "*.png")):   # pictures of an earlier build no longer kept
        if os.path.basename(f)[:-4] not in {px[:16] for px in kept_px}:
            os.remove(f)
    all_px = list(by_pic)

    def dist_hist(where: str, kind: str, which) -> Dict[str, int]:
        return histogram(distances[px][where][kind] for px in which)

    manifest = {
        "name": "v04_confirm",
        "role": "frozen v0.4 confirmation set (held out; not for training, monitoring or temperature fitting)",
        "frozen": datetime.date.today().isoformat(),
        "source": {"dataset": "derek-thomas/ScienceQA", "split": "test",
                   "snapshot": os.path.basename(os.path.dirname(os.path.dirname(parquets["test"][0]))),
                   "parquet_sha256": {os.path.basename(f): sha256(f) for s in parquets for f in parquets[s]}},
        "selection": {"rule": "test rows with a picture whose skill is listed; all of them; no model output used, "
                              "no content edited, no replacements",
                      "skills": SKILLS},
        "conversion": "v3_build.make_sample (hint -> state_text; options shuffled with Random('<seed>:test-<row>'))",
        "label_source": "ScienceQA answer index",
        "isolation": {
            "stages": {"exact": "decoded pixels equal a ScienceQA train / validation picture",
                       "near_sqa": "dHash or pHash (original or mirrored candidate) within threshold of a ScienceQA "
                                   "train / validation picture",
                       "recolor": "colour-agnostic (saturated-pixel mask) dHash and pHash within threshold of a "
                                  "ScienceQA train / validation picture or a ScienceQA picture in data/ (mask needs "
                                  ">= %.0f%% of pixels)" % (100 * MASK_MIN_FILL),
                       "used_id": "test row already used by id in a data/ JSONL (and the other rows of its picture)",
                       "near_data": "dHash or pHash within threshold of a picture referenced by a data/ JSONL",
                       "unusable_options": "make_sample rejected the options"},
            "threshold_bits": a.threshold,
            "rule": "excluded when one reference picture is within threshold bits by both dHash and pHash "
                    "(min over references of max(dHash, pHash distance), each the smaller of original / mirrored "
                    "candidate); histograms: 'both' is that deciding distance, dhash / phash the nearest per hash",
            "nearest_distance_histogram": {
                "all_candidate_pictures": {"n": len(all_px), **{"%s_%s" % (w, k): dist_hist(w, k, all_px)
                                                                  for w in ("sqa", "mask", "data") for k in ("both", "dhash", "phash")}},
                "kept_pictures": {"n": len(kept_px), **{"%s_%s" % (w, k): dist_hist(w, k, kept_px)
                                                         for w in ("sqa", "mask", "data") for k in ("both", "dhash", "phash")}}},
        },
        "mismatched_control_missing_for": no_control,
        "candidate_pictures": [{"image_id": "img:" + px[:16], "subset": g[0][2], "rows": [c[0] for c in g],
                                "nearest": distances[px], "nearest_match": matches[px], "kept": px in kept_px,
                                "reasons": sorted({f["stage"] for f in reasons[px]})}
                               for px, g in sorted(by_pic.items(), key=lambda kv: kv[1][0][0])],
        "excluded_by_any_reason": {st: {k: {"questions": sum(1 for e in exclusions if e["subset"] == k and
                                                             any(f["stage"] == st for f in e["all_reasons"])),
                                            "images": len({e["image_id"] for e in exclusions if e["subset"] == k and
                                                           any(f["stage"] == st for f in e["all_reasons"])})}
                                        for k in SUBSETS} for st in stages},
        "counts_note": "counts: a funnel, each excluded row under the first stage that excludes it (stage order "
                       "exact, near_sqa, recolor, used_id, near_data); excluded_by_any_reason: every stage that "
                       "matched",
        "counts": {s: {k: {"questions": funnel[s][k], "images": len(pics[s][k])} for k in SUBSETS}
                   for s in funnel},
        "coverage": {
            "scienceqa_train_validation_pictures": len(sqa_names),
            "scienceqa_train_validation_unique_pixels": len(sqa_pixels),
            "data_jsonl_files": len(files),
            "data_jsonl_list": [os.path.relpath(f, root) for f in files],
            "skipped_dirs": ["raw", OUT],
            "data_pictures_referenced": len(paths),
            "data_pictures_hashed": len(ref_names),
            "data_pictures_unreadable_or_missing": len(bad),
            "colour_agnostic_references": len(mask_names),
            "scienceqa_named_data_pictures": len(sqa_paths),
            "unreadable_examples": [{"image": b[0], "file": b[1], "error": b[2]} for b in bad[:20]],
            "test_rows_used_by_id": len(used),
            "jevbench_scienceqa_rows": len(jev),
            "jevbench_rows_confirmed_test_split": sum(jev_checked.values()),
        },
        "files": {"confirm.jsonl": {"rows": len(rows), "sha256": sha256(os.path.join(out, "confirm.jsonl"))},
                  "confirm_mism.jsonl": {"rows": len(mism), "sha256": sha256(os.path.join(out, "confirm_mism.jsonl"))},
                  "EXCLUSIONS.json": {"rows": len(exclusions)}},
        "images": {os.path.basename(f): sha256(f) for f in sorted(glob.glob(os.path.join(out, "images", "*.png")))},
        "options_per_question": {k: dict(Counter(len(s["questions"]["q"]["criteria"]) for s in rows if s["kind"] == k))
                    for k in SUBSETS},
    }
    with open(os.path.join(out, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
    summary = {k: manifest[k] for k in ("counts", "files")}
    summary["histogram"] = manifest["isolation"]["nearest_distance_histogram"]
    summary["coverage"] = {k: v for k, v in manifest["coverage"].items() if k != "data_jsonl_list"}
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
