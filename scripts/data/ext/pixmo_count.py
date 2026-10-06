"""Extension pack, PixMo-Count (web photos from Flickr; one object class per picture with its count, found by the
Detic detector and filtered by AI2 -- the train split is not human-verified, only its val / test are). Licence
ODC-BY 1.0.

    uv run python scripts/data/ext/pixmo_count.py --root data [--workers 4] [--download-minutes 120] [--force]

Flickr throttles downloads (HTTP 429 to most requests after a burst of a few thousand), so a run downloads for at
most --download-minutes and builds train.jsonl from the pictures it has; `--force` later fetches more (pictures
already there and links known dead are skipped) and rebuilds. MANIFEST "download" says how many are still missing.

Source: allenai/pixmo-count, train split only (its validation / test splits are counting benchmarks: never used,
and train rows whose picture -- URL or SHA256 -- is also in them are dropped). Each row: picture URL + SHA256,
an Objects365 class name (naively pluralised, e.g. "butterflys", "bowl/basins"; cleaned by `plural`, unclear ones
dropped) and the count. Per picture (one row each):
  - count 0..10 -> "How many <label> are there?" choice with nearby numbers (common.count_options); counts > 10
    are dropped; counts flattened (no answer above twice the mean);
  - "Are there any <label> in the image?" noul, yes from count > 0, no from count 0, balanced yes = no per label
    (otherwise the label gives the answer away: "people" is almost always present).
Pictures are downloaded once to data/ext/pixmo_count/images/<sha256>.<ext>; links found dead are remembered in
data/raw/ext/pixmo_count/dead.json and not retried. Flickr re-encodes its files, so the downloaded bytes almost
never match the dataset's SHA256: a mismatch is counted, not dropped; instead a picture is dropped when it does not
decode, when the same bytes come back for several dataset pictures (a "photo unavailable" placeholder), when a labelled point lies
outside it (another picture or size), or when it is within NEAR_BITS of a held-out photo (common.HeldOut.near_held).
"""
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (HeldOut, already_built, balance_yes_no, base_args, choice, count_options,  # noqa: E402
                    finish, flatten_answers, hf_files, noul, raw_dir)

SOURCE = "pixmo_count"
REPO = "allenai/pixmo-count"
LICENCE = "ODC-BY 1.0 (pictures: their web owners' terms)"
PLACEHOLDERS = 3      # identical bytes behind this many different dataset pictures = a placeholder picture

# ---------------------------------------------------------------- labels

# PixMo-Count's labels are Objects365 class names with an "s" appended; the ones that do not read as a plural noun
_FIX = {
    "butterflys": "butterflies", "strawberrys": "strawberries", "cherrys": "cherries", "candys": "candies",
    "trophys": "trophies", "toiletrys": "toiletries", "ballons": "balloons", "noddles": "noodles", "bus": "buses",
    "wine glass": "wine glasses", "trash bin cans": "trash cans", "head phones": "headphones", "suvs": "SUVs",
    "cds": "CDs", "canneds": "cans", "soccers": "soccer balls", "curlings": "curling stones",
    "billards": "billiard balls", "tennis": "tennis balls", "table tennis": "table tennis balls",
    "formula 1s": "Formula 1 cars", "cues": "cue sticks", "skiboards": "skis", "corns": "ears of corn",
    "garlics": "garlic bulbs", "okras": "okra pods", "baozis": "baozi", "sushis": "pieces of sushi",
    "kiwi fruits": "kiwis", "broccolis": "heads of broccoli", "extractors": "range hoods",
    "computer boxes": "computer towers", "machinery vehicles": "construction vehicles",
    "lifesavers": "life buoys", "poker cards": "playing cards", "notepapers": "sticky notes",
    "skating and skiing shoes": "skates or ski boots", "bowl/basins": "bowls", "barrel/buckets": "barrels or buckets",
    "cabinet/shelves": "cabinets or shelves", "pen/pencils": "pens or pencils", "monitor/tvs": "monitors or TVs",
    "cigar/cigarettes": "cigars or cigarettes", "handbag/satchels": "handbags", "picture/frames": "picture frames",
    "router/modems": "routers or modems", "wallet/purses": "wallets or purses",
    "orange/tangerines": "oranges or tangerines", "cutting/chopping boards": "cutting boards",
    "blackboard/whiteboards": "blackboards or whiteboards", "tape measure/ rulers": "tape measures or rulers",
    "washing machine/drying machines": "washing machines or dryers",
    "cosmetics brush/eyeliner pencils": "makeup brushes or eyeliner pencils",
}
# no clear plural noun for these (French horn or fries? which "other" balls? rice / pasta are not counted in units)
_DROP = {"frenches", "other balls", "other fish", "other shoes", "converters", "rices", "pastas"}


def plural(label: str) -> Optional[str]:
    """A PixMo-Count class name as a plural noun phrase for "How many ... are there?", or None when unclear."""
    k = " ".join(label.split()).lower()
    if k in _DROP or not k:
        return None
    return _FIX.get(k, k)


# ---------------------------------------------------------------- conversion

def convert(row: dict, image: str, rng: random.Random) -> Tuple[List[dict], str]:
    """One PixMo-Count row -> (samples, reason when none). The exist question comes with every row and is
    thinned by the per-label yes = no balance later."""
    name = plural(row["label"])
    if name is None:
        return [], "unclear label"
    n = int(row["count"])
    if n > 10:
        return [], "count > 10"
    key = row["image_sha256"][:16]
    image_id = "%s:%s" % (SOURCE, key)
    sid = "ext-%s:%s:%s" % (SOURCE, key, re.sub(r"[^a-z0-9]+", "-", name.lower()))
    out = [choice(sid + ":count", SOURCE, "count", image_id, image,
                  "How many %s are there?" % name, count_options(n, rng), str(n), rng, group="count"),
           noul(sid + ":exist", SOURCE, "exist", image_id, image,
                "Are there any %s in the image?" % name, n > 0, group="exist:%s" % name)]
    return out, ""


# ---------------------------------------------------------------- pictures (also used by pixmo_points)

_EXT = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp")


def picture_rel(source: str, url: str, sha: str) -> str:
    ext = os.path.splitext(url.split("?")[0])[1].lower()
    return os.path.join("ext", source, "images", sha + (ext if ext in _EXT else ".jpg"))


def _check(path: str) -> list:
    """[bytes sha256, width, height] of a picture file, or ["bad", reason] when it does not decode."""
    from PIL import Image
    try:
        with Image.open(path) as im:
            im.load()
            w, h = im.size
    except Exception as e:      # truncated, HTML error page, SVG, decompression bomb ...
        return ["bad", type(e).__name__]
    if min(w, h) < 32:
        return ["bad", "tiny"]
    h256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h256.update(chunk)
    return [h256.hexdigest(), w, h]


def fetch_politely(root: str, todo: List[Tuple[str, str]], rel: Dict[str, str], source: str, workers: int,
                   dead_path: str, dead: set, minutes: float = 0) -> Tuple[set, int]:
    """Download `todo` pictures like common.fetch_pictures, but rate-limit aware (Flickr answers 429 to a burst):
    a 429 / 503 pauses that host (Retry-After, else 15 s doubling up to 5 min) and the picture is tried again (up to
    8 times; still limited -> left for the next run, not dead). A 4xx answer, 3 server errors or 2 failed
    connections = dead.
    `minutes` > 0 bounds the run: pictures not fetched by then are left for the next run as well.
    Returns (dead URLs, pictures left for a re-run); dead.json is updated every ~1000 pictures."""
    import threading
    import urllib.error
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from common import save_bytes
    lock = threading.Lock()
    pause: Dict[str, float] = defaultdict(float)
    strikes: Counter = Counter()
    deadline = time.time() + 60 * minutes if minutes > 0 else float("inf")

    def one(url: str) -> str:
        host = url.split("/")[2] if "//" in url else url
        errors = 0
        for _ in range(8):
            wait = pause[host] - time.time()
            if time.time() + max(0.0, wait) > deadline:
                return "limited"
            if wait > 0:
                time.sleep(wait)
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=20) as r:
                    data = r.read()
                save_bytes(os.path.join(root, rel[url]), data)
                with lock:
                    strikes[host] = 0
                return "ok"
            except urllib.error.HTTPError as e:
                if e.code in (429, 503):
                    with lock:
                        strikes[host] += 1
                        after = e.headers.get("Retry-After") if e.headers else None
                        delay = float(after) if after and after.isdigit() else min(300.0, 15.0 * 2 ** (strikes[host] - 1))
                        pause[host] = max(pause[host], time.time() + delay)
                    continue
                if e.code == 408 or e.code >= 500:
                    errors += 1
                    if errors < 3:
                        continue
                return "dead"
            except Exception:       # DNS, refused, timeout, reset, truncated read
                errors += 1
                if errors >= 2:
                    return "dead"
                time.sleep(2)
        return "limited"

    died, limited = set(), 0
    total, done, start, last = len(todo), 0, time.time(), time.time()
    print("%s pictures: %d to download" % (source, total), flush=True)
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(one, u): u for u, _ in todo}
        for f in as_completed(futures):
            done += 1
            r = f.result()
            if r == "dead":
                died.add(futures[f])
            limited += r == "limited"
            now = time.time()
            if done % 1000 == 0 or done == total:
                json.dump(sorted(dead | died), open(dead_path, "w"))
            if done == total or done % max(1, total // 20) == 0 or now - last >= 60:
                last = now
                print("  %s pictures: %d / %d (%d dead, %d rate limited), %.0f s" % (source, done, total, len(died),
                      limited, now - start), flush=True)
    return died, limited


def get_pictures(root: str, source: str, items: Iterable[Tuple[str, str]], workers: int = 32,
                 require_sha: bool = False, minutes: float = 0) -> Tuple[Dict[str, Tuple[int, int]], dict]:
    """Download the pictures of (url, dataset sha256) pairs. Returns ({rel: (width, height)} of the usable ones,
    stats). `require_sha`: drop a picture whose bytes differ from the dataset's SHA256 (else only counted).
    Dead links are remembered in raw/ext/<source>/dead.json, per-file checks in checks.json (both only added to),
    so a re-run downloads and decodes nothing twice."""
    items = sorted(set(items))
    raw = raw_dir(root, source)
    dead_path, check_path = os.path.join(raw, "dead.json"), os.path.join(raw, "checks.json")
    dead = set(json.load(open(dead_path))) if os.path.exists(dead_path) else set()
    checks = json.load(open(check_path)) if os.path.exists(check_path) else {}
    rel = {u: picture_rel(source, u, s) for u, s in items}
    todo = [(u, s) for u, s in items if u not in dead and not os.path.exists(os.path.join(root, rel[u]))]
    start = time.time()
    deferred = 0
    if todo:
        died, deferred = fetch_politely(root, todo, rel, source, workers, dead_path, dead, minutes)
        dead |= died
    seconds = time.time() - start
    from concurrent.futures import ProcessPoolExecutor
    new = [r for u, r in rel.items() if u not in dead and r not in checks and os.path.exists(os.path.join(root, r))]
    if new:
        with ProcessPoolExecutor(8) as ex:
            for r, c in zip(new, ex.map(_check, [os.path.join(root, r) for r in new], chunksize=64)):
                checks[r] = c
        json.dump(checks, open(check_path, "w"))
    stats: Counter = Counter()
    by_bytes: Dict[str, set] = defaultdict(set)      # downloaded bytes -> the dataset pictures (SHA256s) behind them
    for u, s in items:
        if u not in dead and rel[u] in checks and checks[rel[u]][0] != "bad":
            by_bytes[checks[rel[u]][0]].add(s)
    ok = {}
    for u, s in items:
        if u in dead:
            stats["dead link"] += 1
            continue
        if rel[u] not in checks:
            stats["not downloaded yet (rate limited / time limit)"] += 1
            continue
        c = checks[rel[u]]
        if c[0] == "bad":
            stats["not a picture (%s)" % c[1]] += 1
        elif len(by_bytes[c[0]]) >= PLACEHOLDERS:
            stats["placeholder (same bytes for %d+ dataset pictures)" % PLACEHOLDERS] += 1
        elif require_sha and c[0] != s:
            stats["bytes differ from the dataset SHA256"] += 1
        else:
            stats["usable"] += 1
            stats["usable, bytes match the dataset SHA256"] += c[0] == s
            ok[rel[u]] = (c[1], c[2])
    size = sum(os.path.getsize(os.path.join(root, r)) for r in ok)
    return ok, {"pictures": len(items), **dict(stats), "usable_gb": round(size / 1e9, 2),
                "download_seconds_this_run": round(seconds),
                "rate_limited_left_for_a_rerun": deferred, "dead_rate": round(stats["dead link"] / max(1, len(items)), 4)}


def points_inside(points: dict, width: int, height: int, slack: float = 1.02) -> bool:
    """PixMo-Count's points are pixels of the annotated picture: all inside the downloaded one (2% slack) is a
    weak check that it is the same picture at the same size, since Flickr's bytes no longer match the SHA256."""
    xs, ys = points.get("x") or [], points.get("y") or []
    return all(0 <= x <= width * slack for x in xs) and all(0 <= y <= height * slack for y in ys)


def benchmark_pictures(root: str) -> Tuple[set, set]:
    """URLs and SHA256s of the pictures of PixMo-Count's validation / test splits (Molmo's counting benchmarks)."""
    import pyarrow.parquet as pq
    urls, shas = set(), set()
    for split in ("validation", "test"):
        for p in hf_files(root, REPO, "data/%s-*.parquet" % split):
            t = pq.read_table(p, columns=["image_url", "image_sha256"]).to_pydict()
            urls |= set(t["image_url"])
            shas |= set(t["image_sha256"])
    return urls, shas


def main(argv=None):
    p = base_args(__doc__)
    p.add_argument("--workers", type=int, default=4, help="Flickr throttles (HTTP 429) a burst of downloads")
    p.add_argument("--download-minutes", type=float, default=120,
                   help="stop downloading after this long; a re-run with --force fetches the rest (0: no limit)")
    a = p.parse_args(argv)
    if already_built(a.root, SOURCE, a.force):
        return
    import pyarrow.parquet as pq
    rng = random.Random(a.seed)
    rows = [r for f in hf_files(a.root, REPO, "data/train-*.parquet")
            for r in pq.read_table(f, columns=["image_url", "image_sha256", "count", "label", "points"]).to_pylist()]
    steps, dropped = Counter(rows=len(rows)), Counter()
    bench_urls, bench_shas = benchmark_pictures(a.root)
    keep, seen = [], set()
    for r in rows:
        if (r["image_sha256"], plural(r["label"])) in seen:
            dropped["same picture and label again"] += 1
        elif r["image_url"] in bench_urls or r["image_sha256"] in bench_shas:
            dropped["picture in pixmo-count validation / test"] += 1
        elif plural(r["label"]) is None:
            dropped["unclear label"] += 1
        elif r["count"] > 10:
            dropped["count > 10"] += 1
        else:
            keep.append(r)
            seen.add((r["image_sha256"], plural(r["label"])))
    steps["usable_rows"] = len(keep)
    ok, pictures = get_pictures(a.root, SOURCE, [(r["image_url"], r["image_sha256"]) for r in keep], a.workers,
                                minutes=a.download_minutes)
    near = HeldOut(a.root).near_held(ok, SOURCE)
    steps["pictures_downloaded_usable"] = len(ok)
    steps["pictures_near_held_out"] = len(near)
    out = []
    for r in keep:
        rel = picture_rel(SOURCE, r["image_url"], r["image_sha256"])
        if rel not in ok:
            dropped["picture unusable"] += 1
            continue
        if rel in near:
            dropped["near a held-out photo"] += 1
            continue
        if not points_inside(r["points"], *ok[rel]):
            dropped["points outside the downloaded picture"] += 1
            continue
        samples, _ = convert(r, rel, rng)
        out += samples
    steps["questions"] = len(out)
    out = balance_yes_no(out)
    steps["after_yes_no_balance"] = len(out)
    counts = flatten_answers([r for r in out if r["kind"] == "count"])
    out = [r for r in out if r["kind"] != "count"] + counts
    steps["after_count_flatten"] = len(out)
    finish(a.root, SOURCE, out, dict(steps), {
        "licence": LICENCE, "dropped_rows": dict(dropped), "download": pictures,
        "count_answers": dict(sorted(Counter(r["answer_key"] for r in counts).items(), key=lambda kv: int(kv[0]))),
        "exist_labels": len({r["group"] for r in out if r["kind"] == "exist"})})


if __name__ == "__main__":
    main()
