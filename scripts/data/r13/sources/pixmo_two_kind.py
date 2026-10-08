"""Round 13 source: PixMo-Points two-kind counting (one web picture, two different counted things: ask one, the
other's count is among the options). Licence ODC-BY 1.0. Plan 1.4.

    uv run python scripts/data/r13/sources/pixmo_two_kind.py --root data [--limit 15000] [--fetch-minutes 40]

Source: allenai/pixmo-points (train, Hugging Face cache data/raw/hf), **counting collection only** (people pointed
at every instance; `count` = number of points). Per picture, the usable labels are those that
  - clean into a noun phrase with ext/pixmo_points.phrase (qualifiers kept verbatim: "shoes on top rack" stays,
    never shortened to "shoes"; a short singular phrase gets its last word pluralised);
  - have count > 0 and exactly `count` points (an empty count / empty points is not "absent": skipped);
  - are not asked twice with different counts (such phrases dropped).
A question needs two usable labels A and B on the picture with different counts and A's count in 4..10 (the
counts this collection has start at 4): "How many <A> are there?" choice among A's count, **B's count**, and
neighbours of A's count (within 3, >= 1); 3-5 options. B prefers a label sharing a word with A ("red cards" vs
"cards": the qualifier decides). When B's count is also in 4..10 the picture is asked both ways (A with B's
count, B with A's count). Labels that are synonyms with different counts cannot be detected and are a known
source of noise (see the spot check).
Pictures: the extension pack's PixMo-Points cache (data/ext/pixmo_points/images, dead links in
data/raw/ext/pixmo_points/dead.json) first, then new downloads with ext/pixmo_count.get_pictures (rate-limit
aware) after a first pass with a hard 60 s limit per picture (common.fetch_pictures), bounded by --fetch-minutes; the bytes must match the dataset's SHA256 and the
picture must not be within dHash 2 of a held-out photo; PixMo-Count validation / test pictures are excluded.
"""
import hashlib
import os
import random
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from base import (Identity, already_built, base_args, choice, contact_sheet, describe, finish, hf_files,  # noqa: E402
                  spot_dir)
from pixmo_count import benchmark_pictures, get_pictures, picture_rel  # noqa: E402
from pixmo_points import phrase  # noqa: E402

SOURCE = "pixmo_two_kind"
CACHE = "pixmo_points"           # the extension pack's picture cache for this dataset
REPO = "allenai/pixmo-points"
LOW, HIGH = 4, 10
QUANTIFIERS = {"several", "many", "some", "few", "multiple", "various", "all", "number", "numbers", "total", "amount",
               "examples", "types", "kinds", "different", "individual", "visible", "distinct"}
# looked at 30 questions of the first build (pixmo_two_kind_1/2.png): 12 clearly right; 14 not checkable at sheet
# size (many small things: "purple circles", "tabs", "pointy ends of tools", "parking spots"); 4 wrong or odd
# ("book pages" = 5 for 5 books, "blues", "How many several examples of barrista art", closed-toe 10 + open-toed
# 6 shoes on a sheet of 15). Quantifier-led labels are dropped since then.
SPOT = (30, 12, 14, 4)


def usable_labels(rows: List[dict], dropped: Counter) -> Dict[str, Tuple[str, int]]:
    """phrase (lower case) -> (phrase as written, count) of a picture's counting rows."""
    by: Dict[str, set] = defaultdict(set)
    shown: Dict[str, str] = {}
    for r in rows:
        c = r["count"]
        if not c or c <= 0 or not r["points"]:
            dropped["label with an empty count / no points"] += 1
            continue
        if len(r["points"]) != c:
            dropped["label whose points differ from its count"] += 1
            continue
        name = phrase(r["label"])
        if name and name.split()[0].lower() in QUANTIFIERS:
            dropped["label starts with a quantifier ('several examples of ...')"] += 1
            continue
        if not name:
            dropped["label not a clean noun phrase"] += 1
            continue
        by[name.lower()].add(c)
        shown.setdefault(name.lower(), name)
    out = {}
    for k, v in by.items():
        if len(v) == 1:
            out[k] = (shown[k], v.pop())
        else:
            dropped["phrase asked twice with different counts"] += 1
    return out


def pairs(labels: Dict[str, Tuple[str, int]], rng: random.Random) -> List[Tuple[Tuple[str, int], Tuple[str, int]]]:
    """[(A, B)]: at most one pair per picture, asked both ways when both counts are in range."""
    items = sorted(labels.values())
    best = None
    for a in items:
        if not LOW <= a[1] <= HIGH:
            continue
        for b in items:
            if b[1] == a[1] or b[0].lower() == a[0].lower():
                continue
            shared = bool(set(a[0].lower().split()) & set(b[0].lower().split()))
            key = (not shared, rng.random())
            if best is None or key < best[0]:
                best = (key, a, b)
    if best is None:
        return []
    _, a, b = best
    return [(a, b), (b, a)] if LOW <= b[1] <= HIGH else [(a, b)]


def options(n: int, other: int, rng: random.Random) -> List[str]:
    k = rng.choice([3, 4, 4, 5])
    pool = [m for m in range(max(1, n - 3), n + 4) if m not in (n, other)]
    rng.shuffle(pool)
    return [str(n), str(other)] + [str(m) for m in pool[: k - 2]]


def main(argv=None):
    p = base_args(__doc__)
    p.add_argument("--limit", type=int, default=15000)
    p.add_argument("--fetch-minutes", type=float, default=40)
    p.add_argument("--workers", type=int, default=64)
    a = p.parse_args(argv)
    if already_built(a.root, "r13/%s/train.jsonl" % SOURCE, a.force):
        return
    import pyarrow.parquet as pq
    rng = random.Random(a.seed)
    steps, dropped = Counter(), Counter()
    by_url: Dict[str, list] = defaultdict(list)
    sha_of: Dict[str, str] = {}
    for f in hf_files(a.root, REPO, "data/train-*.parquet"):
        t = pq.read_table(f, columns=["image_url", "image_sha256", "count", "label", "collection_method",
                                      "points"]).to_pydict()
        steps["rows_read"] += len(t["image_url"])
        for u, s, c, lab, m, pts in zip(t["image_url"], t["image_sha256"], t["count"], t["label"],
                                        t["collection_method"], t["points"]):
            if m != "counting":
                continue
            steps["counting_rows"] += 1
            by_url[u].append({"label": lab, "count": c, "points": pts})
            sha_of[u] = s
    steps["counting_pictures"] = len(by_url)
    bench_urls, bench_shas = benchmark_pictures(a.root)
    first: Dict[str, str] = {}
    for u in sorted(by_url):
        first.setdefault(sha_of[u], u)
    plans = {}
    for s, u in first.items():
        if u in bench_urls or s in bench_shas:
            dropped["picture in PixMo-Count validation / test"] += 1
            continue
        pr = pairs(usable_labels(by_url[u], dropped), random.Random(_h(u)))
        if pr:
            plans[u] = pr
    steps["pictures_with_two_kinds"] = len(plans)

    # pictures already in the cache first, then new ones in a fixed order
    on_disk = [u for u in plans if os.path.exists(os.path.join(a.root, picture_rel(CACHE, u, sha_of[u])))]
    rest = sorted((u for u in plans if u not in set(on_disk)), key=_h)
    need = int(a.limit * 1.1)          # about 1.33 questions per usable picture, about 75% of pictures usable
    chosen = sorted(on_disk, key=_h)[:need]
    chosen += rest[: max(0, need - len(chosen))]
    steps["pictures_on_disk_before"] = len(on_disk)
    steps["pictures_tried"] = len(chosen)
    prefetch(a.root, [(u, sha_of[u]) for u in chosen], a.workers)
    ok, stats = get_pictures(a.root, CACHE, [(u, sha_of[u]) for u in chosen], a.workers, require_sha=True,
                             minutes=a.fetch_minutes)
    ident = Identity(a.root)
    near = ident.near_held(ok, CACHE)
    steps["pictures_usable"] = len(ok)
    dropped["picture within dHash 2 of a held-out photo"] = len(near)
    rows = []
    for u in sorted(chosen, key=_h):
        rel = picture_rel(CACHE, u, sha_of[u])
        if rel not in ok or rel in near:
            continue
        key = sha_of[u][:16]
        iid = "pixmo:%s" % sha_of[u]
        for k, (qa, qb) in enumerate(plans[u]):
            if len(rows) >= a.limit:
                break
            rows.append(choice("r13-pixmo2:%s:%d" % (key, k), SOURCE, "count_two_kind", iid, rel,
                               "How many %s are there?" % qa[0], options(qa[1], qb[1], rng), str(qa[1]), rng,
                               group=ident.canonical(iid), other_label=qb[0], other_count=qb[1],
                               shares_word=bool(set(qa[0].lower().split()) & set(qb[0].lower().split()))))
    steps["questions"] = len(rows)
    sheets = spot(a.root, rows)
    finish(a.root, SOURCE, rows, dict(steps), dict(dropped), {
        "licence": "ODC-BY 1.0", "picture_fetch": stats, "identity_backend": ident.backend,
        "limit": a.limit, "limit_reached": len(rows) >= a.limit,
        "count_answers": dict(sorted(Counter(r["answer_key"] for r in rows).items(), key=lambda kv: int(kv[0]))),
        "shares_word": sum(r["shares_word"] for r in rows), "spot_sheets": sheets},
        report=REPORT.format(n=len(rows), steps=dict(steps), dropped=dict(dropped), stats=stats, sheets=sheets,
                             spot=SPOT))


REPORT = """# PixMo-Points two-kind counting (round 13)

Built by `scripts/data/r13/sources/pixmo_two_kind.py`; counts in MANIFEST.json.

- Kept {n} questions. Steps: {steps}
- Dropped per rule: {dropped}
- Picture fetch: {stats}
- Spot check sheets: {sheets}
- Spot check of the first build (looked at, clearly right, not checkable at sheet size, wrong / odd): {spot}.
  Label noise is real (counts by crowd workers, synonyms with different counts cannot be detected); the
  quantifier filter was added after it.
"""


def prefetch(root: str, items: List[Tuple[str, str]], workers: int) -> None:
    """Download the pictures not on disk with a hard per-picture limit (common.fetch_pictures: 60 s per picture;
    pixmo_count's polite fetcher has none, and a server that drips bytes held a run for over 10 minutes). Failures
    are added to the cache's dead list, so get_pictures does not try them again."""
    import json
    from base import fetch_pictures
    dead_path = os.path.join(root, "raw", "ext", CACHE, "dead.json")
    dead = set(json.load(open(dead_path))) if os.path.exists(dead_path) else set()
    todo = [(u, s) for u, s in items if u not in dead and not os.path.exists(os.path.join(root, picture_rel(CACHE, u, s)))]
    if not todo:
        return
    rel = {u: picture_rel(CACHE, u, s) for u, s in todo}
    failed = fetch_pictures(root, todo, lambda it: it[0], lambda it: rel[it[0]], "r13 pixmo", workers)
    inv = {r: u for u, r in rel.items()}
    json.dump(sorted(dead | {inv[r] for r in failed}), open(dead_path, "w"))


def _h(x) -> str:
    return hashlib.sha1(("%s%s" % (SOURCE, x)).encode()).hexdigest()


def spot(root: str, rows: List[dict]) -> List[str]:
    rng = random.Random(1)
    pick = rng.sample(rows, min(30, len(rows)))
    paths = []
    for k in range(0, len(pick), 15):
        path = os.path.join(spot_dir(), "%s_%d.png" % (SOURCE, k // 15 + 1))
        contact_sheet(root, [{"image": r["image"], "text": describe(r) + " | other: %s = %d" % (
            r["other_label"], r["other_count"])} for r in pick[k:k + 15]], path, cols=5)
        paths.append(path)
    return paths


if __name__ == "__main__":
    main()
