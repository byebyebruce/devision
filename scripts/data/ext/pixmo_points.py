"""Extension pack, PixMo-Points (web pictures of every kind; people asked to point at every instance of a phrase).
Licence ODC-BY 1.0.

    uv run python scripts/data/ext/pixmo_points.py --root data [--images 60000] [--per-picture 3] [--workers 32]

Source: allenai/pixmo-points (train only; 2.38M rows over 228k pictures). Rows come from two collections:
  - "pointing" (2.06M rows): a free referring expression and its points, no count -- the points need not cover
    every instance and a row without points does not say the thing is absent ("sky", "water"), so not used;
  - "counting" (319k rows over 72.8k pictures): a phrase, all its instances pointed and `count` = number of points;
    count 0 means the annotators marked the thing as absent. Counts are 0 or >= 4 (the collection targets
    high-frequency objects: no 1-3).
Only counting rows are used. A deterministic sample of --images pictures (by URL hash) among those with a usable
row; per picture at most --per-picture questions, one per phrase:
  - count 4..10 -> "How many <phrase> are there?" choice among nearby counts that can occur (4..10, `options`):
    offering 1-3, which never occur, would let a model rule them out without looking; counts flattened;
  - "Are there any <phrase> in the image?" noul: yes from count > 0 (any count), no from count 0; balanced
    yes = no per phrase, so the phrase does not give the answer away (absent phrases are more specific).
Phrases are cleaned by `phrase` (plural noun phrase, at most 6 words, plain characters, not starting with a number;
others dropped).
Pictures are downloaded once (pixmo_count.get_pictures) to data/ext/pixmo_points/images/<sha256>.<ext>; a picture
is dropped when its link is dead, it does not decode, its bytes differ from the dataset's SHA256 (the page changed),
or it is within NEAR_BITS of a held-out photo. Pictures of PixMo-Count's validation / test splits are excluded.
"""
import hashlib
import os
import random
import re
import sys
from collections import Counter, defaultdict
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (HeldOut, already_built, balance_yes_no, base_args, choice, finish,  # noqa: E402
                    flatten_answers, hf_files, noul)
from pixmo_count import LICENCE, benchmark_pictures, get_pictures, picture_rel  # noqa: E402

SOURCE = "pixmo_points"
REPO = "allenai/pixmo-points"
SUPPORT = list(range(4, 11))        # the counts 1..10 this collection has

# ---------------------------------------------------------------- phrases

_IRREGULAR = {"person": "people", "man": "men", "woman": "women", "child": "children", "foot": "feet",
              "tooth": "teeth", "mouse": "mice", "goose": "geese", "leaf": "leaves", "knife": "knives",
              "wife": "wives", "life": "lives", "shelf": "shelves", "loaf": "loaves", "half": "halves",
              "wolf": "wolves", "calf": "calves", "potato": "potatoes", "tomato": "tomatoes", "hero": "heroes",
              "cactus": "cacti", "bus": "buses", "glass": "glasses", "dress": "dresses", "box": "boxes"}
_SAME = {"sheep", "deer", "fish", "people", "men", "women", "children", "feet", "teeth", "mice", "geese", "cattle",
         "police", "glasses", "pants", "jeans", "scissors", "clothes", "shorts", "sunglasses", "aircraft", "series",
         "species", "silverware", "cutlery", "furniture", "food", "hair", "luggage", "jewelry", "equipment"}
_FUNCTION = {"the", "of", "in", "on", "with", "and", "or", "a", "an", "to", "at", "by", "for", "from", "that",
             "which", "who", "is", "are", "this", "these", "those", "under", "behind", "near", "without"}
_MASS = {"food", "hair", "silverware", "cutlery", "furniture", "luggage", "jewelry", "equipment", "water", "sky",
         "grass", "text", "writing"}
_CHARS = re.compile(r"^[A-Za-z0-9 '&$\-]+$")
_NUMBER = re.compile(r"^(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b", re.I)


def looks_plural(word: str) -> bool:
    w = word.lower()
    if "'" in w or w in _MASS:
        return False
    return w in _SAME or (len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")))


def pluralize(word: str) -> str:
    w = word.lower()
    if w in _IRREGULAR:
        return _IRREGULAR[w]
    if w in _SAME:
        return word
    if re.search(r"[^aeiou]y$", w):
        return word[:-1] + "ies"
    if w.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"
    return word + "s"


def _case(word: str) -> str:
    """Lower case, except short all-capital words (acronyms: TV, BT21)."""
    return word if word.isupper() and len(word) <= 4 and len(word) > 1 else word.lower()


def phrase(label: str) -> Optional[str]:
    """A PixMo-Points counting label as a plural noun phrase for "How many ... are there?", or None.
    A phrase with a plural-looking word is kept as written ("people wearing glasses", "gold coins"); a short
    singular phrase without function words gets its last word pluralised ("purple grape" -> "purple grapes");
    anything else (long, odd characters, a mass noun, a singular clause) is dropped."""
    text = " ".join(label.split()).strip(" .,")
    if not text or not _CHARS.match(text) or _NUMBER.match(text):     # "14 features" would give the count away
        return None
    words = [_case(w) for w in text.split()]
    if len(words) > 6 or not any(re.search(r"[a-z]", w.lower()) for w in words):
        return None
    if any(looks_plural(w) for w in words):
        return " ".join(words)
    if len(words) > 3 or any(w.lower() in _FUNCTION for w in words) or words[-1].lower() in _MASS \
            or not words[-1].isalpha():
        return None
    return " ".join(words[:-1] + [pluralize(words[-1])])


# ---------------------------------------------------------------- conversion

def options(n: int, rng: random.Random) -> List[str]:
    """`n` (4..10) and 1-4 distractors within 3 of it among the counts that occur (4..10)."""
    k = rng.choice([2, 3, 3, 4, 4, 5])
    pool = [m for m in SUPPORT if m != n and abs(m - n) <= 3]
    rng.shuffle(pool)
    return [str(n)] + [str(m) for m in pool[:k - 1]]


def picture_questions(key: str, image: str, rows: List[dict], cap: int, rng: random.Random) -> List[dict]:
    """The questions of one picture: one per phrase (duplicates of a phrase with different counts dropped), at most
    `cap`. A row with count 4..10 asks the count or, half the time, existence (yes); count 0 asks existence (no);
    count > 10 asks existence (yes)."""
    by: dict = defaultdict(set)
    for r in rows:
        name = phrase(r["label"])
        if name:
            by[name.lower()].add((name, int(r["count"])))
    usable = sorted(v.pop() for v in by.values() if len(v) == 1)
    rng.shuffle(usable)
    image_id = "%s:%s" % (SOURCE, key)
    out = []
    for i, (name, n) in enumerate(usable[:cap]):
        sid = "ext-%s:%s:%d" % (SOURCE, key, i)
        if 4 <= n <= 10 and rng.random() < 0.5:
            out.append(choice(sid, SOURCE, "count", image_id, image, "How many %s are there?" % name,
                              options(n, rng), str(n), rng, group="count"))
        else:
            out.append(noul(sid, SOURCE, "exist", image_id, image, "Are there any %s in the image?" % name, n > 0,
                            group="exist:%s" % name.lower()))
    return out


def sample_pictures(urls, n: int) -> List[str]:
    """`n` of `urls`, the same ones on every run (smallest SHA1 of the URL)."""
    return sorted(urls, key=lambda u: hashlib.sha1((SOURCE + u).encode()).hexdigest())[:n]


def main(argv=None):
    p = base_args(__doc__)
    p.add_argument("--images", type=int, default=60000, help="pictures to sample (before dead links)")
    p.add_argument("--per-picture", type=int, default=3)
    p.add_argument("--workers", type=int, default=64, help="downloads in parallel (over thousands of hosts)")
    a = p.parse_args(argv)
    if already_built(a.root, SOURCE, a.force):
        return
    import pyarrow.parquet as pq
    rng = random.Random(a.seed)
    steps, dropped = Counter(), Counter()
    by_url: dict = defaultdict(list)
    sha_of = {}
    for f in hf_files(a.root, REPO, "data/train-*.parquet"):
        t = pq.read_table(f, columns=["image_url", "image_sha256", "count", "label", "collection_method"]).to_pydict()
        steps["rows"] += len(t["image_url"])
        for u, s, c, lab, m in zip(t["image_url"], t["image_sha256"], t["count"], t["label"], t["collection_method"]):
            if m != "counting":
                continue
            steps["counting_rows"] += 1
            by_url[u].append({"label": lab, "count": c})
            sha_of[u] = s
    steps["counting_pictures"] = len(by_url)
    bench_urls, bench_shas = benchmark_pictures(a.root)
    usable = [u for u, rows in by_url.items() if any(phrase(r["label"]) for r in rows)]
    steps["pictures_with_a_usable_phrase"] = len(usable)
    bench = {u for u in usable if u in bench_urls or sha_of[u] in bench_shas}
    dropped["picture in pixmo-count validation / test"] = len(bench)
    first = {}
    for u in sorted(usable):          # one URL per picture (a few pictures sit behind two URLs)
        first.setdefault(sha_of[u], u)
    dropped["same picture behind another URL"] = len(usable) - len(first)
    chosen = sample_pictures([u for u in first.values() if u not in bench], a.images)
    steps["pictures_sampled"] = len(chosen)
    ok, pictures = get_pictures(a.root, SOURCE, [(u, sha_of[u]) for u in chosen], a.workers, require_sha=True)
    near = HeldOut(a.root).near_held(ok, SOURCE)
    steps["pictures_near_held_out"] = len(near)
    out = []
    for u in sorted(chosen):
        rel = picture_rel(SOURCE, u, sha_of[u])
        if rel not in ok or rel in near:
            continue
        steps["pictures_used"] += 1
        out += picture_questions(sha_of[u][:16], rel, by_url[u], a.per_picture, rng)
    steps["questions"] = len(out)
    out = balance_yes_no(out)
    steps["after_yes_no_balance"] = len(out)
    counts = flatten_answers([r for r in out if r["kind"] == "count"])
    out = [r for r in out if r["kind"] != "count"] + counts
    steps["after_count_flatten"] = len(out)
    finish(a.root, SOURCE, out, dict(steps), {
        "licence": LICENCE, "dropped": dict(dropped), "download": pictures,
        "count_answers": dict(sorted(Counter(r["answer_key"] for r in counts).items(), key=lambda kv: int(kv[0]))),
        "exist_phrases": len({r["group"] for r in out if r["kind"] == "exist"})})


if __name__ == "__main__":
    main()
