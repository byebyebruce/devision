"""Public-data pack, Amazon Berkeley Objects (ABO, CC BY 4.0): catalog photos of products with listing metadata.

    uv run python scripts/data/pub/abo.py --root data

Downloads abo-images-small.tar (~3.3 GB, pictures at most 256 px -- our input size) and abo-listings.tar (~83 MB),
both resumable, and extracts the small pictures once. Per product (its main picture), only what the photo shows:
  - colour: the English colour value names exactly one basic colour (black, white, red, blue, green, yellow,
    orange, pink, purple, brown, grey) and nothing else ("Black/Red", "Multicolor", "Navy Blue" dropped) ->
    "What color is this <product>?" among it and 3 other basic colours;
  - kind: the product type is one of the frequent, plainly named types -> "What kind of product is this?" among it
    and 3 other such types.
Products are the unit: a product's pictures stay on one side of the train / dev split. Catalog product shots are
not near any held-out photo by construction (studio pictures of products); the photo check is still run.
"""
import csv
import gzip
import io
import json
import os
import random
import re
import sys
import tarfile
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ext"))
sys.path.insert(0, HERE)
from common import already_built, base_args, choice, download, finish, raw_dir  # noqa: E402
from held import get_held, split_dev  # noqa: E402

SOURCE = "abo"
BASE = "https://amazon-berkeley-objects.s3.amazonaws.com/archives/"
LICENCE = "CC BY 4.0 (Amazon Berkeley Objects)"
COLOURS = ("black", "white", "red", "blue", "green", "yellow", "orange", "pink", "purple", "brown", "grey")
MIN_TYPE = 300
BAD_TYPE = re.compile(r"HOME|ACCESSORY|ACCESSORIES|OTHER|PRODUCT|SUPPLIES|ABIS|_KIT|MISC")


def basic_colour(values):
    """The one basic colour an English colour value names, else None."""
    for v in values:
        words = re.findall(r"[a-z]+", v.lower().replace("gray", "grey"))
        hits = [w for w in words if w in COLOURS]
        if len(words) == 1 and len(hits) == 1:
            return hits[0]
    return None


def type_name(t: str) -> str:
    return t.lower().replace("_", " ")


def english(field):
    return [x["value"] for x in field or [] if str(x.get("language_tag", "")).startswith("en")]


def extract_small(tar_path: str, out_root: str) -> None:
    """Extract the pictures and the image metadata once (existing files skipped; an interrupted run continues)."""
    marker = os.path.join(out_root, ".extracted")
    if os.path.exists(marker):
        return
    with tarfile.open(tar_path) as t:
        for m in t:
            if not m.isfile():
                continue
            dest = os.path.join(out_root, m.name)
            if os.path.exists(dest):
                continue
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with t.extractfile(m) as src, open(dest + ".tmp", "wb") as f:
                f.write(src.read())
            os.replace(dest + ".tmp", dest)
    open(marker, "w").close()


def main(argv=None):
    p = base_args(__doc__)
    a = p.parse_args(argv)
    if already_built(a.root, SOURCE, a.force):
        return
    rng = random.Random(a.seed)
    raw = raw_dir(a.root, SOURCE)
    small = download(BASE + "abo-images-small.tar", os.path.join(raw, "abo-images-small.tar"))
    listings = download(BASE + "abo-listings.tar", os.path.join(raw, "abo-listings.tar"))
    pics = os.path.join(a.root, "ext", SOURCE)
    extract_small(small, pics)
    with gzip.open(os.path.join(pics, "images", "metadata", "images.csv.gz"), "rt") as f:
        path_of = {r["image_id"]: r["path"] for r in csv.DictReader(f)}
    items, steps = [], Counter()
    with tarfile.open(listings) as t:
        for m in t:
            if m.name.endswith(".json.gz"):
                with gzip.open(io.BytesIO(t.extractfile(m).read()), "rt") as f:
                    for line in f:
                        d = json.loads(line)
                        steps["listings"] += 1
                        img = d.get("main_image_id")
                        if not img or img not in path_of:
                            continue
                        ptype = (d.get("product_type") or [{}])[0].get("value", "")
                        items.append((d["item_id"], img, ptype, basic_colour(english(d.get("color")))))
    seen, uniq = set(), []
    for it in sorted(items):            # a product listed in several marketplaces: once
        if it[0] not in seen:
            seen.add(it[0])
            uniq.append(it)
    steps["products_with_a_picture"] = len(uniq)
    types = Counter(it[2] for it in uniq)
    good_types = sorted(t for t, n in types.items() if n >= MIN_TYPE and t and not BAD_TYPE.search(t))
    held = get_held(a.root)
    rels = {it[1]: os.path.join("ext", SOURCE, "images", "small", path_of[it[1]]) for it in uniq}
    near = held.near_held([rels[it[1]] for it in uniq if it[3] or it[2] in good_types], SOURCE)
    rows = []
    for item, img, ptype, colour in uniq:
        rel = rels[img]
        if rel in near:
            steps["near a held-out photo"] += 1
            continue
        image_id = "abo:%s" % img
        noun = type_name(ptype) if ptype in good_types else "product"
        if colour:
            others = [c for c in COLOURS if c != colour]
            rng.shuffle(others)
            rows.append(choice("pub-abo:%s:colour" % item, SOURCE, "colour", image_id, rel,
                               "What color is this %s?" % noun, [colour] + others[:3], colour, rng, group=item))
        if ptype in good_types:
            others = [type_name(x) for x in good_types if x != ptype]
            rng.shuffle(others)
            rows.append(choice("pub-abo:%s:kind" % item, SOURCE, "kind", image_id, rel, "What kind of product is this?",
                               [type_name(ptype)] + others[:3], type_name(ptype), rng, group=item))
    steps["questions"] = len(rows)
    rows, steps["dev_questions"] = split_dev(a.root, SOURCE, rows, key=lambda r: r["group"])
    finish(a.root, SOURCE, rows, dict(steps), {"licence": LICENCE, "types": [type_name(t) for t in good_types]})


if __name__ == "__main__":
    main()
