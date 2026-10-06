"""Extension pack, SpatialSense (Yang et al., ICCV 2019: spatial relations between two objects in Flickr and NYU
photos, crowdsourced adversarially so that the relation cannot be guessed from the words alone). Licence: Zenodo
record CC BY 2.0 (annotations); pictures Flickr (their owners' terms) / NYU Depth v2.

    uv run python scripts/data/ext/spatialsense.py --root data [--per-picture 3] [--archive] [--force]

Source: the original release, Zenodo record 8104370 (annotations.json, checked against the record's MD5; a short
read is resumed, see `fetch_checked`). Only the "train" split is used (the "valid" and "test" splits are left
alone). Pictures: the Flickr ones (6,820 of 7,645 train pictures) are fetched from their staticflickr URLs (a
deleted photo, served as a non-JPEG placeholder, is dropped); the NYU ones are only in the release's
images.tar.gz (1.8 GB, Zenodo serves it at ~15 KB/s), so they are used only with --archive, which downloads that
archive and takes every picture from it. Each (subject, predicate, object, label) triple -> a yes / no question
"Is the cup on the table?" (label True = yes). Balance: per predicate, as many yes as no; at most --per-picture
questions per picture. "to the left of" / "to the right of" are kind "relation-leftright" (the mixer caps
left / right questions), all others kind "relation". Every picture also passes `HeldOut.near_held` (photos that
are near-duplicates of a held-out photo are dropped).
"""
import hashlib
import json
import os
import random
import re
import sys
import tarfile
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (HeldOut, already_built, balance_yes_no, base_args, cap_per_picture, download,  # noqa: E402
                    fetch_pictures, finish, noul, out_dir, raw_dir)

SOURCE = "spatialsense"
ZENODO = "https://zenodo.org/api/records/8104370/files/%s/content"
MD5 = {"annotations.json": "36ca163a8f6383a3650df0f3d9dfab49", "images.tar.gz": "a8a83146f16c79fe2b836b840146e8a0"}
LEFT_RIGHT = {"to the left of", "to the right of"}


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_checked(url: str, path: str, want_md5: str, tries: int = 5) -> str:
    """common.download, then the file's MD5 must match: a short file (the server closed the connection early, which
    common.download does not notice) is moved back to .part and resumed; a wrong file of full size is fetched anew.
    The check is remembered in <path>.md5ok so a re-run does not hash 1.8 GB again."""
    ok = path + ".md5ok"
    for _ in range(tries):
        download(url, path)
        if os.path.exists(ok) and open(ok).read().strip() == want_md5:
            return path
        if md5(path) == want_md5:
            with open(ok, "w") as f:
                f.write(want_md5)
            return path
        print("  %s: MD5 mismatch at %.1f MB, resuming" % (os.path.basename(path), os.path.getsize(path) / 1e6),
              flush=True)
        os.replace(path, path + ".part")
        if os.path.getsize(path + ".part") >= 2e9:  # larger than any file here: start over
            os.remove(path + ".part")
    raise SystemExit("%s: still not matching MD5 %s" % (path, want_md5))


def article(name: str) -> str:
    """'Is' for 'cup', 'Are' for 'shoes' / 'people' (a plural guess from the last word)."""
    return "Are" if plural(name) else "Is"


def plural(name: str) -> bool:
    last = name.split()[-1] if name.split() else name
    return (last.endswith("s") and not last.endswith(("ss", "us", "is", "ous"))) or last in {
        "people", "children", "men", "women", "feet", "teeth", "mice", "geese"}


def clean(name: str) -> str:
    return " ".join(re.sub(r"[^\w\s'-]", " ", name.lower()).split())


def question(subject: str, predicate: str, obj: str) -> str:
    """('cat', 'on', 'ground') -> 'Is the cat on the ground?'."""
    s, o = clean(subject), clean(obj)
    return "%s the %s %s the %s?" % (article(s), s, predicate.strip().lower(), o)


def picture_rel(url: str) -> str:
    """The picture's path relative to the data root (the release keeps flickr/<name> and nyu/<name>)."""
    sub = "flickr" if url.startswith("http") else "nyu"
    return os.path.join("ext", SOURCE, "images", sub, url.split("/")[-1])


def convert(img: dict) -> list:
    """One annotations.json entry (its "train" relations) -> noul samples."""
    rel = picture_rel(img["url"])
    image_id = "spatialsense:%s/%s" % (rel.split("/")[-2], os.path.splitext(rel.split("/")[-1])[0])
    out = []
    for a in img["annotations"]:
        s, o, p = a["subject"]["name"], a["object"]["name"], a["predicate"]
        if not clean(s) or not clean(o) or clean(s) == clean(o):
            continue
        kind = "relation-leftright" if p in LEFT_RIGHT else "relation"
        out.append(noul("ext-spatialsense:%s" % a["_id"], SOURCE, kind, image_id, rel, question(s, p, o),
                        bool(a["label"]), group=p))
    return out


def usable(path: str) -> bool:
    """The picture is there and is what its name says (a deleted Flickr photo comes back as a PNG / GIF
    placeholder under the .jpg name)."""
    if not os.path.exists(path):
        return False
    with open(path, "rb") as f:
        head = f.read(8)
    return head.startswith(b"\xff\xd8") if path.endswith(".jpg") else head.startswith(b"\x89PNG")


def extract(root: str, tar_path: str, wanted: set) -> int:
    """Extract the wanted pictures (rel paths) from the archive once; returns how many were written."""
    todo = {os.path.join(*w.split("/")[-2:]): w for w in wanted if not os.path.exists(os.path.join(root, w))}
    if not todo:
        return 0
    n = 0
    print("extracting %d pictures from %s" % (len(todo), tar_path), flush=True)
    with tarfile.open(tar_path, "r:gz") as t:
        for m in t:
            if not m.isfile():
                continue
            key = os.path.join(*m.name.split("/")[-2:])
            if key in todo:
                path = os.path.join(root, todo.pop(key))
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path + ".tmp", "wb") as f:
                    f.write(t.extractfile(m).read())  # pyright: ignore[reportOptionalMemberAccess]
                os.replace(path + ".tmp", path)
                n += 1
                if not todo:
                    break
    return n


def main(argv=None):
    p = base_args(__doc__)
    p.add_argument("--per-picture", type=int, default=3)
    p.add_argument("--archive", action="store_true", help="take all pictures (NYU too) from the 1.8 GB Zenodo archive")
    a = p.parse_args(argv)
    if already_built(a.root, SOURCE, a.force):
        return
    rng = random.Random(a.seed)
    raw = raw_dir(a.root, SOURCE)
    ann = fetch_checked(ZENODO % "annotations.json", os.path.join(raw, "annotations.json"), MD5["annotations.json"])
    imgs = json.load(open(ann))
    steps, dropped = Counter(), Counter()
    steps["pictures_all"] = len(imgs)
    steps["relations_all"] = sum(len(i["annotations"]) for i in imgs)
    train = [i for i in imgs if i["split"] == "train"]
    steps["pictures_train"] = len(train)
    steps["relations_train"] = sum(len(i["annotations"]) for i in train)
    out_dir(a.root, SOURCE)
    if a.archive:
        tar = fetch_checked(ZENODO % "images.tar.gz", os.path.join(raw, "images.tar.gz"), MD5["images.tar.gz"])
        steps["pictures_extracted_now"] = extract(a.root, tar, {picture_rel(i["url"]) for i in train})
    else:
        flickr = [i for i in train if i["url"].startswith("http")]
        dropped["NYU picture (only in the archive, see --archive)"] = len(train) - len(flickr)
        fetch_pictures(a.root, flickr, lambda i: i["url"], lambda i: picture_rel(i["url"]), "spatialsense pictures")
        train = flickr
    present = [i for i in train if usable(os.path.join(a.root, picture_rel(i["url"])))]
    dropped["picture missing or a Flickr placeholder"] = len(train) - len(present)
    near = HeldOut(a.root).near_held([picture_rel(i["url"]) for i in present], SOURCE)
    dropped["near a held-out photo"] = len(near)
    rows = []
    for i in present:
        if picture_rel(i["url"]) in near:
            continue
        rows += convert(i)
    steps["questions"] = len(rows)
    dropped["same or empty subject / object"] = sum(len(i["annotations"]) for i in present
                                                    if picture_rel(i["url"]) not in near) - len(rows)
    rows = cap_per_picture(rows, a.per_picture, rng)
    steps["after_per_picture_cap"] = len(rows)
    rows = balance_yes_no(rows)
    steps["after_balance"] = len(rows)
    finish(a.root, SOURCE, rows, dict(steps), {
        "dropped": dict(dropped), "predicates": dict(Counter(r["group"] for r in rows).most_common()),
        "licence": "CC BY 2.0 (Zenodo record 8104370); pictures Flickr (owners' terms) / NYU Depth v2"})


if __name__ == "__main__":
    main()
