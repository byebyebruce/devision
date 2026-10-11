"""Round 13 bench (evaluation only): What'sUp Controlled A and B (photos the authors took of two household objects
in four arrangements). Licence MIT. Plan 1.5.

    uv run python scripts/data/r13/sources/whatsup.py --root data [--force]

Source: github.com/amitakamath/whatsup_vlms; annotation JSONs and picture archives from the authors' Google Drive
(fetched with `uvx gdown`, no project dependency): controlled_images (A: on / under / left of / right of) and
controlled_clevr (B: left of / right of / in front of / behind). Upstream the first caption is always the correct
one; here the four captions are a choice in a seeded random order ("Which caption describes the image?").
`group` = the object pair's set (the four pictures of the same two objects, "whatsup_a:<obj1>_<obj2>"), `pair` = the
two pictures that differ only in the opposite preposition (left/right, on/under, front/behind), `kind` = the
gold preposition, `subset` = A / B. Writes data/r13/bench/whatsup.jsonl.
"""
import json
import os
import random
import subprocess
import sys
import tarfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from base import already_built, base_args, finish_bench, raw_dir  # noqa: E402

FILES = {"A": ("controlled_images_dataset.json", "1ap8mmmpQjLIjPGuplkpBgc1hoEHCj4hm",
               "controlled_images.tar.gz", "19KGYVQjrV3syb00GgcavB2nZTW5NXX0H"),
         "B": ("controlled_clevr_dataset.json", "1unNNosLbdy9NDjgj4l8fsQP3WiAAGA6z",
               "controlled_clevr.tar.gz", "13jdBpg8t3NqW3jrL6FK8HO93vwsUjDxG")}
OPPOSITE = {"left_of": "lr", "right_of": "lr", "on": "ou", "under": "ou", "in-front_of": "fb", "behind": "fb"}
QUESTION = "Which caption describes the image?"


def gdown(file_id: str, path: str) -> str:
    if not os.path.exists(path):
        subprocess.run(["uvx", "gdown", file_id, "-O", path + ".part"], check=True)
        os.replace(path + ".part", path)
    return path


def preposition(name: str) -> str:
    stem = name.rsplit(".", 1)[0]
    for p in ("in-front_of", "left_of", "right_of", "behind", "under", "on"):
        if "_%s_" % p in stem:
            return p
    raise ValueError(name)


def main(argv=None):
    p = base_args(__doc__)
    a = p.parse_args(argv)
    if already_built(a.root, "r13/bench/whatsup.jsonl", a.force):
        return
    raw = raw_dir(a.root, "whatsup")
    rng = random.Random(a.seed)
    rows, missing = [], 0
    for sub, (js, js_id, tar, tar_id) in FILES.items():
        data = json.load(open(gdown(js_id, os.path.join(raw, js))))
        out_dir = os.path.join(a.root, "r13", "bench", "images", "whatsup")
        with tarfile.open(gdown(tar_id, os.path.join(raw, tar))) as t:
            for m in t.getmembers():
                dest = os.path.join(out_dir, m.name)
                if m.isfile() and not os.path.exists(dest):
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    with open(dest, "wb") as f:
                        f.write(t.extractfile(m).read())
        for d in data:
            folder, name = d["image_path"].split("/")[-2:]
            rel = "r13/bench/images/whatsup/%s/%s" % (folder, name)
            if not os.path.exists(os.path.join(a.root, rel)):
                missing += 1
                continue
            prep = preposition(name)
            stem = name.rsplit(".", 1)[0]
            obj1, obj2 = stem.split("_")[0], stem.split("_")[-1]
            gold = d["caption_options"][0]
            opts = list(dict.fromkeys(d["caption_options"]))
            rng.shuffle(opts)
            set_id = "whatsup_%s:%s_%s" % (sub.lower(), obj1, obj2)
            rows.append({"id": "whatsup:%s:%s" % (sub, stem), "source": "whatsup", "kind": prep, "subset": sub,
                         "image_id": "whatsup:%s:%s" % (sub, stem), "image": rel, "group": set_id,
                         "pair": "%s:%s" % (set_id, OPPOSITE[prep]),
                         "questions": {"q": {"type": "choice", "instructions": QUESTION,
                                             "criteria": {o: None for o in opts}}},
                         "gold": {"q": {"probabilities": {o: float(o == gold) for o in opts}}}, "answer_key": gold})
    finish_bench(a.root, "whatsup", rows, {"licence": "MIT", "picture_missing": len(rows) and missing,
                                           "subsets": {s: sum(1 for r in rows if r["subset"] == s) for s in "AB"},
                                           "note": "first caption correct upstream; shuffled here"})


if __name__ == "__main__":
    main()
