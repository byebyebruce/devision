"""Left/right questions on mirrored pictures (kept out of the repo): counterfactual pairs for round 6.

    uv run python scripts/data/v6_flip.py --root data

Every left/right question of the COCO position and relation pools (data/v2/coco_{position,relation}.jsonl,
axis lr) is asked again on the picture flipped left to right, with the answer swapped. The pair is the
same picture, the same words and opposite answers, so neither the names nor the option order can answer
it; only where the object is in the picture can. Pictures are written once to
data/v6/images/flip/<original file name>. Text in a picture is mirrored too; the questions never refer to
it. Writes data/v6/flip_lr.jsonl (the flipped questions only; mix them with the originals); with --dev,
data/v6/dev_flip_lr.jsonl from the dev left/right questions (for monitoring).
"""
import argparse
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

SWAP = {"left": "right", "right": "left", "to the left of": "to the right of", "to the right of": "to the left of"}


def flipped(s: dict, image: str) -> dict:
    """The same question on the mirrored picture, with the gold answer swapped."""
    q = s["questions"]["q"]
    g = s["gold"]["q"]["probabilities"]
    if q["type"] == "choice":
        assert set(g) <= set(SWAP), g
        gold = {o: g[SWAP[o]] for o in g}          # option order unchanged, probabilities swapped
    else:
        gold = {"false": g["true"], "true": g["false"]}
    out = dict(s, id="flip:" + s["id"], image=image, image_id=s["image_id"] + ":flip",
               gold={"q": {"probabilities": gold}}, flipped=True)
    if "answer_key" in s and s["answer_key"] in SWAP:
        out["answer_key"] = SWAP[s["answer_key"]]
    return out


def main(argv=None):
    from PIL import Image, ImageOps

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="data")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--dev", action="store_true", help="mirror the dev left/right questions instead -> dev_flip_lr.jsonl")
    p.add_argument("--files", nargs="*", help="data/v2/<file>.jsonl stems to mirror instead (with --out)")
    p.add_argument("--out", help="output file name under data/v6/ (with --files)")
    a = p.parse_args(argv)
    rows = []
    files = a.files or (("dev_coco_position", "dev_coco_relation") if a.dev else ("coco_position", "coco_relation"))
    for f in files:
        with open(os.path.join(a.root, "v2", f + ".jsonl")) as fh:
            rows += [r for r in map(json.loads, fh) if r.get("axis") == "lr"]
    target = lambda rel: "v6/images/flip/" + os.path.basename(rel)

    def mirror(rel):
        dst = os.path.join(a.root, target(rel))
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with Image.open(os.path.join(a.root, rel)) as im:
                ImageOps.mirror(im.convert("RGB")).save(dst, quality=92)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(mirror, sorted({r["image"] for r in rows})))
    out = [flipped(r, target(r["image"])) for r in rows]
    name = a.out or ("dev_flip_lr.jsonl" if a.dev else "flip_lr.jsonl")
    with open(os.path.join(a.root, "v6", name), "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in out)
    print("%s: %d questions on %d mirrored pictures; answers %s" % (
        name, len(out), len({r["image"] for r in out}), dict(Counter(r.get("answer_key") for r in out))))


if __name__ == "__main__":
    main()
