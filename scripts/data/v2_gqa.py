"""v2 class 5: GQA questions about things visible at 256 px (kept out of the repo).

    uv run python scripts/data/v2_gqa.py --root data --limit 40000 --out data/v2/gqa.jsonl

Same conversion as convert.convert_gqa (yes/no -> noul, choose -> 2-option choice), with GQA's scene
graphs (data/raw/gqa/sceneGraphs.zip, train_sceneGraphs.json) used to check every object a question's
program refers to: each must cover >= 1.5% of the image, or the question is dropped (a spot check
of v1 found 3-4 of 8 GQA questions about things too small to see or not findable at 256 px).
Yes/no questions are dropped by default (--keep-noul to keep them): their "no" answers come from
scene graphs that miss objects and attributes, and a spot check found 2 of 8 wrong ("no printer"
next to a printer, "no bandana" on a dog wearing one); yes/no data comes from VQAv2 and COCO instead.
Questions about left / right are kept but marked kind "spatial" (the model cannot yet find a named
object, see the experiment log), everything else is kind "gqa". Evaluation images are skipped.
"""
import argparse
import json
import random
import re
import zipfile
from collections import Counter
from typing import Dict, Iterable, List, Optional, Set

MIN_OBJECT_AREA = 0.015
OBJECT_REF = re.compile(r"\((\d+)\)")
SPATIAL = re.compile(r"\b(left|right)\b")


def referenced_objects(program: Iterable[dict]) -> Set[str]:
    return {m for step in program for m in OBJECT_REF.findall(step.get("argument", ""))}


def visible(rec: dict, graph: Optional[dict]) -> bool:
    """Every object the question refers to is in the scene graph and covers >= MIN_OBJECT_AREA."""
    if graph is None:
        return False
    area = float(graph["width"] * graph["height"])
    for oid in referenced_objects(rec.get("semantic") or []):
        obj = graph["objects"].get(oid)
        if obj is None or obj["w"] * obj["h"] / area < MIN_OBJECT_AREA:
            return False
    return True


def tag(sample: dict) -> dict:
    q = sample["questions"]["q"]
    text = q["instructions"] + " " + " ".join(q.get("criteria") or [])
    sample["kind"] = "spatial" if SPATIAL.search(text.lower()) else "gqa"
    sample["source"] = "gqa-v2"
    g = sample["gold"]["q"]["probabilities"]
    sample["group"] = re.sub(r"\s+", " ", q["instructions"].lower().strip())
    sample["answer_key"] = max(g, key=g.get)
    return sample


# The pictures the project's GQA pools came from: 3 of the 21 train shards of lmms-lab/GQA.
TRAIN_SHARDS = ["train-%05d-of-00021.parquet" % i for i in range(3)]


def main(argv=None) -> None:
    import glob
    import os

    from convert import convert_gqa, same_images, select
    from fetch import gqa_records, vg_to_coco
    from prepare import _write_jsonl, read_jsonl

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="data")
    p.add_argument("--limit", type=int, default=40000)
    p.add_argument("--out", required=True)
    p.add_argument("--keep-noul", action="store_true", help="also keep yes/no questions (noisier)")
    p.add_argument("--split", default="train", choices=["train", "val"], help="GQA balanced split")
    p.add_argument("--spatial-share", type=float, default=0.25, help="left/right questions, as a share of --limit")
    p.add_argument("--exclude", nargs="*", default=[], help="more JSONL files whose images must not be used")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-shards", nargs="+", default=TRAIN_SHARDS,
                   help="lmms-lab/GQA train_balanced_images shards to take pictures from (downloaded if missing)")
    a = p.parse_args(argv)
    from fetch import GQA_SCENE_GRAPHS_URL, ensure_file
    with zipfile.ZipFile(ensure_file(GQA_SCENE_GRAPHS_URL, os.path.join(a.root, "raw", "gqa", "sceneGraphs.zip"))) as z:
        graphs: Dict[str, dict] = json.load(z.open("%s_sceneGraphs.json" % a.split))
    evals = [f for f in glob.glob(os.path.join(a.root, "*.jsonl")) + glob.glob(os.path.join(a.root, "v2", "*.jsonl"))
             if os.path.basename(f).startswith(("val", "pope", "eval", "dev", "test_", "bench_"))]
    held_out = same_images({s["image_id"] for f in evals + a.exclude for s in read_jsonl(f)}, vg_to_coco(a.root))
    if a.split == "train":
        shards = list(a.train_shards)   # a fixed list, not whatever happens to be cached
    else:
        from huggingface_hub import list_repo_files
        shards = sorted(os.path.basename(f) for f in list_repo_files("lmms-lab/GQA", repo_type="dataset")
                        if f.startswith("val_balanced_images/"))
    seen = kept = 0
    samples: List[dict] = []
    for qid, rec in gqa_records(a.root, a.split, shards):
        s = convert_gqa(qid, rec)
        if s is None:
            continue
        seen += 1
        if s["questions"]["q"]["type"] == "noul" and not a.keep_noul:
            continue
        if visible(rec, graphs.get(str(rec["imageId"]))):
            kept += 1
            samples.append(tag(s))
    rng = random.Random(a.seed)
    out = []
    for kind in ("gqa", "spatial"):  # select() balances yes / no within each kind
        share = a.limit if kind == "gqa" else int(a.limit * a.spatial_share)
        out += select([s for s in samples if s["kind"] == kind], share, held_out, a.seed)
    rng.shuffle(out)
    if not out:   # an empty GQA pool would quietly change every mix built from it
        raise SystemExit("v2_gqa: no question left for %s (shards %s); check the downloads" % (a.out, shards))
    print("%s: %d questions (%s); %d of %d convertible questions refer only to visible objects (%d shards)" % (
        a.out, _write_jsonl(a.out, out), dict(Counter(s["kind"] for s in out)), kept, seen, len(shards)))


if __name__ == "__main__":
    main()
