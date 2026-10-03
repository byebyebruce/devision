"""Lay out the project's own question sets as a Hugging Face dataset repository (nothing is uploaded).

    uv run python scripts/package_dataset.py --data data --out ../lookfirst

Only questions this project generated or rebalanced (data/v2 plus the scene-graph above/below questions)
go in; published benchmarks (POPE, laya-vision's sets) and plain re-formats of other datasets (A-OKVQA,
ScienceQA, AI2D, TQA) stay out. Images are not redistributed: every row names its COCO or Visual Genome
picture, and the dataset ships fetch_images.py to download them from the original hosts.

One config per ability, each with train / validation / test (the data-v2 pools, dev_* and eval_*), in
Parquet with flat columns so the Hub viewer works. `to_sample` in the card turns a row back into the
training format (devision.train.samples).
"""
import argparse
import json
import os
import shutil
from collections import Counter

CONFIGS = {   # config -> (train pools, dev file, test file) under data/v2
    "exist": (["coco_exist"], "dev_coco_exist", "eval_coco_exist"),
    "position": (["coco_position"], "dev_coco_position", "eval_coco_position"),
    "relation": (["coco_relation", "vg_relation_tb"], "dev_coco_relation", "eval_coco_relation"),
    "size": (["coco_size"], "dev_coco_size", "eval_coco_size"),
    "vqa_yesno": (["vqa_yesno"], "dev_vqa_yesno", "eval_vqa_yesno"),
    "vqa_choice": (["vqa_choice"], "dev_vqa_choice", "eval_vqa_choice"),
    "gqa": (["gqa"], "dev_gqa", "eval_gqa"),
}
SPLITS = ("train", "validation", "test")


def read(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def to_row(s: dict) -> dict:
    """A training sample as one flat row."""
    q = s["questions"]["q"]
    probs = s["gold"]["q"]["probabilities"]
    options = list(probs) if q["type"] == "choice" else ["false", "true"]
    p = [float(probs.get(o, 0.0)) for o in options]
    image = s["image"]
    if image.startswith("coco/"):
        source_image = "coco_" + image.split("/")[1]          # coco_train2014 / coco_val2014
    else:
        source_image = "visual_genome"                          # gqa/images/<vg id>.jpg
    return {"id": s["id"], "question_type": q["type"], "question": q["instructions"], "options": options,
            "probabilities": p, "answer": options[max(range(len(p)), key=p.__getitem__)],
            "image_id": s["image_id"], "image_file": image, "image_source": source_image,
            "source": s.get("source", ""), "kind": s.get("kind", ""), "axis": s.get("axis", ""),
            "group": s.get("group", "")}


def write_parquet(rows, path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data")
    p.add_argument("--out", required=True)
    p.add_argument("--card", default="docs/dataset-card.md")
    a = p.parse_args(argv)
    v2 = os.path.join(a.data, "v2")
    if os.path.exists(a.out):
        for f in os.listdir(a.out):
            if f != ".git":
                shutil.rmtree(os.path.join(a.out, f)) if os.path.isdir(os.path.join(a.out, f)) else os.remove(os.path.join(a.out, f))
    os.makedirs(a.out, exist_ok=True)
    stats = {}
    for name, (train, dev, test) in CONFIGS.items():
        parts = {"train": [r for f in train for r in read(os.path.join(v2, f + ".jsonl"))],
                 "validation": read(os.path.join(v2, dev + ".jsonl")),
                 "test": read(os.path.join(v2, test + ".jsonl"))}
        for split, rows in parts.items():
            out_rows = [to_row(s) for s in rows]
            assert len({r["id"] for r in out_rows}) == len(out_rows), (name, split)
            write_parquet(out_rows, os.path.join(a.out, "data", name, "%s.parquet" % split))
            stats.setdefault(name, {})[split] = {
                "questions": len(out_rows), "pictures": len({r["image_id"] for r in out_rows}),
                "types": dict(Counter(r["question_type"] for r in out_rows))}
    # no picture is in two splits, across every config (COCO and Visual Genome ids of one picture match)
    ident_path = os.path.join(v2, "image_identity.json")
    ident = json.load(open(ident_path)) if os.path.exists(ident_path) else {}
    seen = {sp: set() for sp in SPLITS}
    for name in CONFIGS:
        for sp in SPLITS:
            import pyarrow.parquet as pq
            ids = pq.read_table(os.path.join(a.out, "data", name, "%s.parquet" % sp), columns=["image_id"]).column(0).to_pylist()
            seen[sp] |= {ident.get(i, i) for i in ids}
    for x, y in (("train", "validation"), ("train", "test"), ("validation", "test")):
        shared = seen[x] & seen[y]
        assert not shared, "%d pictures in both %s and %s, e.g. %s" % (len(shared), x, y, sorted(shared)[:3])
    with open(os.path.join(a.out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    manifest = os.path.join(v2, "MANIFEST.json")
    if os.path.exists(manifest):
        tb = json.load(open(manifest)).get("text_baselines", {})
        keep = {k: v.get("all_files") for k, v in tb.items() if k.startswith(("eval_", "dev_")) and "pope" not in k}
        with open(os.path.join(a.out, "question_only_baselines.json"), "w") as f:
            json.dump(keep, f, indent=1)
    shutil.copy(os.path.join(os.path.dirname(__file__), "dataset_fetch_images.py"), os.path.join(a.out, "fetch_images.py"))
    shutil.copy(a.card, os.path.join(a.out, "README.md"))
    for name, s in stats.items():
        print("%-11s" % name, "  ".join("%s %6d q / %5d pic" % (k, v["questions"], v["pictures"]) for k, v in s.items()))


if __name__ == "__main__":
    main()
