"""Round 10 data (docs: critic/plan-round10.md, revision 1): the four ScienceQA diagram question types.

    uv run python scripts/data/v10_build.py --root data

- `diagram_type` names the four types by their question text (the plan's label table): attract / repel, which
  pair has the larger magnetic force, which sample has the higher temperature, which solution has the higher
  concentration. Everything else (diffusion included) is None and stays out.
- Monitoring set data/v10/dev_sqa_diagram.jsonl: those questions from ScienceQA's official **test** split, which
  neither our training (train split) nor laya-vision's benchmark (validation split) uses. A test picture whose
  decoded pixels equal any train or validation picture is dropped (diagrams match by identical pixels only; a
  perceptual hash cannot tell template maps apart). Every held-out file we keep comes from those two splits, so
  this also keeps it apart from dev_mix, dev_scienceqa and the training pool.
- data/v10/MANIFEST.json: kept / dropped per type, answers per type, the pixel-isolation counts.
"""
import argparse
import glob
import json
import os
import random
from collections import Counter
from typing import Optional

from v3_build import make_sample, pixel_hash_bytes, picture_id, read_parquets, save_image

TYPES = {"repel": "will these magnets attract or repel each other?",
         "force": "think about the magnetic force between the magnets in each pair.",
         "temp": "compare the average kinetic energies of the particles in each sample.",
         "conc": "which solution has a higher concentration of "}


def diagram_type(question: str) -> Optional[str]:
    q = " ".join(question.lower().split())
    for name, prefix in TYPES.items():
        if q.startswith(prefix):
            return name
    return None


def split_files(root: str, split: str):
    return os.path.join(root, "raw", "hf", "datasets--derek-thomas--ScienceQA", "snapshots", "*", "data",
                        "%s-*.parquet" % split)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    if not glob.glob(split_files(a.root, "test")):
        from fetch import hf_dataset_files
        hf_dataset_files(a.root, "derek-thomas/ScienceQA", "data/test-*.parquet")
    seen = set()
    for split in ("train", "validation"):
        for r in read_parquets(split_files(a.root, split), columns=["image"]):
            if r["image"] is not None:
                seen.add(pixel_hash_bytes(r["image"]["bytes"]))
    rng = random.Random(a.seed)
    rows, kept, dropped, answers = [], Counter(), Counter(), Counter()
    for i, r in enumerate(read_parquets(split_files(a.root, "test"))):
        kind = diagram_type(r["question"])
        if kind is None:
            continue
        if r["image"] is None:
            dropped[kind + ": no picture"] += 1
            continue
        data = r["image"]["bytes"]
        if pixel_hash_bytes(data) in seen:
            dropped[kind + ": same pixels as a train / validation picture"] += 1
            continue
        pic = picture_id(data)
        rel = "v10/images/sqa_test/%s.jpg" % pic[4:]
        s = make_sample("sqa-test", "test-%d" % i, pic, rel, r["question"], list(r["choices"]), int(r["answer"]), rng,
                        r.get("hint") or "", {"subject": r.get("subject"), "category": r.get("category"),
                                              "diagram_type": kind})
        if s is None:
            dropped[kind + ": unusable options"] += 1
            continue
        s["source"] = "scienceqa-test"
        save_image(a.root, rel, data)
        rows.append(s)
        kept[kind] += 1
        answers[kind + ": " + s["answer_key"][:40]] += 1
    out = os.path.join(a.root, "v10")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "dev_sqa_diagram.jsonl"), "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    manifest = {"dev_sqa_diagram": {"role": "monitoring (selects recipes; never a held-out result)",
                                    "source": "ScienceQA official test split",
                                    "kept": dict(kept), "questions": len(rows), "dropped": dict(dropped),
                                    "answers": dict(sorted(answers.items())),
                                    "train_validation_pixel_hashes": len(seen)}}
    with open(os.path.join(out, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
    print(json.dumps(manifest, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
