"""laya-vision's public benchmark, rebuilt as our samples (kept out of the repo):

    uv run python scripts/data/lv_bench.py --root data

Writes data/lv_bench/{pope,vqav2_yesno,aokvqa,scienceqa}.jsonl -- exactly the questions laya-vision
reports on -- and converts their published per-question predictions (results/raw/*.predictions.jsonl.gz,
downloaded to data/raw/laya_vision/ and checked against their SHA256SUMS) into our record format
(data/lv_bench/<run>.<set>.details.jsonl), so `devision-compare` can pair them with ours question by
question. Nothing of laya-vision is installed or run.

Sources, as named in their .meta.json: VQAv2 yes/no = their 5,000-question re-split of VQAv2 val (ids
are VQAv2 question ids; gold from the VQAv2 annotations); A-OKVQA = HuggingFaceM4/A-OKVQA validation
(their 1,138 of 1,145, by question_id); ScienceQA = derek-thomas/ScienceQA validation rows with an image
(their "val-<i>" = row i; the hint goes in as text state); POPE = all 9,000 questions (they publish only
aggregate POPE numbers, no rows). Every rebuilt label is checked against theirs.

Also reports which of these pictures our training data used: COCO / VG ids for VQAv2 and POPE; for
A-OKVQA, whose parquet has no COCO id, a perceptual hash against every COCO train2014 / val2014 and
GQA image on disk.
"""
import argparse
import ast
import glob
import gzip
import io
import json
import os
import shutil
from collections import Counter, defaultdict
from typing import Dict, List

LAYA_VISION_RAW = "https://raw.githubusercontent.com/r33drichards/laya-vision/main/results/raw/"
RUNS = {"laya-vision-201m": "smolvlm-autoresearch-full-long-sep24-b64-best",
        "laya-vision-237m-previous": "smolvlm-cauldron-score-2ep-bidir-full-best"}


def their_rows(root: str, run_file: str) -> Dict[str, Dict[str, dict]]:
    path = os.path.join(root, "raw", "laya_vision", run_file + ".vqa-val.predictions.jsonl.gz")
    out: Dict[str, Dict[str, dict]] = defaultdict(dict)
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            out[r["dataset"]][r["id"]] = r
    return out


def dhash(img, size: int = 8) -> int:
    from PIL import Image
    g = img.convert("L").resize((size + 1, size), Image.BILINEAR)
    px = list(g.getdata())
    bits = 0
    for y in range(size):
        for x in range(size):
            bits = (bits << 1) | (px[y * (size + 1) + x] < px[y * (size + 1) + x + 1])
    return bits


def main(argv=None) -> None:
    import pyarrow.parquet as pq
    from PIL import Image

    from convert import convert_vqav2, picture
    from fetch import download_coco_images, vg_to_coco, vqav2_records
    from prepare import _write_jsonl, read_jsonl

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="data")
    a = p.parse_args(argv)
    root = a.root
    out_dir = os.path.join(root, "lv_bench")
    os.makedirs(os.path.join(out_dir, "images", "aokvqa"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "images", "scienceqa"), exist_ok=True)
    # inputs, downloaded on first use: laya-vision's published per-question rows, and the val splits
    from fetch import ensure_file, hf_dataset_files
    raw = os.path.join(root, "raw", "laya_vision")
    for f in ["SHA256SUMS"] + [r + ".vqa-val." + x for r in RUNS.values() for x in ("predictions.jsonl.gz", "meta.json")]:
        ensure_file(LAYA_VISION_RAW + f, os.path.join(raw, f))
    hf_dataset_files(root, "HuggingFaceM4/A-OKVQA", "data/validation-*.parquet")
    hf_dataset_files(root, "derek-thomas/ScienceQA", "data/validation-*.parquet")
    theirs = {name: their_rows(root, f) for name, f in RUNS.items()}
    ref = theirs["laya-vision-201m"]
    report = {"label_mismatches": {}, "counts": {}}

    # VQAv2 yes/no: their ids, our conversion (soft target), their label order is (no, yes)
    ids = set(ref["vqav2_yesno"])
    vqa = []
    for q, ann in vqav2_records(root, "Val"):
        if str(q["question_id"]) in ids:
            s = convert_vqav2(q, ann, "val2014")
            if s and s["gold"]["q"]["probabilities"]["true"] == 0.5:
                report.setdefault("vqav2_ties_left_out", []).append(q["question_id"])
                continue   # 5 of 10 annotators each way: no right answer (theirs picks one; both sides drop them)
            if s:
                s["id"], s["source"], s["kind"] = "lv-vqav2:%d" % q["question_id"], "lv-vqav2-yesno", "yesno"
                vqa.append(s)
    download_coco_images(root, vqa)
    mism = sum(1 for s in vqa if (s["gold"]["q"]["probabilities"]["true"] > 0.5) != (ref["vqav2_yesno"][s["id"][9:]]["label"] == 1))
    report["label_mismatches"]["vqav2_yesno"] = {"mismatch": mism, "of": len(vqa),
                                                 "annotator_ties_left_out": len(report.get("vqav2_ties_left_out", []))}

    # A-OKVQA
    path = glob.glob(os.path.join(root, "raw", "hf", "datasets--HuggingFaceM4--A-OKVQA", "snapshots", "*", "data",
                                  "validation-*.parquet"))[0]
    aok, aok_hash = [], {}
    for r in pq.read_table(path).to_pylist():
        if r["question_id"] not in ref["aokvqa"]:
            continue
        rel = "lv_bench/images/aokvqa/%s.jpg" % r["question_id"]
        img = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
        img.save(os.path.join(root, rel), quality=95)
        aok_hash[r["question_id"]] = dhash(img)
        choices = ast.literal_eval(r["choices"]) if isinstance(r["choices"], str) else list(r["choices"])
        gold = int(r["correct_choice_idx"])
        aok.append({"id": "lv-aokvqa:" + r["question_id"], "source": "lv-aokvqa", "kind": "aokvqa",
                    "image_id": "aokvqa:" + r["question_id"], "image": rel,
                    "questions": {"q": {"type": "choice", "instructions": r["question"],
                                        "criteria": {c: None for c in choices}}},
                    "gold": {"q": {"probabilities": {c: float(i == gold) for i, c in enumerate(choices)}}}})
        if len(set(choices)) != len(choices):
            report.setdefault("aokvqa_duplicate_choices", []).append(r["question_id"])
    report["label_mismatches"]["aokvqa"] = {"mismatch": sum(
        1 for s in aok if list(s["questions"]["q"]["criteria"]).index(max(s["gold"]["q"]["probabilities"],
        key=s["gold"]["q"]["probabilities"].get)) != ref["aokvqa"][s["id"][10:]]["label"]), "of": len(aok)}

    # ScienceQA: rows with an image, "val-<row>"
    path = glob.glob(os.path.join(root, "raw", "hf", "datasets--derek-thomas--ScienceQA", "snapshots", "*", "data",
                                  "validation-*.parquet"))[0]
    sqa = []
    for i, r in enumerate(pq.read_table(path).to_pylist()):
        key = "val-%d" % i
        if key not in ref["scienceqa"] or r["image"] is None:
            continue
        rel = "lv_bench/images/scienceqa/%d.png" % i
        Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB").save(os.path.join(root, rel))
        choices = ast.literal_eval(r["choices"]) if isinstance(r["choices"], str) else list(r["choices"])
        gold = int(r["answer"])
        sample = {"id": "lv-scienceqa:" + key, "source": "lv-scienceqa", "kind": "scienceqa",
                  "image_id": "scienceqa:" + key, "image": rel,
                  "questions": {"q": {"type": "choice", "instructions": r["question"],
                                      "criteria": {c: None for c in choices}}},
                  "gold": {"q": {"probabilities": {c: float(j == gold) for j, c in enumerate(choices)}}}}
        if r.get("hint"):
            sample["state_text"] = r["hint"]
        sqa.append(sample)
    report["label_mismatches"]["scienceqa"] = {"mismatch": sum(
        1 for s in sqa if list(s["gold"]["q"]["probabilities"].values()).index(1.0) != ref["scienceqa"][s["id"][13:]]["label"]),
        "of": len(sqa)}

    pope = read_jsonl(os.path.join(root, "v2", "pope_full_9000.jsonl"))
    for name, rows in (("vqav2_yesno", vqa), ("aokvqa", aok), ("scienceqa", sqa), ("pope", pope)):
        report["counts"][name] = _write_jsonl(os.path.join(out_dir, name + ".jsonl"), rows)

    # their published rows -> our record format (correctness only needs their argmax and the label)
    sample_of = {"vqav2_yesno": {s["id"][9:]: s for s in vqa}, "aokvqa": {s["id"][10:]: s for s in aok},
                 "scienceqa": {s["id"][13:]: s for s in sqa}}
    for run, rows in theirs.items():
        for ds, by_id in rows.items():
            recs = []
            for rid, r in by_id.items():
                s = sample_of[ds].get(rid)
                if s is None:
                    continue
                probs = r["probs_calibrated"]
                opts = ["false", "true"] if r["qtype"] == "noul" else list(s["questions"]["q"]["criteria"])
                pred = max(range(len(probs)), key=probs.__getitem__)
                recs.append({"sample_id": s["id"], "qid": "q", "image_id": s["image_id"], "source": s["source"],
                             "kind": s["kind"], "type": r["qtype"], "options": opts,
                             "probabilities": dict(zip(opts, probs)), "prediction": opts[pred],
                             "gold_answer": opts[r["label"]], "correct": pred == r["label"],
                             "p_prediction": probs[pred], "control": "none", "published_by": "laya-vision"})
            with open(os.path.join(out_dir, "%s.%s.details.jsonl" % (run, ds)), "w") as f:
                f.writelines(json.dumps(x) + "\n" for x in recs)
            report.setdefault("their_accuracy", {}).setdefault(run, {})[ds] = round(
                sum(x["correct"] for x in recs) / max(1, len(recs)), 4)

    # which of these pictures our training used
    mapping = vg_to_coco(root)
    ours = set()
    for f in [x for x in glob.glob(os.path.join(root, "*.jsonl")) + glob.glob(os.path.join(root, "v2", "*.jsonl"))
              if os.path.basename(x).startswith(("train", "align_", "cocoqa_train", "coco_", "vqa_", "gqa", "match"))]:
        ours |= {picture(s["image_id"], mapping) for s in read_jsonl(f)}
    seen = {"vqav2_yesno": {s["id"] for s in vqa if picture(s["image_id"], mapping) in ours},
            "pope": {s["id"] for s in pope if picture(s["image_id"], mapping) in ours}}
    local = {}
    for d in ("coco/train2014", "coco/val2014", "gqa/images"):
        for fpath in glob.glob(os.path.join(root, d, "*.jpg")):
            try:
                with Image.open(fpath) as im:
                    local.setdefault(dhash(im), []).append(fpath)
            except OSError:
                pass
    hits = {qid: local[h] for qid, h in aok_hash.items() if h in local}
    trained_files = {os.path.basename(s["image"]) for f in glob.glob(os.path.join(root, "v2", "*.jsonl"))
                     if not os.path.basename(f).startswith(("eval_", "dev_", "pope", "test_", "bench_"))
                     for s in read_jsonl(f)} | {os.path.basename(s["image"]) for f in
                                                [os.path.join(root, x) for x in ("train_100k.jsonl", "train_cocoqa_mix.jsonl",
                                                                                 "align_full_train.jsonl")]
                                                if os.path.exists(f) for s in read_jsonl(f)}
    seen["aokvqa"] = {"lv-aokvqa:" + qid for qid, v in hits.items() if any(os.path.basename(x) in trained_files for x in v)}
    seen["scienceqa"] = set()
    report["aokvqa_pictures_found_locally_by_hash"] = len(hits)
    # the same sets without any picture our training used (A-OKVQA by perceptual hash, so approximate)
    report["seen_by_our_training"] = {k: len(v) for k, v in seen.items()}
    for name, rows in (("vqav2_yesno", vqa), ("aokvqa", aok), ("scienceqa", sqa), ("pope", pope)):
        report["counts"][name + ".unseen"] = _write_jsonl(os.path.join(out_dir, name + ".unseen.jsonl"),
                                                          [s for s in rows if s["id"] not in seen[name]])
    with open(os.path.join(out_dir, "REPORT.json"), "w") as f:
        json.dump(report, f, indent=1)
    shutil.copy(os.path.join(root, "raw", "laya_vision", "SHA256SUMS"), os.path.join(out_dir, "laya_vision_SHA256SUMS"))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
