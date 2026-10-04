"""data-v5: real-photo relations and grounded questions from VSR and Visual7W telling.

    uv run python scripts/data/v5_build.py --root data

- VSR (cambridgeltl/vsr_random, all three splits pooled): a COCO picture and a statement about two objects
  ("The cat is inside the refrigerator."), true or false -> a noul question ("Is the cat inside the
  refrigerator?"). 66 relations in seven meta-categories (VSR's rel_meta_category_dict.txt).
- Visual7W telling (dataset_v7w_telling.json): what / where / who / how / why / when questions with the
  answer and three human-written wrong answers -> a four-option choice, options in a random order. Its
  pictures are Visual Genome ids; half are also COCO pictures (VG's image_data.json gives the COCO id).

Held out, by one picture identity for both sources and every other file (COCO id, VG ids mapped to it;
A-OKVQA pictures, which carry no id, by perceptual hash):
1. every evaluation / monitoring / calibration picture (test_*, dev_*, bench_*, lv_bench, val_*, pope,
   align_val) is dropped from everything built here;
2. the strict test and the dev sets take only pictures the starting checkpoint (round 7) never trained on:
   not in caption alignment (align_full_train), nor in the decision data of rounds 3, 5, 6, 7 (mirrored
   pictures count as their original);
3. the test and dev pictures are chosen first and frozen; training takes only the remaining pictures.

Writes data/v5/{vsr,v7w}.jsonl (training pools), test_{vsr,v7w}.jsonl, dev_{vsr,v7w}.jsonl and
MANIFEST.json (counts at every step, text-only baselines -- diagnostics, never used to drop questions).
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
import urllib.request
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable, List, Optional, Set

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert import picture as picture_of, same_images  # noqa: E402
from v2_common import file_stats, require_benchmarks, write  # noqa: E402
from v3_build import clean, make_sample  # noqa: E402

VSR_REPO = "cambridgeltl/vsr_random"
VSR_META_URL = ("https://raw.githubusercontent.com/cambridgeltl/visual-spatial-reasoning/"
                "b27a0af0ee1462d2b6b92c8c83e869d9254a241a/analysis_scripts/rel_meta_category_dict.txt")
V7W_URL = "https://ai.stanford.edu/~yukez/papers/resources/dataset_v7w_telling.zip"
LR_RELATIONS = {"left of", "right of", "at the left side of", "at the right side of"}

# held out and history files (relative to the data root)
HELD_PREFIXES = ("test_", "dev", "bench_", "val", "pope", "align_val")
HISTORY = ["align_full_train.jsonl", "v2/train_mix.jsonl", "v3/train_cont.jsonl", "v6/train_flip_mix.jsonl",
           "v4/train_sqa_mix.jsonl"]
AOKVQA_HISTORY = ["v3/aokvqa.jsonl", "v4/aokvqa.jsonl"]

TEST_SIZE = {"vsr": 1000, "v7w": 1000}
DEV_SIZE = {"vsr": 300, "v7w": 300}
# Visual7W: questions per type in the test / dev (stratified) and in the training pool (weighted to where /
# what / who -- position, attribute, reference -- while keeping every type)
V7W_TEST_SHARE = {"what": 0.30, "where": 0.20, "who": 0.15, "how": 0.15, "why": 0.10, "when": 0.10}
V7W_TRAIN = {"what": 2600, "where": 1800, "who": 1000, "how": 1100, "why": 500, "when": 500}
PER_PICTURE = 3


# ---------------------------------------------------------------- conversion

def vsr_question(caption: str) -> Optional[str]:
    """VSR's statement as a yes/no question: "The cat is inside the fridge." -> "Is the cat inside the fridge?"."""
    c = caption.strip().rstrip(".").strip()
    m = re.match(r"^The (.+?) (is|are) (.+)$", c)
    if m:
        return "%s the %s %s?" % (m.group(2).capitalize(), m.group(1), m.group(3))
    m = re.match(r"^The (.+?) (contains|has|consists of|touches) (.+)$", c)
    if m:
        verb = {"contains": "contain", "has": "have", "consists of": "consist of", "touches": "touch"}[m.group(2)]
        return "Does the %s %s %s?" % (m.group(1), verb, m.group(3))
    return None


def meta_categories(text: str) -> Dict[str, str]:
    """relation -> meta-category, from VSR's rel_meta_category_dict.txt (a relation listed twice keeps the first)."""
    out: Dict[str, str] = {}
    for line in text.splitlines():
        if ":" in line:
            cat, rels = line.split(":", 1)
            for r in rels.split(","):
                out.setdefault(r.strip(), cat.strip())
    return out


def coco_train2014_ids(annotations_zip: str) -> Set[int]:
    """The official train2014 image ids (from the COCO 2014 captions file), so a picture's split never depends
    on which images happen to be on disk."""
    with zipfile.ZipFile(annotations_zip) as z:
        return {im["id"] for im in json.load(z.open("annotations/captions_train2014.json"))["images"]}


def coco_path(coco_id: int, train2014: Set[int]) -> str:
    split = "train2014" if coco_id in train2014 else "val2014"
    return "coco/%s/COCO_%s_%012d.jpg" % (split, split, coco_id)


def convert_vsr(rec: dict, key: str, meta: Dict[str, str], train2014: Set[int]) -> Optional[dict]:
    q = vsr_question(rec["caption"])
    if q is None:
        return None
    coco = int(rec["image"].split(".")[0])
    answer = "true" if rec["label"] == 1 else "false"
    rel = rec["relation"]
    return {"id": "v5-vsr:%s" % key, "source": "vsr", "kind": "vsr", "category": rel,
            "meta_category": meta.get(rel, "Unallocated"), "axis": "lr" if rel in LR_RELATIONS else "",
            "group": rel, "answer_key": answer, "image_id": "coco:%d" % coco, "image": coco_path(coco, train2014),
            "questions": {"q": {"type": "noul", "instructions": q}},
            "gold": {"q": {"probabilities": {"true": float(answer == "true"), "false": float(answer == "false")}}}}


def convert_v7w(qa: dict, image_id: str, image: str, rng: random.Random) -> Optional[dict]:
    options = [qa["answer"]] + list(qa["multiple_choices"])
    if len({clean(o).lower() for o in options}) != len(options):
        return None
    s = make_sample("v7w", str(qa["qa_id"]), image_id, image, qa["question"], options, 0, rng,
                    extra={"category": qa["type"]})
    if s is not None:
        s["id"] = "v5-v7w:%d" % qa["qa_id"]
        s["group"] = qa["type"]
    return s


# ---------------------------------------------------------------- picture identity and splits

def picture_key(image_id: str, vg2coco: Dict[str, str]) -> str:
    return picture_of(image_id, vg2coco)


def take_by_picture(rows: List[dict], n: int, key, salt: str) -> List[dict]:
    """Whole pictures in a fixed pseudo-random order (hash of salt + picture) until at least `n` questions."""
    by: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by[key(r)].append(r)
    out: List[dict] = []
    for p in sorted(by, key=lambda p: hashlib.sha1((salt + p).encode()).hexdigest()):
        if len(out) >= n:
            break
        out += by[p]
    return out


def balance_answers(rows: List[dict], rng: random.Random) -> List[dict]:
    """Equal true / false over the file, dropping from the larger side at random (no label changed, nothing copied)."""
    by: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by[r["answer_key"]].append(r)
    k = min(len(v) for v in by.values()) if len(by) > 1 else 0
    out = []
    for v in by.values():
        v = v[:]
        rng.shuffle(v)
        out += v[:k]
    return out


def equalize_by_group(rows: List[dict], rng: random.Random) -> List[dict]:
    """Per relation, as many true as false (training pool only); relations with one answer go."""
    by: Dict[str, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["group"]][r["answer_key"]].append(r)
    out = []
    for answers in by.values():
        if len(answers) < 2:
            continue
        k = min(len(v) for v in answers.values())
        for v in answers.values():
            v = v[:]
            rng.shuffle(v)
            out += v[:k]
    return out


def cap_per_picture(rows: List[dict], cap: int, key, rng: random.Random) -> List[dict]:
    rows = rows[:]
    rng.shuffle(rows)
    seen: Counter = Counter()
    out = []
    for r in rows:
        if seen[key(r)] < cap:
            seen[key(r)] += 1
            out.append(r)
    return out


def stratified(rows: List[dict], quota: Dict[str, int], rng: random.Random) -> List[dict]:
    by: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by[r["category"]].append(r)
    out = []
    for cat, n in quota.items():
        v = by.get(cat, [])[:]
        rng.shuffle(v)
        out += v[:n]
    return out


# ---------------------------------------------------------------- text-only baselines (diagnostics)

def majority_baseline(train: List[dict], evaluate: List[dict], field: str) -> dict:
    """Accuracy of answering every question with the most common training answer of its `field` value
    (fitted on `train` only)."""
    counts: Dict[str, Counter] = defaultdict(Counter)
    overall: Counter = Counter()
    for r in train:
        counts[r.get(field, "")][r["answer_key"]] += 1
        overall[r["answer_key"]] += 1
    if not evaluate or not overall:
        return {}
    hit = 0
    for r in evaluate:
        c = counts.get(r.get(field, "")) or overall
        hit += c.most_common(1)[0][0] == r["answer_key"]
    return {"n": len(evaluate), "accuracy": round(hit / len(evaluate), 4)}


def option_baselines(train: List[dict], evaluate: List[dict]) -> dict:
    """Choice questions: how often the longest / shortest option is right, and picking the option whose text
    was most often the right answer in `train` (per question type). Fitted on `train`, measured on `evaluate`."""
    freq: Dict[str, Counter] = defaultdict(Counter)
    for r in train:
        freq[r["category"]][r["answer_key"].lower()] += 1
    if not evaluate:
        return {}
    longest = shortest = frequent = 0
    for r in evaluate:
        opts = list(r["gold"]["q"]["probabilities"])
        gold = r["answer_key"]
        longest += max(opts, key=len) == gold
        shortest += min(opts, key=len) == gold
        frequent += max(opts, key=lambda o: freq[r["category"]][o.lower()]) == gold
    n = len(evaluate)
    return {"n": n, "longest": round(longest / n, 4), "shortest": round(shortest / n, 4),
            "train_answer_frequency": round(frequent / n, 4), "chance": 0.25}


# ---------------------------------------------------------------- sources and pictures

def read(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def held_files(root: str) -> List[str]:
    out = []
    for d in ("", "v2", "v3", "v4", "v6", "lv_bench"):
        base = os.path.join(root, d)
        for f in sorted(os.listdir(base)):
            if f.endswith(".jsonl") and (d == "lv_bench" or f.startswith(HELD_PREFIXES)):
                out.append(os.path.join(base, f))
    return out


def ids_of(paths: Iterable[str]) -> Set[str]:
    return {json.loads(l)["image_id"] for p in paths for l in open(p) if l.strip()}


def dhashes(root: str, rels: Iterable[str]) -> Set[int]:
    from PIL import Image
    from lv_bench import dhash
    out = set()
    for rel in sorted(set(rels)):
        try:
            with Image.open(os.path.join(root, rel)) as im:
                out.add(dhash(im))
        except OSError:
            pass
    return out


def fetch_vg_images(root: str, rows: List[dict], urls: Dict[str, str], workers: int = 16) -> List[str]:
    def one(r) -> str:
        path = os.path.join(root, r["image"])
        if os.path.exists(path):
            return ""
        try:
            with urllib.request.urlopen(urls[r["image_id"]], timeout=30) as resp:
                data = resp.read()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path + ".part", "wb") as f:
                f.write(data)
            os.replace(path + ".part", path)
            return ""
        except (OSError, KeyError):
            return r["id"]
    with ThreadPoolExecutor(workers) as pool:
        return [x for x in pool.map(one, rows) if x]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    root, out_dir = a.root, os.path.join(a.root, "v5")
    require_benchmarks(root)
    from fetch import _download_json_zip, coco_annotations, download_coco_images, ensure_file, hf_dataset_files, VG_IMAGE_DATA_URL
    rng = random.Random(a.seed)
    report: dict = {"steps": {}}

    vg_data: List[dict] = list(_download_json_zip(VG_IMAGE_DATA_URL, os.path.join(root, "raw", "vg")))  # type: ignore[arg-type]
    vg2coco = {"vg:%d" % r["image_id"]: "coco:%d" % r["coco_id"] for r in vg_data if r.get("coco_id")}
    vg_url = {"vg:%d" % r["image_id"]: r["url"] for r in vg_data}
    train2014 = coco_train2014_ids(coco_annotations(root))
    key = lambda r: picture_key(r["image_id"], vg2coco)  # noqa: E731

    # held-out and history pictures
    held = same_images(ids_of(held_files(root)), vg2coco)
    history = same_images(ids_of(os.path.join(root, f) for f in HISTORY), vg2coco)
    held_aok = dhashes(root, [r["image"] for f in held_files(root) for r in read(f) if "aokvqa" in r.get("image", "")])
    hist_aok = dhashes(root, [r["image"] for f in AOKVQA_HISTORY for r in read(os.path.join(root, f))])
    report["held_out"] = {"files": [os.path.relpath(f, root) for f in held_files(root)], "picture_ids": len(held),
                          "aokvqa_dhashes": len(held_aok)}
    report["history"] = {"files": HISTORY + AOKVQA_HISTORY, "picture_ids": len(history), "aokvqa_dhashes": len(hist_aok),
                         "limits": "history files are the current files under data/, not snapshots taken when each "
                                   "round trained; mirrored pictures share the original's image_id"}

    # sources
    meta_path = ensure_file(VSR_META_URL, os.path.join(root, "raw", "vsr", "rel_meta_category_dict.txt"))
    meta = meta_categories(open(meta_path).read())
    vsr_raw = []
    for split in ("train", "dev", "test"):
        (path,) = hf_dataset_files(root, VSR_REPO, split + ".jsonl")
        vsr_raw += [(split, i, r) for i, r in enumerate(read(path))]
    vsr = [s for s in (convert_vsr(r, "%s-%d" % (sp, i), meta, train2014) for sp, i, r in vsr_raw) if s]
    v7w_json = os.path.join(root, "raw", "visual7w", "dataset_v7w_telling.json")
    if not os.path.exists(v7w_json):
        z = ensure_file(V7W_URL, os.path.join(root, "raw", "visual7w", "dataset_v7w_telling.zip"))
        zipfile.ZipFile(z).extractall(os.path.dirname(v7w_json))
    v7w = []
    for im in json.load(open(v7w_json))["images"]:
        vid = "vg:%d" % im["image_id"]
        coco = vg2coco.get(vid)
        image = coco_path(int(coco[5:]), train2014) if coco else "gqa/images/%d.jpg" % im["image_id"]
        for qa in im["qa_pairs"]:
            s = convert_v7w(qa, vid, image, rng)
            if s:
                s["official_split"] = im["split"]
                v7w.append(s)
    report["steps"]["raw"] = {"vsr": len(vsr_raw), "v7w": sum(len(im["qa_pairs"]) for im in json.load(open(v7w_json))["images"])}
    report["steps"]["converted"] = {"vsr": len(vsr), "v7w": len(v7w)}

    # 1. drop held-out pictures (ids), 2. mark pictures the starting checkpoint saw
    vsr = [r for r in vsr if not ({r["image_id"], key(r)} & held)]
    v7w = [r for r in v7w if not ({r["image_id"], key(r)} & held)]
    report["steps"]["held_out_ids_dropped"] = {"vsr": len(vsr), "v7w": len(v7w)}
    unseen = lambda r: not ({r["image_id"], key(r)} & history)  # noqa: E731

    # COCO pictures on disk and their perceptual hash (A-OKVQA pictures are COCO pictures without an id);
    # Visual Genome pictures that are not COCO cannot be A-OKVQA pictures and are fetched after selection
    coco_images = sorted({r["image"] for r in vsr + v7w if r["image"].startswith("coco/")})
    failed = set(download_coco_images(root, [{"id": i, "image": i} for i in coco_images]))
    from PIL import Image
    from lv_bench import dhash
    pic_hash: Dict[str, Optional[int]] = {}
    for rel in coco_images:
        try:
            with Image.open(os.path.join(root, rel)) as im:
                pic_hash[rel] = dhash(im)
        except OSError:
            failed.add(rel)
    ok = lambda r: r["image"] not in failed and pic_hash.get(r["image"]) not in held_aok  # noqa: E731
    vsr, v7w = [r for r in vsr if ok(r)], [r for r in v7w if ok(r)]
    report["steps"]["coco_images_and_aokvqa_hash"] = {"vsr": len(vsr), "v7w": len(v7w), "coco_image_failures": len(failed)}
    strict = lambda r: unseen(r) and (pic_hash.get(r["image"]) is None or pic_hash[r["image"]] not in hist_aok)  # noqa: E731

    # 3. freeze test and dev on strictly unseen pictures (both sources, one picture identity)
    vsr_pool = [r for r in vsr if strict(r)]
    v7w_test_pool = [r for r in v7w if strict(r) and r["official_split"] == "test"]
    v7w_dev_pool = [r for r in v7w if strict(r) and r["official_split"] == "val"]
    report["steps"]["strictly_unseen_pool"] = {"vsr": len(vsr_pool), "v7w_test": len(v7w_test_pool), "v7w_val": len(v7w_dev_pool)}
    test_vsr = take_by_picture(vsr_pool, TEST_SIZE["vsr"], key, "test_vsr")
    taken = {key(r) for r in test_vsr}
    dev_vsr = take_by_picture([r for r in vsr_pool if key(r) not in taken], DEV_SIZE["vsr"], key, "dev_vsr")
    taken |= {key(r) for r in dev_vsr}
    test_vsr, dev_vsr = balance_answers(test_vsr, rng), balance_answers(dev_vsr, rng)

    def v7w_split(pool, size, salt):
        pool = cap_per_picture([r for r in pool if key(r) not in taken], 1, key, random.Random(salt))
        return stratified(pool, {t: round(size * s) for t, s in V7W_TEST_SHARE.items()}, random.Random(salt))
    test_v7w = v7w_split(v7w_test_pool, TEST_SIZE["v7w"], "test_v7w")
    taken |= {key(r) for r in test_v7w}
    dev_v7w = v7w_split(v7w_dev_pool, DEV_SIZE["v7w"], "dev_v7w")
    taken |= {key(r) for r in dev_v7w}

    # training pools: everything else (seen pictures allowed), test / dev pictures of either source excluded
    vsr_train = [r for r in vsr if key(r) not in taken]
    report["steps"]["vsr_train"] = {"after_test_dev_pictures": len(vsr_train)}
    vsr_train = equalize_by_group(vsr_train, rng)
    report["steps"]["vsr_train"]["after_true_false_per_relation"] = len(vsr_train)
    vsr_train = cap_per_picture(vsr_train, PER_PICTURE, key, rng)
    report["steps"]["vsr_train"]["after_per_picture_cap"] = len(vsr_train)
    v7w_train = [r for r in v7w if r["official_split"] == "train" and key(r) not in taken]
    report["steps"]["v7w_train"] = {"after_test_dev_pictures": len(v7w_train)}
    v7w_train = stratified(cap_per_picture(v7w_train, PER_PICTURE, key, rng), V7W_TRAIN, rng)
    report["steps"]["v7w_train"]["after_cap_and_type_quota"] = len(v7w_train)

    vg_rows = [r for r in v7w_train + test_v7w + dev_v7w if r["image"].startswith("gqa/")]
    vg_failed_images = set(fetch_vg_images(
        root, [{"id": i, "image": i, "image_id": v} for i, v in sorted({(r["image"], r["image_id"]) for r in vg_rows})], vg_url))
    v7w_train, test_v7w, dev_v7w = ([r for r in rows if r["image"] not in vg_failed_images]
                                    for rows in (v7w_train, test_v7w, dev_v7w))
    report["steps"]["vg_image_failures"] = len(vg_failed_images)

    files = {"vsr.jsonl": vsr_train, "v7w.jsonl": v7w_train, "test_vsr.jsonl": test_vsr, "test_v7w.jsonl": test_v7w,
             "dev_vsr.jsonl": dev_vsr, "dev_v7w.jsonl": dev_v7w}
    for name, rows in files.items():
        for r in rows:
            r.pop("official_split", None)
        write(os.path.join(out_dir, name), rows)

    # acceptance: picture identity disjoint across train / dev / test and from held-out and (test/dev) history
    pics = {n: {key(r) for r in rows} for n, rows in files.items()}
    train_pics = pics["vsr.jsonl"] | pics["v7w.jsonl"]
    problems = []
    for n in ("test_vsr.jsonl", "test_v7w.jsonl", "dev_vsr.jsonl", "dev_v7w.jsonl"):
        if pics[n] & train_pics:
            problems.append("%s shares %d pictures with training" % (n, len(pics[n] & train_pics)))
        if any(not strict(r) for r in files[n]):
            problems.append("%s has pictures the starting checkpoint trained on" % n)
    for n, rows in files.items():
        if any({r["image_id"], key(r)} & held for r in rows):
            problems.append("%s has held-out pictures" % n)
        if not rows:
            problems.append("%s is empty" % n)
        if len({r["id"] for r in rows}) != len(rows):
            problems.append("%s has duplicate ids" % n)
    report["problems"] = problems

    report["files"] = {n: dict(file_stats(os.path.join(out_dir, n)),
                               by_category=dict(Counter(r["category"] for r in rows).most_common()),
                               by_meta_category=dict(Counter(r.get("meta_category", "") for r in rows)) if "vsr" in n else None,
                               left_right=sum(r.get("axis") == "lr" for r in rows),
                               max_per_picture=max(Counter(key(r) for r in rows).values()) if rows else 0)
                       for n, rows in files.items()}
    report["text_only_baselines"] = {
        "vsr_relation_majority_on_dev": majority_baseline(vsr_train, dev_vsr, "group"),
        "vsr_global_majority_on_dev": majority_baseline(vsr_train, dev_vsr, "none"),
        "v7w_options_on_dev": option_baselines(v7w_train, dev_v7w),
        "v7w_options_on_dev_by_type": {t: option_baselines(v7w_train, [r for r in dev_v7w if r["category"] == t])
                                       for t in V7W_TRAIN},
        "v7w_answer_position_train": dict(Counter(r["answer_position"] for r in v7w_train)),
        "note": "fitted on the training pools, measured on dev; diagnostics only, no question is dropped by them",
    }
    with open(os.path.join(out_dir, "MANIFEST.json"), "w") as f:
        json.dump(report, f, indent=1)
    for n, s in report["files"].items():
        print("%-16s %6d questions %6d pictures" % (n, s["questions"], s["images"]))
    print(json.dumps(report["steps"]))
    if problems:
        raise SystemExit("acceptance failed: %s" % problems)


if __name__ == "__main__":
    main()
