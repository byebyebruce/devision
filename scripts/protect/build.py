"""PROTECT-PREP-01 (critic EXP73-RESULT-01; docs/experiments/2026-10-19-protect-replay-plan.md): the data audit and the
mixed training set of the one-arm replay recovery. CPU only, loads no model, changes no existing file.

  - the new task: data/exp72/train_K.jsonl as it is (3,800 questions on 2,800 rows), every row marked "input": "text":
    scored with image=None, as EXP72-K learned it under the global text_only switch -- including its 200 reading /
    magnet replay rows whose image field names a real-layout picture (source identity only, never loaded);
  - the replay: questions of v0.3's own training set data/r14b/train.jsonl (real pictures), after removing every picture
    -- and its family: the mirrored copy and the COCO / VG twin (data/v5/image_identity.json) -- that any monitoring,
    development, protection, calibration, confirmation or benchmark file uses (every non-training JSONL the configs
    name, plus every non-training file under data/exp*, data/facts, data/real_layout, data/v04_*, data/lv_bench);
    sampled per block with fixed quotas (ALLOC, set before looking at any item) by sha256 of the unit key, whole units:
    a mirrored pair (original + flipped picture of one COCO image) or a group of questions stays together; never chosen
    from any protection panel's wrong answers. In the copy written here a mirrored pair's rows share one training group
    "protect-mirror:<base image>" (the pool's own groups differ: coco:<id> vs coco:<id>:flip; kept as source_group), so
    rlcd.plan_mixed packs each pair into one batch (critic PROTECT-R1-01); data/r14b is not changed;
  - MANIFEST.json: input hashes, the exposure files with their sha256 (the isolation evidence the runner's START binds),
    per block available / chosen counts, the exclusion counts, and the batch plan exactly as training draws it
    (random.seed(1), rlcd.plan_mixed on the items of train_mix.jsonl in file order, micro batch 8): update count, the
    largest unit, no batch mixing inputs, every question once, every mirrored pair in one batch.

    uv run python scripts/protect/build.py [--out data/protect2]
"""
import argparse
import glob
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

import yaml

NEW = "data/exp72/train_K.jsonl"
POOL = "data/r14b/train.jsonl"
IDENTITY = "data/v5/image_identity.json"
# replay questions per block (7,600): science / mirror / counting / vsr floors for the capabilities the seven panels
# protect, the rest of v0.3's mix in its own proportions; fixed before any item was looked at
ALLOC = {"science": 1600, "mirror": 1000, "count_balanced": 800, "count_original": 400, "pixmo": 600, "existence": 900,
         "general": 900, "ext": 1200, "vsr": 200}
MICRO = 8
EXTRA_GLOBS = ("data/exp*/**/*.jsonl", "data/facts/**/*.jsonl", "data/real_layout/**/*.jsonl", "data/v04_*/**/*.jsonl",
               "data/lv_bench/**/*.jsonl")


def sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jsonl(path: str) -> List[Dict]:
    return [json.loads(x) for x in open(path) if x.strip()]


def config_files() -> Tuple[Set[str], Set[str]]:
    """(training files, exposure files) the configs name: a stage's `data` is training, every other JSONL is exposure."""
    train, expo = set(), set()

    def walk(o, path):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, path + [str(k)])
        elif isinstance(o, list):
            for v in o:
                walk(v, path)
        elif isinstance(o, str) and o.endswith(".jsonl"):
            (train if path and path[-1] == "data" and "stages" in path else expo).add(o)
    for f in sorted(glob.glob("configs/*.yaml")):
        try:
            walk(yaml.safe_load(open(f)), [])
        except yaml.YAMLError:
            continue
    return train, expo


def exposure_files() -> List[str]:
    train, expo = config_files()
    extra = {p for g in EXTRA_GLOBS for p in glob.glob(g, recursive=True)
             if not os.path.basename(p).startswith("train") and p not in train}
    return sorted(p for p in (expo | extra) if os.path.exists(p) and p != POOL and p != NEW)


def family(image_id: str, image: str, twins: Dict[str, str]) -> Set[str]:
    """The keys of a picture's family: its id without the mirrored suffix, that id's COCO / VG twin, and the file name."""
    base = image_id[:-len(":flip")] if image_id.endswith(":flip") else image_id
    keys = {"id:" + base, "file:" + os.path.basename(image)}
    if base in twins:
        keys.add("id:" + twins[base])
    return keys


def build(out: str) -> Dict:
    raw = json.load(open(IDENTITY))
    twins = dict(raw)
    twins.update({v: k for k, v in raw.items()})
    files = exposure_files()
    exposed: Set[str] = set()
    per_file = {}
    for p in files:
        n = 0
        for r in jsonl(p):
            if isinstance(r, dict) and isinstance(r.get("image"), str) and r.get("image") and not r["image"].startswith("text-only:"):
                exposed |= family(str(r.get("image_id", "")), r["image"], twins)
                n += 1
        per_file[p] = {"pictures": n, "sha256": sha(p)}
    pool = jsonl(POOL)
    units: Dict[str, Dict[Tuple, List[Dict]]] = defaultdict(lambda: defaultdict(list))
    removed: Counter = Counter()
    for r in pool:
        if family(r["image_id"], r["image"], twins) & exposed:
            removed[r["block"]] += 1
            continue
        base = r["image_id"][:-len(":flip")] if r["image_id"].endswith(":flip") else r["image_id"]
        key = ("pair", base) if r["block"] == "mirror" else ("group", r.get("group") or r["id"])
        units[r["block"]][key].append(r)
    chosen: List[Dict] = []
    blocks = {}
    for block, quota in ALLOC.items():
        us = units.get(block, {})
        if block == "mirror":          # only complete pairs: the original and its mirrored copy
            us = {k: v for k, v in us.items() if {x["image_id"].endswith(":flip") for x in v} == {True, False}}
        order = sorted(us, key=lambda k: hashlib.sha256(json.dumps(k).encode()).hexdigest())
        got: List[Dict] = []
        for k in order:
            if len(got) + len(us[k]) > quota:
                continue
            got.extend(us[k])
            if len(got) == quota:
                break
        blocks[block] = {"available_questions": sum(len(v) for v in us.values()), "available_units": len(us),
                         "quota": quota, "chosen": len(got), "removed_by_exposure": removed[block],
                         "sources": dict(Counter(x["source"] for x in got).most_common())}
        chosen.extend(got)
    new = [dict(r, input="text") for r in jsonl(NEW)]
    chosen = [dict(r, input="picture", **mirror_group(r)) for r in chosen]
    n_new = sum(len(r["questions"]) for r in new)
    mix = new + chosen
    random.Random(0).shuffle(mix)
    plan = batch_plan(mix)
    os.makedirs(out, exist_ok=False)
    with open(os.path.join(out, "replay.jsonl"), "w") as fh:
        fh.write("".join(json.dumps(r) + "\n" for r in chosen))
    with open(os.path.join(out, "train_mix.jsonl"), "w") as fh:
        fh.write("".join(json.dumps(r) + "\n" for r in mix))
    man = {"inputs": {p: sha(p) for p in (NEW, POOL, IDENTITY)}, "exposure_files": {p: per_file[p] for p in files},
           "exposed_family_keys": len(exposed), "pool_questions": len(pool), "removed_by_exposure": dict(removed),
           "allocation": ALLOC, "blocks": blocks,
           "new_task": {"rows": len(new), "questions": n_new, "input": "text (image=None) for every row",
                        "rows_naming_a_picture": sum(not r["image"].startswith("text-only:") for r in new)},
           "replay": {"rows": len(chosen), "questions": sum(len(r["questions"]) for r in chosen),
                      "mirror_pairs": plan["mirrored_pairs"]},
           "plan": plan, "sha256": {n: sha(os.path.join(out, n)) for n in ("replay.jsonl", "train_mix.jsonl")}}
    with open(os.path.join(out, "MANIFEST.json"), "w") as fh:
        json.dump(man, fh, indent=1)
    return man


def mirror_group(r: Dict) -> Dict:
    """A mirrored pair's shared training group (the row's own group kept as source_group); other rows unchanged."""
    if r["block"] != "mirror":
        return {}
    base = r["image_id"][:-len(":flip")] if r["image_id"].endswith(":flip") else r["image_id"]
    return {"group": "protect-mirror:" + base, "source_group": r["group"]}


def batch_plan(mix: List[Dict], seed: int = 1) -> Dict:
    """The first epoch's plan as training draws it: random.seed(seed) then rlcd.plan_mixed on the items of `mix` in its
    order (one per question: group, qid, input -- what rlcd._items gives plan_batches), checked: every question once, no
    batch mixing inputs, no unit larger than a batch (refused: plan_batches would split it), every mirrored pair in one
    batch."""
    sys.path.insert(0, "src")
    from devision.train import rlcd

    items, mirror = [], defaultdict(list)
    for r in mix:
        for q in r["questions"]:
            it = {"qid": q, "input": r["input"]}
            if r.get("group"):
                it["group"] = r["group"]
            if r.get("block") == "mirror":      # by the shared base picture, whatever the rows' groups say
                mirror[r["image_id"][:-len(":flip")] if r["image_id"].endswith(":flip") else r["image_id"]].append(len(items))
            items.append(it)
    sizes = Counter((it.get("group", "#%d" % i), it["qid"]) for i, it in enumerate(items))
    if max(sizes.values()) > MICRO:
        raise SystemExit("units larger than a batch: %s" % [k for k, v in sizes.items() if v > MICRO][:5])
    random.seed(seed)
    plan = rlcd.plan_mixed(items, MICRO)
    flat = sorted(i for b in plan for i in b)
    where = {i: n for n, b in enumerate(plan) for i in b}
    mixed = sum(len({items[i]["input"] for i in b}) > 1 for b in plan)
    split = [g for g, idx in mirror.items() if len({where[i] for i in idx}) > 1]
    if flat != list(range(len(items))) or mixed or split or any(len(v) < 2 for v in mirror.values()):
        raise SystemExit("plan check failed: questions once %s, mixed batches %d, split mirrored pairs %d"
                         % (flat == list(range(len(items))), mixed, len(split)))
    text = [b for b in plan if items[b[0]]["input"] == "text"]
    pic = [b for b in plan if items[b[0]]["input"] == "picture"]
    return {"micro_batch": MICRO, "seed": seed, "drawn_as": "random.seed(seed); rlcd.plan_mixed(items of train_mix.jsonl in file order)",
            "text_batches": len(text), "picture_batches": len(pic), "updates": len(plan),
            "text_partial_batches": sum(len(b) < MICRO for b in text), "picture_partial_batches": sum(len(b) < MICRO for b in pic),
            "largest_unit": max(sizes.values()), "questions": len(items), "mixed_batches": 0,
            "mirrored_pairs": len(mirror), "mirrored_pairs_in_one_batch": len(mirror) - len(split),
            "mirrored_unit_sizes": {str(k): v for k, v in sorted(Counter(len(v) for v in mirror.values()).items())},
            "plan_sha256": hashlib.sha256(json.dumps(plan).encode()).hexdigest()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/protect2")
    a = ap.parse_args(argv)
    man = build(a.out)
    print(json.dumps({k: man[k] for k in ("removed_by_exposure", "blocks", "new_task", "replay", "plan")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
