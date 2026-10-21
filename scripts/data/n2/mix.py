"""data-n2: the r19 training mix and its dev sets (plan docs/experiments/2026-10-21-v04-plan.md, stage 2-3).

    uv run python scripts/data/n2/mix.py --root data [--vqav2 200000] [--gqa 200000] [--drop data/n2/blind_drop.json]

Parts (each optional file is used when it exists; the MANIFEST lists what was read):
- data-n1 (data/n1/train_mix.jsonl): VQAv2 and GQA down-sampled to --vqav2 / --gqa (data-n1 was 83% these two, which
  diluted every other ability); everything else (existence, position, relation, size, A-OKVQA, VSR, ScienceQA, AI2D,
  TQA) kept; the old 7,499 Visual7W questions are replaced by the superset below.
- r17's existence questions of every size (data/n1/exist_train.jsonl).
- data-n2 blocks: n2/v7w.jsonl (Visual7W telling, all), n2/spatial.jsonl (caption choice on / under / in front of /
  behind / left / right), n2/count_train.jsonl (COCO train2014 counts 0..10), n2/sqa_synth*.jsonl (programmatic
  science diagrams, when built).
- extension sources rebuilt here (data/ext/<name>/train.jsonl): iconqa, tallyqa, clevr, spatialsense. MapQA is read
  but capped at 0: in the first r19 run (stopped at step 22,000, runs/archive/r19-n2-stop-mapqa) ScienceQA's
  "which region is highlighted" map questions fell (Thirteen Colonies 0.93 -> 0.34 on dev_sqa) with MapQA's
  choropleth maps in the mix.

Rules applied to every row:
- held out (n2/held.py: ids, all of COCO val2014, dHash of every evaluation picture) -> dropped;
- counting questions whose options are all numbers get the full "0".."10" options (shuffled); a gold above 10 ->
  dropped (gold-dependent windows let a picture-blind rule win, r13);
- counting golds are capped at --count-cap questions each over all sources (COCO and TallyQA have many more small
  counts than large ones);
- --drop: ids to leave out (scripts/data/n2/blind.py: answerable without the picture);
- --repeat source=k: listed sources appear k times (ScienceQA / AI2D / TQA x2 by default: small, and the weakest
  ability);
- about 2% of each new block's pictures (by hash) go to <root>/n2/dev_<block>.jsonl instead (monitoring).
Writes <root>/n2/train_mix.jsonl, dev_*.jsonl and MIX.json.
"""
import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

COUNT_OPTIONS = [str(n) for n in range(11)]
EXT = ["iconqa", "tallyqa", "mapqa", "clevr", "spatialsense"]
NEW_BLOCKS = {"v7w": "n2/v7w.jsonl", "spatial": "n2/spatial.jsonl", "count": "n2/count_train.jsonl"}


def read(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def is_count(r: dict) -> bool:
    q = r["questions"]["q"]
    if q["type"] != "choice":
        return False
    opts = list(q["criteria"])
    text = q["instructions"] if isinstance(q["instructions"], str) else json.dumps(q["instructions"])
    return all(o.strip().isdigit() for o in opts) and ("how many" in text.lower() or r.get("kind") == "count")


def full_count_options(r: dict, rng: random.Random) -> Optional[dict]:
    """The row with options "0".."10" (gold mass moved over, soft golds kept), or None when any gold mass sits on
    a count above 10. Rows that already have exactly these options are returned unchanged."""
    probs = r["gold"]["q"]["probabilities"]
    if set(probs) == set(COUNT_OPTIONS):
        return r
    mass = {str(int(o)): p for o, p in probs.items() if p > 0}
    if any(int(o) > 10 for o in mass):
        return None
    opts = COUNT_OPTIONS[:]
    rng.shuffle(opts)
    r = json.loads(json.dumps(r))
    r["questions"]["q"]["criteria"] = {o: None for o in opts}
    r["gold"]["q"]["probabilities"] = {o: float(mass.get(o, 0.0)) for o in opts}
    r["answer_key"] = max(mass, key=lambda o: mass[o])
    return r


def in_dev(key: str, share: float = 0.02) -> bool:
    """Salted, so it does not pick the pictures data-n1's dev rule (sha1 of the bare id) already holds out."""
    return int(hashlib.sha1(("n2:" + key).encode()).hexdigest()[:8], 16) < share * 0x100000000


def _hash_one(args):
    root, rel = args
    from held import dhash_file
    return rel, dhash_file(os.path.join(root, rel))


def dhashes(root: str, rels: List[str], cache_path: str) -> Dict[str, Optional[int]]:
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    todo = [r for r in rels if r not in cache]
    if todo:
        with ProcessPoolExecutor(max_workers=max(2, (os.cpu_count() or 4) - 2)) as ex:
            for i, (rel, h) in enumerate(ex.map(_hash_one, [(root, r) for r in todo], chunksize=256)):
                cache[rel] = h
                if i % 50000 == 0:
                    print("  hashed %d / %d" % (i, len(todo)), flush=True)
        with open(cache_path, "w") as f:
            json.dump(cache, f)
    return {r: cache[r] for r in rels}


def main(argv=None) -> None:
    import held as held_mod
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    p.add_argument("--vqav2", type=int, default=200000)
    p.add_argument("--gqa", type=int, default=200000)
    p.add_argument("--count-cap", type=int, default=12000)
    p.add_argument("--repeat", action="append", default=["scienceqa=2", "ai2d=2", "tqa=2"])
    p.add_argument("--cap", action="append", default=["ext-clevr=30000", "ext-mapqa=0"],
                   help="part=N: at most N training rows from that part (random)")
    p.add_argument("--drop", default="", help="JSON list (or {ids: [...]}) of ids to leave out")
    p.add_argument("--out", default="n2/train_mix.jsonl")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dev-only", action="store_true", help="rewrite only the dev_*.jsonl files, not the training mix")
    a = p.parse_args(argv)
    rng = random.Random(a.seed)
    root = a.root
    read_files, parts = [], {}

    n1 = read(os.path.join(root, "n1", "train_mix.jsonl"))
    read_files.append("n1/train_mix.jsonl")
    vqa = [r for r in n1 if r["source"] == "vqav2"]
    gqa = [r for r in n1 if r["source"] == "gqa"]
    rng.shuffle(vqa)
    rng.shuffle(gqa)
    parts["n1-vqav2"] = vqa[:a.vqav2]
    parts["n1-gqa"] = gqa[:a.gqa]
    parts["n1-replay"] = [r for r in n1 if r["source"] not in ("vqav2", "gqa", "v7w")]
    if os.path.exists(os.path.join(root, "n1", "exist_train.jsonl")):
        parts["r17-exist"] = read(os.path.join(root, "n1", "exist_train.jsonl"))
        read_files.append("n1/exist_train.jsonl")
    dev: Dict[str, List[dict]] = {}
    blocks = dict(NEW_BLOCKS)
    for f in sorted(os.listdir(os.path.join(root, "n2"))):
        if f.startswith("sqa_synth") and f.endswith(".jsonl"):
            blocks[f[:-6]] = "n2/" + f
    for name in EXT:
        blocks["ext-" + name] = "ext/%s/train.jsonl" % name
    for name, rel in blocks.items():
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            continue
        read_files.append(rel)
        rows = read(path)
        parts[name] = [r for r in rows if not in_dev(r["image_id"])]
        rng.shuffle(parts[name])
        dev[name] = [r for r in rows if in_dev(r["image_id"])]

    drop = set()
    if a.drop:
        d = json.load(open(a.drop))
        drop = set(d["ids"] if isinstance(d, dict) else d)
    steps: Dict[str, Counter] = defaultdict(Counter)
    for name in list(parts):
        out = []
        for r in parts[name]:
            if r["id"] in drop:
                steps[name]["answerable without the picture"] += 1
                continue
            if is_count(r):
                r2 = full_count_options(r, rng)
                if r2 is None:
                    steps[name]["count gold not 0..10"] += 1
                    continue
                r = r2
            out.append(r)
        parts[name] = out

    h = held_mod.load(root)
    rels = sorted({r["image"] for rows in list(parts.values()) + list(dev.values()) for r in rows})
    print("hashing %d pictures" % len(rels), flush=True)
    hashes = dhashes(root, rels, os.path.join(root, "n2", "dhash_cache.json"))

    def keep(name: str, r: dict) -> bool:
        if h.id_held(r["image_id"], r["image"]):
            steps[name]["held out (id / val2014)"] += 1
            return False
        if hashes.get(r["image"]) in h.hashes:
            steps[name]["held out (dHash)"] += 1
            return False
        if hashes.get(r["image"]) is None:
            steps[name]["picture unreadable"] += 1
            return False
        return True

    caps = {k: int(v) for k, v in (x.split("=") for x in a.cap)}
    for name in list(parts):
        parts[name] = [r for r in parts[name] if keep(name, r)]
        if name in caps and len(parts[name]) > caps[name]:
            steps[name]["over the part cap"] += len(parts[name]) - caps[name]
            parts[name] = parts[name][:caps[name]]
    for name in list(dev):
        dev[name] = [r for r in dev[name] if keep("dev-" + name, r)]

    if a.dev_only:
        for name, rows in dev.items():
            if rows:
                with open(os.path.join(root, "n2", "dev_%s.jsonl" % name.replace("ext-", "")), "w") as f:
                    for r in rows:
                        f.write(json.dumps(r) + "\n")
        print(json.dumps({n: len(v) for n, v in dev.items()}))
        return

    # counting golds capped over all sources
    per_gold: Counter = Counter()
    all_rows = [(n, r) for n in parts for r in parts[n]]
    rng.shuffle(all_rows)
    train, seen = [], set()
    for n, r in all_rows:
        if r["id"] in seen:
            steps[n]["duplicate id"] += 1
            continue
        if is_count(r) and set(r["questions"]["q"]["criteria"]) == set(COUNT_OPTIONS):
            probs = r["gold"]["q"]["probabilities"]
            g = max(probs, key=lambda o: probs[o])
            if per_gold[g] >= a.count_cap:
                steps[n]["count gold over the cap"] += 1
                continue
            per_gold[g] += 1
        seen.add(r["id"])
        train.append(r)
    reps = {k: int(v) for k, v in (x.split("=") for x in a.repeat)}
    extra = []
    for r in train:
        for _ in range(reps.get(r["source"], 1) - 1):
            extra.append(r)
    train += extra
    rng.shuffle(train)

    out = os.path.join(root, a.out)
    with open(out, "w") as f:
        for r in train:
            f.write(json.dumps(r) + "\n")
    for name, rows in dev.items():
        if rows:
            with open(os.path.join(root, "n2", "dev_%s.jsonl" % name.replace("ext-", "")), "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
    m = {"built_by": "scripts/data/n2/mix.py", "args": vars(a), "read": read_files, "train": len(train),
         "repeated_rows": len(extra), "by_source": dict(Counter(r["source"] for r in train).most_common()),
         "by_part_after_rules": {n: len(v) for n, v in parts.items()},
         "dev": {n: len(v) for n, v in dev.items()}, "steps": {n: dict(c) for n, c in steps.items()},
         "count_golds": dict(sorted(per_gold.items(), key=lambda kv: int(kv[0])))}
    with open(out.replace(".jsonl", ".MIX.json"), "w") as f:
        json.dump(m, f, indent=1)
    print(json.dumps({k: m[k] for k in ("train", "repeated_rows", "by_source", "dev", "count_golds")}, indent=1))


if __name__ == "__main__":
    main()
