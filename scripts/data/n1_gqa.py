"""data-n1, GQA part: GQA balanced train questions on all 21 picture shards of lmms-lab/GQA, open questions included
(docs/experiments/2026-10-20-r16-cuda-plan.md, "data-n1").

    uv run python scripts/data/n1_gqa.py --root data

data-v2 used 3 of the 21 shards and only GQA's two-option "choose" questions. Here:
  choose  2-option choice, as data-v2 (convert.convert_gqa);
  yes/no  noul for verify / logical / compare questions; questions whose program has an `exist` step and whose
          answer is "no" are dropped (missing objects in the scene graph make some of those "no"s wrong, data-v2's
          spot check);
  query   choice with 3-5 options: the answer, then distractors drawn from the answers of other questions that end
          in the same query ("query: color", "query: name", ...) and start with the same four words (only the same
          query when that gives < 8 answers), weighted by how often each is right. A name
          distractor is never the name of an object in the picture's scene graph; an attribute distractor is never
          an attribute of the queried object; never a word containing the answer or contained in it.
Every object a question refers to must cover >= 1.5% of the picture (data-v2's `visible`). At most 8 questions per
picture, then yes questions dropped at random down to the number of no questions; pictures of every evaluation set under <root> are dropped (COCO and VG ids of a picture both); 2% of the
pictures (v2_common.in_dev on "n1:<image id>", independent of data-v2's 3% dev pictures, which are held out) go to gqa_dev.jsonl. Writes <root>/n1/gqa_train.jsonl, gqa_dev.jsonl, GQA_MANIFEST.json.
"""
import argparse
import glob
import json
import os
import random
import re
import zipfile
from collections import Counter, defaultdict
from typing import Dict, List, Optional

from n1_vqav2 import MIN_POOL, clash
from v2_common import in_dev
from v2_gqa import OBJECT_REF, visible

PER_PICTURE = 8
DEV_SHARE = 0.02


def _program(rec: Dict) -> List[Dict]:
    return rec.get("semantic") or []


def query_key(rec: Dict) -> Optional[str]:
    prog = _program(rec)
    if not prog or not prog[-1]["operation"].startswith("query"):
        return None
    return prog[-1]["operation"] + ":" + prog[-1].get("argument", "")


def queried_object(rec: Dict) -> Optional[str]:
    prog = _program(rec)
    deps = prog[-1].get("dependencies") or [] if prog else []
    if not deps:
        return None
    ids = OBJECT_REF.findall(prog[deps[-1]].get("argument", ""))
    return ids[-1] if ids else None


def excluded(rec: Dict, graph: Dict) -> set:
    """Answers that may be right for this picture besides the gold one (see module doc)."""
    key = query_key(rec) or ""
    if key.startswith("query:name"):
        return {o["name"] for o in graph["objects"].values()}
    obj = graph["objects"].get(queried_object(rec) or "")
    return set(obj.get("attributes", [])) if obj else set()


def yes_no(rec: Dict) -> Optional[Dict]:
    if rec["answer"] not in ("yes", "no"):
        return None
    if rec["answer"] == "no" and any(s["operation"].startswith("exist") for s in _program(rec)):
        return None
    truth = rec["answer"] == "yes"
    return {"question": {"type": "noul", "instructions": rec["question"]},
            "probs": {"false": float(not truth), "true": float(truth)}}


def query(rec: Dict, graph: Dict, pools: Dict[str, Counter], rng: random.Random) -> Optional[Dict]:
    answer = rec["answer"]
    keys = [k for k in pool_keys(rec) if len(pools.get(k) or ()) >= MIN_POOL]
    if not keys or answer in ("yes", "no"):
        return None
    key = keys[0]
    bad = excluded(rec, graph) | {answer}
    words, weights = _weighted(pools, key)
    n = rng.choice([3, 4, 4, 5])
    wrong: List[str] = []
    for w in rng.choices(words, cum_weights=weights, k=50):   # draw by frequency, drop clashes
        if len(wrong) == n - 1:
            break
        if w not in wrong and not any(clash(w, b) for b in bad | set(wrong)):
            wrong.append(w)
    if len(wrong) < 2:
        return None
    options = [answer] + wrong
    rng.shuffle(options)
    return {"question": {"type": "choice", "instructions": rec["question"], "criteria": {o: None for o in options}},
            "probs": {o: float(o == answer) for o in options}}


_WEIGHTED: Dict[str, tuple] = {}


def _weighted(pools: Dict[str, Counter], key: str) -> tuple:
    """(answers, cumulative weights) of pools[key], cached while it is the same Counter object."""
    pool = pools[key]
    hit = _WEIGHTED.get(key)
    if hit is None or hit[0] is not pool:
        words = list(pool)
        total, cum = 0, []
        for w in words:
            total += pool[w]
            cum.append(total)
        hit = _WEIGHTED[key] = (pool, words, cum)
    return hit[1], hit[2]


def convert(qid: str, rec: Dict, graph: Optional[Dict], pools: Dict[str, Counter], rng: random.Random) -> Optional[Dict]:
    from convert import convert_gqa

    if graph is None or not visible(rec, graph):
        return None
    structural = (rec.get("types") or {}).get("structural")
    if structural == "choose":
        s = convert_gqa(qid, rec)
        if s is None:
            return None
        s.update(source="gqa", kind="choose")
        s["id"] = "n1-gqa:%s" % qid
        return s
    made = yes_no(rec) if rec["answer"] in ("yes", "no") else query(rec, graph, pools, rng)
    if made is None:
        return None
    iid = str(rec["imageId"])
    return {"id": "n1-gqa:%s" % qid, "source": "gqa", "kind": "yesno" if made["question"]["type"] == "noul" else
            "query", "image_id": "vg:" + iid, "image": "gqa/images/%s.jpg" % iid,
            "group": re.sub(r"\s+", " ", rec["question"].lower().strip()),
            "questions": {"q": made["question"]}, "gold": {"q": {"probabilities": made["probs"]}}}


def balance_yes_no(samples: List[Dict], rng: random.Random) -> List[Dict]:
    """Drop yes questions at random until there are as many as no questions (dropping existence "no"s tilts GQA
    towards yes; a tilted existence share cost r13 6.5 points on POPE)."""
    yes = [s for s in samples if s["kind"] == "yesno" and s["gold"]["q"]["probabilities"]["true"] == 1.0]
    no = [s for s in samples if s["kind"] == "yesno" and s["gold"]["q"]["probabilities"]["true"] == 0.0]
    rng.shuffle(yes)
    drop = {id(s) for s in yes[len(no):]}
    return [s for s in samples if id(s) not in drop]


def pool_keys(rec: Dict) -> List[str]:
    """Distractor pools of a query question, finest first: its query and its first four words, then its query."""
    key = query_key(rec)
    return [key + "|" + " ".join(rec["question"].lower().split()[:4]), key] if key else []


def answer_pools(records: List[Dict]) -> Dict[str, Counter]:
    pools: Dict[str, Counter] = defaultdict(Counter)
    for rec in records:
        if rec["answer"] not in ("yes", "no"):
            for k in pool_keys(rec):
                pools[k][rec["answer"]] += 1
    return pools


def main(argv=None) -> None:
    from convert import same_images
    from fetch import GQA_SCENE_GRAPHS_URL, ensure_file, gqa_records, vg_to_coco
    from huggingface_hub import list_repo_files
    from prepare import _write_jsonl, read_jsonl

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    rng = random.Random(a.seed)
    out_dir = os.path.join(a.root, "n1")
    with zipfile.ZipFile(ensure_file(GQA_SCENE_GRAPHS_URL, os.path.join(a.root, "raw", "gqa", "sceneGraphs.zip"))) as z:
        graphs: Dict[str, dict] = json.load(z.open("train_sceneGraphs.json"))
    evals = [f for f in glob.glob(os.path.join(a.root, "**", "*.jsonl"), recursive=True)
             if os.path.basename(f).startswith(("val", "pope", "eval", "dev", "test_", "bench_"))
             and os.sep + "raw" + os.sep not in f and not f.startswith(out_dir)]
    held_out = same_images({s["image_id"] for f in evals for s in read_jsonl(f)}, vg_to_coco(a.root))
    shards = sorted(os.path.basename(f) for f in list_repo_files("lmms-lab/GQA", repo_type="dataset")
                    if f.startswith("train_balanced_images/"))
    records = [(qid, rec) for qid, rec in gqa_records(a.root, "train", shards)]
    pools = answer_pools([r for _, r in records])
    by_picture: Dict[str, List[Dict]] = defaultdict(list)
    seen = Counter()
    for qid, rec in records:
        if "vg:%s" % rec["imageId"] in held_out:
            seen["held_out"] += 1
            continue
        s = convert(qid, rec, graphs.get(str(rec["imageId"])), pools, rng)
        seen["kept" if s else "dropped"] += 1
        if s:
            by_picture[s["image_id"]].append(s)
    samples: List[Dict] = []
    for ss in by_picture.values():
        rng.shuffle(ss)
        samples += ss[:PER_PICTURE]
    samples = balance_yes_no(samples, rng)
    rng.shuffle(samples)
    dev = [s for s in samples if in_dev("n1:" + s["image_id"], DEV_SHARE)]
    train = [s for s in samples if not in_dev("n1:" + s["image_id"], DEV_SHARE)]
    os.makedirs(out_dir, exist_ok=True)
    manifest = {"records": len(records), "shards": len(shards), **seen,
                "train": _write_jsonl(os.path.join(out_dir, "gqa_train.jsonl"), train),
                "dev": _write_jsonl(os.path.join(out_dir, "gqa_dev.jsonl"), dev),
                "train_by_kind": dict(Counter(s["kind"] for s in train)),
                "yes_share": round(sum(s["gold"]["q"]["probabilities"]["true"] for s in train if s["kind"] == "yesno")
                                   / max(1, sum(s["kind"] == "yesno" for s in train)), 4),
                "options": dict(sorted(Counter(len(s["gold"]["q"]["probabilities"]) for s in train).items())),
                "pictures": len({s["image_id"] for s in train})}
    with open(os.path.join(out_dir, "GQA_MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
