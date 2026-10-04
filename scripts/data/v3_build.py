"""data-v3: common-sense reasoning and textbook-diagram questions, plus the data-v2 mix (kept out of the repo).

    uv run python scripts/data/v3_build.py --root data

Adds four sources the earlier data never had (laya-vision is trained on them; we are 4 points behind it
on A-OKVQA and 34 behind on ScienceQA):

  aokvqa     A-OKVQA train (HuggingFaceM4/A-OKVQA), 4-way common-sense questions on COCO 2017 pictures
  scienceqa  ScienceQA train rows with an image (derek-thomas/ScienceQA); the hint goes in as state_text
  ai2d       AI2D science diagrams, 4-way (HuggingFaceM4/the_cauldron, ai2d)
  tqa        TQA textbook diagrams, 4-way (HuggingFaceM4/the_cauldron, tqa)

Only train splits are read. A picture is dropped when its perceptual hash matches a picture of any
evaluation, dev or laya-vision benchmark set (A-OKVQA's parquet has no COCO id, and ScienceQA reuses the
same maps across splits), so the laya-vision A-OKVQA / ScienceQA sets stay unseen. Options are shuffled
(seeded) so the answer position carries nothing; rows with duplicate or empty options are dropped.
A dev share (by picture) goes to data/v3/dev_<set>.jsonl.

Writes data/v3/{aokvqa,scienceqa,ai2d,tqa}.jsonl, dev_*.jsonl, dev_mix.jsonl (dev of the new sets plus
data/v2/dev_mix_half.jsonl), train_mix.jsonl (data-v2 train_mix + the above/below relation questions of
round 4 + the new sets) and MANIFEST.json (counts, sha256, answer-position and option-count baselines).
"""
import argparse
import glob
import hashlib
import io
import json
import os
import random
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Set

SETS = ("aokvqa", "scienceqa", "ai2d", "tqa")
DEV_SHARE = 0.03
OPTION = re.compile(r"^([A-Z])\. (.*)$")


def parse_cauldron(user: str, assistant: str):
    """(question, options, answer index) of a Cauldron multiple-choice turn, or None."""
    lines = user.split("\n")
    if not lines or not lines[0].startswith("Question: "):
        return None
    question, options = lines[0][len("Question: "):].strip(), []
    for line in lines[1:]:
        m = OPTION.match(line.strip())
        if m:
            options.append(m.group(2).strip())
    m = re.match(r"Answer: ([A-Z])", assistant.strip())
    if not m or not options:
        return None
    idx = ord(m.group(1)) - ord("A")
    return (question, options, idx) if idx < len(options) else None


def clean(option: str) -> str:
    return option.strip().rstrip(".").strip()


def make_sample(set_name: str, qid: str, picture: str, image: str, question: str, options: List[str],
                answer: int, rng: random.Random, state_text: str = "", extra: Optional[dict] = None) -> Optional[dict]:
    """A choice sample with the options in a random order, or None for unusable options."""
    options = [clean(o) for o in options]
    if any(not o for o in options) or len(set(options)) != len(options) or not 2 <= len(options) <= 10:
        return None
    gold = options[answer]
    order = options[:]
    rng.shuffle(order)
    s = {"id": "v3-%s:%s" % (set_name, qid), "source": set_name, "kind": set_name, "image_id": picture, "image": image,
         "questions": {"q": {"type": "choice", "instructions": question.strip(), "criteria": {o: None for o in order}}},
         "gold": {"q": {"probabilities": {o: float(o == gold) for o in order}}},
         "group": question.strip().lower(), "answer_key": gold, "answer_position": order.index(gold)}
    if state_text.strip():
        s["state_text"] = state_text.strip()
    s.update(extra or {})
    return s


def position_baseline(rows: List[dict]) -> Dict[str, float]:
    """Accuracy of always picking the option at the most common answer position, and chance."""
    if not rows:
        return {}
    pos = Counter(r["answer_position"] for r in rows)
    best = pos.most_common(1)[0][0]
    return {"n": len(rows), "most_common_position": round(sum(r["answer_position"] == best for r in rows) / len(rows), 4),
            "chance": round(sum(1 / len(r["gold"]["q"]["probabilities"]) for r in rows) / len(rows), 4)}


def dhash_bytes(data: bytes) -> int:
    from PIL import Image
    from lv_bench import dhash
    with Image.open(io.BytesIO(data)) as im:
        return dhash(im)


def held_out_hashes(root: str) -> Set[int]:
    """Perceptual hashes of every picture of the evaluation, dev and laya-vision benchmark sets."""
    from PIL import Image
    from lv_bench import dhash
    files = [f for f in glob.glob(os.path.join(root, "*.jsonl")) + glob.glob(os.path.join(root, "v2", "*.jsonl"))
             + glob.glob(os.path.join(root, "lv_bench", "*.jsonl"))
             if os.path.basename(f).startswith(("val", "pope", "eval", "dev", "test_", "bench_", "vqav2_yesno", "aokvqa", "scienceqa"))]
    paths = {json.loads(l)["image"] for f in files for l in open(f) if l.strip()}
    out = set()
    for p in sorted(paths):
        try:
            with Image.open(os.path.join(root, p)) as im:
                out.add(dhash(im))
        except OSError:
            pass
    return out


def save_image(root: str, rel: str, data: bytes) -> None:
    path = os.path.join(root, rel)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        from PIL import Image
        with Image.open(io.BytesIO(data)) as im:
            im.convert("RGB").save(path, quality=92)


def picture_id(data: bytes) -> str:
    return "img:" + hashlib.sha1(data).hexdigest()[:16]


def read_parquets(pattern: str, columns=None) -> Iterable[dict]:
    import pyarrow.parquet as pq
    for f in sorted(glob.glob(pattern)):
        for batch in pq.ParquetFile(f).iter_batches(batch_size=256, columns=columns):
            yield from batch.to_pylist()


HF_SOURCES = {"aokvqa": ("HuggingFaceM4/A-OKVQA", "data/train-*.parquet"),
              "scienceqa": ("derek-thomas/ScienceQA", "data/train-*.parquet"),
              "ai2d": ("HuggingFaceM4/the_cauldron", "ai2d/train-*.parquet"),
              "tqa": ("HuggingFaceM4/the_cauldron", "tqa/train-*.parquet")}


def build_set(root: str, name: str, held: Set[int], rng: random.Random, report: dict) -> List[dict]:
    from fetch import hf_dataset_files
    hf_dataset_files(root, *HF_SOURCES[name])   # into data/raw/hf on first use
    hf = os.path.join(root, "raw", "hf")
    rows: List[dict] = []
    dropped = Counter()

    def add(qid, data, question, options, answer, state_text="", extra=None):
        if dhash_bytes(data) in held:
            dropped["picture in an evaluation set"] += 1
            return
        pic = picture_id(data)
        rel = "v3/images/%s/%s.jpg" % (name, pic[4:])
        s = make_sample(name, qid, pic, rel, question, options, answer, rng, state_text, extra)
        if s is None:
            dropped["unusable options"] += 1
            return
        save_image(root, rel, data)
        rows.append(s)

    if name == "aokvqa":
        for r in read_parquets(os.path.join(hf, "datasets--HuggingFaceM4--A-OKVQA", "snapshots", "*", "data", "train-*.parquet")):
            add(r["question_id"], r["image"]["bytes"], r["question"], list(r["choices"]), int(r["correct_choice_idx"]))
    elif name == "scienceqa":
        for i, r in enumerate(read_parquets(os.path.join(hf, "datasets--derek-thomas--ScienceQA", "snapshots", "*", "data",
                                                         "train-*.parquet"))):
            if r["image"] is None:
                continue
            add("train-%d" % i, r["image"]["bytes"], r["question"], list(r["choices"]), int(r["answer"]),
                state_text=r.get("hint") or "", extra={"subject": r.get("subject"), "category": r.get("category")})
    else:
        for i, r in enumerate(read_parquets(os.path.join(hf, "datasets--HuggingFaceM4--the_cauldron", "snapshots", "*",
                                                         name, "train-*.parquet"))):
            data = r["images"][0]["bytes"]
            for j, t in enumerate(r["texts"]):
                parsed = parse_cauldron(t["user"], t["assistant"])
                if parsed is None:
                    dropped["not a multiple-choice turn"] += 1
                    continue
                add("%d-%d" % (i, j), data, *parsed)
    report[name] = {"kept": len(rows), "dropped": dict(dropped)}
    return rows


def write(path: str, rows: List[dict]) -> int:
    with open(path, "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    return len(rows)


def sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main(argv=None) -> None:
    from v2_common import in_dev

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    p.add_argument("--per-picture", type=int, default=6, help="at most this many questions of a picture in the mix")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    out_dir = os.path.join(a.root, "v3")
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(a.seed)
    from v2_common import require_benchmarks
    require_benchmarks(a.root)
    held = held_out_hashes(a.root)
    report: dict = {"held_out_pictures_hashed": len(held), "sets": {}, "files": {}}
    train_new, dev_all = [], []
    for name in SETS:
        rows = build_set(a.root, name, held, rng, report["sets"])
        dev = [r for r in rows if in_dev(r["image_id"], DEV_SHARE)]
        train = [r for r in rows if not in_dev(r["image_id"], DEV_SHARE)]
        write(os.path.join(out_dir, name + ".jsonl"), train)
        write(os.path.join(out_dir, "dev_%s.jsonl" % name), dev)
        report["sets"][name].update({"train": len(train), "dev": len(dev),
                                     "train_baseline": position_baseline(train), "dev_baseline": position_baseline(dev)})
        train_new += train
        dev_all += dev
    # the mix: data-v2's, round 4's above/below relations, then the new sets, a few questions per picture
    v2 = [json.loads(l) for l in open(os.path.join(a.root, "v2", "train_mix.jsonl"))]
    tb = [json.loads(l) for l in open(os.path.join(a.root, "v2", "train_relation_tb.jsonl"))]
    tb = [r for r in tb if r.get("kind") == "relation" and r.get("axis") == "tb"]
    have = {r["id"] for r in v2}
    per_pic: Counter = Counter()
    extra = []
    shuffled = train_new[:]
    rng.shuffle(shuffled)
    for r in shuffled:
        if per_pic[r["image_id"]] < a.per_picture:
            per_pic[r["image_id"]] += 1
            extra.append(r)
    mix = v2 + [r for r in tb if r["id"] not in have] + extra
    rng.shuffle(mix)
    assert len({r["id"] for r in mix}) == len(mix), "duplicate ids in the mix"
    write(os.path.join(out_dir, "train_mix.jsonl"), mix)
    dev_mix = dev_all + [json.loads(l) for l in open(os.path.join(a.root, "v2", "dev_mix_half.jsonl"))]
    write(os.path.join(out_dir, "dev_mix.jsonl"), dev_mix)
    for f in sorted(glob.glob(os.path.join(out_dir, "*.jsonl"))):
        rows = [json.loads(l) for l in open(f)]
        report["files"][os.path.basename(f)] = {"questions": len(rows), "sha256": sha256(f),
                                                "sources": dict(Counter(r.get("source", "") for r in rows))}
    report["mix"] = {"data_v2_train_mix": len(v2), "relation_above_below": len([r for r in tb if r["id"] not in have]),
                     "new_reasoning_and_diagrams": len(extra), "total": len(mix),
                     "new_share": round(len(extra) / len(mix), 4)}
    report["settings"] = {"seed": a.seed, "per_picture": a.per_picture, "dev_share": DEV_SHARE}
    with open(os.path.join(out_dir, "MANIFEST.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: report[k] for k in ("held_out_pictures_hashed", "sets", "mix")}, indent=1))


if __name__ == "__main__":
    main()
