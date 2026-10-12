"""Public-data pack, VizWiz VQA (CC BY 4.0): photos taken by blind people, each question answered by 10 people.

    uv run python scripts/data/pub/vizwiz.py --root data

Downloads train.zip (~11.3 GB, resumable), val.zip and the annotations; extracts only the pictures that are used.
Kept (answerable questions only; "unanswerable" is not turned into "no"):
  - answer type yes/no, at least 3 yes/no votes -> noul with the vote share as the soft gold
  - "How many ..." with at least 3 votes that are whole numbers 0..10 -> choice over 0..10, soft gold = vote shares
Open answers are dropped. train.json -> train.jsonl; the official val split -> dev.jsonl (monitoring only; the
test split is never used).
"""
import json
import random
import os
import re
import sys
import zipfile
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ext"))
sys.path.insert(0, HERE)
from common import already_built, base_args, download, finish, problems, raw_dir, save_bytes  # noqa: E402
from held import balance_noul, get_held  # noqa: E402

SOURCE = "vizwiz"
BASE = "https://vizwiz.cs.colorado.edu/VizWiz_final/"
LICENCE = "CC BY 4.0 (VizWiz)"
WORDS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten".split())}
WORDS["none"] = 0


def number(a: str):
    a = a.strip().lower().rstrip(".")
    return int(a) if a.isdigit() else WORDS.get(a)


def convert(ann: dict, split: str, rel: str):
    """One VizWiz annotation -> a sample or (None, reason)."""
    if not ann.get("answerable"):
        return None, "unanswerable"
    answers = [x["answer"].strip().lower() for x in ann["answers"]]
    q = " ".join(ann["question"].split())
    sid = "pub-vizwiz:%s:%s" % (split, ann["image"].rsplit(".", 1)[0])
    base = {"id": sid, "source": SOURCE, "image_id": "vizwiz:" + ann["image"], "image": rel,
            "group": "vizwiz:" + ann["image"]}
    if ann["answer_type"] == "yes/no":
        yes, no = answers.count("yes"), answers.count("no")
        if yes + no < 3:
            return None, "fewer than 3 yes/no votes"
        p = yes / (yes + no)
        return dict(base, kind="yesno", answer_key="true" if p >= 0.5 else "false",
                    questions={"q": {"type": "noul", "instructions": q}},
                    gold={"q": {"probabilities": {"false": 1 - p, "true": p}}}), ""
    if re.match(r"(?i)\s*how many\b", q):
        votes = Counter(n for n in map(number, answers) if n is not None and 0 <= n <= 10)
        if sum(votes.values()) < 3:
            return None, "fewer than 3 count votes"
        total = sum(votes.values())
        opts = [str(n) for n in range(11)]
        probs = {o: votes.get(int(o), 0) / total for o in opts}
        return dict(base, kind="count", answer_key=max(probs, key=probs.get), candidates="0..10",
                    questions={"q": {"type": "choice", "instructions": q, "criteria": {o: None for o in opts}}},
                    gold={"q": {"probabilities": probs}}), ""
    return None, "open answer"


def main(argv=None):
    p = base_args(__doc__)
    a = p.parse_args(argv)
    if already_built(a.root, SOURCE, a.force):
        return
    raw = raw_dir(a.root, SOURCE)
    anns = zipfile.ZipFile(download(BASE + "vqa_data/Annotations.zip", os.path.join(raw, "Annotations.zip")))
    held, steps, out = get_held(a.root), Counter(), {}
    for split in ("train", "val"):
        z = zipfile.ZipFile(download(BASE + "images/%s.zip" % split, os.path.join(raw, split + ".zip")))
        names = {os.path.basename(n): n for n in z.namelist() if n.lower().endswith((".jpg", ".png"))}
        rows = []
        for ann in json.load(anns.open(split + ".json")):
            rel = os.path.join("ext", SOURCE, "images", ann["image"])
            r, why = convert(ann, split, rel)
            if r is None:
                steps["%s: %s" % (split, why)] += 1
                continue
            save_bytes(os.path.join(a.root, rel), z.read(names[ann["image"]]))
            rows.append(r)
        near = held.near_held([r["image"] for r in rows], SOURCE)
        steps["%s: near a held-out photo" % split] = len(near)
        out[split] = balance_noul([r for r in rows if r["image"] not in near], random.Random(a.seed))
    bad = [(r["id"], p) for r in out["val"] for p in [problems(r, a.root)] if p]
    if bad:
        raise SystemExit("vizwiz dev: malformed %s" % bad[:3])
    d = os.path.join(a.root, "ext", SOURCE)
    with open(os.path.join(d, "dev.jsonl"), "w") as f:
        for r in out["val"]:
            f.write(json.dumps(r) + "\n")
    steps["dev_questions"] = len(out["val"])
    finish(a.root, SOURCE, out["train"], dict(steps), {"licence": LICENCE})


if __name__ == "__main__":
    main()
