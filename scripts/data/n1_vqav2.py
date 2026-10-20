"""data-n1, VQAv2 part: every usable VQAv2 train question as a decision question, with the annotators' answers as
soft gold (docs/experiments/2026-10-20-r16-cuda-plan.md, "data-n1").

    uv run python scripts/data/n1_vqav2.py --root data

Unlike data-v2 (40k yes/no + 40k choice, >= 7 of 10 annotators agreeing, gold 1.0), every question with a usable
consensus is kept and its gold is the share of the ten answers each option got:
  yes/no  noul; gold P(true) = yes / (yes + no); >= 8 of the 10 answers are yes or no;
  count   "how many" with >= 6 answers in 0..10; options are the whole set 0..10 in a fresh random order (no
          window around the answer, which a no-picture rule could exploit); gold = the answers' shares;
  other   choice with 3-5 options: the answers >= 3 annotators gave (at most 2), then distractors -- from the
          question's category list (data-v2's categories) when it has one, else from the agreed answers of
          other questions that start with the same three words, weighted by how often each is right. A
          distractor is never an answer any annotator gave, a synonym of one, or a word containing one.
          The top answer needs >= 4 votes; gold = shares among the options.
Dropped: reading questions (data-v2's list), questions whose answer is outside the type's rules, pictures of every
evaluation set under <root> (COCO and VG ids of a picture both, convert.same_images). A recurring wording (5 or
more questions) keeps each top answer at most as often as its second most common one, so the wording alone does
not point at an answer. 2% of the pictures (v2_common.in_dev on "n1:<image id>", independent of data-v2's dev pictures, which are held out) go to dev.jsonl for monitoring / temperatures.
Writes <root>/n1/vqav2_train.jsonl, dev.jsonl and MANIFEST.json (counts, option counts, a question-only baseline).
"""
import argparse
import glob
import json
import os
import random
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional

from v2_common import in_dev
from v2_vqa_choice import READING, category_of, listed_alternatives, related, vocabularies

NUMBERS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten".split())}
NUMBERS.update({str(i): i for i in range(11)})
NUMBERS["none"] = 0
DEV_SHARE = 0.02


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.lower().strip().rstrip("?").strip())


def prefix(t: str) -> str:
    return " ".join(norm(t).split()[:3])


def clash(a: str, b: str) -> bool:
    """Too close to be a wrong option next to `b`: the same, a synonym, or one contains the other as a word."""
    return related(a, b) or bool(re.search(r"\b%s\b" % re.escape(a), b)) or bool(re.search(r"\b%s\b" % re.escape(b), a))


def _sample(qid: int, kind: str, iid: int, split: str, question: Dict, probs: Dict[str, float]) -> Dict:
    return {"id": "n1-vqa:%d" % qid, "source": "vqav2", "kind": kind, "image_id": "coco:%d" % iid,
            "image": "coco/%s/COCO_%s_%012d.jpg" % (split, split, iid), "group": norm(question["instructions"]),
            "questions": {"q": question}, "gold": {"q": {"probabilities": probs}}}


def convert(question: Dict, annotation: Dict, split: str, rng: random.Random) -> Optional[Dict]:
    """The sample of one VQAv2 question, or None; `other` samples get their options later (`finish`)."""
    text = question["question"].strip()
    if READING.search(text.lower()):
        return None
    votes = Counter(a["answer"].strip().lower() for a in annotation["answers"])
    qid, iid = question["question_id"], int(question["image_id"])
    if annotation["answer_type"] == "yes/no":
        yes, no = votes["yes"], votes["no"]
        if yes + no < 8:
            return None
        return _sample(qid, "yesno", iid, split, {"type": "noul", "instructions": text},
                       {"false": no / (yes + no), "true": yes / (yes + no)})
    if annotation["answer_type"] == "number":
        if not text.lower().startswith("how many"):
            return None
        counts = Counter()
        for a, k in votes.items():
            if a in NUMBERS:
                counts[NUMBERS[a]] += k
        if sum(counts.values()) < 6:
            return None
        options = [str(i) for i in range(11)]
        rng.shuffle(options)
        total = sum(counts.values())
        return _sample(qid, "count", iid, split, {"type": "choice", "instructions": text,
                                                   "criteria": {o: None for o in options}},
                       {o: counts[int(o)] / total for o in options})
    top, k = votes.most_common(1)[0]
    if k < 4 or top in ("yes", "no") or top in NUMBERS or not top or len(top) > 30:
        return None
    s = _sample(qid, "other", iid, split, {"type": "choice", "instructions": text, "criteria": {}}, {})
    s["_votes"], s["_category"] = dict(votes), category_of(text)
    s["_n"] = rng.choice([3, 4, 4, 5])
    return s


def flatten(samples: List[Dict], rng: random.Random) -> List[Dict]:
    """A wording asked 5+ times keeps each top answer at most as often as its second most common top answer."""
    by_text: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for s in samples:
        p = s["gold"]["q"]["probabilities"] or s["_votes"]
        by_text[s["group"]][max(p, key=p.get)].append(s)
    out: List[Dict] = []
    for answers in by_text.values():
        sizes = sorted((len(v) for v in answers.values()), reverse=True)
        cap = sizes[0] if sum(sizes) < 5 else (sizes[1] if len(sizes) > 1 else 1)
        for v in answers.values():
            rng.shuffle(v)
            out += v[:cap]
    return out


def finish(samples: List[Dict], rng: random.Random) -> List[Dict]:
    """Options and soft gold of the `other` samples; samples without enough distractors are dropped."""
    vocabs = vocabularies()
    freq: Dict[str, Counter] = defaultdict(Counter)       # prefix -> agreed answers
    for s in samples:
        if s["kind"] == "other":
            top = max(s["_votes"], key=s["_votes"].get)
            freq[prefix(s["questions"]["q"]["instructions"])][top] += 1
    out = []
    for s in samples:
        if s["kind"] != "other":
            out.append(s)
            continue
        text, votes, cat = s["questions"]["q"]["instructions"], s.pop("_votes"), s.pop("_category")
        n = s.pop("_n")
        agreed = [a for a, k in sorted(votes.items(), key=lambda x: -x[1]) if k >= 3][:2]
        if len(agreed) == 2 and clash(agreed[0], agreed[1]):
            agreed = agreed[:1]
        if " or " in text.lower():
            listed = listed_alternatives(text, vocabs[cat]) if cat else None
            if not listed or agreed[0] not in listed or not 2 <= len(listed) <= 5:
                continue
            options = listed
        else:
            if cat and agreed[0] in vocabs[cat]:
                weights = Counter({w: 1 + freq[prefix(text)][w] for w in vocabs[cat]})
            else:
                weights = Counter(dict(freq[prefix(text)]))
            pool = [w for w in weights if not any(clash(w, g) for g in votes)]
            wrong: List[str] = []
            while pool and len(agreed) + len(wrong) < n:
                w = rng.choices(pool, weights=[weights[x] for x in pool])[0]
                wrong.append(w)
                pool = [x for x in pool if not clash(x, w)]
            if len(agreed) + len(wrong) < 3:
                continue
            options = agreed + wrong
        rng.shuffle(options)
        total = sum(votes.get(o, 0) for o in options)
        s["questions"]["q"]["criteria"] = {o: None for o in options}
        s["gold"]["q"]["probabilities"] = {o: votes.get(o, 0) / total for o in options}
        out.append(s)
    return out


def text_baseline(train: List[Dict], dev: List[Dict]) -> Dict[str, float]:
    """Accuracy on dev of answering from the wording alone: the most common top answer of the same wording in
    train among the options, else of the same first three words, else the first option."""
    def top(s):
        p = s["gold"]["q"]["probabilities"]
        return max(p, key=p.get)

    by_text, by_prefix = defaultdict(Counter), defaultdict(Counter)
    for s in train:
        t = s["questions"]["q"]["instructions"]
        by_text[norm(t)][top(s)] += 1
        by_prefix[prefix(t)][top(s)] += 1
    right: Counter = Counter()
    n: Counter = Counter()
    for s in dev:
        q = s["questions"]["q"]
        opts = list(s["gold"]["q"]["probabilities"])
        guess = opts[0]
        for table in (by_text[norm(q["instructions"])], by_prefix[prefix(q["instructions"])]):
            ranked = [a for a, _ in table.most_common() if a in opts]
            if ranked:
                guess = ranked[0]
                break
        n[s["kind"]] += 1
        right[s["kind"]] += guess == top(s)
    return {k: round(right[k] / n[k], 4) for k in n}


def main(argv=None) -> None:
    from convert import same_images
    from fetch import vg_to_coco, vqav2_records
    from prepare import _write_jsonl, read_jsonl

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    rng = random.Random(a.seed)
    out_dir = os.path.join(a.root, "n1")
    evals = [f for f in glob.glob(os.path.join(a.root, "**", "*.jsonl"), recursive=True)
             if os.path.basename(f).startswith(("val", "pope", "eval", "dev", "test_", "bench_"))
             and os.sep + "raw" + os.sep not in f and not f.startswith(out_dir)]
    held_out = same_images({s["image_id"] for f in evals for s in read_jsonl(f)}, vg_to_coco(a.root))
    raw = [s for s in (convert(q, ann, "train2014", rng) for q, ann in vqav2_records(a.root, "Train")) if s]
    kept = [s for s in raw if s["image_id"] not in held_out
            and os.path.exists(os.path.join(a.root, s["image"]))]
    samples = finish(flatten(kept, rng), rng)
    rng.shuffle(samples)
    dev = [s for s in samples if in_dev("n1:" + s["image_id"], DEV_SHARE)]
    train = [s for s in samples if not in_dev("n1:" + s["image_id"], DEV_SHARE)]
    os.makedirs(out_dir, exist_ok=True)
    manifest = {
        "converted": len(raw), "held_out_pictures_dropped": len(raw) - len(kept), "eval_files": len(evals),
        "train": _write_jsonl(os.path.join(out_dir, "vqav2_train.jsonl"), train),
        "dev": _write_jsonl(os.path.join(out_dir, "dev.jsonl"), dev),
        "train_by_kind": dict(Counter(s["kind"] for s in train)),
        "options": dict(sorted(Counter(len(s["gold"]["q"]["probabilities"]) for s in train).items())),
        "yes_share": round(sum(s["gold"]["q"]["probabilities"]["true"] for s in train if s["kind"] == "yesno")
                           / max(1, sum(s["kind"] == "yesno" for s in train)), 4),
        "pictures": len({s["image_id"] for s in train}),
        "text_only_baseline_dev": text_baseline(train, dev),
    }
    with open(os.path.join(out_dir, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
