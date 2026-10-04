"""v2 class 4: multiple-choice questions (2-5 options) from VQAv2's open-ended questions (kept out of the repo).

    uv run python scripts/data/v2_vqa_choice.py --root data --limit 40000 --out data/v2/vqa_choice.jsonl
    uv run python scripts/data/v2_vqa_choice.py --root data --split Val --limit 1500 --stratify 150 \\
        --out data/v2/eval_vqa_choice.jsonl

VQAv2's "number" and "other" questions ("how many dogs?", "what color is the bus?", "what fruit is
this?") become choice questions. Every option must be a sensible answer to the question:
  - the question decides the answer category, by its words: "how many" -> counts, "color" ->
    colours, "fruit" -> fruits, "meat" -> meats, "room" -> rooms, "doing" -> activities, ... A question
    that names no category, or whose agreed answer is not in that category's list, is skipped
    (review 2026-10-02: "what fruit is this?" had options yellow / orange / white because "orange" was
    looked up as a colour first);
  - a question that names its alternatives ("are the bananas yellow or green?") gets exactly those as
    options, and is skipped when the agreed answer is not one of them;
  - otherwise distractors come from the same category, weighted by how often each is a right answer
    (so an option's frequency says nothing), never an answer any annotator gave, never a synonym;
  - >= 7 of the 10 annotators agree; 2-5 options; target 1.0 on the agreed answer;
  - recurring wordings ("what color is the sky?") keep their most common answer at most as often as
    the second most common, so the wording alone points at no answer; answers are capped per
    category; reading questions dropped; evaluation images skipped.
`problems()` lists what is wrong with a finished sample; the build fails if any sample has one.
"""
import argparse
import random
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set

# (category, question pattern, answers). The first category whose pattern matches the question wins.
CATEGORIES = [
    ("count", r"^how many\b", "0 1 2 3 4 5 6 7 8 9 10"),
    ("color", r"\bcolou?rs?\b", "white black red blue green yellow brown gray orange pink purple silver tan beige gold"),
    ("sport", r"\bsports?\b", "tennis baseball soccer frisbee skiing snowboarding surfing skateboarding football "
                              "basketball volleyball golf"),
    ("fruit", r"\bfruits?\b", "banana apple orange grape strawberry lemon lime pear pineapple watermelon peach cherry"),
    ("vegetable", r"\bvegetables?\b|\bveggies?\b", "broccoli carrot tomato onion lettuce potato pepper cucumber "
                                                  "celery spinach corn mushroom cauliflower"),
    ("meat", r"\bmeat\b", "chicken beef pork ham bacon sausage turkey fish steak salami pepperoni"),
    ("animal", r"\banimals?\b", "dog cat horse cow sheep elephant giraffe zebra bear bird duck goat pig"),
    ("room", r"\broom\b", "kitchen bathroom bedroom living_room office dining_room"),
    ("vehicle", r"\bvehicles?\b|\btransportation\b", "car bus train truck motorcycle bicycle airplane boat van"),
    ("weather", r"\bweather\b", "sunny cloudy rainy snowy overcast clear foggy"),
    ("material", r"\bmade of\b|\bmade from\b|\bmaterial\b", "wood metal plastic glass brick stone concrete cloth paper ceramic"),
    ("activity", r"\bdoing\b", "eating walking running sitting standing sleeping drinking reading playing riding "
                               "flying swimming surfing skiing skateboarding snowboarding talking cooking cutting posing"),
    ("food", r"\bfood\b|\bdish\b|\bmeal\b|\bdessert\b|\bsnack\b", "pizza sandwich cake donut hot_dog salad soup bread "
                                                                 "rice pasta burger fries cookie pie"),
]
SYNONYMS = [{"gray", "grey", "silver"}, {"tan", "beige", "brown"}, {"bicycle", "bike"}, {"airplane", "plane", "jet"},
            {"donut", "doughnut"}, {"cloudy", "overcast"}, {"sunny", "clear"}, {"cloth", "fabric"},
            {"stone", "concrete", "brick"}, {"lemon", "lime"}, {"soccer", "football"}, {"burger", "sandwich"},
            {"ham", "pork", "bacon"}, {"salami", "pepperoni", "sausage"}, {"standing", "posing"},
            {"car", "van"}, {"cake", "pie"}, {"surfing", "swimming"}]
READING = re.compile(r"\b(say|says|read|written|spell|letter|word|text|brand|logo|label|title|name|time)\b")
MIN_AGREE = 7


def vocabularies() -> Dict[str, List[str]]:
    return {name: [w.replace("_", " ") for w in words.split()] for name, _, words in CATEGORIES}


PATTERNS = [(name, re.compile(pat)) for name, pat, _ in CATEGORIES]


def category_of(question: str) -> Optional[str]:
    """By keyword ("color", "fruit", "how many", ...); failing that, a question naming two or more
    answers of one category as alternatives ("are the bananas yellow or green?") is of that category."""
    q = question.lower()
    for name, pat in PATTERNS:
        if pat.search(q):
            return name
    if " or " in q:
        for name, words in vocabularies().items():
            if len(named_options(q, words)) >= 2:
                return name
    return None


def related(a: str, b: str) -> bool:
    return a == b or any(a in s and b in s for s in SYNONYMS)


def named_options(question: str, vocab: List[str]) -> List[str]:
    """Answers from `vocab` that the question itself names ("yellow or green?" -> yellow, green)."""
    q = question.lower()
    return [w for w in vocab if re.search(r"\b%s\b" % re.escape(w), q)]


ASKS_WHAT_IT_DOES = re.compile(r"\bwhat (is|are) (this|that|the|these|those) animals? \w+ing\b")


def listed_alternatives(question: str, vocab: List[str]) -> Optional[List[str]]:
    """The alternatives a question lists ("yellow or green?", "a car, a bus or a truck?"), when every
    listed phrase holds exactly one word of `vocab`; None when the list is partly outside the
    vocabulary ("dry, snowy, windy or rainy?") or a compound is cut ("a tennis ball or frisbee?").
    The first phrase's word must end it, unless both phrases share their last word ("red bus or blue bus")."""
    tail = question.lower().strip().rstrip("?").strip()
    if " or " not in tail:
        return None
    parts = [p.strip() for p in re.split(r",\s*(?:or\s+)?|\s+or\s+", tail) if p.strip()]
    found: List[str] = []
    for i, part in enumerate(parts):
        words = [w for w in vocab if re.search(r"\b%s\b" % re.escape(w), part)]
        if i == 0:
            last = part.split()[-1]
            words = [w for w in words if part.endswith(w) or (
                part.split()[-2:-1] == [w] and parts[-1].split()[-1:] == [last])]
        if len(words) != 1:
            return None
        found.append(words[0])
    return found if len(set(found)) == len(found) else None


def problems(sample: dict, vocabs: Dict[str, List[str]]) -> List[str]:
    """Why this sample is a bad multiple-choice question; [] when it is fine."""
    q = sample["questions"]["q"]
    opts = list(q["criteria"])
    gold = max(sample["gold"]["q"]["probabilities"], key=sample["gold"]["q"]["probabilities"].get)
    cat = category_of(q["instructions"])
    out = []
    if cat is None or cat != sample["kind"]:
        return ["question category %r, sample says %r" % (cat, sample["kind"])]
    if not 2 <= len(opts) <= 5 or len(set(opts)) != len(opts):
        out.append("option count or duplicates: %s" % opts)
    if gold not in opts:
        out.append("answer not an option")
    off = [o for o in opts if o not in vocabs[cat]]
    if off:
        out.append("options outside the %s list: %s" % (cat, off))
    if " or " in q["instructions"].lower():
        listed = listed_alternatives(q["instructions"], vocabs[cat])
        if listed is None or set(opts) != set(listed):
            out.append("question lists %s, options are %s" % (listed, opts))
    if cat == "animal" and ASKS_WHAT_IT_DOES.search(q["instructions"].lower()):
        out.append("asks what the animal does, not which animal")
    for i, a in enumerate(opts):
        for b in opts[i + 1:]:
            if related(a, b):
                out.append("synonymous options %s / %s" % (a, b))
    return out


def convert(question: dict, annotation: dict, split: str, vocabs, rng: random.Random) -> Optional[dict]:
    if annotation["answer_type"] == "yes/no":
        return None
    text = question["question"]
    if READING.search(text.lower()):
        return None
    cat = category_of(text)
    if cat is None:
        return None
    votes = Counter(a["answer"].strip().lower() for a in annotation["answers"])
    answer, k = votes.most_common(1)[0]
    if k < MIN_AGREE or answer not in vocabs[cat]:
        return None
    if cat == "animal" and ASKS_WHAT_IT_DOES.search(text.lower()):
        return None       # "what is this animal eating?" is not answered by an animal
    fixed: Optional[List[str]] = None
    if " or " in text.lower():
        listed = listed_alternatives(text, vocabs[cat])
        if listed is None or answer not in listed or not 2 <= len(listed) <= 5 or any(
                related(x, y) for i, x in enumerate(listed) for y in listed[i + 1:]):
            return None   # also "cloudy or overcast?": no single right answer
        fixed = listed
    iid = int(question["image_id"])
    return {"id": "v2-vqachoice:%d" % question["question_id"], "source": "vqav2-choice", "kind": cat,
            "image_id": "coco:%d" % iid, "image": "coco/%s/COCO_%s_%012d.jpg" % (split, split, iid),
            "questions": {"q": {"type": "choice", "instructions": text, "criteria": {}}},
            "gold": {"q": {"probabilities": {answer: 1.0}}},
            "_answer": answer, "_given": sorted(votes), "_fixed": fixed,
            "_n_options": rng.choice([2, 3, 3, 4, 4, 5])}


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.lower().strip())


def select(samples: List[dict], limit: int, held_out: Set[str], rng: random.Random, stratify: int = 0) -> List[dict]:
    """Flatten recurring wordings and per-category answers, then sample (see module doc)."""
    samples = [s for s in samples if s["image_id"] not in held_out]
    # 1. a recurring wording keeps its top answer at most as often as its second answer
    by_text: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for s in samples:
        by_text[norm(s["questions"]["q"]["instructions"])][s["_answer"]].append(s)
    flat: List[dict] = []
    for answers in by_text.values():
        counts = sorted((len(v) for v in answers.values()), reverse=True)
        cap = counts[0] if sum(counts) < 5 else (counts[1] if len(counts) > 1 else 1)
        for v in answers.values():
            rng.shuffle(v)
            flat += v[:cap]
    # 2. cap every answer within its category at twice the category's mean
    by = defaultdict(lambda: defaultdict(list))
    for s in flat:
        by[s["kind"]][s["_answer"]].append(s)
    capped: List[dict] = []
    for answers in by.values():
        cap = max(1, 2 * sum(len(v) for v in answers.values()) // len(answers))
        for v in answers.values():
            rng.shuffle(v)
            capped += v[:cap]
    rng.shuffle(capped)
    # 3. optional floor per category (evaluation sets), then the rest; at most 3 questions per image
    if stratify:
        taken: Counter = Counter()
        first, rest = [], []
        for s in capped:
            (first if taken[s["kind"]] < stratify else rest).append(s)
            taken[s["kind"]] += 1
        capped = first + rest
    per_image: Counter = Counter()
    out = []
    for s in capped:
        if per_image[s["image_id"]] < 3 and len(out) < limit:
            per_image[s["image_id"]] += 1
            out.append(s)
    return out


def finish(samples: List[dict], vocabs, rng: random.Random) -> List[dict]:
    """Choose the options: the named ones, or distractors weighted by how often each word is a right
    answer in `samples`. Drops samples without enough sensible distractors and the helper fields."""
    freq: Dict[str, Counter] = defaultdict(Counter)
    for s in samples:
        freq[s["kind"]][s["_answer"]] += 1
    out = []
    for s in samples:
        answer, cat = s["_answer"], s["kind"]
        if s["_fixed"]:
            options = list(s["_fixed"])
        else:
            given = set(s["_given"]) | set(named_options(s["questions"]["q"]["instructions"], vocabs[cat]))
            pool = [w for w in vocabs[cat] if freq[cat][w] and not any(related(w, x) for x in given | {answer})]
            if cat == "count":
                pool = [w for w in pool if abs(int(w) - int(answer)) <= 3]
            wrong: List[str] = []
            while pool and len(wrong) < s["_n_options"] - 1:
                w = rng.choices(pool, weights=[freq[cat][x] for x in pool])[0]
                wrong.append(w)
                pool = [x for x in pool if not related(x, w)]
            if len(wrong) < s["_n_options"] - 1:
                continue
            options = [answer] + wrong
        rng.shuffle(options)
        s = {k: v for k, v in s.items() if not k.startswith("_")}
        s["group"], s["answer_key"] = norm(s["questions"]["q"]["instructions"]), answer
        s["questions"]["q"]["criteria"] = {o: None for o in options}
        s["gold"] = {"q": {"probabilities": {o: float(o == answer) for o in options}}}
        out.append(s)
    return out


def main(argv=None) -> None:
    import glob
    import os

    from convert import same_images
    from fetch import download_coco_images, vg_to_coco, vqav2_records
    from prepare import _write_jsonl, read_jsonl

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    p.add_argument("--split", default="Train", help="VQAv2 split: Train or Val")
    p.add_argument("--limit", type=int, default=40000)
    p.add_argument("--stratify", type=int, default=0, help="take up to N per category first (evaluation sets)")
    p.add_argument("--exclude", nargs="*", default=[], help="more JSONL files whose images must not be used")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    coco_split = "train2014" if a.split == "Train" else "val2014"
    rng = random.Random(a.seed)
    vocabs = vocabularies()
    evals = [f for f in glob.glob(os.path.join(a.root, "*.jsonl")) + glob.glob(os.path.join(a.root, "v2", "*.jsonl"))
             if os.path.basename(f).startswith(("val", "pope", "eval", "dev", "test_", "bench_"))
             and os.path.abspath(f) != os.path.abspath(a.out)]
    held_out = same_images({s["image_id"] for f in evals + a.exclude for s in read_jsonl(f)}, vg_to_coco(a.root))
    samples = [s for s in (convert(q, ann, coco_split, vocabs, rng) for q, ann in vqav2_records(a.root, a.split)) if s]
    out = finish(select(samples, a.limit, held_out, rng, a.stratify), vocabs, rng)
    bad = [(s["id"], problems(s, vocabs)) for s in out if problems(s, vocabs)]
    if bad:
        raise SystemExit("%d bad samples, e.g. %s" % (len(bad), bad[:3]))
    failed = set(download_coco_images(a.root, out))
    out = [s for s in out if s["id"] not in failed]
    print("%s: %d questions from %d candidates; per category %s; options %s; %d images failed" % (
        a.out, _write_jsonl(a.out, out), len(samples), dict(Counter(s["kind"] for s in out)),
        dict(sorted(Counter(len(s["gold"]["q"]["probabilities"]) for s in out).items())), len(failed)))


if __name__ == "__main__":
    main()
