"""v2 class 3: VQAv2 yes/no questions (kept out of the repo).

    uv run python scripts/data/v2_vqa_yesno.py --root data --limit 60000 --out data/v2/vqa_yesno.jsonl

Same conversion as convert.convert_vqav2 (noul, target = share of annotators who said yes), with:
  - questions that need reading text dropped ("does the sign say ...", "is this a 7") -- at 256 px
    letters are not legible;
  - questions whose wording recurs (>= 8 times, "is it sunny?") balanced to equal yes and no, so the
    wording alone answers nothing; rarer questions are taken as they come;
  - overall yes / no balanced, at most 3 questions per image;
  - evaluation images (COCO and Visual Genome ids) skipped.
"""
import argparse
import random
import re
from collections import Counter, defaultdict
from typing import Dict, List, Set

READING = re.compile(r"\b(say|says|said|read|reads|written|write|spell|letter|letters|word|words|text|"
                     r"number|numbers|digit|brand|logo|label|title|name)\b|\b(is this|is that|is it) (a |an |the )?\d")
RECURRING = 8
MAX_PER_IMAGE = 3


def norm(q: str) -> str:
    return re.sub(r"\s+", " ", q.lower().strip())


def needs_reading(question: str) -> bool:
    return bool(READING.search(question.lower()))


def select(samples: List[dict], limit: int, held_out: Set[str], rng: random.Random) -> List[dict]:
    """Balanced, reading-free, eval-image-free subset (see module doc)."""
    def yes(s):
        return s["gold"]["q"]["probabilities"]["true"] > 0.5

    def tie(s):
        return s["gold"]["q"]["probabilities"]["true"] == 0.5

    pool = [s for s in samples if s["image_id"] not in held_out and not tie(s)
            and not needs_reading(s["questions"]["q"]["instructions"])]
    by_text: Dict[str, List[dict]] = defaultdict(list)
    for s in pool:
        by_text[norm(s["questions"]["q"]["instructions"])].append(s)
    kept: List[dict] = []
    for text, ss in by_text.items():
        if len(ss) >= RECURRING:
            y, n = [s for s in ss if yes(s)], [s for s in ss if not yes(s)]
            rng.shuffle(y)
            rng.shuffle(n)
            k = min(len(y), len(n))
            kept += y[:k] + n[:k]
        else:
            kept += ss
    rng.shuffle(kept)
    y, n = [s for s in kept if yes(s)], [s for s in kept if not yes(s)]
    per_image: Counter = Counter()
    out: List[dict] = []
    for pair in zip(y, n):          # alternate, so any prefix stays balanced
        for s in pair:
            if per_image[s["image_id"]] < MAX_PER_IMAGE and len(out) < limit:
                per_image[s["image_id"]] += 1
                out.append(s)
    rng.shuffle(out)
    recurring = Counter(norm(s["questions"]["q"]["instructions"]) for s in out)
    for s in out:
        t = norm(s["questions"]["q"]["instructions"])
        s["group"] = t if recurring[t] >= RECURRING else "_rare"
        s["answer_key"] = "true" if yes(s) else "false"
    return out


def main(argv=None) -> None:
    import glob
    import os

    from convert import convert_vqav2, same_images
    from fetch import download_coco_images, vg_to_coco, vqav2_records
    from prepare import _write_jsonl, read_jsonl

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="data")
    p.add_argument("--limit", type=int, default=60000)
    p.add_argument("--split", default="Train", choices=["Train", "Val"], help="VQAv2 split")
    p.add_argument("--out", required=True)
    p.add_argument("--exclude", nargs="*", default=[], help="more JSONL files whose images must not be used")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    evals = [f for f in glob.glob(os.path.join(a.root, "*.jsonl")) + glob.glob(os.path.join(a.root, "v2", "*.jsonl"))
             if os.path.basename(f).startswith(("val", "pope", "eval", "dev", "test_", "bench_")) and os.path.abspath(f) != os.path.abspath(a.out)]
    held_out = same_images({s["image_id"] for f in evals + a.exclude for s in read_jsonl(f)}, vg_to_coco(a.root))
    coco_split = "train2014" if a.split == "Train" else "val2014"
    samples = [s for s in (convert_vqav2(q, ann, coco_split) for q, ann in vqav2_records(a.root, a.split)) if s]
    for s in samples:
        s["source"], s["kind"] = "vqav2-yesno", "yesno"
    out = select(samples, a.limit, held_out, random.Random(a.seed))
    failed = set(download_coco_images(a.root, out))
    out = [s for s in out if s["id"] not in failed]
    print("%s: %d questions from %d yes/no candidates (%d need reading), %d images failed" % (
        a.out, _write_jsonl(a.out, out), len(samples),
        sum(needs_reading(s["questions"]["q"]["instructions"]) for s in samples), len(failed)))


if __name__ == "__main__":
    main()
