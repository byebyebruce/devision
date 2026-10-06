"""Extension pack, SNLI-VE (Flickr30k photos + an SNLI hypothesis: does the picture entail it or contradict it?).
Licence: SNLI-VE / e-SNLI-VE annotations BSD-3-Clause (SNLI: CC BY-SA 4.0); pictures under the Flickr30k terms
(non-commercial research use).

    uv run python scripts/data/ext/snli_ve.py --root data [--limit 60000] [--per-picture 3] [--rounds 2] [--force]

Source: J1mb0o/e-SNLI-VE, train split (401,717 rows; the e-SNLI-VE release, whose labels were cleaned of the
original SNLI-VE neutral-label errors; pictures embedded, saved once to data/ext/snli_ve/images/<flickr id>.jpg).
Entailment -> yes, contradiction -> no, neutral dropped. Question: 'Is this true of the picture? "<hypothesis>"'.

SNLI hypotheses give the label away from the text alone (negations, "sleeping", "outdoors", "people"...), so:
  1. hypotheses with a negation / absence cue (NEGATION) are dropped;
  2. text-only debiasing, --rounds times: a Naive Bayes model on the hypothesis words and word pairs scores every
     row out-of-fold (2 folds), rows are put in 20 bins by that score and each bin keeps as many yes as no; after
     this the text score says nothing about the answer;
  3. pictures near a held-out photo (HeldOut.near_held) dropped; at most --per-picture questions per picture;
     yes = no; at most --limit questions.
The MANIFEST's text_only_baseline gives a fresh Naive Bayes' accuracy (fitted on one half, measured on the other)
before and after each step.
"""
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (HeldOut, already_built, base_args, cap_per_picture, finish, hf_files, noul,  # noqa: E402
                    save_bytes)

SOURCE = "snli_ve"
REPO = "J1mb0o/e-SNLI-VE"
ENTAILMENT, NEUTRAL, CONTRADICTION = 0, 1, 2
NEGATION = {"not", "no", "n't", "nobody", "nothing", "never", "none", "nowhere", "neither", "nor", "cannot",
            "without", "sleeping", "sleeps", "asleep", "alone", "empty"}


# ---------------------------------------------------------------- pure conversion

def tokens(text: str) -> List[str]:
    """Lower-case words; "isn't" -> ["is", "n't"]."""
    return re.findall(r"n't|[a-z]+", text.lower().replace("n't", " n't"))


def has_negation(text: str) -> bool:
    return bool(set(tokens(text)) & NEGATION)


def question(hypothesis: str) -> str:
    return 'Is this true of the picture? "%s"' % " ".join(hypothesis.split())


def features(text: str) -> Set[str]:
    """Words, word pairs and a length bucket (entailed hypotheses are shorter)."""
    w = tokens(text)
    return set(w) | {a + "_" + b for a, b in zip(w, w[1:])} | {"LEN%d" % min(len(w) // 2, 8)}


class NaiveBayes:
    """Bernoulli-style Naive Bayes over feature sets; score > 0 means "yes" is more likely."""

    def fit(self, X: Sequence[Set[str]], y: Sequence[bool], min_count: int = 3) -> "NaiveBayes":
        c: Dict[bool, Counter] = {True: Counter(), False: Counter()}
        n = Counter(y)
        for f, label in zip(X, y):
            c[label].update(f)
        self.w = {k: math.log((c[True][k] + 1) / (n[True] + 2)) - math.log((c[False][k] + 1) / (n[False] + 2))
                  for k in set(c[True]) | set(c[False]) if c[True][k] + c[False][k] >= min_count}
        self.b = math.log((n[True] + 1) / (n[False] + 1))
        return self

    def score(self, f: Iterable[str]) -> float:
        return self.b + sum(self.w.get(k, 0.0) for k in f)


def text_baseline(texts: Sequence[str], labels: Sequence[bool], seed: int = 0) -> dict:
    """Accuracy of a Naive Bayes on the text alone, fitted on a random half and measured on the other half."""
    idx = list(range(len(texts)))
    random.Random(seed).shuffle(idx)
    fit, test = idx[: len(idx) // 2], idx[len(idx) // 2:]
    m = NaiveBayes().fit([features(texts[i]) for i in fit], [labels[i] for i in fit])
    hit = sum((m.score(features(texts[i])) > 0) == labels[i] for i in test)
    yes = sum(labels) / max(1, len(labels))
    return {"n": len(texts), "yes_share": round(yes, 4), "majority": round(max(yes, 1 - yes), 4),
            "accuracy": round(hit / max(1, len(test)), 4)}


def debias(texts: Sequence[str], labels: Sequence[bool], seed: int, bins: int = 20, folds: int = 2) -> List[int]:
    """Indices to keep: every row is scored by a Naive Bayes fitted on the other folds; in each of `bins` equal
    bins of that score, as many yes as no are kept (the rows dropped are the surplus answer, chosen at random)."""
    idx = list(range(len(texts)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    feats = [features(t) for t in texts]
    scored: List[Tuple[float, int]] = []
    for k in range(folds):
        test = idx[k::folds]
        held = set(test)
        fit = [i for i in idx if i not in held]
        m = NaiveBayes().fit([feats[i] for i in fit], [labels[i] for i in fit])
        scored += [(m.score(feats[i]), i) for i in test]
    scored.sort()
    keep: List[int] = []
    n = len(scored)
    for b in range(bins):
        chunk = [i for _, i in scored[b * n // bins:(b + 1) * n // bins]]
        yes = [i for i in chunk if labels[i]]
        no = [i for i in chunk if not labels[i]]
        k_ = min(len(yes), len(no))
        rng.shuffle(yes)
        rng.shuffle(no)
        keep += yes[:k_] + no[:k_]
    return sorted(keep)


def balance(rows: List[dict], limit: int, rng: random.Random) -> List[dict]:
    """As many yes as no, at most `limit` in total (random choice within each answer)."""
    yes = [r for r in rows if r["answer_key"] == "true"]
    no = [r for r in rows if r["answer_key"] == "false"]
    rng.shuffle(yes)
    rng.shuffle(no)
    k = min(len(yes), len(no), limit // 2)
    return yes[:k] + no[:k]


# ---------------------------------------------------------------- build

def read_text(paths: Sequence[str]) -> List[dict]:
    import pyarrow.parquet as pq
    out = []
    for s, path in enumerate(paths):
        t = pq.read_table(path, columns=["flickr_id", "hypothesis", "gold_label"]).to_pylist()
        shard = os.path.basename(path).split("-of-")[0]
        out += [dict(r, key="%s:%d" % (shard, i)) for i, r in enumerate(t)]
    return out


def save_pictures(root: str, paths: Sequence[str], wanted: Set[str]) -> Dict[str, str]:
    """flickr id -> picture path (relative to root) for every wanted id, each saved once."""
    import pyarrow.parquet as pq
    rels = {f: os.path.join("ext", SOURCE, "images", f) for f in wanted}
    todo = {f for f in wanted if not os.path.exists(os.path.join(root, rels[f]))}
    if todo:
        print("saving %d pictures" % len(todo), flush=True)
    for path in paths:
        if not todo:
            break
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=256, columns=["flickr_id", "image"]):
            for fid, im in zip(batch.column("flickr_id").to_pylist(), batch.column("image").to_pylist()):
                if fid in todo and im and im.get("bytes"):
                    save_bytes(os.path.join(root, rels[fid]), im["bytes"])
                    todo.discard(fid)
    if todo:
        raise SystemExit("%d pictures not found in the parquet files, e.g. %s" % (len(todo), sorted(todo)[:3]))
    return rels


def main(argv=None):
    p = base_args(__doc__)
    p.add_argument("--limit", type=int, default=60000)
    p.add_argument("--per-picture", type=int, default=3)
    p.add_argument("--rounds", type=int, default=2, help="text-only debiasing rounds")
    a = p.parse_args(argv)
    if already_built(a.root, SOURCE, a.force):
        return
    rng = random.Random(a.seed)
    paths = hf_files(a.root, REPO, "data/train-*.parquet")
    rows = read_text(paths)
    steps = Counter(rows=len(rows))
    labels = Counter(r["gold_label"] for r in rows)
    steps.update({"entailment": labels[ENTAILMENT], "neutral (dropped)": labels[NEUTRAL],
                  "contradiction": labels[CONTRADICTION]})
    rows = [r for r in rows if r["gold_label"] in (ENTAILMENT, CONTRADICTION) and r["hypothesis"].strip()]
    baseline = {}

    def measure(name: str, rs: List[dict]) -> None:
        baseline[name] = text_baseline([r["hypothesis"] for r in rs], [r["gold_label"] == ENTAILMENT for r in rs],
                                       a.seed)
        print("text-only baseline %-22s %s" % (name, baseline[name]), flush=True)

    measure("entail_vs_contradict", rows)
    rows = [r for r in rows if not has_negation(r["hypothesis"])]
    steps["after_negation_drop"] = len(rows)
    measure("after_negation_drop", rows)
    for k in range(a.rounds):
        keep = debias([r["hypothesis"] for r in rows], [r["gold_label"] == ENTAILMENT for r in rows], a.seed + k)
        rows = [rows[i] for i in keep]
        steps["after_debias_round_%d" % (k + 1)] = len(rows)
        measure("after_debias_round_%d" % (k + 1), rows)

    rels = save_pictures(a.root, paths, {r["flickr_id"] for r in rows})
    near = HeldOut(a.root).near_held(set(rels.values()), SOURCE)
    rows = [r for r in rows if rels[r["flickr_id"]] not in near]
    steps["after_held_out"] = len(rows)

    samples = [noul("ext-snli_ve:%s" % r["key"], SOURCE, "entailment", "flickr30k:%s" % r["flickr_id"].split(".")[0],
                    rels[r["flickr_id"]], question(r["hypothesis"]), r["gold_label"] == ENTAILMENT, group="snli-ve",
                    hypothesis=r["hypothesis"])
               for r in rows]
    samples = cap_per_picture(samples, a.per_picture, rng)
    steps["after_per_picture_cap"] = len(samples)
    samples = balance(samples, a.limit, rng)
    samples.sort(key=lambda r: r["id"])
    steps["kept"] = len(samples)
    baseline["final"] = text_baseline([r["hypothesis"] for r in samples], [r["answer_key"] == "true" for r in samples],
                                      a.seed)
    print("text-only baseline %-22s %s" % ("final", baseline["final"]), flush=True)
    finish(a.root, SOURCE, samples, dict(steps), {
        "text_only_baseline": baseline,
        "text_only_baseline_note": "Naive Bayes on hypothesis words + word pairs + length, fitted on a random half, "
                                   "accuracy on the other half (chance 0.5 when yes = no)",
        "negation_cues": sorted(NEGATION), "dropped_near_held": len(near),
        "per_picture": dict(sorted(Counter(Counter(r["image_id"] for r in samples).values()).items())),
        "licence": "annotations BSD-3-Clause (SNLI-VE; SNLI CC BY-SA 4.0); pictures Flickr30k terms "
                   "(non-commercial research use)"})


if __name__ == "__main__":
    main()
