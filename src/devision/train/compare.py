"""Paired comparison of two evaluations of the same questions, from their per-question records.

The questions of one picture are not independent (POPE asks 18 per picture), so the bootstrap
resamples pictures, keeping all of a picture's questions together. An interval that covers 0 means the
data cannot tell the two apart -- not that they are equal.
"""
import json
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

Record = Dict[str, Any]


def load_records(path: str) -> List[Record]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _paired(a: List[Record], b: List[Record]) -> List[Tuple[Record, Record]]:
    index = {(r["sample_id"], r["qid"]): r for r in a}
    pairs = [(index[k], r) for r in b if (k := (r["sample_id"], r["qid"])) in index]
    if not pairs:
        raise ValueError("the two evaluations share no questions")
    return pairs


def _interval(pairs: List[Tuple[Record, Record]], resamples: int, rng: random.Random) -> Dict[str, Any]:
    by_picture: Dict[str, List[float]] = defaultdict(list)
    for x, y in pairs:
        by_picture[x["image_id"]].append(float(y["correct"]) - float(x["correct"]))
    pictures = list(by_picture.values())
    sums = [sum(v) for v in pictures]
    counts = [len(v) for v in pictures]
    diff = sum(sums) / sum(counts)
    boot = []
    for _ in range(resamples):
        idx = [rng.randrange(len(pictures)) for _ in pictures]
        boot.append(sum(sums[i] for i in idx) / max(1, sum(counts[i] for i in idx)))
    boot.sort()
    lo, hi = boot[int(0.025 * resamples)], boot[min(resamples - 1, int(0.975 * resamples))]
    return {"questions": len(pairs), "pictures": len(pictures),
            "accuracy_a": sum(float(x["correct"]) for x, _ in pairs) / len(pairs),
            "accuracy_b": sum(float(y["correct"]) for _, y in pairs) / len(pairs),
            "difference": diff, "ci95": [lo, hi], "covers_zero": lo <= 0 <= hi}


def compare(a: List[Record], b: List[Record], by: str = "source", resamples: int = 2000, seed: int = 0,
            only: Optional[Set[str]] = None) -> Dict[str, Any]:
    """`only`: restrict to these sample ids (e.g. the questions neither model's training saw)."""
    pairs = _paired(a, b)
    if only is not None:
        pairs = [(x, y) for x, y in pairs if x["sample_id"] in only]
        if not pairs:
            raise ValueError("no shared question is in `only`")
    rng = random.Random(seed)
    out: Dict[str, Any] = {"all": _interval(pairs, resamples, rng), "by": by, "groups": {}}
    groups: Dict[str, List[Tuple[Record, Record]]] = defaultdict(list)
    for x, y in pairs:
        groups[str(x.get(by) or "-")].append((x, y))
    for g, ps in sorted(groups.items()):
        out["groups"][g] = _interval(ps, resamples, rng)
    return out


# Statistics beyond per-question accuracy (v12 retrain plan, section 10.1). Each is a function of sums over
# pictures, so a bootstrap resample of pictures recomputes it exactly; pictures are resampled whole.
METRICS = ("accuracy", "recall", "fpr", "balanced_accuracy", "nll", "macro_accuracy")


def _require_same(a: List[Record], b: List[Record]) -> None:
    """The judged panels must match question for question: no silent intersection."""
    ka = {(r["sample_id"], r["qid"]): r for r in a}
    kb = {(r["sample_id"], r["qid"]): r for r in b}
    if len(ka) != len(a) or len(kb) != len(b):
        raise ValueError("duplicate (sample_id, qid) in an evaluation")
    if ka.keys() != kb.keys():
        only_a, only_b = len(ka.keys() - kb.keys()), len(kb.keys() - ka.keys())
        raise ValueError("the evaluations differ in questions: %d only in A, %d only in B" % (only_a, only_b))
    bad = [k for k in ka if ka[k]["gold"] != kb[k]["gold"] or ka[k]["image_id"] != kb[k]["image_id"]]
    if bad:
        raise ValueError("%d questions differ in gold or picture, e.g. %s" % (len(bad), bad[0]))


def _positive(r: Record) -> bool:
    if r["type"] != "noul":
        raise ValueError("recall / fpr / balanced accuracy need yes/no questions; %s is %s" % (r["sample_id"], r["type"]))
    return r["gold_answer"] == "true"


def _features(metric: str, r: Record, sources: Sequence[str]) -> List[float]:
    """Per-question sums whose totals over pictures give the metric (for one model's record)."""
    c = float(r["correct"])
    if metric == "accuracy":
        return [c, 1.0]
    if metric == "nll":
        return [float(r["nll"]), 1.0]
    if metric in ("recall", "fpr", "balanced_accuracy"):
        pos = _positive(r)
        return [c if pos else 0.0, float(pos), c if not pos else 0.0, float(not pos)]
    if metric == "macro_accuracy":
        if r["source"] not in sources:
            raise ValueError("source %r is not in the fixed list %s" % (r["source"], list(sources)))
        i = list(sources).index(r["source"])
        out = [0.0] * (2 * len(sources))
        out[2 * i], out[2 * i + 1] = c, 1.0
        return out
    raise ValueError("unknown metric %r (one of %s)" % (metric, METRICS))


def _value(metric: str, s: Sequence[float]) -> float:
    def div(x: float, y: float) -> float:
        return x / y if y else float("nan")
    if metric in ("accuracy", "nll"):
        return div(s[0], s[1])
    recall, tnr = div(s[0], s[1]), div(s[2], s[3])
    if metric == "recall":
        return recall
    if metric == "fpr":
        return 1.0 - tnr
    if metric == "balanced_accuracy":
        return (recall + tnr) / 2
    return sum(div(s[2 * i], s[2 * i + 1]) for i in range(len(s) // 2)) / (len(s) // 2)   # macro


def paired_metric(a: List[Record], b: List[Record], metric: str = "accuracy", sources: Optional[Sequence[str]] = None,
                  resamples: int = 2000, seed: int = 0) -> Dict[str, Any]:
    """B - A for `metric` on exactly the same questions, with a 95% interval from resampling pictures.

    recall / fpr / balanced_accuracy: yes/no questions, positives = gold "true"; a picture's positive and
    negative questions are resampled together. nll: mean of the records' `nll` (soft gold). macro_accuracy:
    accuracy per source, averaged with equal weight over the fixed `sources` (all must be present); pictures
    are resampled within strata of the sources they carry questions of, so every source stays in every
    resample and a picture shared by two sources moves with all of its questions."""
    if metric not in METRICS:
        raise ValueError("unknown metric %r (one of %s)" % (metric, METRICS))
    if metric == "macro_accuracy":
        if not sources:
            raise ValueError("macro_accuracy needs the fixed list of sources")
        missing = set(sources) - {r["source"] for r in a}
        if missing:
            raise ValueError("sources without questions: %s" % sorted(missing))
    _require_same(a, b)
    srcs = list(sources or [])
    index = {(r["sample_id"], r["qid"]): r for r in a}
    pics: Dict[str, List[List[float]]] = {}
    strata: Dict[str, Set[str]] = defaultdict(set)
    for y in b:
        x = index[(y["sample_id"], y["qid"])]
        fa, fb = _features(metric, x, srcs), _features(metric, y, srcs)
        cur = pics.setdefault(x["image_id"], [[0.0] * len(fa), [0.0] * len(fb)])
        cur[0] = [u + v for u, v in zip(cur[0], fa)]
        cur[1] = [u + v for u, v in zip(cur[1], fb)]
        strata[x["image_id"]].add(x["source"] if metric == "macro_accuracy" else "")
    names = sorted(pics)
    groups: Dict[Tuple[str, ...], List[str]] = defaultdict(list)
    for n in names:
        groups[tuple(sorted(strata[n]))].append(n)

    pos = {n: i for i, n in enumerate(names)}
    fa = np.array([pics[n][0] for n in names])
    fb = np.array([pics[n][1] for n in names])
    va, vb = _value(metric, fa.sum(0).tolist()), _value(metric, fb.sum(0).tolist())
    members = [np.array([pos[n] for n in g]) for g in groups.values()]
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(resamples):
        w = np.zeros(len(names))
        for m in members:   # each stratum keeps its size
            np.add.at(w, m[rng.integers(0, len(m), len(m))], 1.0)
        boot.append(_value(metric, (w @ fb).tolist()) - _value(metric, (w @ fa).tolist()))
    boot = sorted(x for x in boot if x == x)
    if not boot:
        raise ValueError("the metric is undefined in every resample")
    lo, hi = boot[int(0.025 * len(boot))], boot[min(len(boot) - 1, int(0.975 * len(boot)))]
    return {"metric": metric, "questions": len(b), "pictures": len(names), "a": va, "b": vb,
            "difference": vb - va, "ci95": [lo, hi], "resamples": len(boot),
            **({"sources": srcs} if metric == "macro_accuracy" else {})}
