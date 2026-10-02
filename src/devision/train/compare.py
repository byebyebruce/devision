"""Paired comparison of two evaluations of the same questions, from their per-question records.

The questions of one picture are not independent (POPE asks 18 per picture), so the bootstrap
resamples pictures, keeping all of a picture's questions together. An interval that covers 0 means the
data cannot tell the two apart -- not that they are equal.
"""
import json
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

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
