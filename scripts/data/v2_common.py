"""Shared last steps of the v2 build: final balancing, dev split by picture, acceptance checks, manifest.

Every sample carries "group" (what the question is about, e.g. a category or a wording) and
"answer_key" (its answer in that group). Balancing runs on the final rows, after every filter, so
nothing downstream can undo it (review 2026-10-02: per-image caps applied after balancing had broken it).
"""
import hashlib
import json
import os
from collections import Counter, defaultdict
from typing import Callable, Dict, Iterable, List

Sample = dict


def equalize(rows: List[Sample]) -> List[Sample]:
    """Per group, the same number of rows for every answer the group has; groups with one answer go."""
    by: Dict[str, Dict[str, List[Sample]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["group"]][r["answer_key"]].append(r)
    keep = set()
    for answers in by.values():
        if len(answers) < 2:
            continue
        k = min(len(v) for v in answers.values())
        for v in answers.values():
            keep |= {id(r) for r in v[:k]}
    return [r for r in rows if id(r) in keep]


def flatten_top(rows: List[Sample], min_group: int = 5) -> List[Sample]:
    """Per group of >= `min_group` rows, no answer more frequent than the group's second answer
    (a single-answer group keeps one row)."""
    by: Dict[str, Dict[str, List[Sample]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["group"]][r["answer_key"]].append(r)
    keep = set()
    for answers in by.values():
        counts = sorted((len(v) for v in answers.values()), reverse=True)
        cap = counts[0] if sum(counts) < min_group else (counts[1] if len(counts) > 1 else 1)
        for v in answers.values():
            keep |= {id(r) for r in v[:cap]}
    return [r for r in rows if id(r) in keep]


def balance_overall(rows: List[Sample]) -> List[Sample]:
    """Equal counts of every answer over the whole file (yes / no)."""
    by: Dict[str, List[Sample]] = defaultdict(list)
    for r in rows:
        by[r["answer_key"]].append(r)
    k = min(len(v) for v in by.values()) if by else 0
    keep = {id(r) for v in by.values() for r in v[:k]}
    return [r for r in rows if id(r) in keep]


def in_dev(picture_id: str, share: float) -> bool:
    """Deterministic, by picture: the same picture is always on the same side, whatever file it is in."""
    h = int(hashlib.sha1(picture_id.encode()).hexdigest()[:8], 16)
    return h < share * 0x100000000


def split_dev(rows: List[Sample], picture: Callable[[str], str], share: float):
    dev = [r for r in rows if in_dev(picture(r["image_id"]), share)]
    train = [r for r in rows if not in_dev(picture(r["image_id"]), share)]
    return train, dev


def check(rows: List[Sample], name: str, balance: str) -> List[str]:
    """Acceptance problems of a finished file: duplicate ids, and the balance it promises."""
    out = []
    dup = [i for i, n in Counter(r["id"] for r in rows).items() if n > 1]
    if dup:
        out.append("%s: %d duplicate ids, e.g. %s" % (name, len(dup), dup[:3]))
    if balance not in ("equal", "flat"):
        return out
    by: Dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        by[r["group"]][r["answer_key"]] += 1
    if balance == "equal":
        bad = [g for g, c in by.items() if len(c) < 2 or len(set(c.values())) > 1]
        if bad:
            out.append("%s: %d groups not balanced, e.g. %s" % (name, len(bad), [(g, dict(by[g])) for g in bad[:3]]))
    if balance == "flat":
        bad = [g for g, c in by.items() if sum(c.values()) >= 5 and len(c) > 1
               and sorted(c.values())[-1] > sorted(c.values())[-2]]
        if bad:
            out.append("%s: %d groups with a dominant answer, e.g. %s" % (name, len(bad), [(g, dict(by[g])) for g in bad[:3]]))
    return out


def write(path: str, rows: Iterable[Sample]) -> int:
    rows = list(rows)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            # no sort_keys: the order of a choice's criteria is the order the options are asked in
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def file_stats(path: str) -> dict:
    rows = [json.loads(l) for l in open(path)]
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    answers = Counter()
    for r in rows:
        g = r["gold"]["q"]["probabilities"]
        answers["choice" if r["questions"]["q"]["type"] == "choice" else max(g, key=g.get)] += 1
    return {"questions": len(rows), "images": len({r["image_id"] for r in rows}),
            "kinds": dict(Counter(r.get("kind", "") for r in rows)), "answers": dict(answers),
            "options": dict(sorted(Counter(len(r["gold"]["q"]["probabilities"]) for r in rows).items())),
            "sha256": digest}
