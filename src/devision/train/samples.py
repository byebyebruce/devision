"""The training / evaluation sample format. The scripts that produce samples are in scripts/data/ (not in the package).

One JSON object per JSONL line:

    {"id": "gqa:02241348", "source": "gqa",
     "image_id": "vg:2384884",            # namespaced ("coco:42", "vg:..."), used to keep eval images out of train
     "image": "gqa/images/2384884.jpg",   # relative to the data root
     "questions": {"q": {"type": "choice", "instructions": "...", "criteria": {"a": null, "b": null}}},
     "gold": {"q": {"probabilities": {"a": 0.0, "b": 1.0}}}}

`questions` follows the Jev request format (noul / choice); `gold` is a distribution over the
options (noul: "false" / "true") and is used as a soft target.
An optional "state_text" is text that comes with the image (e.g. a ScienceQA hint): decide() gets it as
[image, state_text], training feeds it to the encoder the same way.

The alignment stage (`align.py`) reads caption samples instead, one image per line:

    {"id": "coco-cap:42", "source": "coco-captions", "image_id": "coco:42",
     "image": "coco/train2014/COCO_train2014_000000000042.jpg",
     "captions": ["A dog on a couch.", "..."]}
"""
import json
import math
from typing import Any, Dict, List

Sample = Dict[str, Any]


def read_jsonl(path: str) -> List[Sample]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def sample_errors(s: Sample) -> List[str]:
    """What is wrong with one decision sample (empty when it is usable): every question has a known type, its
    gold names only real options, and the probabilities are finite, non-negative and not all zero."""
    errors = []
    if not isinstance(s.get("image"), str) or not s["image"]:
        errors.append("no image path")
    qs, gold = s.get("questions"), s.get("gold")
    if not isinstance(qs, dict) or not qs:
        return errors + ["no questions"]
    if not isinstance(gold, dict):
        return errors + ["no gold"]
    for qid, q in qs.items():
        t = q.get("type") if isinstance(q, dict) else None
        if t == "noul":
            keys = {"false", "true"}
        elif t == "choice" and isinstance(q.get("criteria"), dict) and len(q["criteria"]) >= 2:
            keys = set(q["criteria"])
        else:
            errors.append("question %r: type must be noul, or choice with >= 2 options" % qid)
            continue
        probs = (gold.get(qid) or {}).get("probabilities") if isinstance(gold.get(qid), dict) else None
        if not isinstance(probs, dict) or not probs:
            errors.append("question %r: no gold probabilities" % qid)
            continue
        extra = set(probs) - keys
        if extra:
            errors.append("question %r: gold names %s, not options" % (qid, sorted(extra)))
        values = list(probs.values())
        if not all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in values):
            errors.append("question %r: probabilities must be finite and >= 0" % qid)
        elif sum(values) <= 0:
            errors.append("question %r: probabilities sum to 0" % qid)
    for qid in gold:
        if qid not in qs:
            errors.append("gold for unknown question %r" % qid)
    return errors


def read_decision_jsonl(path: str) -> List[Sample]:
    """Decision samples from `path`, checked before any model is loaded; raises ValueError naming file, line
    and sample id for every bad sample (up to 20)."""
    samples, bad = [], []
    with open(path) as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            s = json.loads(line)
            for e in sample_errors(s):
                bad.append("%s:%d (%s): %s" % (path, n, s.get("id", "?"), e))
            samples.append(s)
    if bad:
        raise ValueError("%d bad sample(s):\n  %s" % (len(bad), "\n  ".join(bad[:20])))
    return samples
