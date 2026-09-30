"""Rule-based conversion of VQA datasets into Jev-format decision samples.

A sample is one JSONL line:
    {"id", "source", "image_id", "image", "questions": {qid: Question}, "gold": {qid: {"probabilities"}}}
`image` is relative to the data root; `image_id` is namespaced ("coco:42", "vg:2354786") so
evaluation images can be excluded from every training source by id.
"""
import random
from typing import Any, Dict, Iterable, List, Optional, Set

Sample = Dict[str, Any]

YES_NO = {"yes": "true", "no": "false"}


def _sample(source: str, sid: str, image_id: str, image: str, question: Dict[str, Any],
            probabilities: Dict[str, float]) -> Sample:
    return {"id": "%s:%s" % (source, sid), "source": source, "image_id": image_id, "image": image,
            "questions": {"q": question}, "gold": {"q": {"probabilities": probabilities}}}


def _noul(text: str) -> Dict[str, Any]:
    return {"type": "noul", "instructions": text}


def _choice_options(program: List[Dict[str, Any]]) -> Optional[List[str]]:
    """Options of a GQA choose question, read from the argument of its last `choose ...` step."""
    for step in reversed(program):
        if step.get("operation", "").startswith("choose"):
            for part in step.get("argument", "").split(","):
                if "|" in part:
                    return [o.strip() for o in part.split("|")]
            return None
    return None


def convert_gqa(qid: str, rec: Dict[str, Any]) -> Optional[Sample]:
    """GQA balanced question -> noul (yes/no answers) or 2-option choice (`choose` questions)."""
    image_id, answer = str(rec["imageId"]), rec["answer"]
    image = "gqa/images/%s.jpg" % image_id
    if answer in YES_NO:
        truth = YES_NO[answer]
        probs = {"false": float(truth == "false"), "true": float(truth == "true")}
        return _sample("gqa", qid, "vg:" + image_id, image, _noul(rec["question"]), probs)
    if rec.get("types", {}).get("structural") != "choose":
        return None
    options = _choice_options(rec.get("semantic", []))
    if not options or len(set(options)) < 2:
        return None
    if answer not in options:  # "right" for the option "to the right of"
        matches = [o for o in options if answer in o.split()]
        if len(matches) != 1:
            return None
        answer = matches[0]
    random.Random(qid).shuffle(options)  # the program may list options in a telling order
    question = {"type": "choice", "instructions": rec["question"], "criteria": {o: None for o in options}}
    return _sample("gqa", qid, "vg:" + image_id, image, question,
                   {o: float(o == answer) for o in options})


def convert_vqav2(question: Dict[str, Any], annotation: Dict[str, Any], split: str) -> Optional[Sample]:
    """VQAv2 yes/no question -> noul; target is the share of annotators who said yes."""
    if annotation["answer_type"] != "yes/no":
        return None
    votes = [a["answer"] for a in annotation["answers"] if a["answer"] in YES_NO]
    if not votes:
        return None
    p_yes = round(votes.count("yes") / len(votes), 4)
    image_id = int(question["image_id"])
    image = "coco/%s/COCO_%s_%012d.jpg" % (split, split, image_id)
    return _sample("vqav2", str(question["question_id"]), "coco:%d" % image_id, image,
                   _noul(question["question"]), {"false": round(1 - p_yes, 4), "true": p_yes})


def select(samples: Iterable[Optional[Sample]], limit: int, exclude_image_ids: Set[str],
           seed: int) -> List[Sample]:
    """Drop excluded images, balance noul yes/no, take up to `limit` samples reproducibly.

    Yes, no and choice samples are drawn round-robin, so any prefix stays balanced.
    """
    rng = random.Random(seed)
    yes: List[Sample] = []
    no: List[Sample] = []
    other: List[Sample] = []
    for s in samples:
        if s is None or s["image_id"] in exclude_image_ids:
            continue
        (q,), (gold,) = s["questions"].values(), s["gold"].values()
        if q["type"] == "noul":
            (yes if gold["probabilities"]["true"] > 0.5 else no).append(s)
        else:
            other.append(s)
    for group in (yes, no, other):
        rng.shuffle(group)
    k = min(len(yes), len(no))
    pools = [yes[:k], no[:k], other]
    picked: List[Sample] = []
    i = 0
    while len(picked) < limit and any(i < len(p) for p in pools):
        picked.extend(p[i] for p in pools if i < len(p))
        i += 1
    picked = picked[:limit]
    rng.shuffle(picked)
    return picked


def convert_pope(rec: Dict[str, Any]) -> Sample:
    """POPE existence question (evaluation only) -> noul on its COCO val2014 image."""
    source = rec["image_source"]
    split = source.split("_")[1]
    truth = YES_NO[rec["answer"].strip().lower()]
    return _sample("pope", "%s:%s" % (rec["category"], rec["question_id"]),
                   "coco:%d" % int(source.rsplit("_", 1)[1]), "coco/%s/%s.jpg" % (split, source),
                   _noul(rec["question"]),
                   {"false": float(truth == "false"), "true": float(truth == "true")})
