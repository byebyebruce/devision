"""Evaluation through `Decider.decide`, i.e. the code path production requests take.

`evaluate` asks every question once and returns one record per question (prediction, full
probabilities, gold distribution, image used, ...) plus a summary computed only from those records,
so later slices and comparisons need no new inference (`summarize`, `devision-compare`).

Controls (same questions, one thing changed):
  mismatched  each picture's questions are asked about another picture (fixed-seed derangement of the
              pictures; the pairing is in the records) -- a model that reads the image drops towards
              its no-image level;
  reversed    choice options in reverse order (the option keys and gold are unchanged); compared with
              the plain run by `order_sensitivity` -- answer flips mean the order matters.
"""
import base64
import math
import os
import platform
import random
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from laya.common import ece_score

from .samples import Sample
from ..model import Decider

CONTROLS = ("none", "mismatched", "reversed")
WARMUP = 3


def _probabilities(answer: Dict[str, Any]) -> Dict[str, float]:
    if answer["type"] == "noul":
        return {"false": 1.0 - answer["noul"], "true": answer["noul"]}
    return dict(answer["probabilities"])


def picture_pairs(samples: Sequence[Sample], seed: int = 0) -> Dict[str, str]:
    """image_id -> the image path to show instead: a fixed-seed derangement of the distinct pictures."""
    first: Dict[str, str] = {}
    for s in samples:
        first.setdefault(s["image_id"], s["image"])
    ids = sorted(first)
    if len(ids) < 2:
        return {i: first[i] for i in ids}
    order = ids[:]
    random.Random(seed).shuffle(order)
    return {order[k]: first[order[(k + 1) % len(order)]] for k in range(len(order))}


def evaluate_records(decider: Decider, samples: Sequence[Sample], data_root, control: str = "none",
                     seed: int = 0) -> List[Dict[str, Any]]:
    if control not in CONTROLS:
        raise ValueError("control must be one of %s" % (CONTROLS,))
    pairs: Dict[str, str] = picture_pairs(samples, seed) if control == "mismatched" else {}
    records: List[Dict[str, Any]] = []
    warm = 0
    for s in samples:
        image: str = pairs[s["image_id"]] if s["image_id"] in pairs else s["image"]
        questions = s["questions"]
        if control == "reversed":
            questions = {qid: dict(q, criteria=dict(reversed(list(q["criteria"].items()))))
                         if q["type"] == "choice" else q for qid, q in questions.items()}
        with open(os.path.join(str(data_root), image), "rb") as f:
            state = [{"type": "image", "base64": base64.b64encode(f.read()).decode()}]
        while warm < WARMUP:  # first calls pay one-off costs; keep them out of the latency
            decider.decide(state=state, questions=questions)
            warm += 1
        t0 = time.perf_counter()
        res = decider.decide(state=state, questions=questions)
        ms = (time.perf_counter() - t0) * 1000
        for n, (qid, q) in enumerate(questions.items()):
            gold = s["gold"][qid]["probabilities"]
            probs = _probabilities(res["answers"][qid])
            pred = max(probs, key=probs.__getitem__)
            records.append({
                "sample_id": s["id"], "qid": qid, "image_id": s["image_id"], "image_used": image,
                "source": s.get("source", ""), "kind": s.get("kind", ""), "axis": s.get("axis", ""),
                "type": q["type"], "options": list(q["criteria"]) if q["type"] == "choice" else ["false", "true"],
                "gold": gold, "gold_answer": max(gold, key=gold.__getitem__), "probabilities": probs,
                "prediction": pred, "p_prediction": probs[pred], "control": control,
                "latency_ms": ms if n == 0 else None, "questions_in_request": len(questions)})
    for r in records:
        r["correct"] = r["prediction"] == r["gold_answer"]
        r["nll"] = -sum(g * math.log(max(r["probabilities"].get(o, 0.0), 1e-12)) for o, g in r["gold"].items() if g > 0)
        r["brier"] = sum((r["probabilities"].get(o, 0.0) - r["gold"].get(o, 0.0)) ** 2 for o in r["gold"])
    return records


def _acc(rs: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"n": len(rs), "images": len({r["image_id"] for r in rs}),
            "accuracy": float(np.mean([r["correct"] for r in rs])) if rs else None}


def _pope(rs: List[Dict[str, Any]]) -> Dict[str, float]:
    """POPE's own metrics, "yes" (true) being the positive class."""
    tp = sum(r["prediction"] == "true" and r["gold_answer"] == "true" for r in rs)
    fp = sum(r["prediction"] == "true" and r["gold_answer"] == "false" for r in rs)
    fn = sum(r["prediction"] == "false" and r["gold_answer"] == "true" for r in rs)
    tn = sum(r["prediction"] == "false" and r["gold_answer"] == "false" for r in rs)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "accuracy": (tp + tn) / max(1, len(rs)),
            "precision": precision, "recall": recall, "f1": f1, "yes_ratio": (tp + fp) / max(1, len(rs))}


def _reliability(records: List[Dict[str, Any]], bins: int = 10) -> List[Dict[str, Any]]:
    """Per confidence bin (top probability): count, mean confidence, accuracy."""
    out = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        rs = [r for r in records if lo <= r["p_prediction"] < hi or (b == bins - 1 and r["p_prediction"] == 1.0)]
        if rs:
            out.append({"bin": [lo, hi], "n": len(rs), "confidence": float(np.mean([r["p_prediction"] for r in rs])),
                        "accuracy": float(np.mean([r["correct"] for r in rs]))})
    return out


def _above(records: List[Dict[str, Any]], t: float) -> Dict[str, Any]:
    """Answering only when the top probability is >= t: share answered and accuracy on those."""
    rs = [r for r in records if r["p_prediction"] >= t]
    return {"coverage": len(rs) / max(1, len(records)),
            "accuracy": float(np.mean([r["correct"] for r in rs])) if rs else None}


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Every reported number, from the per-question records alone."""
    out: Dict[str, Any] = {"n": len(records), "images": len({r["image_id"] for r in records})}
    out["accuracy_all"] = _acc(records)["accuracy"]
    out["accuracy"] = {t: _acc([r for r in records if r["type"] == t])["accuracy"] for t in ("noul", "choice")}
    out["nll"] = float(np.mean([r["nll"] for r in records])) if records else None
    out["brier"] = float(np.mean([r["brier"] for r in records])) if records else None
    out["ece"] = ece_score(np.array([r["p_prediction"] for r in records]),
                           np.array([r["correct"] for r in records], dtype=float)) if records else None
    out["ece_bins"] = 15
    out["reliability"] = _reliability(records)
    out["thresholds"] = {"%.1f" % t: _above(records, t) for t in (0.6, 0.7, 0.8, 0.9)}
    for key, field in (("by_source", "source"), ("by_kind", "kind"), ("by_axis", "axis"), ("by_options", None)):
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in records:
            groups[str(len(r["options"])) if field is None else (r[field] or "-")].append(r)
        out[key] = {k: _acc(v) for k, v in sorted(groups.items())}
    noul = [r for r in records if r["type"] == "noul"]
    if noul:
        out["noul_confusion"] = {k: v for k, v in _pope(noul).items() if k in ("tp", "fp", "fn", "tn", "yes_ratio")}
    pope = defaultdict(list)
    for r in records:
        if r["source"].startswith("pope"):
            pope[r["source"]].append(r)
    if pope:
        out["pope"] = {k: _pope(v) for k, v in sorted(pope.items())}
    lat = [r["latency_ms"] for r in records if r["latency_ms"] is not None]
    if lat:
        out["latency_ms"] = {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95)),
                             "requests": len(lat), "warmup_excluded": WARMUP}
    out["control"] = records[0]["control"] if records else "none"
    return out


def order_sensitivity(plain: List[Dict[str, Any]], reversed_: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Choice questions asked with options in the original and reversed order, matched by question and
    aligned by option key: share of changed answers, accuracy difference, mean |probability change|."""
    a = {(r["sample_id"], r["qid"]): r for r in plain if r["type"] == "choice"}
    pairs = [(a[k], r) for r in reversed_ if (k := (r["sample_id"], r["qid"])) in a and r["type"] == "choice"]
    if not pairs:
        return {"n": 0}
    flips = [x["prediction"] != y["prediction"] for x, y in pairs]
    dp = [abs(x["probabilities"][o] - y["probabilities"].get(o, 0.0)) for x, y in pairs for o in x["probabilities"]]
    return {"n": len(pairs), "answer_flip_rate": float(np.mean(flips)),
            "accuracy_plain": float(np.mean([x["correct"] for x, _ in pairs])),
            "accuracy_reversed": float(np.mean([y["correct"] for _, y in pairs])),
            "mean_abs_probability_change": float(np.mean(dp))}


def evaluate(decider: Decider, samples: Sequence[Sample], data_root, control: str = "none", seed: int = 0,
             records_out: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Summary of `evaluate_records` (accuracy per type / source / kind / axis / option count, NLL,
    Brier, ECE on the top probability with 15 bins, POPE precision / recall / F1 / yes ratio, latency).
    Pass a list as `records_out` to also get the per-question records."""
    records = evaluate_records(decider, samples, data_root, control, seed)
    if records_out is not None:
        records_out.extend(records)
    out = summarize(records)
    out["model"] = decider.cfg.model_name
    out["temperature"] = {"choice": decider.cfg.temperature[0], "noul": decider.cfg.temperature[2]}
    out["environment"] = {"threads": _threads(), "machine": platform.machine(), "system": platform.system()}
    return out


def _threads() -> int:
    import torch

    return torch.get_num_threads()
