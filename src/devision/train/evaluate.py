"""Evaluation through `Decider.decide`, i.e. the code path production requests take."""
import base64
import os
import time
from typing import Any, Dict, List, Sequence

import numpy as np
from laya.common import ece_score

from .data import Sample
from ..model import Decider


def _gold_answer(q: Dict[str, Any], probs: Dict[str, float]) -> str:
    return max(probs, key=probs.__getitem__)


def _predicted(answer: Dict[str, Any]):
    """(predicted option, probability mass on it)."""
    if answer["type"] == "noul":
        p = answer["noul"]
        return ("true", p) if p >= 0.5 else ("false", 1 - p)
    return answer["choice"], answer["probabilities"][answer["choice"]]


def evaluate(decider: Decider, samples: Sequence[Sample], data_root) -> Dict[str, Any]:
    """Accuracy per question type, ECE on max-probability confidence, decide() latency."""
    correct: Dict[str, List[bool]] = {"noul": [], "choice": []}
    conf: List[float] = []
    hits: List[bool] = []
    latency: List[float] = []
    for s in samples:
        with open(os.path.join(str(data_root), s["image"]), "rb") as f:
            state = [{"type": "image", "base64": base64.b64encode(f.read()).decode()}]
        t0 = time.perf_counter()
        res = decider.decide(state=state, questions=s["questions"])
        latency.append((time.perf_counter() - t0) * 1000)
        for qid, q in s["questions"].items():
            pred, p = _predicted(res["answers"][qid])
            ok = pred == _gold_answer(q, s["gold"][qid]["probabilities"])
            correct[q["type"]].append(ok)
            conf.append(p)
            hits.append(ok)
    lat = np.array(latency) if latency else np.zeros(1)
    return {
        "n": len(samples),
        "accuracy": {t: (float(np.mean(v)) if v else None) for t, v in correct.items()},
        "accuracy_all": float(np.mean(hits)) if hits else None,
        "ece": ece_score(np.array(conf), np.array(hits, dtype=float)),
        "latency_ms": {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95))},
        "model": decider.cfg.model_name,
    }
