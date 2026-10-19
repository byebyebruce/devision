"""exp66's fixed batch plan (docs/experiments/2026-10-19-exp66-layout-plan.md section 3). CPU only, no model.

EXP62's generator (scripts/exp62/build.plan: per update 24 (parent, crop) pairs next to each other + 16 fit_O / fit_L
replay rows; one seeded torch.Generator per pool, seeds 0 / 1; a fresh permutation when a pool runs out; 2,000 updates)
with the pairs given explicitly: EXP62's 1,200 (train_P parent, train_V crop) and exp66's 1,200 (train_C_parent,
train_C), sorted by crop key. Only the pool changed, so with EXP62's pairs alone it reproduces data/exp62/plan.jsonl
(checked by the fixtures); every pair is now seen 20 times instead of 40. Writes data/exp66/plan.jsonl and PLAN.json
(sha256, counts); data/exp66's accepted MANIFEST and questions are not rewritten.

    uv run python scripts/exp66/plan.py --root data
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
from collections import Counter
from typing import Dict, List, Tuple

HERE = os.path.dirname(__file__)
STEPS, PAIRS_PER_BATCH, REPLAY_PER_BATCH, SEED = 2000, 24, 16, 0


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, rel))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


b62 = _load("exp62_build", "../exp62/build.py")


def jsonl(p: str) -> List[Dict]:
    return [json.loads(x) for x in open(p) if x.strip()]


def sha(p: str) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def plan(pairs: List[Tuple[str, str]], replay_keys: List[str], steps: int = STEPS) -> List[List[str]]:
    """b62.plan with explicit (parent, crop) pairs: the same generators, draws and row order."""
    import torch
    if (PAIRS_PER_BATCH, REPLAY_PER_BATCH, SEED) != (b62.PAIRS_PER_BATCH, b62.REPLAY_PER_BATCH, b62.SEED):
        raise SystemExit("EXP62's plan constants changed")
    gp, gr = torch.Generator().manual_seed(SEED), torch.Generator().manual_seed(SEED + 1)
    qp: List[int] = []
    qr: List[int] = []
    out = []
    for _ in range(steps):
        while len(qp) < PAIRS_PER_BATCH:
            qp += torch.randperm(len(pairs), generator=gp).tolist()
        while len(qr) < REPLAY_PER_BATCH:
            qr += torch.randperm(len(replay_keys), generator=gr).tolist()
        pk, qp = qp[:PAIRS_PER_BATCH], qp[PAIRS_PER_BATCH:]
        rk, qr = qr[:REPLAY_PER_BATCH], qr[REPLAY_PER_BATCH:]
        rows: List[str] = []
        for i in pk:
            rows += list(pairs[i])
        out.append(rows + [replay_keys[i] for i in rk])
    return out


def pools(root: str = "data") -> Tuple[List[Tuple[str, str]], List[str], Dict[str, Dict]]:
    """(pairs sorted by crop key, replay keys sorted, key -> record of every planned row)."""
    p62 = jsonl(os.path.join(root, "exp62", "parents.jsonl"))
    q66 = jsonl(os.path.join(root, "exp66", "questions.jsonl"))
    v61 = {q["key"]: q for q in jsonl(os.path.join(root, "exp61", "questions.jsonl"))}
    rec = {q["key"]: q for q in p62 + q66}
    rec.update({q["pair"]: v61[q["pair"]] for q in p62})
    pairs = [(q["key"], q["pair"]) for q in p62] + [(q["key"], q["pair"]) for q in q66 if q["set"] == "train_C_parent"]
    pairs.sort(key=lambda p: p[1])
    replay = [q for q in jsonl(os.path.join(root, "exp49", "questions.jsonl")) if q["set"] in ("fit_O", "fit_L")]
    rec.update({q["key"]: q for q in replay})
    return pairs, sorted(q["key"] for q in replay), rec


def plan_stats(batches: List[List[str]], pairs: List[Tuple[str, str]], replay: List[str], rec: Dict[str, Dict]) -> Dict:
    flat = [k for b in batches for k in b]
    expo = Counter(flat)
    pair_of = dict(pairs)
    split = 0
    for b in batches:
        for i, k in enumerate(b):
            if k in pair_of and (i + 1 >= len(b) or b[i + 1] != pair_of[k]):
                split += 1
    kind = lambda k: k.split("|")[0]  # noqa: E731
    return {"updates": len(batches), "batch_sizes": {str(k): v for k, v in Counter(map(len, batches)).items()}, "exposures": len(flat),
            "by_kind": dict(Counter(kind(k) for k in flat)), "split_pairs": split,
            "pair_exposure_range": [min(expo[p] for p, _ in pairs), max(expo[p] for p, _ in pairs)],
            "replay_exposure_range": [min(expo[k] for k in replay), max(expo[k] for k in replay)],
            "every_row_seen": all(expo[k] > 0 for k in [x for p in pairs for x in p] + replay),
            "by_field": dict(Counter(rec[k]["field"] for k in flat)), "families_seen": len({rec[k]["family"] for k in flat}),
            "pairs": {"exp62": sum(1 for p, _ in pairs if p.startswith("train_P|")), "exp66": sum(1 for p, _ in pairs if p.startswith("train_C_parent|"))}}


def build(root: str = "data") -> Dict:
    pairs, replay, rec = pools(root)
    batches = plan(pairs, replay)
    st = plan_stats(batches, pairs, replay, rec)
    if st["split_pairs"] or st["exposures"] != STEPS * 64 or st["batch_sizes"] != {"64": STEPS} or not st["every_row_seen"]:
        raise SystemExit("plan: %s" % st)
    dest = os.path.join(root, "exp66")
    path = os.path.join(dest, "plan.jsonl")
    with open(path, "w") as fh:
        fh.write("".join(json.dumps(b) + "\n" for b in batches))
    m = {"stats": st, "plan_sha256": sha(path), "sources": {"exp62_parents": sha(os.path.join(root, "exp62", "parents.jsonl")),
                                                            "exp66_questions": sha(os.path.join(root, "exp66", "questions.jsonl")),
                                                            "exp66_manifest": sha(os.path.join(root, "exp66", "MANIFEST.json")),
                                                            "exp61_questions": sha(os.path.join(root, "exp61", "questions.jsonl")),
                                                            "exp49_questions": sha(os.path.join(root, "exp49", "questions.jsonl"))}}
    json.dump(m, open(os.path.join(dest, "PLAN.json"), "w"), indent=1)
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data")
    print(json.dumps(build(ap.parse_args().root), indent=1))
    sys.exit(0)
