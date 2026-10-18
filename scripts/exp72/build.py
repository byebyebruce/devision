"""exp72 CPU data and audit (critic EXP72-PREP-01; docs/experiments/2026-10-19-exp72-k-supervision-plan.md). No model
weights (the plan and the truncation check use EXP68-X's tokenizer / config only).

Two arms from the same EXP68-X end, identical in every row, question, group, order and batch plan; the only difference
is the temperature state: raw (the hint + the four-number table) or K (the same + EXP70's fixed K block, computed from the
state's own numbers -- scripts/exp70/build.k_block). Magnet rows are byte-identical in both arms.

Training data (both arms, data/exp72/train_{raw,K}.jsonl), comparison-led:
  comparisons  the 1,200 temperature comparison questions of data/exp68/train.jsonl (600 combinations x two views, six
               classes x 200), each row reduced to its comparison question, in TWO passes (pass 2: ids / groups "#p2")
               = 2,400 items (each comparison seen twice);
  readings     100 temperature families by sha256("exp72-read:" + family) (both views, the four reading questions) = 800
               items, a fixed replay so the arms keep reading from the table;
  magnets      100 magnet families by sha256("exp72-mag:" + family) (both views, all three questions, byte-identical rows)
               = 600 items.
  Total 3,800 items; EXP63's code path plans them (random.seed(1), plan_batches(8), units (group, qid) whole).
Readout sets (fixed now, report-time only; the K arm gets the K block on every temperature state):
  diag68       data/exp68/diag.jsonl (480; raw) / data/exp70/diag68_K.jsonl (480; K) -- the untrained new diagnostic;
  train_slice  20 families per class by sha256("exp72-fit:" + family), comparison only, both views = 240 per arm
               (an end-of-training fit check on the arm's own combinations);
  fit_old      data/facts/fit.jsonl (400 questions; temperature rows with K for the K arm) -- the old facts fit, reported apart;
  G            data/facts/tune_G.jsonl (600; gold tables -- K is always defined; EXP56's live P tables may hold
               "unreadable" and are not used here).
Audit (MANIFEST): exposures and share of the training loss per kind and per comparison class; both arms' plans equal;
no state truncated / options distinct (both arms, every file); every K block equals the exact products of its table.

    uv run python scripts/exp72/build.py --root data
"""
import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, cast

HERE = os.path.dirname(__file__)
X_RUN = "runs/exp68-x"
READ_FAMILIES, MAG_FAMILIES, FIT_PER_CLASS, PASSES = 100, 100, 20, 2


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, rel))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


b70 = _load("exp70_build", "../exp70/build.py")
b68, fb = b70.b68, b70.fb
CLASSES = b68.CLASSES


def h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def sha(p: str) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def comparison_qid(r: Dict) -> str:
    return next(qid for qid, q in r["questions"].items() if q["instructions"] not in fb.TEMPLATES)


def only(r: Dict, qids: List[str], suffix: str = "") -> Dict:
    out = dict(r, questions={q: r["questions"][q] for q in qids}, gold={q: r["gold"][q] for q in qids})
    if suffix:
        out = dict(out, id=r["id"] + suffix, group=r["group"] + suffix)
    return out


def k_arm(rows: List[Dict]) -> List[Dict]:
    """The K arm's version: every temperature row's state gets EXP70's K block; magnet rows unchanged."""
    return [b70.with_k(r) if r.get("kind") == "temp" else r for r in rows]


def pick(fams: List[str], n: int, salt: str) -> List[str]:
    return sorted(fams, key=lambda f: h(salt + f))[:n]


def build(root: str = "data") -> Dict:
    rows = fb.jsonl(os.path.join(root, "exp68", "train.jsonl"))
    temp = [r for r in rows if r["kind"] == "temp"]
    force_lines = [x for x in open(os.path.join(root, "exp68", "train.jsonl")).read().splitlines() if json.loads(x)["kind"] == "force"]
    force = [json.loads(x) for x in force_lines]
    comps = [only(r, [comparison_qid(r)]) for r in temp]
    passes = comps + [only(r, list(r["questions"]), "#p2") for r in comps]
    tfams = sorted({r["family"] for r in temp})
    rfam = set(pick(tfams, READ_FAMILIES, "exp72-read:"))
    reads = [only(r, [q for q in r["questions"] if r["questions"][q]["instructions"] in fb.TEMPLATES]) for r in temp if r["family"] in rfam]
    mfam = set(pick(sorted({r["family"] for r in force}), MAG_FAMILIES, "exp72-mag:"))
    mags = [r for r in force if r["family"] in mfam]
    mag_lines = [x for x, r in zip(force_lines, force) if r["family"] in mfam]
    train_raw = passes + reads + mags
    train_k = k_arm(passes + reads) + mags
    by_class: Dict[str, List[str]] = defaultdict(list)
    for r in temp:
        if r["family"] not in by_class[r["axis"]]:
            by_class[r["axis"]].append(r["family"])
    sfam = {f for c in CLASSES for f in pick(by_class[c], FIT_PER_CLASS, "exp72-fit:")}
    slice_raw = [r for r in comps if r["family"] in sfam]
    fit_old = fb.jsonl(os.path.join(root, "facts", "fit.jsonl"))
    g = fb.jsonl(os.path.join(root, "facts", "tune_G.jsonl"))
    dest = os.path.join(root, "exp72")
    os.makedirs(dest, exist_ok=True)
    files = {"train_raw.jsonl": train_raw, "train_K.jsonl": train_k, "slice_raw.jsonl": slice_raw, "slice_K.jsonl": k_arm(slice_raw),
             "fit_old_K.jsonl": k_arm(fit_old), "G_K.jsonl": k_arm(g)}
    for n, rs in files.items():
        with open(os.path.join(dest, n), "w") as fh:
            fh.write("".join(json.dumps(x) + "\n" for x in rs))
    for x in mag_lines:                                                 # magnets byte for byte in both arms
        line = x
        for n in ("train_raw.jsonl", "train_K.jsonl"):
            if line not in open(os.path.join(dest, n)).read().splitlines():
                raise SystemExit("magnet rows are not the source bytes")
    for r in train_k + files["slice_K.jsonl"] + files["fit_old_K.jsonl"] + files["G_K.jsonl"]:
        if r.get("kind") == "temp":
            ka, kb = b70.k_values(r["state_text"])
            a, b = b70.b65.state_values(r["state_text"].split("\n" + b70.K_HEADER)[0])
            if [ka, kb] != [int(a[0]) * int(a[1]) ** 2, int(b[0]) * int(b[1]) ** 2]:
                raise SystemExit("%s: K does not match its table" % r["id"])
    m = {"items": {n: sum(len(r["questions"]) for r in rs) for n, rs in files.items()},
         "training": audit(train_raw), "families": {"reading_replay": len(rfam), "magnet_replay": len(mfam), "train_slice": len(sfam)},
         "readout": {"diag68_raw": "data/exp68/diag.jsonl", "diag68_K": "data/exp70/diag68_K.jsonl", "fit_old_raw": "data/facts/fit.jsonl",
                     "G_raw": "data/facts/tune_G.jsonl"},
         "sources": {"exp68_train": sha(os.path.join(root, "exp68", "train.jsonl")), "exp68_diag": sha(os.path.join(root, "exp68", "diag.jsonl")),
                     "exp70_diag68_K": sha(os.path.join(root, "exp70", "diag68_K.jsonl")), "facts_fit": sha(os.path.join(root, "facts", "fit.jsonl")),
                     "facts_tune_G": sha(os.path.join(root, "facts", "tune_G.jsonl"))},
         "sha256": {n: sha(os.path.join(dest, n)) for n in files}}
    json.dump(m, open(os.path.join(dest, "MANIFEST.json"), "w"), indent=1)
    return m


def audit(train: List[Dict]) -> Dict:
    """Exposures (= share of the per-item mean CE loss) per kind and per comparison class."""
    kinds: Counter = Counter()
    classes: Counter = Counter()
    for r in train:
        for qid, q in r["questions"].items():
            if r["kind"] == "force":
                kinds["magnet"] += 1
            elif q["instructions"] in fb.TEMPLATES:
                kinds["temperature reading"] += 1
            else:
                kinds["temperature comparison"] += 1
                classes[r["axis"]] += 1
    n = sum(kinds.values())
    return {"items": n, "by_kind": {k: {"items": v, "share": round(v / n, 4)} for k, v in kinds.items()},
            "comparison_by_class": {c: {"items": classes[c], "share": round(classes[c] / n, 4)} for c in CLASSES},
            "exp68_comparison": {"items": 1200, "share": round(1200 / 9600, 4), "per_class_share": round(200 / 9600, 4)}}


def plan_and_truncation(root: str = "data") -> Dict:
    """Both arms through EXP63's code path (read_decision_jsonl -> rlcd._items -> random.seed(1) -> plan_batches(8)):
    the plans must be equal item for item (id, qid); every readout / training file: no state token dropped, options
    distinct."""
    from types import SimpleNamespace

    from transformers import AutoTokenizer

    from devision.model import ModelConfig
    from devision.train.rlcd import _items, plan_batches
    from devision.train.samples import read_decision_jsonl
    st = os.path.join(X_RUN, "stage2")
    cfg = json.load(open(os.path.join(st, "config.json")))
    d = SimpleNamespace(tok=AutoTokenizer.from_pretrained(os.path.join(st, "tokenizer")),
                        cfg=ModelConfig(**{k: v for k, v in cfg.items() if k in ModelConfig.__dataclass_fields__}), model=None)
    plans = {}
    for arm in ("raw", "K"):
        items = _items(cast(Any, d), read_decision_jsonl(os.path.join(root, "exp72", "train_%s.jsonl" % arm)))
        random.seed(1)
        p = plan_batches(items, 8)
        plans[arm] = [[(items[i]["sample_id"], items[i]["qid"]) for i in b] for b in p]
        units: Dict = defaultdict(list)
        for i, it in enumerate(items):
            units[(it["group"], it["qid"])].append(i)
        where = {i: k for k, b in enumerate(p) for i in b}
        split = sum(1 for v in units.values() if len({where[i] for i in v}) > 1)
        plans[arm + "_stats"] = {"items": len(items), "updates": len(p), "batch_sizes": dict(Counter(map(len, p))), "split_units": split,
                                 "every_item_once": sorted(i for b in p for i in b) == list(range(len(items)))}
    if plans["raw"] != plans["K"]:
        raise SystemExit("the two arms' plans differ")
    trunc = {n: fb.truncation(cast(Any, d), os.path.join(root, "exp72", n)) for n in
             ("train_raw.jsonl", "train_K.jsonl", "slice_raw.jsonl", "slice_K.jsonl", "fit_old_K.jsonl", "G_K.jsonl")}
    trunc["fit_old_raw"] = fb.truncation(cast(Any, d), os.path.join(root, "facts", "fit.jsonl"))
    if any(v["state_dropped"] or v["options_not_distinct"] for v in trunc.values()):
        raise SystemExit("truncation: %s" % trunc)
    out = {"plans_equal": True, "raw": plans["raw_stats"], "K": plans["K_stats"], "seed": 1, "micro_batch": 8,
           "plan_sha256": h(json.dumps(plans["raw"])), "truncation": trunc}
    m = json.load(open(os.path.join(root, "exp72", "MANIFEST.json")))
    m["plan"] = out
    json.dump(m, open(os.path.join(root, "exp72", "MANIFEST.json"), "w"), indent=1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data")
    a = ap.parse_args()
    print(json.dumps(build(a.root), indent=1))
    print(json.dumps({k: v for k, v in plan_and_truncation(a.root).items() if k != "truncation"}, indent=1))
    sys.exit(0)
