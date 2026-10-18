"""Decision-side comparison supervision: CPU data and the real batch plan (critic DECISION-CMP-PREP-01, -R1-01 / 02;
docs/experiments/2026-10-18-decision-cmp-plan.md). No model, no training. From data/real_layout/questions_train.jsonl only:

  - families chosen by a stable hash (sha256 of "cmp-v1:" + family), 600 temperature families stratified by axis in
    proportion to the train split, and 500 magnet families;
  - every row (both views of a family) gets its gold number table -- the row's own reading questions' gold options,
    thousands commas removed then exp55's separators (devision.reading.fmt / serialise, the exp55 interface) -- appended to
    its state_text; temperature rows keep all 5 questions (4 readings + the comparison), magnet rows only their 2
    readings (no magnet comparison: sizes / poles are not in the table);
  - options reordered per question by a fixed seed (random.Random(sha256 of row id + qid)) to break position habits;
    the gold follows its option;
  - a training-internal fit slice of 100 whole temperature families (by the same hash, stratified by axis), fixed now;
  - the gold comparison answer checked against the row's own gold readings (m v^2; data preparation only).
Writes data/cmp/train.jsonl, data/cmp/fit_slice.json, data/cmp/MANIFEST.json, and the real plan on rlcd's code path
(read_decision_jsonl -> rlcd._items with EXP50-T's tokenizer / config -> random.seed(1) -> plan_batches(8)) to
data/cmp/PLAN.json: updates, per-question exposure, comparison count, (family, qid) pairs never split, the tail batch.

    uv run python scripts/cmp/build.py --root data
"""
import argparse
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from types import SimpleNamespace

from devision.reading import TEMPLATES, fmt, serialise

SEED = "cmp-v1:"
N_TEMP, N_MAG, N_FIT = 600, 500, 100
INIT = "runs/exp50-t/stage2"


def h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def number_of(option: str) -> str:
    return option.split(" ")[0].replace(",", "")


def stratified(fams: dict, n: int, salt: str = "") -> list:
    """n families by stable hash, each axis in proportion to its share (largest remainders)."""
    by = defaultdict(list)
    for f, ax in fams.items():
        by[ax].append(f)
    total = sum(len(v) for v in by.values())
    quota = {ax: n * len(v) / total for ax, v in by.items()}
    take = {ax: int(q) for ax, q in quota.items()}
    for ax in sorted(by, key=lambda a: -(quota[a] - take[a]))[: n - sum(take.values())]:
        take[ax] += 1
    out = []
    for ax, v in sorted(by.items()):
        out += sorted(v, key=lambda f: h(SEED + salt + f))[: take[ax]]
    return sorted(out)


def temperature_gold(table) -> str:
    """The answer the row's own gold readings imply (data preparation only): higher mean kinetic energy m v^2, or
    neither."""
    v = {(o, q): float(x.replace(",", "")) for o, q, x in table}
    ka = v[("Sample A", "mass of each particle")] * v[("Sample A", "average particle speed")] ** 2
    kb = v[("Sample B", "mass of each particle")] * v[("Sample B", "average particle speed")] ** 2
    return "neither" if ka == kb else "sample A" if ka > kb else "sample B"


def build(root: str) -> dict:
    src = os.path.join(root, "real_layout", "questions_train.jsonl")
    rows = [json.loads(x) for x in open(src)]
    temp = {r["family"]: r["axis"] for r in rows if r["kind"] == "temp"}
    mag = {r["family"]: r["axis"] for r in rows if r["kind"] == "force"}
    t_f, m_f = stratified(temp, N_TEMP), stratified(mag, N_MAG)
    fit = stratified({f: temp[f] for f in t_f}, N_FIT, salt="fit:")
    chosen = set(t_f) | set(m_f)
    out, gold_checked = [], Counter()
    for r in rows:
        if r["family"] not in chosen:
            continue
        table = []
        reading = {q["instructions"]: qid for qid, q in r["questions"].items() if q["instructions"] in TEMPLATES}
        for t in TEMPLATES:
            if t in reading:
                g = next(o for o, p in r["gold"][reading[t]]["probabilities"].items() if p == 1.0)
                table.append([*TEMPLATES[t], fmt(number_of(g))])
        keep = list(r["questions"]) if r["kind"] == "temp" else list(reading.values())
        qs, gold = {}, {}
        for qid in keep:
            q = r["questions"][qid]
            opts = list(q["criteria"])
            random.Random(h(r["id"] + "|" + qid)).shuffle(opts)
            qs[qid] = {**q, "criteria": {o: None for o in opts}}
            gold[qid] = {"probabilities": {o: r["gold"][qid]["probabilities"][o] for o in opts}}
            if r["kind"] == "temp" and qid not in reading.values():
                g = next(o for o, p in gold[qid]["probabilities"].items() if p == 1.0)
                gold_checked[temperature_gold(table) == ("neither" if g.startswith("neither") else g)] += 1
        out.append({**{k: r[k] for k in ("id", "source", "kind", "axis", "image_id", "image", "family", "case")},
                    "group": r["family"], "state_text": serialise(r.get("state_text", ""), table), "questions": qs, "gold": gold})
    if gold_checked.get(False):
        raise SystemExit("%d comparison golds disagree with the case's physics" % gold_checked[False])
    dest = os.path.join(root, "cmp")
    os.makedirs(dest, exist_ok=True)
    path = os.path.join(dest, "train.jsonl")
    with open(path, "w") as fh:
        fh.write("".join(json.dumps(x) + "\n" for x in out))
    json.dump({"families": fit, "rule": "100 whole temperature families of the 600, by hash, stratified by axis"},
              open(os.path.join(dest, "fit_slice.json"), "w"), indent=1)
    cmp_gold = Counter()
    for x in out:
        for qid, q in x["questions"].items():
            if q["instructions"] not in TEMPLATES:
                g = next(o for o, p in x["gold"][qid]["probabilities"].items() if p == 1.0)
                cmp_gold[(x["axis"], "neither" if g.startswith("neither") else g)] += 1
    m = {"source": h(open(src).read()), "families": {"temperature": len(t_f), "magnets": len(m_f), "fit_slice": len(fit)},
         "rows": len(out), "questions": sum(len(x["questions"]) for x in out),
         "comparison_questions": sum(cmp_gold.values()),
         "comparison_gold": {k: sum(v for (a, g), v in cmp_gold.items() if g == k) for k in ("sample A", "sample B", "neither")},
         "comparison_by_axis": {"%s|%s" % k: v for k, v in sorted(cmp_gold.items())},
         "axes_temperature": dict(Counter(temp[f] for f in t_f)), "gold_checked_by_physics": dict(gold_checked),
         "option_shuffle": "random.Random(sha256(row id + '|' + qid))", "sha256": h(open(path).read())}
    json.dump(m, open(os.path.join(dest, "MANIFEST.json"), "w"), indent=1)
    return m


def plan(root: str) -> dict:
    from transformers import AutoTokenizer

    from devision.model import ModelConfig
    from devision.train.rlcd import _items, plan_batches
    from devision.train.samples import read_decision_jsonl
    cfg = json.load(open(os.path.join(INIT, "config.json")))
    d = SimpleNamespace(tok=AutoTokenizer.from_pretrained(os.path.join(INIT, "tokenizer")),
                        cfg=ModelConfig(**{k: v for k, v in cfg.items() if k in ModelConfig.__dataclass_fields__}), model=None)
    samples = read_decision_jsonl(os.path.join(root, "cmp", "train.jsonl"))
    items = _items(d, samples)
    random.seed(1)
    p = plan_batches(items, 8)
    flat = sorted(i for b in p for i in b)
    units: dict = defaultdict(list)
    for i, it in enumerate(items):
        units[(it["group"], it["qid"])].append(i)
    where = {i: k for k, b in enumerate(p) for i in b}
    split = [u for u, v in units.items() if len({where[i] for i in v}) > 1]
    exposure = Counter(flat.count(i) for i in range(len(items)))
    n_cmp = sum(1 for it in items if it["source"].endswith("temp") and "temperature" in it["keys"][0].lower() or it["keys"][0].startswith("sample"))
    out = {"items": len(items), "updates": len(p), "batch_sizes": dict(Counter(map(len, p))), "units": len(units),
           "unit_sizes": dict(Counter(map(len, units.values()))), "split_units": len(split),
           "exposure_per_question": {str(k): v for k, v in exposure.items()}, "comparison_items": n_cmp,
           "tail_batch": len(p[-1]), "every_item_once": flat == list(range(len(items))), "seed": 1, "micro_batch": 8,
           "tokenizer_config": INIT}
    json.dump(out, open(os.path.join(root, "cmp", "PLAN.json"), "w"), indent=1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data")
    a = ap.parse_args()
    print(json.dumps(build(a.root), indent=1))
    print(json.dumps(plan(a.root), indent=1))
