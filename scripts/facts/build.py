"""Complete-fact decision CE data (critic DECISION-FACTS-PREP-02; docs/experiments/2026-10-19-decision-facts-prep.md). CPU only,
no model weights (the batch plan uses EXP57's tokenizer / config only). From data/real_layout only -- never a ScienceQA,
dev_sqa, EXP59 or EXP60 picture:

  train  600 temperature + 600 magnet families of the TRAIN split, by stable hash ("facts-v1:" + family), stratified by
         axis, excluding EXP57's 1,100 data/cmp families; both views. Each row's state = its own state_text + its reading
         questions' gold numbers (exp55's table, devision.reading.serialise / fmt) + for magnets one line "Also shown in the
         picture: Pair 1: ... Pair 2: ..." with each pair's magnet sizes from the generator's own caption
         (captions_train.jsonl; never a force relation). Questions: temperature = the comparison + 4 readings; magnets =
         the comparison + 2 distance readings. Options reordered by random.Random(sha256(row id + qid)); the gold follows.
  fit    100 + 100 whole families of the train set (by hash, stratified by axis), fixed now (G condition).
  tune   the TUNE split's 150 + 150 families (data/real_layout/questions_tune.jsonl), never trained, three conditions:
         G  gold numbers + gold sizes;
         H  EXP56's registered live tables (automatic numbers) + gold sizes  -- a HYBRID, never the product path;
         P  EXP56's live tables only (fully automatic; magnets then have no size fact: the product path today).
Labels are the generator's (one factor at a time for magnets: distance / size / equal); nothing is computed from a force
formula. Writes data/facts/{train.jsonl, fit.jsonl, tune_G.jsonl, tune_H.jsonl, tune_P.jsonl, PLAN.json, MANIFEST.json}.

    uv run python scripts/facts/build.py --root data
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from types import SimpleNamespace
from typing import Any, Dict, List, cast

from devision.reading import TEMPLATES, fmt, serialise

SEED = "facts-v1:"
N_TEMP, N_FORCE, N_FIT = 600, 600, 100
INIT = "runs/exp57/stage2"
FACTS_HEADER = "Also shown in the picture:"
SIZES = re.compile(r"Pair (\d): (two large magnets|one large magnet and one small magnet)")


def h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def jsonl(p: str) -> List[Dict]:
    return [json.loads(x) for x in open(p) if x.strip()]


def stratified(fams: Dict[str, str], n: int, salt: str = "") -> List[str]:
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


def number_of(option: str) -> str:
    return option.split(" ")[0].replace(",", "")


def facts_line(caption: str) -> str:
    """The pair sizes of a generator caption, as facts (no distance, no relation)."""
    got = dict(SIZES.findall(caption))
    if set(got) != {"1", "2"}:
        raise SystemExit("caption without both pairs' sizes: %r" % caption)
    return "%s Pair 1: %s. Pair 2: %s." % (FACTS_HEADER, got["1"], got["2"])


def gold_class(kind: str, answer: str) -> str:
    """The canonical class of a comparison answer: temperature sample A / sample B / same; magnets the pair with the
    stronger force (P1 / P2) or same -- "weaker / smaller in Pair 1" means P2."""
    a = answer.rstrip(".")
    if kind == "temp":
        return "same" if a.startswith("neither") else a
    if "same" in a:
        return "same"
    pair = a[-1]
    strong = any(w in a for w in ("stronger", "greater"))
    return "P" + (pair if strong else ("2" if pair == "1" else "1"))


def gold_table(r: Dict) -> List[List[str]]:
    reading = {q["instructions"]: qid for qid, q in r["questions"].items() if q["instructions"] in TEMPLATES}
    out = []
    for t in TEMPLATES:
        if t in reading:
            g = next(o for o, p in r["gold"][reading[t]]["probabilities"].items() if p == 1.0)
            out.append([*TEMPLATES[t], fmt(number_of(g))])
    return out


def row_of(r: Dict, state: str, facts: str = "") -> Dict:
    qs, gold = {}, {}
    for qid, q in r["questions"].items():
        opts = list(q["criteria"])
        random.Random(h(r["id"] + "|" + qid)).shuffle(opts)
        qs[qid] = {**q, "criteria": {o: None for o in opts}}
        gold[qid] = {"probabilities": {o: r["gold"][qid]["probabilities"][o] for o in opts}}
    out = {**{k: r[k] for k in ("id", "source", "kind", "axis", "image_id", "image", "family", "case")}, "group": r["family"],
           "state_text": state + ("\n" + facts if facts else ""), "questions": qs, "gold": gold}
    return out


def build(root: str = "data") -> Dict:
    rl = os.path.join(root, "real_layout")
    rows = jsonl(os.path.join(rl, "questions_train.jsonl"))
    caps = {c["id"]: c["captions"][0] for c in jsonl(os.path.join(rl, "captions_train.jsonl"))}
    used = {r["family"] for r in jsonl(os.path.join(root, "cmp", "train.jsonl"))}
    temp = {r["family"]: r["axis"] for r in rows if r["kind"] == "temp" and r["family"] not in used}
    force = {r["family"]: r["axis"] for r in rows if r["kind"] == "force" and r["family"] not in used}
    t_f, m_f = stratified(temp, N_TEMP), stratified(force, N_FORCE)
    fit = stratified({f: temp[f] for f in t_f}, N_FIT, "fit:") + stratified({f: force[f] for f in m_f}, N_FIT, "fit:")
    chosen = set(t_f) | set(m_f)
    train = []
    for r in rows:
        if r["family"] in chosen:
            st = serialise(r.get("state_text", ""), gold_table(r))
            train.append(row_of(r, st, facts_line(caps[r["id"]]) if r["kind"] == "force" else ""))
    dest = os.path.join(root, "facts")
    os.makedirs(dest, exist_ok=True)

    def write(name, rs):
        with open(os.path.join(dest, name), "w") as fh:
            fh.write("".join(json.dumps(x) + "\n" for x in rs))
    write("train.jsonl", train)
    write("fit.jsonl", [x for x in train if x["family"] in set(fit)])
    tune = jsonl(os.path.join(rl, "questions_tune.jsonl"))
    tcaps = {c["id"]: c["captions"][0] for c in jsonl(os.path.join(rl, "captions_tune.jsonl"))}
    live = {x["id"]: x for x in jsonl(os.path.join("runs", "exp56", "inputs_P.jsonl"))}
    g, hy, p = [], [], []
    for r in tune:
        f = facts_line(tcaps[r["id"]]) if r["kind"] == "force" else ""
        base = r.get("state_text", "")
        g.append(row_of(r, serialise(base, gold_table(r)), f))
        auto = live[r["id"]]
        if auto["state"] != serialise(base, auto["table"]):
            raise SystemExit("%s: EXP56's live state is not its table serialised" % r["id"])
        hy.append(row_of(r, auto["state"], f))
        p.append(row_of(r, auto["state"]))
    write("tune_G.jsonl", g)
    write("tune_H.jsonl", hy)
    write("tune_P.jsonl", p)
    cmp_gold = Counter()
    for x in train:
        for qid, q in x["questions"].items():
            if q["instructions"] not in TEMPLATES:
                a = next(o for o, v in x["gold"][qid]["probabilities"].items() if v == 1.0)
                cmp_gold["%s|%s|%s" % (x["kind"], x["axis"], gold_class(x["kind"], a))] += 1
    tune_fams = {r["family"] for r in tune}
    tune_cases = {r["case"] for r in tune}
    m = {"source": {"questions_train": h(open(os.path.join(rl, "questions_train.jsonl")).read()),
                    "captions_train": h(open(os.path.join(rl, "captions_train.jsonl")).read()),
                    "questions_tune": h(open(os.path.join(rl, "questions_tune.jsonl")).read()),
                    "captions_tune": h(open(os.path.join(rl, "captions_tune.jsonl")).read()),
                    "exp56_inputs_P": h(open(os.path.join("runs", "exp56", "inputs_P.jsonl")).read())},
         "families": {"temp": len(t_f), "force": len(m_f), "fit": len(fit), "excluded_exp57_cmp": len(used)},
         "axes": {"temp": dict(Counter(temp[f] for f in t_f)), "force": dict(Counter(force[f] for f in m_f))},
         "rows": len(train), "questions": sum(len(x["questions"]) for x in train),
         "comparison_gold": dict(sorted(cmp_gold.items())),
         "isolation": {"train_tune_families": len(chosen & tune_fams), "train_tune_cases": len({x["case"] for x in train} & tune_cases),
                       "train_exp57_families": len(chosen & used)},
         "tune": {"rows": len(tune), "conditions": {"G": "gold numbers + gold sizes", "H": "EXP56 live numbers + gold sizes (hybrid)",
                                                    "P": "EXP56 live numbers only (product path today)"}},
         "sha256": {n: h(open(os.path.join(dest, n)).read()) for n in ("train.jsonl", "fit.jsonl", "tune_G.jsonl", "tune_H.jsonl", "tune_P.jsonl")}}
    if any(m["isolation"].values()):
        raise SystemExit("isolation: %s" % m["isolation"])
    json.dump(m, open(os.path.join(dest, "MANIFEST.json"), "w"), indent=1)
    return m


def truncation(d, path: str) -> Dict:
    """exp55's check with EXP57's tokenizer and sequence budget (no weights): questions whose state loses tokens or
    whose options are not all distinct."""
    from laya.common import build_sequence

    from devision.infer import _split_state, serialize_state
    n = {"questions": 0, "state_dropped": 0, "options_not_distinct": 0, "max_state_tokens": 0}
    for r in jsonl(path):
        text_state, _ = _split_state(r["state_text"])
        for q in r["questions"].values():
            lq = {"t": q["type"], "ins": serialize_state(q["instructions"]), "crit": q.get("criteria")}
            res: Any = build_sequence(d.tok, text_state, lq, d.cfg.max_len, d.cfg.head_max_len, return_stats=True, return_truncation_stats=True)
            markers, stats, tr = res[1], res[2], res[3]
            n["questions"] += 1
            n["state_dropped"] += tr["state_tokens_dropped"] > 0
            n["options_not_distinct"] += stats.get("options_distinct", len(markers)) < len(markers)
            n["max_state_tokens"] = max(n["max_state_tokens"], tr["state_tokens"])
    return n


def plan(root: str = "data") -> Dict:
    """EXP57's code path: read_decision_jsonl -> rlcd._items (EXP57's tokenizer / config) -> random.seed(1) ->
    plan_batches(8); plus a truncation check of every train / tune state (tokens only)."""
    from transformers import AutoTokenizer

    from devision.model import ModelConfig
    from devision.train.rlcd import _items, plan_batches
    from devision.train.samples import read_decision_jsonl
    cfg = json.load(open(os.path.join(INIT, "config.json")))
    d = SimpleNamespace(tok=AutoTokenizer.from_pretrained(os.path.join(INIT, "tokenizer")),
                        cfg=ModelConfig(**{k: v for k, v in cfg.items() if k in ModelConfig.__dataclass_fields__}), model=None)
    items = _items(cast(Any, d), read_decision_jsonl(os.path.join(root, "facts", "train.jsonl")))
    random.seed(1)
    p = plan_batches(items, 8)
    units: Dict = defaultdict(list)
    for i, it in enumerate(items):
        units[(it["group"], it["qid"])].append(i)
    where = {i: k for k, b in enumerate(p) for i in b}
    split = [u for u, v in units.items() if len({where[i] for i in v}) > 1]
    flat = sorted(i for b in p for i in b)
    trunc = {name: truncation(d, os.path.join(root, "facts", name)) for name in ("train.jsonl", "tune_G.jsonl", "tune_H.jsonl", "tune_P.jsonl")}
    if any(v["state_dropped"] or v["options_not_distinct"] for v in trunc.values()):
        raise SystemExit("states lose tokens / options not distinct: %s" % trunc)
    out = {"items": len(items), "updates": len(p), "batch_sizes": dict(Counter(map(len, p))), "units": len(units),
           "unit_sizes": dict(Counter(map(len, units.values()))), "split_units": len(split), "every_item_once": flat == list(range(len(items))),
           "by_kind": dict(Counter(it["source"] for it in items)), "seed": 1, "micro_batch": 8, "tokenizer_config": INIT, "truncation": trunc}
    json.dump(out, open(os.path.join(root, "facts", "PLAN.json"), "w"), indent=1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data")
    a = ap.parse_args()
    print(json.dumps(build(a.root), indent=1))
    print(json.dumps(plan(a.root), indent=1))
    sys.exit(0)
