"""exp68 CPU data (critic EXP68-PREP-01; docs/experiments/2026-10-19-exp68-combinations-plan.md). No model weights (the
batch plan uses EXP63's tokenizer / config only).

The question: with the same 9,600 questions / 1,200 updates as EXP65's C arm, do new anti-shortcut temperature value
combinations -- never seen in any earlier data -- let both conflict classes generalise? Only the temperature part of
data/facts/train.jsonl is replaced; the 1,200 magnet rows (3 questions each) are copied byte for byte.

  train  temperature: 6 classes (equal, same mass, same speed, same direction, opposite / mass wins, opposite / speed
         wins) x 100 distinct canonical value combinations, each as two views (A / B swapped) = 1,200 rows x 5 questions
         (the comparison + 4 readings), built by the generator's own items (scripts/data/real_layout.temp_items: question,
         shuffled options, shared reading candidates) and facts' row_of (state = the hint + the gold table, exp55's
         serialise / fmt; options reordered by sha256(row id + qid)). The two conflict classes come as 100 pairs that
         share three of the four numbers: a mass-wins combination and a speed-wins partner made by changing ONE number
         (even pairs: the light object's speed raised; odd pairs: the heavy object's mass lowered, still heavier), so
         neither class is the range of a single number. Values only from the train set's own mass / speed vocabulary.
         Labels by the generator's rule (higher m * v^2; equal when both are equal). Rows are text only: the image
         path "text-only:exp68" is never opened (text_only training never loads a picture).
  diag   a new development diagnostic, 40 canonical combinations per class (seed 6868), each asked in both orders
         (480 requests, one question each, EXP65's diag format), disjoint from the new train combinations and every
         earlier one. Not a release confirmation set.
Excluded everywhere: the canonical combinations of data/real_layout (train / tune / confirm), data/cmp, data/facts,
EXP65's diag and the 24 transcribed pictures. Both sets are stratified by the kinetic-energy ratio (near < 2 <= large),
reported only. Writes data/exp68/{train.jsonl, diag.jsonl, MANIFEST.json, PLAN.json}.

    uv run python scripts/exp68/build.py --root data
"""
import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple, cast

HERE = os.path.dirname(__file__)
CLASSES = ("equal", "same_mass", "same_speed", "same_direction", "opposite_mass_wins", "opposite_speed_wins")
CONFLICT = ("opposite_mass_wins", "opposite_speed_wins")
PER_CLASS, DIAG_PER_CLASS, SEED, DIAG_SEED = 100, 40, 68, 6868
IMAGE = "text-only:exp68"
INIT = "runs/exp63/stage2"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, rel))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


b65 = _load("exp65_build", "../exp65/build.py")
fb, rl = b65.fb, b65.rl
ke, class_of, canonical = b65.ke, b65.class_of, b65.canonical


def h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def sha(p: str) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def guarded(fn, *a):
    """real_layout imports scripts/exp28/build.py by the bare name "build" at call time: set any other "build" aside."""
    saved = sys.modules.pop("build", None)
    try:
        return fn(*a)
    finally:
        sys.modules.pop("build", None)
        if saved is not None:
            sys.modules["build"] = saved


def vocabulary(root: str) -> Tuple[List[int], List[int]]:
    rows = [r for r in fb.jsonl(os.path.join(root, "facts", "train.jsonl")) if r["kind"] == "temp"]
    ms = sorted({int(p[0]) for r in rows for p in b65.values_of(r)})
    vs = sorted({int(p[1]) for r in rows for p in b65.values_of(r)})
    return ms, vs


def excluded(root: str) -> Dict[str, set]:
    old = guarded(b65.existing_cases, root)
    diag = {canonical(*b65.state_values(r["state_text"])) for r in fb.jsonl(os.path.join(root, "exp65", "diag.jsonl"))}
    tr = set()
    for k in guarded(rl.transcribed_keys):
        if k[0] == "temp":
            tr.add(canonical((k[1], k[2]), (k[3], k[4])))
    return {"data": {tuple(tuple(map(float, p)) for p in c) for c in old}, "exp65_diag": {tuple(tuple(map(float, p)) for p in c) for c in diag},
            "transcribed": {tuple(tuple(map(float, p)) for p in c) for c in tr}}


def key(a, b) -> Tuple:
    return tuple(sorted([(float(a[0]), float(a[1])), (float(b[0]), float(b[1]))]))


def stratum(a, b) -> str:
    x, y = sorted([ke(a), ke(b)])
    return "equal" if x == y else ("near" if y < 2 * x else "large")


def conflict_pairs(rng: random.Random, ms: List[int], vs: List[int], banned: set, n: int) -> List[Tuple[Tuple, Tuple]]:
    """n (mass-wins, speed-wins) pairs differing in one number: heavy = (mh, v_slow), light = (ml, v_fast), mh > ml,
    v_slow < v_fast, heavy wins; partner = the light speed raised (even k) or the heavy mass lowered (odd k) until the light
    one wins, the value drawn uniformly among those that flip it."""
    out: List[Tuple[Tuple, Tuple]] = []
    seen = set(banned)
    tries = 0
    while len(out) < n:
        tries += 1
        if tries > 5_000_000:
            raise SystemExit("not enough conflict pairs: %d" % len(out))
        mh, ml = rng.choice(ms), rng.choice(ms)
        vsl, vf = rng.choice(vs), rng.choice(vs)
        if not (mh > ml and vsl < vf):
            continue
        heavy, light = (mh, vsl), (ml, vf)
        if class_of(heavy, light) != "opposite_mass_wins":
            continue
        if len(out) % 2 == 0:
            cand = [(heavy, (ml, v)) for v in vs if v > vf and ke((ml, v)) > ke(heavy)]
        else:
            cand = [((m, vsl), light) for m in ms if ml < m < mh and ke((m, vsl)) < ke(light)]
        cand = [c for c in cand if class_of(*c) == "opposite_speed_wins" and key(*c) not in seen]
        if not cand or key(heavy, light) in seen:
            continue
        partner = rng.choice(cand)
        out.append(((heavy, light), partner))
        seen |= {key(heavy, light), key(*partner)}
    return out


def sample_class(rng: random.Random, c: str, ms: List[int], vs: List[int], seen: set, n: int) -> List[Tuple]:
    """b65.diag's draw for one class: a, then b by the class's constraint; no tie unless equal; unseen."""
    out: List[Tuple] = []
    tries = 0
    while len(out) < n:
        tries += 1
        if tries > 5_000_000:
            raise SystemExit("not enough %s combinations: %d" % (c, len(out)))
        a = (rng.choice(ms), rng.choice(vs))
        if c == "equal":
            b = a
        elif c == "same_mass":
            b = (a[0], rng.choice(vs))
        elif c == "same_speed":
            b = (rng.choice(ms), a[1])
        else:
            b = (rng.choice(ms), rng.choice(vs))
        if class_of(a, b) != c or (c != "equal" and ke(a) == ke(b)) or key(a, b) in seen:
            continue
        seen.add(key(a, b))
        out.append((a, b))
    return out


def rows_of(k: int, cls: str, combo: Tuple, orient: bool) -> List[Dict]:
    """The two views of one combination as facts rows (generator items -> real_layout row -> facts row_of)."""
    p1, p2 = combo if orient else (combo[1], combo[0])
    views = [{"A": {"mass": p1[0], "speed": p1[1]}, "B": {"mass": p2[0], "speed": p2[1]}, "speed_first": False},
             {"A": {"mass": p2[0], "speed": p2[1]}, "B": {"mass": p1[0], "speed": p1[1]}, "speed_first": False}]
    fam = "exp68-c%03d" % k
    out = []
    for v, scene in enumerate(views):
        _, items = guarded(rl.temp_items, scene, 10_000 + k, views)
        rid = "exp68:c%03d:v%d" % (k, v)
        r = {"id": rid, "source": "real-layout-temp", "kind": "temp", "axis": cls, "image_id": rid, "image": IMAGE, "family": fam, "group": fam,
             "case": repr(rl.case_key("temp", scene)), "state_text": rl.TEMP_HINT,
             "questions": {"q%d" % j: {"type": "choice", "instructions": it["q"], "criteria": {o: None for o in it["options"]}} for j, it in enumerate(items)},
             "gold": {"q%d" % j: {"probabilities": {o: float(o == it["gold"]) for o in it["options"]}} for j, it in enumerate(items)}}
        out.append(fb.row_of(r, fb.serialise(r["state_text"], fb.gold_table(r))))
    return out


def check_row(r: Dict) -> None:
    """Every temperature row: the four table numbers = the reading golds; the comparison gold = the rule."""
    a, b = b65.values_of(r)
    sv = b65.state_values(r["state_text"])
    if (a, b) != sv:
        raise SystemExit("%s: table %s, golds %s" % (r["id"], sv, (a, b)))
    q0 = r["gold"]["q0"]["probabilities"]
    g = next(o for o, p in q0.items() if p == 1.0)
    w = b65.winner(a, b)
    if not (g.startswith("neither") if w == "neither" else g == w):
        raise SystemExit("%s: comparison gold %r, rule %r" % (r["id"], g, w))
    if class_of(a, b) != r["axis"]:
        raise SystemExit("%s: class %s, axis %s" % (r["id"], class_of(a, b), r["axis"]))


def separability(combos: Dict[str, List[Tuple]]) -> Dict[str, float]:
    """Mass-wins vs speed-wins: the best single-threshold accuracy of each single feature (heavy / light mass, slow /
    fast speed, their ratios) -- how far one number alone tells the class."""
    pts = []
    for c in CONFLICT:
        for a, b in combos[c]:
            heavy, light = (a, b) if a[0] > b[0] else (b, a)
            pts.append((c, {"heavy_mass": heavy[0], "light_mass": light[0], "slow_speed": heavy[1], "fast_speed": light[1],
                            "mass_ratio": heavy[0] / light[0], "speed_ratio": light[1] / heavy[1]}))
    out = {}
    for f in pts[0][1]:
        xs = sorted((p[1][f], p[0] == CONFLICT[0]) for p in pts)
        n, best = len(xs), 0.0
        pos = sum(y for _, y in xs)
        left_pos = 0
        for i in range(n + 1):
            if i:
                left_pos += xs[i - 1][1]
            acc = max(left_pos + (n - i - (pos - left_pos)), (i - left_pos) + (pos - left_pos)) / n
            best = max(best, acc)
        out[f] = round(best, 4)
    return out


def old_conflicts(root: str) -> Dict[str, List[Tuple]]:
    out: Dict[str, set] = defaultdict(set)
    for r in fb.jsonl(os.path.join(root, "facts", "train.jsonl")):
        if r["kind"] == "temp":
            a, b = b65.values_of(r)
            c = class_of(a, b)
            if c in CONFLICT:
                out[c].add(key(a, b))
    return {c: [tuple(x) for x in sorted(v)] for c, v in out.items()}


def diag_rows(combos: Dict[str, List[Tuple]], template: Dict) -> List[Dict]:
    """EXP65's diag format: one comparison question, both orders, options by sha256(id)."""
    state, q_text, opts = template["state"], template["q"], template["opts"]
    same_opt = next(o for o in opts if o.startswith("neither"))
    out = []
    for c in CLASSES:
        for k, (p1, p2) in enumerate(combos[c]):
            for order, (a, b) in enumerate(((p1, p2), (p2, p1))):
                rid = "exp68-diag:%s:%02d:%d" % (c, k, order)
                table = [["Sample A", "mass of each particle", fb.fmt("%g" % a[0])], ["Sample B", "mass of each particle", fb.fmt("%g" % b[0])],
                         ["Sample A", "average particle speed", fb.fmt("%g" % a[1])], ["Sample B", "average particle speed", fb.fmt("%g" % b[1])]]
                o = list(opts)
                random.Random(h(rid)).shuffle(o)
                g = b65.winner(a, b)
                gold = same_opt if g == "neither" else g
                out.append({"id": rid, "kind": "temp", "axis": c, "family": "%s:%02d" % (c, k), "case": str((p1, p2)), "source": "exp68-diag",
                            "stratum": stratum(p1, p2), "image": "", "image_id": rid, "state_text": fb.serialise(state, table),
                            "questions": {"q0": {"type": "choice", "instructions": q_text, "criteria": {x: None for x in o}}},
                            "gold": {"q0": {"probabilities": {x: float(x == gold) for x in o}}}})
    return out


def build(root: str = "data") -> Dict:
    ms, vs = vocabulary(root)
    ex = excluded(root)
    banned = set().union(*ex.values())
    rng = random.Random(SEED)
    pairs = conflict_pairs(rng, ms, vs, banned, PER_CLASS)
    combos: Dict[str, List[Tuple]] = {"opposite_mass_wins": [p[0] for p in pairs], "opposite_speed_wins": [p[1] for p in pairs]}
    seen = banned | {key(*c) for v in combos.values() for c in v}
    for c in CLASSES[:4]:
        combos[c] = sample_class(rng, c, ms, vs, seen, PER_CLASS)
    facts_lines = open(os.path.join(root, "facts", "train.jsonl")).read().splitlines()
    force = [x for x in facts_lines if json.loads(x)["kind"] == "force"]
    temp: List[Dict] = []
    k = 0
    for c in CLASSES:
        for combo in combos[c]:
            for r in rows_of(k, c, combo, bool(rng.random() < 0.5)):
                check_row(r)
                temp.append(r)
            k += 1
    dest = os.path.join(root, "exp68")
    os.makedirs(dest, exist_ok=True)
    with open(os.path.join(dest, "train.jsonl"), "w") as fh:
        fh.write("".join(x + "\n" for x in force) + "".join(json.dumps(r) + "\n" for r in temp))
    drng = random.Random(DIAG_SEED)
    dseen = set(seen)
    dcombos = {c: sample_class(drng, c, ms, vs, dseen, DIAG_PER_CLASS) for c in CLASSES}
    old = fb.jsonl(os.path.join(root, "exp65", "diag.jsonl"))
    tq = next(q for q in old[0]["questions"].values())
    template = {"state": old[0]["state_text"].split("\n")[0], "q": tq["instructions"], "opts": sorted(tq["criteria"])}
    diag = diag_rows(dcombos, template)
    with open(os.path.join(dest, "diag.jsonl"), "w") as fh:
        fh.write("".join(json.dumps(r) + "\n" for r in diag))
    train_keys = {key(*c) for v in combos.values() for c in v}
    diag_keys = {key(*c) for v in dcombos.values() for c in v}
    m = {"rows": {"force": len(force), "temp": len(temp)}, "questions": len(force) * 3 + sum(len(r["questions"]) for r in temp),
         "combinations": {c: len(v) for c, v in combos.items()}, "distinct_train_combinations": len(train_keys),
         "pairs": {"n": len(pairs), "changed": {"light_speed_raised": sum(1 for i in range(len(pairs)) if i % 2 == 0),
                                                "heavy_mass_lowered": sum(1 for i in range(len(pairs)) if i % 2 == 1)},
                   "shared_numbers": sorted(Counter(len({x for p in a for x in p} & {x for p in b for x in p}) for a, b in pairs).items())},
         "strata": {"train": {c: dict(Counter(stratum(*x) for x in v)) for c, v in combos.items()},
                    "diag": {c: dict(Counter(stratum(*x) for x in v)) for c, v in dcombos.items()}},
         "single_feature_separability": {"exp68_train": separability(combos), "facts_train_old": separability(old_conflicts(root))},
         "old_conflict_combinations": {c: len(v) for c, v in old_conflicts(root).items()},
         "vocabulary": {"masses": len(ms), "speeds": len(vs)},
         "excluded": {k_: len(v) for k_, v in ex.items()},
         "isolation": {"train_vs_excluded": len(train_keys & banned), "diag_vs_excluded": len(diag_keys & banned), "train_vs_diag": len(train_keys & diag_keys)},
         "diag": {"requests": len(diag), "per_class": DIAG_PER_CLASS, "seed": DIAG_SEED,
                  "gold": dict(Counter(next(o for o, p in r["gold"]["q0"]["probabilities"].items() if p == 1.0) for r in diag))},
         "sources": {"facts_train": sha(os.path.join(root, "facts", "train.jsonl")), "exp65_diag": sha(os.path.join(root, "exp65", "diag.jsonl")),
                     "real_layout_train": sha(os.path.join(root, "real_layout", "questions_train.jsonl"))},
         "sha256": {n: sha(os.path.join(dest, n)) for n in ("train.jsonl", "diag.jsonl")}, "seeds": {"train": SEED, "diag": DIAG_SEED}}
    if any(m["isolation"].values()):
        raise SystemExit("isolation: %s" % m["isolation"])
    json.dump(m, open(os.path.join(dest, "MANIFEST.json"), "w"), indent=1)
    return m


def plan(root: str = "data") -> Dict:
    """EXP63's code path on the new file: read_decision_jsonl -> rlcd._items (EXP63's tokenizer / config) ->
    random.seed(1) -> plan_batches(8); units (group, qid) never split; truncation of train and diag."""
    from types import SimpleNamespace

    from transformers import AutoTokenizer

    from devision.model import ModelConfig
    from devision.train.rlcd import _items, plan_batches
    from devision.train.samples import read_decision_jsonl
    cfg = json.load(open(os.path.join(INIT, "config.json")))
    d = SimpleNamespace(tok=AutoTokenizer.from_pretrained(os.path.join(INIT, "tokenizer")),
                        cfg=ModelConfig(**{k: v for k, v in cfg.items() if k in ModelConfig.__dataclass_fields__}), model=None)
    items = _items(cast(Any, d), read_decision_jsonl(os.path.join(root, "exp68", "train.jsonl")))
    random.seed(1)
    p = plan_batches(items, 8)
    units: Dict = defaultdict(list)
    for i, it in enumerate(items):
        units[(it["group"], it["qid"])].append(i)
    where = {i: k for k, b in enumerate(p) for i in b}
    split = [u for u, v in units.items() if len({where[i] for i in v}) > 1]
    flat = sorted(i for b in p for i in b)
    trunc = {n: fb.truncation(d, os.path.join(root, "exp68", n)) for n in ("train.jsonl", "diag.jsonl")}
    if any(v["state_dropped"] or v["options_not_distinct"] for v in trunc.values()):
        raise SystemExit("states lose tokens / options not distinct: %s" % trunc)
    out = {"items": len(items), "updates": len(p), "batch_sizes": dict(Counter(map(len, p))), "units": len(units),
           "unit_sizes": dict(Counter(map(len, units.values()))), "split_units": len(split), "every_item_once": flat == list(range(len(items))),
           "by_kind": dict(Counter(it["source"] for it in items)), "seed": 1, "micro_batch": 8, "tokenizer_config": INIT, "truncation": trunc,
           "train_sha256": sha(os.path.join(root, "exp68", "train.jsonl"))}
    if out["updates"] != 1200 or out["split_units"] or not out["every_item_once"] or out["items"] != 9600:
        raise SystemExit("plan: %s" % {k: out[k] for k in ("items", "updates", "split_units", "every_item_once")})
    json.dump(out, open(os.path.join(root, "exp68", "PLAN.json"), "w"), indent=1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data")
    a = ap.parse_args()
    print(json.dumps(build(a.root), indent=1))
    print(json.dumps({k: v for k, v in plan(a.root).items() if k != "truncation"}, indent=1))
    sys.exit(0)
