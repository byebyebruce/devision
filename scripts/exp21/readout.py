"""exp21 statistics shared by precheck.py and report.py (docs/experiments/2026-10-15-exp21-reading-plan.md).

Synthetic records (sample id "exp21:<family>:v<view>:h<height>[:erased|:text]", qids readA / readB / cmp) are
resampled by scene family: a family's two value views, four heights and three questions move together.
Every statistic reports its point value, a 95% interval and its denominator (questions and families).
"""
import random
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

HEIGHTS = (6, 8, 11, 16)
PASS, FAIL, NEAR = "pass", "fail", "near threshold"
# temp : force weights of the real transfer readout = the ScienceQA validation counts of these two question forms
# (68 temperature, 114 force; critic EXP21-R1-03), fixed before running
WEIGHTS = {"temp": 68 / 182, "force": 114 / 182}


def parse(sample_id: str) -> Tuple[str, int, int]:
    """(family, view, height) of a synthetic sample id; height 0 for the erased / text controls."""
    parts = sample_id.split(":")
    return parts[1], int(parts[2][1:]), int(parts[3][1:])


def bootstrap(units: Dict[str, List[Tuple[float, float]]], stat: Callable[[List[Tuple[float, float]]], float],
              resamples: int = 2000, seed: int = 0) -> Dict:
    """`units`: resampling unit -> list of per-unit (numerator, denominator) cells, one per component; `stat` maps
    the summed cells to a number. Returns value, ci95, units."""
    keys = sorted(units)
    if not keys:
        return {"value": None, "ci95": [None, None], "units": 0}
    width = len(units[keys[0]])

    def total(sel: Iterable[str]) -> List[Tuple[float, float]]:
        acc = [[0.0, 0.0] for _ in range(width)]
        for k in sel:
            for j, (a, b) in enumerate(units[k]):
                acc[j][0] += a
                acc[j][1] += b
        return [(a, b) for a, b in acc]

    rng = random.Random(seed)
    boot = sorted(stat(total(rng.choice(keys) for _ in keys)) for _ in range(resamples))
    return {"value": stat(total(keys)), "ci95": [boot[int(0.025 * resamples)], boot[int(0.975 * resamples) - 1]],
            "units": len(keys)}


def ratio(cells: List[Tuple[float, float]]) -> float:
    a, b = cells[0]
    return a / b if b else float("nan")


def syn_index(records: Sequence[Dict]) -> Dict[Tuple[str, int, int], Dict[str, bool]]:
    """(family, view, height) -> {qid: correct}."""
    out: Dict[Tuple[str, int, int], Dict[str, bool]] = defaultdict(dict)
    for r in records:
        out[parse(r["sample_id"])][r["qid"]] = bool(r["correct"])
    return out


def syn_stat(index, height: Optional[int], what: str, resamples: int = 2000) -> Dict:
    """One synthetic statistic at one height, resampled by family:
      read      -- readA and readB accuracy (questions);
      readA / readB / cmp -- that question's accuracy;
      both_read -- readA and readB both right on a picture;
      cmp_given_both -- compare right on the pictures where both readings are right (a locating clue, not a
                        measurement of an internal computation; its denominator is reported);
      cf_read / cf_cmp -- right in both value views (readB, cmp): the answer follows the drawn values."""
    units: Dict[str, List[Tuple[float, float]]] = defaultdict(lambda: [(0.0, 0.0)])
    rows = [(k, v) for k, v in index.items() if height is None or k[2] == height]

    def add(fam, a, b):
        x, y = units[fam][0]
        units[fam] = [(x + a, y + b)]
    if what in ("cf_read", "cf_cmp"):
        q = "readB" if what == "cf_read" else "cmp"
        per = defaultdict(dict)
        for (fam, view, h), v in rows:
            per[(fam, h)][view] = v.get(q)
        for (fam, h), views in per.items():
            if len(views) == 2 and None not in views.values():
                add(fam, float(views[0] and views[1]), 1.0)
    else:
        for (fam, view, h), v in rows:
            if what == "read":
                for q in ("readA", "readB"):
                    if q in v:
                        add(fam, float(v[q]), 1.0)
            elif what in ("readA", "readB", "cmp"):
                if what in v:
                    add(fam, float(v[what]), 1.0)
            elif what == "both_read":
                add(fam, float(v["readA"] and v["readB"]), 1.0)
            elif what == "cmp_given_both":
                if v["readA"] and v["readB"]:
                    add(fam, float(v["cmp"]), 1.0)
            else:
                raise ValueError(what)
    res = bootstrap(dict(units), ratio, resamples)
    res["n"] = int(sum(c[0][1] for c in units.values()))
    return res


def height_gap(index, high: int, low: int, resamples: int = 2000) -> Dict:
    """Paired read accuracy (high - low), the same families at both heights."""
    units = defaultdict(lambda: [(0.0, 0.0), (0.0, 0.0)])
    for (fam, view, h), v in index.items():
        if h in (high, low):
            j = 0 if h == high else 1
            for q in ("readA", "readB"):
                c = units[fam]
                c[j] = (c[j][0] + float(v[q]), c[j][1] + 1.0)
    return bootstrap(dict(units), lambda c: c[0][0] / c[0][1] - c[1][0] / c[1][1], resamples)


def classify(stat: Dict) -> str:
    """The screening rule, written before running: pass = point >= 0.80 and lower bound >= 0.75; fail = point
    < 0.60 and upper bound < 0.65; anything else (an interval across a threshold) is near threshold."""
    v, (lo, hi) = stat["value"], stat["ci95"]
    if v is None:
        return NEAR
    if v >= 0.80 and lo >= 0.75:
        return PASS
    if v < 0.60 and hi < 0.65:
        return FAIL
    return NEAR


def accuracy(records: Sequence[Dict]) -> Dict:
    units = defaultdict(lambda: [(0.0, 0.0)])
    for r in records:
        a, b = units[r["image_id"]][0]
        units[r["image_id"]] = [(a + float(r["correct"]), b + 1.0)]
    res = bootstrap(dict(units), ratio)
    res["n"] = len(records)
    return res


def _by_key(records):
    return {(r["sample_id"], r["qid"]): r for r in records}


def transfer(step0: Mapping[str, Sequence[Dict]], step1: Mapping[str, Sequence[Dict]], resamples: int = 2000,
             seed: int = 0) -> Dict:
    """Real transfer, fixed before running (critic EXP21-R1-03): the 68:114-weighted temp / force accuracy change
    (step1 - step0) and the change of the picture gain ((true - mismatched) at step1 minus at step0), each set's
    pictures resampled within the set, the four records of a question (true / mismatched x two steps) together.
    `step0` / `step1`: {"temp", "force", "temp_mism", "force_mism"} -> records."""
    rng = random.Random(seed)
    cells = {}
    for kind in ("temp", "force"):
        t0, t1 = _by_key(step0[kind]), _by_key(step1[kind])
        m0, m1 = _by_key(step0[kind + "_mism"]), _by_key(step1[kind + "_mism"])
        if not (set(t0) == set(t1) == set(m0) == set(m1)):
            raise ValueError("%s: the four record sets hold different questions" % kind)
        per = defaultdict(lambda: [0.0, 0.0, 0.0])   # picture -> [acc change, gain change, questions]
        for k, r in t0.items():
            c = per[r["image_id"]]
            acc = float(t1[k]["correct"]) - float(r["correct"])
            gain = (float(t1[k]["correct"]) - float(m1[k]["correct"])) - (float(r["correct"]) - float(m0[k]["correct"]))
            c[0] += acc
            c[1] += gain
            c[2] += 1
        cells[kind] = list(per.values())

    def value(sel):
        out = [0.0, 0.0]
        parts = {}
        for kind, rows in sel.items():
            n = sum(r[2] for r in rows)
            parts[kind] = (sum(r[0] for r in rows) / n, sum(r[1] for r in rows) / n)
            out[0] += WEIGHTS[kind] * parts[kind][0]
            out[1] += WEIGHTS[kind] * parts[kind][1]
        return out, parts

    point, parts = value(cells)
    boot = [value({k: [rng.choice(v) for _ in v] for k, v in cells.items()}) for _ in range(resamples)]
    def ci(get):
        xs = sorted(get(b) for b in boot)
        return [xs[int(0.025 * resamples)], xs[int(0.975 * resamples) - 1]]
    res = {"accuracy_change": {"value": point[0], "ci95": ci(lambda b: b[0][0])},
           "gain_change": {"value": point[1], "ci95": ci(lambda b: b[0][1])}}
    for kind in cells:
        res[kind] = {"questions": int(sum(r[2] for r in cells[kind])), "pictures": len(cells[kind]),
                     "accuracy_change": {"value": parts[kind][0], "ci95": ci(lambda b, k=kind: b[1][k][0])},
                     "gain_change": {"value": parts[kind][1], "ci95": ci(lambda b, k=kind: b[1][k][1])}}
    return res


def paired_by_occurrence(a: Sequence[Dict], b: Sequence[Dict], field: str = "correct", resamples: int = 2000,
                         seed: int = 0) -> Dict:
    """B - A of `field` on a set whose file may hold a question id twice (data/r13b/val.jsonl: 242 ids, original
    and rewritten options; critic EXP21-I01): each copy is matched with the same copy (n-th occurrence) of the other
    run, and the pair must agree on picture, options and gold; nothing is dropped. Pictures resampled."""
    from devision.train.compare import _occurrences
    oa, ob = _occurrences(list(a)), _occurrences(list(b))
    if oa.keys() != ob.keys():
        raise ValueError("the two runs hold different questions (%d vs %d)" % (len(oa), len(ob)))
    units: Dict[str, List[Tuple[float, float]]] = defaultdict(lambda: [(0.0, 0.0)])
    for k, x in oa.items():
        y = ob[k]
        for f in ("image_id", "options", "gold"):
            if x.get(f) != y.get(f):
                raise ValueError("%s: copy %d differs in %s" % (k[0], k[2], f))
        s, n = units[x["image_id"]][0]
        units[x["image_id"]] = [(s + float(y[field]) - float(x[field]), n + 1.0)]
    res = bootstrap(dict(units), ratio, resamples, seed)
    return {"difference": res["value"], "ci95": res["ci95"], "pictures": res["units"], "questions": len(oa)}


def fmt_stat(s: Dict, digits: int = 3) -> str:
    """Point [interval] n=..., or "not estimable (n=0)" when a conditional statistic has no cases (critic
    EXP21-I02); the JSON keeps null."""
    if s.get("value") is None:
        return "not estimable (n=0)"
    text = "%.*f [%.*f, %.*f]" % (digits, s["value"], digits, s["ci95"][0], digits, s["ci95"][1])
    return text + (" n=%d" % s["n"] if "n" in s else "")
