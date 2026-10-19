"""Routed-path calibration analysis (docs/experiments/2026-10-19-v04-reader-evidence-prep03.md 5.3 and 5.5; critic
V04-EVIDENCE-R4-REVIEW-01, V04-K-PREP-02). CPU only, loads no model: it reads the per-question RAW probabilities that the
full reading entry produced on rl_calib (devision.load / Reading with the K rule, use_raw_temperatures, every comparison
question route "read") and computes everything from them.

  - log p as logits: every raw probability must be finite and strictly positive (refused otherwise, never clamped) -- the
    softmax of log p / T equals the softmax of logit / T, so temperatures fit on log p are the logits' ones;
  - three probability sets from the same raw outputs: raw (T = 1), ordinary (the candidate config's own temperature for the
    question's type / option count, i.e. what the 3,892-question ordinary calibration chose) and routed (one temperature
    per domain, fitted out of fold: families split in two by sha256(family), fit on one fold, scored on the other);
  - per family (both views together) the NLL differences Delta_R = routed - ordinary, Delta_O = ordinary - raw,
    Delta_B = routed - raw, pooled and per domain, each with a 95 % percentile interval of 2,000 family resamples (seed 0),
    resampled directly from the per-family differences;
  - the four pre-written branches (tolerance 0.02 nats / question): 1 separate routed temperature, 2 keep the current
    temperatures, 3 clear degradation (calibration fails), 4 insufficient evidence. ECE (15 bins) is reported, never decides.

  collect: the raw records themselves -- the candidate (ordinary-calibrated config) with a temporary bundle OUTSIDE it
  (same components and K rule, bound to that config; scripts/release/rebind.py's rules), loaded by the verified entry
  Reading.load(full_rules), use_raw_temperatures, devision.train.evaluate.evaluate_records on data/rl_calib/questions.jsonl;
  every comparison question must have route "read" (refused otherwise); each record gets its row's family. The temporary
  bundle's identity and the code are recorded; its probabilities are calibration data only, never evidence for the final
  combination.

    uv run python scripts/release/route_calib.py collect --candidate <dir> --bundle <tmp bundle> --rows data/rl_calib/questions.jsonl --out <records.jsonl>
    uv run python scripts/release/route_calib.py analyse --records <records.jsonl> --config <candidate>/config.json --out <report.json>
"""
import argparse
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

TOL = 0.02
DOMAIN_POINT = 0.01
RESAMPLES, SEED = 2000, 0
T_GRID = [math.exp(x / 200.0) for x in range(-600, 601)]          # 0.05 .. 20, multiplicative steps of 0.5 %


def check_records(recs: Sequence[Dict]) -> None:
    """Refuses records that cannot be calibrated: not a read question, missing family / domain, a probability that is not
    finite and > 0, probabilities not summing to 1, or a gold outside the options."""
    for r in recs:
        p = r.get("probabilities") or {}
        if r.get("route") != "read" or r.get("reading_domain") not in ("temperature", "magnets") or not r.get("family"):
            raise SystemExit("%s/%s: not a routed comparison record with a family" % (r.get("sample_id"), r.get("qid")))
        if not p or not all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in p.values()):
            raise SystemExit("%s/%s: a raw probability is zero or not finite -- log p undefined, refused" % (r.get("sample_id"), r.get("qid")))
        if abs(sum(p.values()) - 1) > 1e-4 or r.get("gold_answer") not in p:
            raise SystemExit("%s/%s: probabilities do not sum to 1 or the gold is not an option" % (r.get("sample_id"), r.get("qid")))


def scaled(p: Dict[str, float], t: float) -> Dict[str, float]:
    z = {o: math.log(v) / t for o, v in p.items()}
    m = max(z.values())
    e = {o: math.exp(v - m) for o, v in z.items()}
    s = sum(e.values())
    return {o: v / s for o, v in e.items()}


def nll(p: Dict[str, float], gold: str, t: float) -> float:
    return -math.log(scaled(p, t)[gold])


def fit_t(recs: Sequence[Dict]) -> float:
    """The temperature minimising the mean NLL of `recs` on a fixed grid (deterministic)."""
    return min(T_GRID, key=lambda t: sum(nll(r["probabilities"], r["gold_answer"], t) for r in recs))


def fold_of(family: str) -> int:
    return int(hashlib.sha256(family.encode()).hexdigest(), 16) % 2


def ordinary_t(r: Dict, config: Dict) -> float:
    """The candidate config's temperature for the record (decide's rule: the (type, option count) bucket, else the type)."""
    k = len(r["probabilities"])
    by = config.get("temperature_by_options") or {}
    for b, t in by.items():
        typ, rng = b.split(":")
        lo, _, hi = rng.partition("-")
        if typ == "choice" and int(lo) <= k <= int(hi or lo):
            return float(t)
    return float(config["temperature"][0])


def interval(values: List[float]) -> List[float]:
    rng = random.Random(SEED)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(RESAMPLES))
    return [means[int(0.025 * RESAMPLES)], means[int(0.975 * RESAMPLES) - 1]]


def ece(recs: Sequence[Dict], key: str, bins: int = 15) -> float:
    by: Dict[int, List] = defaultdict(list)
    for r in recs:
        p = r[key]
        top = max(p, key=lambda o: p[o])
        by[min(bins - 1, int(p[top] * bins))].append((p[top], top == r["gold_answer"]))
    return sum(abs(sum(c for c, _ in v) / len(v) - sum(a for _, a in v) / len(v)) * len(v) for v in by.values()) / len(recs)


def analyse(recs: List[Dict], config: Dict) -> Dict:
    check_records(recs)
    domains = sorted({r["reading_domain"] for r in recs})
    temps = {}
    for d in domains:                                          # out-of-fold routed temperatures, per domain
        for f in (0, 1):
            train = [r for r in recs if r["reading_domain"] == d and fold_of(r["family"]) != f]
            if not train:
                raise SystemExit("domain %s: fold %d has no training families" % (d, 1 - f))
            temps[(d, f)] = fit_t(train)
    rows = []
    for r in recs:
        t_ord, t_rt = ordinary_t(r, config), temps[(r["reading_domain"], fold_of(r["family"]))]
        g = r["gold_answer"]
        rows.append(dict(r, nll_raw=nll(r["probabilities"], g, 1.0), nll_ord=nll(r["probabilities"], g, t_ord),
                         nll_rt=nll(r["probabilities"], g, t_rt), p_raw=r["probabilities"], p_ord=scaled(r["probabilities"], t_ord),
                         p_rt=scaled(r["probabilities"], t_rt)))

    def deltas(sel: List[Dict]) -> Dict:
        fam: Dict[str, List[Dict]] = defaultdict(list)
        for x in sel:
            fam[x["family"]].append(x)
        out = {}
        for name, a, b in (("R", "nll_rt", "nll_ord"), ("O", "nll_ord", "nll_raw"), ("B", "nll_rt", "nll_raw")):
            per = [sum(x[a] - x[b] for x in v) / len(v) for v in fam.values()]
            out[name] = {"value": sum(per) / len(per), "ci95": interval(per)}
        out.update(families=len(fam), questions=len(sel), nll={k: sum(x["nll_" + k] for x in sel) / len(sel) for k in ("raw", "ord", "rt")},
                   ece={k: ece(sel, "p_" + k) for k in ("raw", "ord", "rt")})
        return out
    res = {"pooled": deltas(rows), "domains": {d: deltas([x for x in rows if x["reading_domain"] == d]) for d in domains},
           "routed_temperatures_out_of_fold": {"%s:fold%d" % k: v for k, v in temps.items()}}
    res["branch"] = branch(res["pooled"], {d: v["R"]["value"] for d, v in res["domains"].items()})
    if res["branch"].startswith("1"):
        res["routed_temperature_all_data"] = {d: fit_t([r for r in recs if r["reading_domain"] == d]) for d in domains}
    return res


def branch(pooled: Dict, domain_r: Dict[str, float]) -> str:
    """5.3's four branches (with R4's Delta_B condition), from the pooled intervals and the per-domain Delta_R points."""
    r, o, b = pooled["R"], pooled["O"], pooled["B"]
    if r["value"] <= -TOL and r["ci95"][1] < 0 and all(v <= DOMAIN_POINT for v in domain_r.values()) and b["ci95"][1] <= TOL:
        return "1 separate routed temperature: routed is better than ordinary by >= 0.02 (upper bound < 0), no domain worse, and not worse than raw beyond 0.02 -- needs the temperature_by_route product change reviewed before it ships"
    if r["ci95"][0] >= -TOL and o["ci95"][1] <= TOL:
        return "2 keep the current temperatures: a routed temperature gains at most 0.02 and ordinary is not worse than raw beyond 0.02"
    if o["ci95"][0] > TOL:
        return "3 clear degradation: ordinary is worse than raw beyond 0.02 on the routed path -- calibration fails, the current temperatures are not releasable there; the next choice is ruled separately"
    return "4 insufficient evidence: intervals too wide to decide -- routed-path calibration not validated, not claimed complete; nothing released on it"


def collect(reading, rows: List[Dict], data_root: str) -> List[Dict]:
    """Raw-probability records of rl_calib through the given reading combination (already loaded by the verified entry):
    raw temperatures, the formal evaluation path, every question read, each record with its row's family."""
    from devision.train.evaluate import evaluate_records
    reading.use_raw_temperatures()
    if not getattr(reading, "k_rule", False):
        raise SystemExit("the reading combination has no K rule: not the release combination")
    fam = {r["id"]: r["family"] for r in rows}
    recs = evaluate_records(reading, rows, data_root)
    bad = [(r["sample_id"], r["qid"]) for r in recs if r.get("route") != "read" or r.get("reading_domain") is None]
    if bad:
        raise SystemExit("%d rl_calib questions were not read, e.g. %s" % (len(bad), bad[0]))
    return [dict(r, family=fam[r["sample_id"]]) for r in recs]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    for k in ("candidate", "bundle", "rows", "out"):
        c.add_argument("--" + k, required=True)
    c.add_argument("--data-root", default="data")
    c.add_argument("--device", default="auto")
    n = sub.add_parser("analyse")
    for k in ("records", "config", "out"):
        n.add_argument("--" + k, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "collect":
        from devision.reading import Reading
        reading = Reading.load(a.candidate, a.bundle, device=a.device, full_rules=True)
        rows = [json.loads(x) for x in open(a.rows) if x.strip()]
        recs = collect(reading, rows, a.data_root)
        with open(a.out, "w") as fh:
            fh.write("".join(json.dumps(r) + "\n" for r in recs))
        print("%d records" % len(recs))
        return 0
    recs = [json.loads(x) for x in open(a.records) if x.strip()]
    res = analyse(recs, json.load(open(a.config)))
    json.dump(res, open(a.out, "w"), indent=1)
    print(res["branch"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
