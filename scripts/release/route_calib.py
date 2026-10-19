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

  The formal entries are registered (critic V04-K-PREP-02-R1-03): collect checks and records the calibration identity
  before loading (identity: source, candidate files / config, temporary bundle bound to that config, rows, pictures, code)
  and writes records.jsonl + CALIB.json; analyse re-verifies CALIB.json against the files and the records' completeness
  against the source rows (every question once, options / gold / family / domain, both domains, both views) before any
  statistic.

    uv run python scripts/release/route_calib.py collect --candidate <dir> --bundle <tmp bundle> --rows data/rl_calib/questions.jsonl --out-dir <calib dir>
    uv run python scripts/release/route_calib.py analyse --calib-dir <calib dir> --out <report.json>
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
    """-log softmax(log p / t)[gold], by a stable logsumexp: a legal tiny raw probability (1e-30) at a low temperature
    never underflows to log(0) (critic V04-K-PREP-02-R1-04)."""
    z = {o: math.log(v) / t for o, v in p.items()}
    m = max(z.values())
    return -(z[gold] - m - math.log(sum(math.exp(v - m) for v in z.values())))


def fit_t(recs: Sequence[Dict]) -> float:
    """The temperature minimising the mean NLL of `recs` on a fixed grid (deterministic)."""
    return min(T_GRID, key=lambda t: sum(nll(r["probabilities"], r["gold_answer"], t) for r in recs))


def fold_of(family: str) -> int:
    return int(hashlib.sha256(family.encode()).hexdigest(), 16) % 2


def ordinary_t(r: Dict, config: Dict) -> float:
    """The candidate config's temperature for the record by the product's own rule (devision.model.pick_temperature: the
    exact (type, option count) bucket name, e.g. choice:3-5 / choice:11+, else the type's) -- finite and > 0 or refused."""
    from laya.common import QTYPES

    from devision.model import pick_temperature
    t = float(pick_temperature(config["temperature"], config.get("temperature_by_options") or {}, QTYPES["choice"], len(r["probabilities"])))
    if not math.isfinite(t) or t <= 0:
        raise SystemExit("the config's temperature for %d options is %r" % (len(r["probabilities"]), t))
    return t


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


CODE = ("scripts/release/route_calib.py", "scripts/release/rebind.py", "scripts/release/rl_calib.py", "src/devision/reading.py",
        "src/devision/infer.py", "src/devision/model.py", "src/devision/train/evaluate.py")
DOMAIN_OF_KIND = {"temp": "temperature", "force": "magnets"}


def _sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _rb():
    import importlib.util
    import os
    spec = importlib.util.spec_from_file_location("calib_rebind", os.path.join(os.path.dirname(os.path.abspath(__file__)), "rebind.py"))
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def identity(candidate: str, bundle: str, rows_path: str, data_root: str, pins: Optional[Dict[str, str]] = None, root: str = ".") -> Dict:
    """What a calibration run is bound to (critic V04-K-PREP-02-R1-03): the verified source record, the candidate's actual
    inference files and its (ordinary-calibrated) config, the temporary bundle (bound to that config, outside the candidate),
    the source rows and every picture they show, and the code. Refuses a candidate that is not the source's (weights /
    non-whitelisted config) or a bundle not bound to the candidate's current config."""
    import os
    rb = _rb()
    rec = json.load(open(os.path.join(rb.lineage_dir(candidate), "source.json")))
    bad = rb.check_source(rec, pins) + rb.candidate_problems(candidate, rec)
    rj = json.load(open(os.path.join(bundle, "reading.json")))
    if rj["sha256"].get("config.json") != _sha(os.path.join(candidate, "config.json")) or rj["sha256"].get("model.safetensors") != _sha(os.path.join(candidate, "model.safetensors")):
        bad.append("the temporary bundle is not bound to the candidate's current config / weights")
    if "k_rule" not in rj:
        bad.append("the temporary bundle has no K rule")
    if bad:
        raise SystemExit("calibration identity refused:\n  " + "\n  ".join(bad))
    rows = [json.loads(x) for x in open(rows_path) if x.strip()]
    return {"source_json": _sha(os.path.join(rb.lineage_dir(candidate), "source.json")),
            "candidate": {f: rb.digest(os.path.join(candidate, f)) for f in rb.FILES},
            "temporary_bundle": _sha(os.path.join(bundle, "reading.json")), "rows": _sha(rows_path),
            "pictures": {r["image"]: _sha(os.path.join(data_root, r["image"])) for r in rows},
            "code": {f: _sha(os.path.join(root, f)) for f in CODE if os.path.exists(os.path.join(root, f))}, "temperature": "raw (1)"}


def register(out_dir: str, ident: Dict, recs: List[Dict], paths: Dict[str, str]) -> Dict:
    import os
    os.makedirs(out_dir, exist_ok=False)
    rp = os.path.join(out_dir, "records.jsonl")
    with open(rp, "w") as fh:
        fh.write("".join(json.dumps(r) + "\n" for r in recs))
    reg = {**ident, "records": _sha(rp), "paths": {k: os.path.abspath(v) for k, v in paths.items()}}
    json.dump(reg, open(os.path.join(out_dir, "CALIB.json"), "w"), indent=1)
    return reg


def completeness(recs: List[Dict], rows: List[Dict]) -> List[str]:
    """Every source question exactly once with its options, gold, family and domain; both domains; both views of every
    family (critic V04-K-PREP-02-R1-03). An unreadable read that led to a wrong answer is a normal record, kept."""
    want = {(r["id"], q): (list(x["criteria"]), max(r["gold"][q]["probabilities"], key=r["gold"][q]["probabilities"].get), r["family"],
                           DOMAIN_OF_KIND.get(r["kind"])) for r in rows for q, x in r["questions"].items()}
    out, seen = [], set()
    for x in recs:
        k = (x.get("sample_id"), x.get("qid"))
        if k in seen:
            out.append("%s/%s: duplicate record" % k)
            continue
        seen.add(k)
        w = want.get(k)
        if w is None:
            out.append("%s/%s: not a source question" % k)
            continue
        if list(x.get("probabilities") or {}) != w[0] or x.get("gold_answer") != w[1] or x.get("family") != w[2] or x.get("reading_domain") != w[3]:
            out.append("%s/%s: options / gold / family / domain differ from the source" % k)
    missing = set(want) - seen
    if missing:
        out.append("%d source questions missing, e.g. %s" % (len(missing), sorted(missing)[0]))
    views: Dict[str, set] = defaultdict(set)
    for r in rows:
        views[r["family"]].add(r["image_id"].rsplit(":", 1)[-1])
    out += ["family %s: views %s, not v0 + v1" % (f, sorted(v)) for f, v in views.items() if v != {"v0", "v1"}]
    doms = {w[3] for w in want.values()}
    if doms != {"temperature", "magnets"}:
        out.append("the source holds domains %s, not both" % sorted(d for d in doms if d))
    return out


def analyse_registered(out_dir: str, pins: Optional[Dict[str, str]] = None, root: str = ".") -> Dict:
    """The formal analysis: CALIB.json re-verified against the files now (identity recomputed and equal, records hash),
    the records complete against the source rows, then `analyse` with the candidate's own config."""
    import os
    reg = json.load(open(os.path.join(out_dir, "CALIB.json")))
    P = reg["paths"]
    now = identity(P["candidate"], P["bundle"], P["rows"], P["data_root"], pins, root)
    changed = sorted(k for k in now if now[k] != reg.get(k))
    if changed or _sha(os.path.join(out_dir, "records.jsonl")) != reg["records"]:
        raise SystemExit("calibration registration does not verify: %s" % (changed or ["records"]))
    recs = [json.loads(x) for x in open(os.path.join(out_dir, "records.jsonl")) if x.strip()]
    rows = [json.loads(x) for x in open(P["rows"]) if x.strip()]
    bad = completeness(recs, rows)
    if bad:
        raise SystemExit("calibration records incomplete / inconsistent:\n  " + "\n  ".join(bad[:10]))
    res = analyse(recs, json.load(open(os.path.join(P["candidate"], "config.json"))))
    res["registration"] = {"CALIB.json": _sha(os.path.join(out_dir, "CALIB.json")), "records": reg["records"]}
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    for k in ("candidate", "bundle", "rows", "out_dir"):
        c.add_argument("--" + k.replace("_", "-"), required=True)
    c.add_argument("--data-root", default="data")
    c.add_argument("--device", default="auto")
    n = sub.add_parser("analyse")
    for k in ("calib_dir", "out"):
        n.add_argument("--" + k.replace("_", "-"), required=True)
    a = ap.parse_args(argv)
    if a.cmd == "collect":
        from devision.reading import Reading
        ident = identity(a.candidate, a.bundle, a.rows, a.data_root)          # before anything is loaded
        reading = Reading.load(a.candidate, a.bundle, device=a.device, full_rules=True)
        rows = [json.loads(x) for x in open(a.rows) if x.strip()]
        recs = collect(reading, rows, a.data_root)
        register(a.out_dir, ident, recs, {"candidate": a.candidate, "bundle": a.bundle, "rows": a.rows, "data_root": a.data_root})
        print("%d records registered in %s" % (len(recs), a.out_dir))
        return 0
    res = analyse_registered(a.calib_dir)
    json.dump(res, open(a.out, "w"), indent=1)
    print(res["branch"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
