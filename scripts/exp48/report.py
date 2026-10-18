"""exp48 readout (critic EXP48-PREP-04; rules written before the run). C = gold CE, X = CE + the interaction loss, both from
EXP45-G's frozen end (runs/exp45-g, whose temperature-1 records on the four sets are reused after their frozen check);
decide, original question and picture, temperature 1 -- fit_O / fit_L / rl_tune / rl_tune_L with exp42's statistics
(different, both, twin, same answer; field-equal macros; case-linked clusters, 2,000 resamples, seed 0) and the strict
four-corner quads all right (the readout set's quads by scripts/exp48/build.quads_of: both objects in both views,
golds exchanged; quads that do not qualify are counted and left out), by field and layout. Differences X - C, X - G,
C - G. exist / science: risk only. Identity checks as before; anything missing "incomplete".

Conditions (pre-written; only to discuss a larger candidate training, not a release), all on the original rl_tune and
"incomplete" unless both arms' four sets are read:
  progress_X: X different >= 0.60; both >= 0.20 with X - G lower bound > 0; twin >= 0.20 with X - G lower bound > 0;
    quads all right >= 0.10 with X - C lower bound > 0;
  attribution: X - C both >= 0.10 with lower bound > 0;
  stop_flag: both arms' fit_L different < 0.60 (no more steps or weight on this setting).

    uv run python scripts/exp48/report.py
"""
import importlib.util
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List

HERE = os.path.dirname(__file__)


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, rel))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


r42 = _load("exp42_report", "../exp42/report.py")
b48 = _load("exp48_build", "build.py")
r41, frozen = r42.r41, r42.frozen
SETS, RISK = r42.SETS, r42.RISK
G_RUN, G_CKPT = "runs/exp45-g", "runs/exp45-g/stage2"
G_SHA = {"model.safetensors": "c34b43ddde846af40e154aefb8ca1a381c27325e95b03aca55d42b75d4e7c15e",
         "config.json": "2a5dfa64d33846052bf8c13862bd0def53888fce7160a18c25d38279ce89ab08"}
RUNS = {"G": G_RUN, "C": "runs/exp48-c", "X": "runs/exp48-x"}
WEIGHT = {"C": 0.0, "X": 1.0}
QUADS = "data/exp48/quads.jsonl"


def arm_identity(arm: str, run: str, man: Dict, quads_sha: str) -> str:
    try:
        s = json.load(open(os.path.join(run, "START.json")))
        st = json.load(open(os.path.join(run, "run.json")))["config"]["stages"][0]
        tr = json.load(open(os.path.join(run, "stage2", "train_report.json")))
    except (OSError, ValueError, KeyError) as e:
        return "incomplete: %s" % e
    par = st.get("params") or {}
    if s.get("init") != G_CKPT or s.get("init_sha256") != G_SHA or s.get("data_sha256") != man["files"]["train"] \
            or s.get("interaction_weight") != WEIGHT[arm] or s.get("quads_sha256") != quads_sha:
        return "wrong arm: START %s" % s
    if st.get("data") != "data/exp42/train.jsonl" or st.get("init") != G_CKPT or par.get("objective") != "ce" \
            or par.get("interaction_file") != QUADS or par.get("interaction_weight") != WEIGHT[arm] or par.get("region_file"):
        return "wrong arm: run.json %s" % par
    it = tr.get("interaction") or {}
    if tr.get("objective") != "ce" or it.get("weight") != WEIGHT[arm] or it.get("file_sha256") != quads_sha or "region" in tr:
        return "wrong arm: train_report %s" % it
    return ""


def quad_units(name: str, recs: List[Dict]) -> Dict:
    """family -> {"quad:<field>", "quad:<field>:<layout>"} [all four right, quads] for the set's strict quads."""
    rows = r42.load(SETS[name])
    res = b48.quads_of(rows, lambda r: (r["family"],), strict=False)
    right = {}
    for r in recs:
        right[(r["key"])] = r["right"]
    fam: Dict = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
    for (family, field), qd in res["quads"].items():
        ok = all(right[(rid, q)] for rid, q in qd["corners"].values())
        for k in ("quad:" + field,):
            fam[family][k][0] += ok
            fam[family][k][1] += 1
    return {"units": fam, "excluded": res["excluded"], "quads": len(res["quads"])}


def records(run: str, name: str, ckpt: str) -> List[Dict]:
    out = r42.eval_records(run, name, SETS[name], ckpt)
    rows = {r["id"]: r for r in r42.load(SETS[name])}
    for r, x in zip(out, [(rid, q) for rid, row in rows.items() for q, v in row["questions"].items()
                         if r42.b40.field_of(v["instructions"]) != "cmp"]):
        r["key"] = x
    return out


def quad_value(fam, families, field=None) -> float:
    keys = ["quad:" + field] if field else ["quad:mass", "quad:speed", "quad:distance"]
    vals = []
    for k in keys:
        n = sum(fam[f][k][0] for f in families if k in fam[f])
        d = sum(fam[f][k][1] for f in families if k in fam[f])
        if d:
            vals.append(n / d)
    return sum(vals) / len(vals) if vals else float("nan")


def build(runs: Dict[str, str] = RUNS) -> Dict:
    man = json.load(open("data/exp42/MANIFEST.json"))
    qman = json.load(open("data/exp48/MANIFEST.json"))
    if r41.sha(QUADS) != qman["sha256"]:
        raise SystemExit("data/exp48/quads.jsonl is not the MANIFEST's")
    if frozen.check(runs["G"], list(SETS) + list(RISK))["weights"] != G_SHA:
        raise SystemExit("EXP45-G is not its frozen identity")
    out: Dict = {"identity": {"G": G_SHA}, "runs": {}, "readout": {}, "differences": {}, "quads": {}, "risk": {}}
    ckpt = {"G": G_CKPT}
    done_by: Dict[str, List[str]] = {"G": list(SETS) + list(RISK)}
    for arm in ("C", "X"):
        run = runs[arm]
        why = arm_identity(arm, run, man, qman["sha256"]) if os.path.exists(run) else "incomplete: not started"
        if why.startswith("wrong"):
            raise SystemExit("%s: %s" % (arm, why))
        st = r42.states(run + "-status.txt")
        if why or st.get("train") != "completed" or not os.path.exists(frozen.path(run)):
            out["runs"][arm] = why or "incomplete: training"
            continue
        out["identity"][arm] = json.load(open(frozen.path(run)))["weights"]
        ckpt[arm] = os.path.join(run, "stage2")
        done = [n for n in list(SETS) + list(RISK) if st.get("eval:" + n) == "completed"]
        frozen.check(run, done)
        done_by[arm] = done
        out["runs"][arm] = {"completed": done, "missing": [n for n in list(SETS) + list(RISK) if n not in done]}
    recs: Dict = {}
    for who, done in done_by.items():
        for n in done:
            if n in SETS:
                recs[(who, n)] = records(runs[who], n, ckpt[who])
            else:
                rows = r42.load(RISK[n])
                got = r41.check_details("%s/%s" % (runs[who], n), rows, r42.load(os.path.join(runs[who], "eval", n + ".details.jsonl")))
                out["risk"].setdefault(n, {})[who] = sum(g["correct"] for g in got.values()) / len(got)
    fams = {k: r42.units(v) for k, v in recs.items()}
    qfams = {k: quad_units(k[1], v) for k, v in recs.items()}
    for (who, n), fam in fams.items():
        out["readout"].setdefault(n, {})[who] = r42.summary(fam)
        q = qfams[(who, n)]
        fs = sorted(q["units"])
        out["quads"].setdefault(n, {})[who] = {"macro": quad_value(q["units"], fs), "quads": q["quads"], "excluded": q["excluded"],
                                               **{f: {"right": sum(q["units"][x]["quad:" + f][0] for x in fs if "quad:" + f in q["units"][x]),
                                                      "of": sum(q["units"][x]["quad:" + f][1] for x in fs if "quad:" + f in q["units"][x])}
                                                  for f in ("mass", "speed", "distance")}}
    for n in SETS:
        for a, b in (("X", "C"), ("X", "G"), ("C", "G")):
            if (a, n) in fams and (b, n) in fams:
                ca, cb = r42.clusters(recs[(a, n)]), r42.clusters(recs[(b, n)])
                if ca != cb:
                    raise SystemExit("%s: %s and %s on different questions" % (n, a, b))
                d = {s: r42.diff(fams[(a, n)], fams[(b, n)], ca, s) for s in ("different", "both", "twin")}
                keys = sorted(ca)
                expand = lambda ks, ca=ca: [f for k in ks for f in sorted(ca[k])]  # noqa: E731
                qa, qb = qfams[(a, n)]["units"], qfams[(b, n)]["units"]
                d["quad"] = r41.interval(lambda fs, qa=qa, qb=qb: quad_value(qa, fs) - quad_value(qb, fs), keys, expand)
                out["differences"]["%s|%s-%s" % (n, a, b)] = d
    out["conditions"] = conditions(out)
    return out


def conditions(out: Dict) -> Dict:
    rd, d, q = out["readout"], out["differences"], out["quads"]
    complete = all(arm in rd.get(n, {}) for n in SETS for arm in ("C", "X")) and \
        all("%s|%s" % (n, k) in d for n in SETS for k in ("X-C", "X-G"))
    if not complete:
        return {"progress_X": "incomplete", "attribution": "incomplete", "stop_flag": "incomplete"}
    x, xg, xc = rd["rl_tune"]["X"], d["rl_tune|X-G"], d["rl_tune|X-C"]
    return {"progress_X": bool(x["different"] >= 0.60 and x["both"] >= 0.20 and xg["both"]["ci95"][0] > 0
                               and x["twin"] >= 0.20 and xg["twin"]["ci95"][0] > 0
                               and q["rl_tune"]["X"]["macro"] >= 0.10 and xc["quad"]["ci95"][0] > 0),
            "attribution": bool(xc["both"]["value"] >= 0.10 and xc["both"]["ci95"][0] > 0),
            "stop_flag": bool(rd["fit_L"]["C"]["different"] < 0.60 and rd["fit_L"]["X"]["different"] < 0.60)}


def main() -> int:
    out = build()
    json.dump(out, open("runs/exp48-report.json", "w"), indent=1, default=str)
    print("identity %s\nruns %s" % (out["identity"], out["runs"]))
    for n, by in out["readout"].items():
        for who, s in by.items():
            qd = out["quads"][n][who]
            print("%-9s %s different %.3f both %.3f twin %.3f same_answer %.3f quads %.3f %s (excluded %s)" % (
                n, who, s["different"], s["both"], s["twin"], s["same_answer"], qd["macro"],
                {f: "%d/%d" % (qd[f]["right"], qd[f]["of"]) for f in ("mass", "speed", "distance")}, qd["excluded"]))
    for k, v in out["differences"].items():
        print("%s %s" % (k, {s: "%+.3f [%+.3f, %+.3f]" % (x["value"], *x["ci95"]) for s, x in v.items()}))
    print("risk %s\nconditions %s" % (out["risk"], out["conditions"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
