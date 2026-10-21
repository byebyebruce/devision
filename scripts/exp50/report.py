"""exp50 readout (critic EXP50-PREP-04; rules written before the run). T = local views from the original picture, C = from
the reduced picture, both from EXP48-X's frozen end (runs/exp48-x, whose temperature-1 records on the four sets and
exist / science are reused after their frozen check); ordinary decide, original question and picture, temperature 1 --
fit_O / fit_L / rl_tune / rl_tune_L with exp42's statistics (different, both, twin, same answer; field-equal macros;
case-linked clusters, 2,000 resamples, seed 0) and EXP48's strict four-corner quads (scripts/exp48/report.quad_units;
quads that do not qualify are counted and left out). Differences T - X, C - X, T - C. exist / science: risk only
(their summaries checked as the main sets': checkpoint, temperature 1, no control, data sha). Report-only slice: the 15
rl_tune distance questions no single view holds whole under n12 / 0.25 (data/exp50/geometry.json "cut_slice"; critic
EXP50-R1-01) -- each model's accuracy on them; they stay in rl_tune and in every condition's denominator.
Identity checks as before (START / run.json / train_report / checkpoint config of each arm); anything missing
"incomplete". A baseline or arm on other questions is refused.

Conditions (pre-written; "worth a candidate training", not a release), on the original rl_tune, per arm A in {T, C},
"incomplete" unless that arm's and X's four sets are read:
  progress_A: different >= 0.60; both >= 0.20 with A - X lower bound > 0; twin >= 0.20 with A - X lower bound > 0;
    quads all right >= 0.10 with A - X lower bound > 0;
  attribution (local original detail): T - C both >= +0.10 with lower bound > 0 (needs both arms).
If C passes and T does not, C's result is what it is -- not a local-detail success.

    uv run python scripts/exp50/report.py
"""
import importlib.util
import json
import os
import sys
from typing import Dict, List

HERE = os.path.dirname(__file__)


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, rel))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


r48 = _load("exp48_report", "../exp48/report.py")
r42, r41, frozen = r48.r42, r48.r41, r48.frozen
SETS, RISK = r42.SETS, r42.RISK
X_RUN, X_CKPT = "runs/exp48-x", "runs/exp48-x/stage2"
X_SHA = {"model.safetensors": "a8b92cb93a56aa4653f516cbf2a4a38e88fbd64d69ac0f820acc6307f519b232",
         "config.json": "c017a469b35fcccf7e2fb2953dcd334be345bfd4fac493d64f4fd302155140df"}
RUNS = {"X": X_RUN, "T": "runs/exp50-t", "C": "runs/exp50-c"}
QUADS = "data/exp48/quads.jsonl"
VIEWS, OVERLAP = 12, 0.25
GEOMETRY = "data/exp50/geometry.json"


def arm_identity(arm: str, run: str, man: Dict, quads_sha: str) -> str:
    try:
        s = json.load(open(os.path.join(run, "START.json")))
        st = json.load(open(os.path.join(run, "run.json")))["config"]["stages"][0]
        tr = json.load(open(os.path.join(run, "stage2", "train_report.json")))
        cfg = json.load(open(os.path.join(run, "stage2", "config.json")))
    except (OSError, ValueError, KeyError) as e:
        return "incomplete: %s" % e
    par = st.get("params") or {}
    if s.get("init") != X_CKPT or s.get("init_sha256") != X_SHA or s.get("data_sha256") != man["files"]["train"] \
            or s.get("interaction_weight") != 1.0 or s.get("quads_sha256") != quads_sha or s.get("local_source") != arm:
        return "wrong arm: START %s" % s
    if st.get("data") != "data/exp42/train.jsonl" or st.get("init") != X_CKPT or par.get("objective") != "ce" \
            or par.get("interaction_file") != QUADS or par.get("interaction_weight") != 1.0 or par.get("region_file") \
            or par.get("vision_unfreeze_top") or (par.get("local_views"), par.get("local_overlap"), par.get("local_source")) != (VIEWS, OVERLAP, arm):
        return "wrong arm: run.json %s" % par
    it, lo = tr.get("interaction") or {}, tr.get("local") or {}
    if tr.get("objective") != "ce" or it.get("weight") != 1.0 or it.get("file_sha256") != quads_sha or "region" in tr \
            or (lo.get("views"), lo.get("overlap"), lo.get("source")) != (VIEWS, OVERLAP, arm) or lo.get("vision_trainable") != 0:
        return "wrong arm: train_report %s %s" % (it, lo)
    if (cfg.get("local_views"), cfg.get("local_overlap"), cfg.get("local_source")) != (VIEWS, OVERLAP, arm):
        return "wrong arm: checkpoint config %s" % {k: cfg.get(k) for k in ("local_views", "local_overlap", "local_source")}
    return ""


def build(runs: Dict[str, str] = RUNS) -> Dict:
    man = json.load(open("data/exp42/MANIFEST.json"))
    qsha = r41.sha(QUADS)
    if qsha != json.load(open("data/exp48/MANIFEST.json"))["sha256"]:
        raise SystemExit("data/exp48/quads.jsonl is not the MANIFEST's")
    if frozen.check(runs["X"], list(SETS) + list(RISK))["weights"] != X_SHA:
        raise SystemExit("EXP48-X is not its frozen identity")
    out: Dict = {"identity": {"X": X_SHA}, "runs": {}, "readout": {}, "differences": {}, "quads": {}, "risk": {}}
    ckpt = {"X": X_CKPT}
    done_by: Dict[str, List[str]] = {"X": list(SETS) + list(RISK)}
    for arm in ("T", "C"):
        run = runs[arm]
        why = arm_identity(arm, run, man, qsha) if os.path.exists(run) else "incomplete: not started"
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
                recs[(who, n)] = r48.records(runs[who], n, ckpt[who])
            else:
                info = json.load(open(os.path.join(runs[who], "eval", n + ".json")))["run"]
                if os.path.normpath(info.get("checkpoint", "")) != os.path.normpath(ckpt[who]) or info.get("temperature") != "none (raw)" \
                        or info.get("control") != "none" or info.get("data_sha256") != r41.sha(RISK[n]):
                    raise SystemExit("%s/%s: not %s at temperature 1, no control, on %s" % (runs[who], n, ckpt[who], RISK[n]))
                rows = r42.load(RISK[n])
                got = r41.check_details("%s/%s" % (runs[who], n), rows, r42.load(os.path.join(runs[who], "eval", n + ".details.jsonl")))
                out["risk"].setdefault(n, {})[who] = sum(g["correct"] for g in got.values()) / len(got)
    fams = {k: r42.units(v) for k, v in recs.items()}
    qfams = {k: r48.quad_units(k[1], v) for k, v in recs.items()}
    for (who, n), fam in fams.items():
        out["readout"].setdefault(n, {})[who] = r42.summary(fam)
        q = qfams[(who, n)]
        fs = sorted(q["units"])
        out["quads"].setdefault(n, {})[who] = {"macro": r48.quad_value(q["units"], fs), "quads": q["quads"], "excluded": q["excluded"],
                                               **{f: {"right": sum(q["units"][x]["quad:" + f][0] for x in fs if "quad:" + f in q["units"][x]),
                                                      "of": sum(q["units"][x]["quad:" + f][1] for x in fs if "quad:" + f in q["units"][x])}
                                                  for f in ("mass", "speed", "distance")}}
    for n in SETS:
        for a, b in (("T", "X"), ("C", "X"), ("T", "C")):
            if (a, n) in fams and (b, n) in fams:
                ca, cb = r42.clusters(recs[(a, n)]), r42.clusters(recs[(b, n)])
                if ca != cb:
                    raise SystemExit("%s: %s and %s on different questions" % (n, a, b))
                d = {s: r42.diff(fams[(a, n)], fams[(b, n)], ca, s) for s in ("different", "both", "twin")}
                keys = sorted(ca)
                expand = lambda ks, ca=ca: [f for k in ks for f in sorted(ca[k])]  # noqa: E731
                qa, qb = qfams[(a, n)]["units"], qfams[(b, n)]["units"]
                d["quad"] = r41.interval(lambda fs, qa=qa, qb=qb: r48.quad_value(qa, fs) - r48.quad_value(qb, fs), keys, expand)
                out["differences"]["%s|%s-%s" % (n, a, b)] = d
    out["cut_slice"] = cut_slice(recs)
    out["conditions"] = conditions(out)
    return out


def cut_slice(recs: Dict) -> Dict:
    """Per model, right / of on the rl_tune questions of the geometry's cut slice (report only)."""
    keys = {(x["id"], x["qid"]) for x in json.load(open(GEOMETRY))["cut_slice"] if x["set"] == "rl_tune"}
    out = {}
    for (who, n), rs in recs.items():
        if n == "rl_tune":
            got = [r["right"] for r in rs if tuple(r["key"]) in keys]
            if len(got) != len(keys):
                raise SystemExit("%s: %d of the %d cut-slice questions in rl_tune" % (who, len(got), len(keys)))
            out[who] = {"right": sum(got), "of": len(got)}
    return out


def conditions(out: Dict) -> Dict:
    rd, d, q = out["readout"], out["differences"], out["quads"]
    res: Dict = {}
    for arm in ("T", "C"):
        if not all(arm in rd.get(n, {}) and "X" in rd.get(n, {}) for n in SETS) or "rl_tune|%s-X" % arm not in d:
            res["progress_" + arm] = "incomplete"
            continue
        a, ax = rd["rl_tune"][arm], d["rl_tune|%s-X" % arm]
        res["progress_" + arm] = bool(a["different"] >= 0.60 and a["both"] >= 0.20 and ax["both"]["ci95"][0] > 0
                                      and a["twin"] >= 0.20 and ax["twin"]["ci95"][0] > 0
                                      and q["rl_tune"][arm]["macro"] >= 0.10 and ax["quad"]["ci95"][0] > 0)
    tc = d.get("rl_tune|T-C")
    res["attribution"] = "incomplete" if tc is None or "incomplete" in (res["progress_T"], res["progress_C"]) else \
        bool(tc["both"]["value"] >= 0.10 and tc["both"]["ci95"][0] > 0)
    return res


def main() -> int:
    out = build()
    json.dump(out, open("runs/exp50-report.json", "w"), indent=1, default=str)
    print("identity %s\nruns %s" % (out["identity"], out["runs"]))
    for n, by in out["readout"].items():
        for who, s in by.items():
            qd = out["quads"][n][who]
            print("%-9s %s different %.3f both %.3f twin %.3f same_answer %.3f quads %.3f %s (excluded %s)" % (
                n, who, s["different"], s["both"], s["twin"], s["same_answer"], qd["macro"],
                {f: "%d/%d" % (qd[f]["right"], qd[f]["of"]) for f in ("mass", "speed", "distance")}, qd["excluded"]))
    for k, v in out["differences"].items():
        print("%s %s" % (k, {s: "%+.3f [%+.3f, %+.3f]" % (x["value"], *x["ci95"]) for s, x in v.items()}))
    print("cut slice (report only, in rl_tune) %s" % out["cut_slice"])
    print("risk %s\nconditions %s" % (out["risk"], out["conditions"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
