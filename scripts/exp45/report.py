"""exp45 readout (critic EXP45-PREP-02; rules written before the run). S = exp42-C's recipe with the zero-initialised
question conditioning module (region weight 0), G = the same with the region KL at 0.5; both from exp40-M's frozen
end; M (runs/exp42-start) and C (runs/exp42-c) as references from their frozen records. Main evidence: decide,
original question and picture, temperature 1 -- fit_O / fit_L / rl_tune / rl_tune_L with exp42's statistics, case
clusters; S - M, G - M, G - S (and S - C, G - C for reference). exist / science: risk only.

Identity (refused otherwise): exp40-M / exp42-C frozen; each arm's START.json / run.json / train_report.json name
objective ce, question conditioning, its region weight (S 0.0, G 0.5) and data/exp45/regions.jsonl with the MANIFEST
sha, exp40-M's weights and data/exp42/train.jsonl's sha; every arm file recorded under the arm's frozen identity; every
summary the right checkpoint, "none (raw)", no control and the set's sha; every record set complete and consistent.
Anything missing: "incomplete".

Conditions (pre-written; investment, not release), both "incomplete" unless both arms' four main sets are read:
  progress_<arm>: rl_tune different >= 0.60; both >= 0.20 with (arm - M) lower bound > 0; twin >= 0.20 with
    (arm - M) lower bound > 0;
  region_attribution: rl_tune both G - S >= 0.10 with lower bound > 0 (if S also progresses and G adds nothing
    reliable, the gain is not the region loss's).
Attention (mechanism only; scripts/exp45/attention.py on rl_tune and fit_O): mass on the asked box vs the other
object's same-field box vs the box's uniform share, the most attended patch in the box, and whether the A / B (Pair
1 / 2) questions of a picture attend to different top patches -- by field and layout; never a gate.

    uv run python scripts/exp45/report.py
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
r41, frozen = r42.r41, r42.frozen
SETS, RISK, INIT, INIT_SHA = r42.SETS, r42.RISK, r42.INIT, r42.INIT_SHA
RUNS = {"M": "runs/exp42-start", "C": "runs/exp42-c", "S": "runs/exp45-s", "G": "runs/exp45-g"}
WEIGHT = {"S": 0.0, "G": 0.5}
C_SHA = {"model.safetensors": "6596907825f0b9c1ba9c0e9eb8dc136a91d6c2a48b331bc50ba34ea60ad36ca4",
         "config.json": "e5421f88ff74a665d4730725e56bb49246eed419002ca064d7f308baab0f89f9"}
ATTN = {"attn_rl_tune": ("data/real_layout/questions_tune.jsonl", "rl_tune"), "attn_fit_O": ("data/exp42/fit_O.jsonl", "fit_O")}
REGIONS = "data/exp45/regions.jsonl"


def arm_identity(arm: str, run: str, man: Dict, regions_sha: str) -> str:
    try:
        s = json.load(open(os.path.join(run, "START.json")))
        st = json.load(open(os.path.join(run, "run.json")))["config"]["stages"][0]
        tr = json.load(open(os.path.join(run, "stage2", "train_report.json")))
    except (OSError, ValueError, KeyError) as e:
        return "incomplete: %s" % e
    par = st.get("params") or {}
    if s.get("init") != INIT or s.get("init_sha256") != INIT_SHA or s.get("data_sha256") != man["files"]["train"] \
            or s.get("region_weight") != WEIGHT[arm] or s.get("regions_sha256") != regions_sha:
        return "wrong arm: START %s" % s
    if st.get("data") != "data/exp42/train.jsonl" or st.get("init") != INIT or par.get("objective") != "ce" \
            or par.get("question_conditioning") is not True or par.get("region_file") != REGIONS or par.get("region_weight") != WEIGHT[arm]:
        return "wrong arm: run.json %s" % par
    reg = tr.get("region") or {}
    if tr.get("objective") != "ce" or reg.get("weight") != WEIGHT[arm] or reg.get("file_sha256") != regions_sha or reg.get("smooth") != 0.1:
        return "wrong arm: train_report %s" % reg
    return ""


def attention_summary(run: str, name: str, ckpt: str, regions_readout_sha: str) -> Dict:
    data, _ = ATTN[name]
    info = json.load(open(os.path.join(run, "eval", name + ".json")))["run"]
    if os.path.normpath(info.get("checkpoint", "")) != os.path.normpath(ckpt) or info.get("data_sha256") != r41.sha(data) \
            or info.get("regions_sha256") != regions_readout_sha:
        raise SystemExit("%s/%s: wrong checkpoint, data or regions" % (run, name))
    recs = r42.load(os.path.join(run, "eval", name + ".details.jsonl"))
    keys = [(r["sample_id"], r["qid"]) for r in recs]
    if len(set(keys)) != len(keys):
        raise SystemExit("%s/%s: repeated questions" % (run, name))
    out: Dict = {}
    groups = defaultdict(list)
    for r in recs:
        for g in ("all", r["field"], "%s/%s" % (r["field"], r["layout"]), "side_" + r["side"]):
            groups[g].append(r)
    for g, rs in sorted(groups.items()):
        n = len(rs)
        out[g] = {"n": n, "mass_target": sum(r["mass_target"] for r in rs) / n, "mass_other": sum(r["mass_other"] for r in rs) / n,
                  "uniform_share": sum(r["uniform_share"] for r in rs) / n, "top_in_target": sum(r["top_in_target"] for r in rs) / n}
    by_pic = defaultdict(list)
    for r in recs:
        by_pic[(r["picture"], r["field"])].append(r["top_patch"])
    pairs = [v for v in by_pic.values() if len(v) == 2]
    out["ab_top_patch_differs"] = {"pairs": len(pairs), "share": sum(v[0] != v[1] for v in pairs) / len(pairs) if pairs else None}
    return out


def build(runs: Dict[str, str] = RUNS) -> Dict:
    man = json.load(open("data/exp42/MANIFEST.json"))
    rman = json.load(open("data/exp45/MANIFEST.json"))
    regions_sha, readout_sha = rman["regions"]["sha256"], rman["regions_readout"]["sha256"]
    if r41.sha(REGIONS) != regions_sha:
        raise SystemExit("data/exp45/regions.jsonl is not the MANIFEST's")
    if frozen.check("runs/exp40-m", ["rl_tune"])["weights"] != INIT_SHA or frozen.check(runs["C"], list(SETS))["weights"] != C_SHA:
        raise SystemExit("exp40-M or exp42-C is not its frozen identity")
    out: Dict = {"identity": {"M": INIT_SHA, "C": C_SHA}, "runs": {}, "readout": {}, "differences": {}, "attention": {}, "risk": {}}
    ckpt = {"M": INIT, "C": os.path.join(runs["C"], "stage2")}
    done_by: Dict[str, List[str]] = {"M": list(SETS) + list(RISK), "C": list(SETS) + list(RISK)}
    for arm in ("S", "G"):
        run = runs[arm]
        why = arm_identity(arm, run, man, regions_sha) if os.path.exists(run) else "incomplete: not started"
        if why.startswith("wrong"):
            raise SystemExit("%s: %s" % (arm, why))
        st = r42.states(run + "-status.txt")
        if why or st.get("train") != "completed" or not os.path.exists(frozen.path(run)):
            out["runs"][arm] = why or "incomplete: training"
            continue
        out["identity"][arm] = json.load(open(frozen.path(run)))["weights"]
        ckpt[arm] = os.path.join(run, "stage2")
        names = list(SETS) + list(RISK) + list(ATTN)
        done = [n for n in names if st.get("eval:" + n) == "completed"]
        frozen.check(run, done)
        done_by[arm] = done
        out["runs"][arm] = {"completed": done, "missing": [n for n in names if n not in done]}
    recs: Dict = {}
    for who, done in done_by.items():
        run = runs[who]
        for n in done:
            if n in SETS:
                recs[(who, n)] = r42.eval_records(run, n, SETS[n], ckpt[who])
            elif n in RISK:
                rows = r42.load(RISK[n])
                got = r41.check_details("%s/%s" % (run, n), rows, r42.load(os.path.join(run, "eval", n + ".details.jsonl")))
                out["risk"].setdefault(n, {})[who] = sum(g["correct"] for g in got.values()) / len(got)
            elif n in ATTN:
                out["attention"].setdefault(n, {})[who] = attention_summary(run, n, ckpt[who], readout_sha)
    fams = {k: r42.units(v) for k, v in recs.items()}
    for (who, n), fam in fams.items():
        out["readout"].setdefault(n, {})[who] = r42.summary(fam)
    for n in SETS:
        for a, b in (("S", "M"), ("G", "M"), ("G", "S"), ("S", "C"), ("G", "C")):
            if (a, n) in fams and (b, n) in fams:
                ca, cb = r42.clusters(recs[(a, n)]), r42.clusters(recs[(b, n)])
                if ca != cb:
                    raise SystemExit("%s: %s and %s on different questions" % (n, a, b))
                out["differences"]["%s|%s-%s" % (n, a, b)] = {s: r42.diff(fams[(a, n)], fams[(b, n)], ca, s) for s in ("different", "both", "twin")}
    out["conditions"] = conditions(out)
    return out


def conditions(out: Dict) -> Dict:
    d, rd = out["differences"], out["readout"]
    complete = all(arm in rd.get(n, {}) for n in SETS for arm in ("S", "G")) and \
        all("%s|%s" % (n, k) in d for n in SETS for k in ("S-M", "G-M", "G-S"))
    if not complete:
        return {"progress_S": "incomplete", "progress_G": "incomplete", "region_attribution": "incomplete"}
    res: Dict = {}
    for arm in ("S", "G"):
        x, k = rd["rl_tune"][arm], "rl_tune|%s-M" % arm
        res["progress_" + arm] = bool(x["different"] >= 0.60 and x["both"] >= 0.20 and d[k]["both"]["ci95"][0] > 0
                                      and x["twin"] >= 0.20 and d[k]["twin"]["ci95"][0] > 0)
    gs = d["rl_tune|G-S"]["both"]
    res["region_attribution"] = bool(gs["value"] >= 0.10 and gs["ci95"][0] > 0)
    return res


def main() -> int:
    out = build()
    json.dump(out, open("runs/exp45-report.json", "w"), indent=1, default=str)
    print("identity %s\nruns %s" % (out["identity"], out["runs"]))
    for n, by in out["readout"].items():
        for who, s in by.items():
            print("%-9s %s different %.3f both %.3f twin %.3f same_answer %.3f | %s" % (
                n, who, s["different"], s["both"], s["twin"], s["same_answer"],
                {k: "%d/%d" % (v["right"], v["of"]) for k, v in s.items() if isinstance(v, dict) and not k.startswith("same")}))
    for k, v in out["differences"].items():
        print("%s %s" % (k, {s: "%+.3f [%+.3f, %+.3f]" % (x["value"], *x["ci95"]) for s, x in v.items()}))
    for n, by in out["attention"].items():
        for who, s in by.items():
            print("%s %s: %s; A/B top patch differs %s" % (n, who, {g: "target %.3f other %.3f uniform %.3f top-in %.3f" % (
                v["mass_target"], v["mass_other"], v["uniform_share"], v["top_in_target"]) for g, v in s.items() if g != "ab_top_patch_differs"},
                s["ab_top_patch_differs"]))
    print("risk %s\nconditions %s" % (out["risk"], out["conditions"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
