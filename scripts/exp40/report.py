"""exp40 endpoint report (critic EXP40-READOUT-01; rules written before the run, docs/experiments/2026-10-18-exp40-
pair-plan.md). Per arm (O = original picture twice, M = original + L), from devision-eval runs of the arm's final
checkpoint (fitted temperatures) written by scripts/exp40/chain.sh:

  - identity first: the arm's START.json (written by the chain before training) must name exp37-L's weights /
    config sha256 and this arm's own data file and sha256 (data/exp40/MANIFEST.json), and the stage's run.json must
    have trained on that file from runs/exp37-l/stage1 -- anything else is refused ("wrong arm");
  - candidate B's five-part screen, unchanged (scripts/v04b/report.build: transfer 68 : 114 >= +10 points with
    lower bound > 0 and picture-gain lower bound > 0, no pre-written protection cost on the 7 panels, dev16 read and
    compare pass with a valid erased control), bound to the arm's frozen weights / config and every required file;
    a missing step gives "incomplete", never a pass;
  - the main scientific comparison, only when both arms' rl_tune is complete: on the original-picture real-layout
    tune decision questions, M - O of "both objects right where their values differ" (equal-weight macro over mass /
    speed / distance), paired bootstrap over canonical-case clusters (families linked through a shared canonical case;
    seed 0, 2,000 resamples; a cluster's families, views and fields together); also per arm the reading accuracy per field and macro, both twin views right where that
    object's value changed, and the share of different-value pairs where the two objects got the same prediction.
    A one-sided gain or an interval across 0 is not a bridge;
  - report only: rl_tune_L (the same questions on exp37's L pictures: is the large-digit ability kept), exp31,
    exp34 semantics tune.

Candidate choice, pre-written: M if M's screen passes; else O if O's passes (then not a large-digit bridge); else
none -- no further training, no lowered gate. A pass only admits the arm to the 29-set full evaluation, the final
confirmation and release checks.

    uv run python scripts/exp40/report.py
"""
import hashlib
import importlib.util
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List

HERE = os.path.dirname(__file__)
INIT = "runs/exp37-l/stage1"
INIT_SHA = {"model.safetensors": "f9fc93c25b4bd96c55d41fd7c974c43408ea31034eac2c3a1ade2bcf5950da31",
            "config.json": "3a10bacd56e7f48aa07ad981821a592fff10c5eb317c39a9591ad425c175c324"}
RUNS = {"M": "runs/exp40-m", "O": "runs/exp40-o"}
TUNE = "data/real_layout/questions_tune.jsonl"
TUNE_L = "data/exp40/rl_tune_L.jsonl"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, rel))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


v04b = _load("v04b_report", "../v04b/report.py")
r36 = _load("exp36_report", "../exp36/report.py")
build_mod = _load("exp40_build", "build.py")


def sha(path: str):
    return hashlib.sha256(open(path, "rb").read()).hexdigest() if os.path.exists(path) else None


def identity(arm: str, run: str, manifest: Dict) -> str:
    """'' when the arm's run started from exp37-L and trained on its own data, else why not."""
    start = os.path.join(run, "START.json")
    if not os.path.exists(start):
        return "no START.json"
    s = json.load(open(start))
    data = "data/exp40/train_%s.jsonl" % arm
    if s.get("arm") != arm or s.get("data") != data or s.get("data_sha256") != manifest["arms"][arm]["sha256"]:
        return "wrong arm: START names %s %s %s" % (s.get("arm"), s.get("data"), s.get("data_sha256"))
    if s.get("init") != INIT or s.get("init_sha256") != INIT_SHA:
        return "wrong start: %s %s" % (s.get("init"), s.get("init_sha256"))
    rj = os.path.join(run, "run.json")
    if os.path.exists(rj):
        st = json.load(open(rj))["config"]["stages"][0]
        if st.get("data") != data or st.get("init") != INIT:
            return "wrong arm: run.json trained on %s from %s" % (st.get("data"), st.get("init"))
    return ""


def numbers(details: str, data_path: str) -> List[Dict]:
    """Per reading question: family, canonical case, field, object, gold text, prediction, right."""
    meta = {}
    for s in map(json.loads, open(data_path)):
        for qid, q in s["questions"].items():
            f = build_mod.field_of(q["instructions"])
            if f != "cmp":
                meta[(s["id"], qid)] = (s["family"], s["case"], f, build_mod.object_of(q["instructions"]))
    out = []
    for r in map(json.loads, open(details)):
        m = meta.get((r["sample_id"], r["qid"]))
        if m:
            out.append({"sample_id": r["sample_id"], "family": m[0], "case": m[1], "field": m[2], "object": m[3],
                        "text": r["gold_answer"], "prediction": r["prediction"], "right": bool(r["correct"]),
                        "mode": "real", "wrong_picture_kind": "none"})
    return out


def same_prediction(recs: List[Dict]) -> Dict:
    by = defaultdict(list)
    for r in recs:
        by[(r["sample_id"], r["field"])].append(r)
    pairs = [v for v in by.values() if len(v) == 2 and v[0]["text"] != v[1]["text"]]
    return {"pairs": len(pairs), "same_prediction": sum(v[0]["prediction"] == v[1]["prediction"] for v in pairs) / len(pairs) if pairs else None}


def clusters(recs: List[Dict]) -> Dict[str, set]:
    """Resampling units: families linked through a shared canonical case (a family's two views have two cases, and
    other families can share either), as connected components; key = the component's smallest case."""
    parent: Dict[str, str] = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for r in recs:
        a, b = find("f:" + r["family"]), find("c:" + r["case"])
        if a != b:
            parent[max(a, b)] = min(a, b)
    out: Dict[str, set] = defaultdict(set)
    for r in recs:
        out[find("f:" + r["family"])].add(r["family"])
    return dict(out)


def binding(recs: List[Dict]):
    fam = r36.units(recs)
    cases = clusters(recs)
    expand = lambda cs: [f for c in cs for f in sorted(cases[c])]  # noqa: E731
    stats = {s: r36.value(fam, sorted(fam), s) for s in ["whole", "both", "twin"] + ["%s:%s" % (k, f) for k in ("whole", "both", "twin") for f in r36.FIELDS]}
    stats["denominators"] = {s: sum(fam[f][s][1] for f in fam if s in fam[f]) for s in ["%s:%s" % (k, f) for k in ("whole", "both", "twin") for f in r36.FIELDS]}
    stats.update(same_prediction(recs))
    return fam, dict(cases), expand, stats


def compare(m_recs: List[Dict], o_recs: List[Dict]) -> Dict:
    fm, cm, expand, sm = binding(m_recs)
    fo, co, _, so = binding(o_recs)
    if cm != co:
        raise SystemExit("the two arms were evaluated on different questions")
    cases = sorted(cm)
    out = {"M": sm, "O": so, "n_clusters": len(cases), "n_families": sum(len(v) for v in cm.values())}
    out["M_minus_O"] = {s: r36.interval(lambda cs, s=s: r36.value(fm, expand(cs), s) - r36.value(fo, expand(cs), s), cases)
                        for s in ("both", "whole", "twin")}
    d = out["M_minus_O"]["both"]
    out["bridge"] = d["value"] > 0 and d["ci95"][0] > 0
    return out


def arm_screen(arm: str, run: str, manifest: Dict) -> Dict:
    why = identity(arm, run, manifest)
    if why:
        if why == "no START.json":
            return {"outcome": "incomplete: not started", "identity_check": why}
        raise SystemExit("%s: %s" % (arm, why))
    out = v04b.build(run, INIT_SHA["model.safetensors"], os.path.join(os.path.dirname(run), "%s-status.txt" % os.path.basename(run)))
    out["identity"]["start_config_sha256"] = INIT_SHA["config.json"]
    out["identity"]["data_sha256"] = manifest["arms"][arm]["sha256"]
    st = out["status"]
    ro = out.get("report_only", {})
    if st.get("eval:rl_tune_L") == "completed" and os.path.exists(v04b.frozen.path(run)):
        v04b.frozen.check(run, ["rl_tune_L"])
        ro["rl_tune_L"] = binding(numbers(os.path.join(run, "eval", "rl_tune_L.details.jsonl"), TUNE_L))[3]
    out["report_only"] = ro
    return out


def build(runs: Dict[str, str] = RUNS, manifest_path: str = "data/exp40/MANIFEST.json") -> Dict:
    manifest = json.load(open(manifest_path))
    out: Dict = {"arms": {arm: arm_screen(arm, run, manifest) for arm, run in runs.items()}}
    passed = {arm: a["outcome"] == "candidate screen passed" for arm, a in out["arms"].items()}
    complete = {arm: "checks" in a for arm, a in out["arms"].items()}
    if all(complete.values()):
        recs = {arm: numbers(os.path.join(runs[arm], "eval", "rl_tune.details.jsonl"), TUNE) for arm in runs}
        out["comparison"] = compare(recs["M"], recs["O"])
    else:
        out["comparison"] = "incomplete: %s" % ", ".join(a for a, c in complete.items() if not c)
    out["candidate"] = "M" if passed["M"] else "O (not a large-digit bridge)" if passed["O"] else "none"
    return out


def main(argv=None):
    out = build()
    for arm, a in out["arms"].items():      # each arm's screen where release/delivery.py can read it
        json.dump(a, open("runs/exp40-%s-screen.json" % arm.lower(), "w"), indent=1, default=str)
    json.dump(out, open("runs/exp40-report.json", "w"), indent=1, default=str)
    for arm, a in out["arms"].items():
        print("%s: %s" % (arm, a["outcome"]))
        if "checks" in a:
            t = a["transfer"]
            print("   transfer accuracy %+.3f %s gain %+.3f %s" % (t["accuracy_change"]["value"], t["accuracy_change"]["ci95"],
                                                                   t["gain_change"]["value"], t["gain_change"]["ci95"]))
            print("   protection %s costs %s" % ({k: round(100 * v["difference"], 2) for k, v in a["protection"]["panels"].items()},
                                                  a["protection"]["costs"] or "none"))
            print("   dev16 read %.3f (%s) cmp %.3f (%s); erased valid %s" % (
                a["dev16"]["read"]["value"], a["dev16"]["read_class"], a["dev16"]["cmp"]["value"], a["dev16"]["cmp_class"], a["erased"]["valid"]))
            print("   report only %s" % a["report_only"])
    c = out["comparison"]
    if isinstance(c, dict):
        for arm in ("M", "O"):
            s = c[arm]
            print("rl_tune %s: reading %.3f (mass %.3f speed %.3f distance %.3f) both %.3f twin %.3f same prediction %.3f of %d" % (
                arm, s["whole"], s["whole:mass"], s["whole:speed"], s["whole:distance"], s["both"], s["twin"], s["same_prediction"], s["pairs"]))
        for k, v in c["M_minus_O"].items():
            print("M - O %s %+.3f %s" % (k, v["value"], [round(x, 3) for x in v["ci95"]]))
        print("bridge (M - O both-objects lower bound > 0): %s" % c["bridge"])
    else:
        print("comparison: %s" % c)
    print("candidate: %s" % out["candidate"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
