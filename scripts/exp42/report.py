"""exp42 readout (critic EXP42-PREP-01; rules written before the run). R = objective rlcd, C = objective ce, both from
exp40-M's frozen end; "start" = that end itself. Every set evaluated by devision-eval at temperature 1 (--no-temperature).

Identity (refused otherwise): exp40-M still its frozen identity; each arm's START.json names its objective, exp40-M's
weights / config and data/exp42/train.jsonl with the MANIFEST sha; its run.json trained that file from
runs/exp40-m/stage2 with that objective, and its train_report.json says the same objective; every arm file recorded
under the arm's frozen weights / config (scripts/v04b/freeze.py); every summary names the right checkpoint,
"none (raw)" temperature, no control and the set file's sha256; every record set holds each question exactly once
with the data's gold / picture, valid probabilities, prediction = argmax and correct = (prediction == gold)
(scripts/exp41/report.check_details). A run or set missing gives "incomplete", never a pass.

Readout per set -- fit_O / fit_L (the trained questions: fitting, not generalisation), rl_tune / rl_tune_L (the
600 tune pictures, original / L) -- and field (mass / speed / distance), on the reading questions:
  different: single-question accuracy where the two objects' golds differ (same picture and field);
  both: both objects right there;  twin: both views right where that object's gold changes between the views;
  same_answer: the two objects given the same answer where their golds differ.
Field-equal macros; differences C - R, R - start, C - start resampled over case-linked clusters (exp40's algorithm;
seed 0, 2,000 resamples). exist / science: accuracy against the start, risk only.

Pre-written conditions (next-stage investment, not a release gate):
  progress (per arm): rl_tune different macro >= 0.60; both macro >= 0.20 with (arm - start) lower bound > 0;
    twin macro >= 0.20 with (arm - start) lower bound > 0 -- never met through same-value questions;
  CE attribution: rl_tune both macro C - R >= 0.10 with lower bound > 0 (if both arms improve, the concentrated
    supervision itself is the first explanation);
  stop: both arms' fit_L different macro < 0.60 -- this start / settings / 576 updates did not learn it; stop this
    objective route (not evidence of missing features or of needing a larger model);
  fit-only or L-only gains: no long original-picture candidate training on this recipe.

    uv run python scripts/exp42/report.py
"""
import hashlib
import importlib.util
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List

HERE = os.path.dirname(__file__)
M_RUN, INIT = "runs/exp40-m", "runs/exp40-m/stage2"
INIT_SHA = {"model.safetensors": "3fb4232e27bff5fae767b8d62230884bae8dbaae0e92cca76de89be9e3b151dd",
            "config.json": "74e32e7cd4a7d08e9236cb95a2b6cd1a8c407056f07c7008a3f2dc5679b98a32"}
RUNS = {"start": "runs/exp42-start", "R": "runs/exp42-r", "C": "runs/exp42-c"}
OBJECTIVE = {"R": "rlcd", "C": "ce"}
SETS = {"fit_O": "data/exp42/fit_O.jsonl", "fit_L": "data/exp42/fit_L.jsonl",
        "rl_tune": "data/real_layout/questions_tune.jsonl", "rl_tune_L": "data/exp40/rl_tune_L.jsonl"}
RISK = {"exist": "data/r13b/panel_exist.jsonl", "science": "data/r13b/panel_science.jsonl"}
FIELDS = ("mass", "speed", "distance")
STATS = ("different", "both", "twin", "same_answer")


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, rel))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


r41 = _load("exp41_report", "../exp41/report.py")
frozen, clusters, check_details = r41.frozen, r41.clusters, r41.check_details
b40 = _load("exp40_build", "../exp40/build.py")


def sha(path: str) -> str:
    return r41.sha(path)


def load(path: str) -> List[Dict]:
    return [json.loads(x) for x in open(path) if x.strip()]


def states(path: str) -> Dict[str, str]:
    return dict(line.split() for line in open(path) if line.strip()) if os.path.exists(path) else {}


def arm_identity(arm: str, run: str, manifest: Dict) -> str:
    """'' when the arm started from exp40-M on data/exp42/train.jsonl with its own objective, else why not."""
    try:
        s = json.load(open(os.path.join(run, "START.json")))
        st = json.load(open(os.path.join(run, "run.json")))["config"]["stages"][0]
        tr = json.load(open(os.path.join(run, "stage2", "train_report.json")))
    except (OSError, ValueError, KeyError) as e:
        return "incomplete: %s" % e
    if s.get("objective") != OBJECTIVE[arm] or s.get("init") != INIT or s.get("init_sha256") != INIT_SHA \
            or s.get("data") != "data/exp42/train.jsonl" or s.get("data_sha256") != manifest["files"]["train"]:
        return "wrong arm: START %s" % s
    if st.get("data") != "data/exp42/train.jsonl" or st.get("init") != INIT or (st.get("params") or {}).get("objective") != OBJECTIVE[arm]:
        return "wrong arm: run.json %s %s %s" % (st.get("data"), st.get("init"), (st.get("params") or {}).get("objective"))
    if tr.get("objective") != OBJECTIVE[arm]:
        return "wrong arm: train_report objective %s" % tr.get("objective")
    return ""


def eval_records(run: str, name: str, data_path: str, checkpoint: str) -> List[Dict]:
    info = json.load(open(os.path.join(run, "eval", name + ".json")))["run"]
    if os.path.normpath(info.get("checkpoint", "")) != os.path.normpath(checkpoint) or info.get("temperature") != "none (raw)" \
            or info.get("control") != "none" or info.get("data_sha256") != sha(data_path):
        raise SystemExit("%s/%s: not %s at temperature 1, no control, on %s" % (run, name, checkpoint, data_path))
    rows = load(data_path)
    got = check_details("%s/%s" % (run, name), rows, load(os.path.join(run, "eval", name + ".details.jsonl")))
    out = []
    for r in rows:
        view = r.get("view") or r["id"].split(":")[2]
        picture = r.get("orig_id", r["id"])
        for q, v in r["questions"].items():
            f = b40.field_of(v["instructions"])
            if f == "cmp":
                continue
            g = got[(r["id"], q)]
            out.append({"family": r["family"], "case": r["case"], "field": f, "object": b40.object_of(v["instructions"]),
                        "view": view, "picture": picture, "gold": g["gold_answer"], "prediction": g["prediction"],
                        "right": bool(g["correct"])})
    return out


def units(recs: List[Dict]) -> Dict:
    fam: Dict = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))

    def add(f, key, v):
        fam[f][key][0] += v
        fam[f][key][1] += 1
    pairs, twins = defaultdict(list), defaultdict(list)
    for r in recs:
        pairs[(r["picture"], r["field"])].append(r)
        twins[(r["family"], r["field"], r["object"])].append(r)
    for (_, field), v in pairs.items():
        if len(v) != 2:
            raise SystemExit("not two objects on %s %s" % (v[0]["picture"], field))
        a, b = v
        if a["gold"] != b["gold"]:
            add(a["family"], "different:" + field, a["right"])
            add(a["family"], "different:" + field, b["right"])
            add(a["family"], "both:" + field, a["right"] and b["right"])
            add(a["family"], "same_answer:" + field, a["prediction"] == b["prediction"])
    for (family, field, _), v in twins.items():
        if len(v) == 2 and v[0]["view"] != v[1]["view"] and v[0]["gold"] != v[1]["gold"]:
            add(family, "twin:" + field, v[0]["right"] and v[1]["right"])
    return fam


def value(fam, families, stat) -> float:
    if ":" not in stat:          # field-equal macro over the fields present
        vals = [value(fam, families, stat + ":" + f) for f in FIELDS]
        vals = [x for x in vals if x == x]
        return sum(vals) / len(vals) if vals else float("nan")
    n = sum(fam[f][stat][0] for f in families if stat in fam[f])
    d = sum(fam[f][stat][1] for f in families if stat in fam[f])
    return n / d if d else float("nan")


def summary(fam) -> Dict:
    fs = sorted(fam)
    out: Dict = {s: value(fam, fs, s) for s in STATS}
    for s in STATS:
        for f in FIELDS:
            k = "%s:%s" % (s, f)
            d = sum(fam[x][k][1] for x in fs if k in fam[x])
            if d:
                out[k] = {"value": value(fam, fs, k), "right": sum(fam[x][k][0] for x in fs if k in fam[x]), "of": d}
    return out


def diff(fa, fb, cl, stat) -> Dict:
    keys = sorted(cl)
    expand = lambda ks: [f for k in ks for f in sorted(cl[k])]  # noqa: E731
    return r41.interval(lambda fs: value(fa, fs, stat) - value(fb, fs, stat), keys, expand)


def build(runs: Dict[str, str] = RUNS, manifest_path: str = "data/exp42/MANIFEST.json") -> Dict:
    manifest = json.load(open(manifest_path))
    m_frozen = frozen.check(M_RUN, ["rl_tune"])["weights"]
    if m_frozen != INIT_SHA:
        raise SystemExit("exp40-M's frozen identity is not 3fb4232e / 74e32e7c")
    out: Dict = {"identity": {"start": m_frozen}, "runs": {}, "readout": {}, "differences": {}, "risk": {}}
    ckpt = {"start": INIT}
    for arm in ("R", "C"):
        run = runs[arm]
        why = arm_identity(arm, run, manifest) if os.path.exists(run) else "incomplete: not started"
        if why.startswith("wrong"):
            raise SystemExit("%s: %s" % (arm, why))
        st = states(run + "-status.txt")
        if why or st.get("train") != "completed" or not os.path.exists(frozen.path(run)):
            out["runs"][arm] = why or "incomplete: training"
            continue
        out["identity"][arm] = json.load(open(frozen.path(run)))["weights"]
        ckpt[arm] = os.path.join(run, "stage2")
    recs: Dict = {}
    for who, run in runs.items():
        if who not in ckpt:
            continue
        st = states(run + "-status.txt")
        done = [n for n in list(SETS) + list(RISK) if st.get("eval:" + n) == "completed"]
        if who != "start":
            frozen.check(run, done)
        out["runs"][who] = {"completed": done, "missing": [n for n in list(SETS) + list(RISK) if n not in done]}
        for n in done:
            r = eval_records(run, n, SETS[n], ckpt[who]) if n in SETS else None
            if r is not None:
                recs[(who, n)] = r
            else:
                info = json.load(open(os.path.join(run, "eval", n + ".json")))["run"]
                if os.path.normpath(info.get("checkpoint", "")) != os.path.normpath(ckpt[who]) or info.get("data_sha256") != sha(RISK[n]):
                    raise SystemExit("%s/%s: wrong checkpoint or data" % (run, n))
                rows = load(RISK[n])
                got = check_details("%s/%s" % (run, n), rows, load(os.path.join(run, "eval", n + ".details.jsonl")))
                out["risk"].setdefault(n, {})[who] = sum(g["correct"] for g in got.values()) / len(got)
    fams = {k: units(v) for k, v in recs.items()}
    for (who, n), fam in fams.items():
        out["readout"].setdefault(n, {})[who] = summary(fam)
    for n in SETS:
        for a, b in (("C", "R"), ("R", "start"), ("C", "start")):
            if (a, n) in fams and (b, n) in fams:
                ca, cb = clusters(recs[(a, n)]), clusters(recs[(b, n)])
                if ca != cb:
                    raise SystemExit("%s: %s and %s were read on different questions" % (n, a, b))
                out["differences"]["%s|%s-%s" % (n, a, b)] = {s: diff(fams[(a, n)], fams[(b, n)], ca, s) for s in ("different", "both", "twin")}
    out["conditions"] = conditions(out)
    return out


def conditions(out: Dict) -> Dict:
    rd, d = out["readout"].get("rl_tune", {}), out["differences"]
    res: Dict = {}
    for arm in ("R", "C"):
        k = "rl_tune|%s-start" % arm
        if arm not in rd or k not in d:
            res["progress_" + arm] = "incomplete"
            continue
        x = rd[arm]
        res["progress_" + arm] = bool(x["different"] >= 0.60 and x["both"] >= 0.20 and d[k]["both"]["ci95"][0] > 0
                                      and x["twin"] >= 0.20 and d[k]["twin"]["ci95"][0] > 0)
    k = "rl_tune|C-R"
    res["ce_attribution"] = "incomplete" if k not in d else bool(d[k]["both"]["value"] >= 0.10 and d[k]["both"]["ci95"][0] > 0)
    fl = out["readout"].get("fit_L", {})
    res["stop_route"] = "incomplete" if not ("R" in fl and "C" in fl) else bool(fl["R"]["different"] < 0.60 and fl["C"]["different"] < 0.60)
    return res


def main() -> int:
    out = build()
    json.dump(out, open("runs/exp42-report.json", "w"), indent=1, default=str)
    print("identity %s" % out["identity"])
    print("runs %s" % out["runs"])
    for n, by in out["readout"].items():
        for who, s in by.items():
            print("%-9s %-5s different %.3f both %.3f twin %.3f same_answer %.3f | %s" % (
                n, who, s["different"], s["both"], s["twin"], s["same_answer"],
                {k: "%d/%d" % (v["right"], v["of"]) for k, v in s.items() if isinstance(v, dict)}))
    for k, v in out["differences"].items():
        print("%s %s" % (k, {s: "%+.3f [%+.3f, %+.3f]" % (x["value"], *x["ci95"]) for s, x in v.items()}))
    print("risk %s" % out["risk"])
    print("conditions %s" % out["conditions"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
