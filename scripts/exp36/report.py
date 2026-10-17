"""exp36 readout (critic EXP36-READOUT-01; docs/experiments/2026-10-17-exp36-number-mask-plan.md; rules written
before running). Arm A = original random mask, arm B = full number mask on the real-layout captions; both 1,000 steps
from exp32's A2 end. From the per-number records of scripts/exp36/readout.py (600 tune pictures; the 200-picture
fit slice reported apart, never mixed in):

  - per arm, real picture: whole-number accuracy per field and their equal-weight macro; both objects right where
    their values differ (per picture and field) and macro; both twin views right where that field / object's value
    changed; picture dependence on the filtered numbers whose wrong picture is the item's own twin AND whose value
    actually changed between the views: whole-number accuracy real - wrong, interval by family;
  - B - A, paired by family (seed 0, 2,000 resamples, a family's two views and all its fields together): the
    whole-number macro and the both-objects macro;
  - continuation conditions (worth more training on the full-number route; not a delivery pass): B - A
    whole-number macro >= +0.10 with lower bound > 0, B - A both-objects macro >= +0.10 with lower bound > 0, B's
    mass and speed whole-number accuracy each >= 0.20, and B's filtered twin real - wrong lower bound > 0;
  - protection, apart from the number endpoint: the 7 panels of each arm against v0.3 (a cost when <= -3 points
    with the upper bound < 0; exp30's readout).

    uv run python scripts/exp36/report.py
"""
import importlib.util
import json
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List, Sequence

FIELDS = ("mass", "speed", "distance")
ARMS = {"A": "runs/exp36-a", "B": "runs/exp36-b"}

_spec = importlib.util.spec_from_file_location("exp30_report", os.path.join(os.path.dirname(__file__), "..", "exp30", "report.py"))
assert _spec is not None and _spec.loader is not None
_exp30 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_exp30)


def load(path: str) -> List[Dict]:
    return [json.loads(x) for x in open(path) if x.strip()]


def units(recs: Sequence[Dict]) -> Dict[str, Dict[str, List[float]]]:
    """family -> statistic -> [right, of] (summable), from one arm's records."""
    fam: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
    by_pic = defaultdict(list)
    by_obj = defaultdict(dict)
    real = {}
    for r in recs:
        key = (r["sample_id"], r["field"], r["object"])
        if r["mode"] == "real":
            f = fam[r["family"]]["whole:" + r["field"]]
            f[0] += r["right"]
            f[1] += 1
            by_pic[(r["sample_id"], r["field"])].append(r)
            by_obj[(r["family"], r["field"], r["object"])][r["sample_id"]] = r
            real[key] = r
    for (sid, field), rs in by_pic.items():
        if len(rs) == 2 and rs[0]["text"] != rs[1]["text"]:
            f = fam[rs[0]["family"]]["both:" + field]
            f[0] += rs[0]["right"] and rs[1]["right"]
            f[1] += 1
    changed = set()
    for (family, field, obj), views in by_obj.items():
        if len(views) == 2:
            a, b = views.values()
            if a["text"] != b["text"]:
                f = fam[family]["twin:" + field]
                f[0] += a["right"] and b["right"]
                f[1] += 1
                changed |= {(a["sample_id"], field, obj), (b["sample_id"], field, obj)}
    for r in recs:   # picture dependence: wrong picture is the own twin and this value changed between the views
        key = (r["sample_id"], r["field"], r["object"])
        if r["mode"] == "wrong" and r["wrong_picture_kind"].startswith("twin") and key in changed:
            f = fam[r["family"]]["dep"]
            f[0] += float(real[key]["right"]) - float(r["right"])
            f[1] += 1
    return fam


def value(fam: Dict, families: Sequence[str], stat: str) -> float:
    if stat in ("whole", "both"):
        vals = [value(fam, families, stat + ":" + f) for f in FIELDS]
        vals = [v for v in vals if v == v]
        return sum(vals) / len(vals) if vals else float("nan")
    num = sum(fam[f][stat][0] for f in families if stat in fam[f])
    den = sum(fam[f][stat][1] for f in families if stat in fam[f])
    return num / den if den else float("nan")


def interval(fn, families: Sequence[str], resamples: int = 2000, seed: int = 0) -> Dict:
    rng = random.Random(seed)
    boots = sorted(fn([rng.choice(families) for _ in families]) for _ in range(resamples))
    boots = [b for b in boots if b == b]
    return {"value": fn(list(families)), "ci95": [boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]]}


def arm_stats(fam: Dict) -> Dict:
    families = sorted(fam)
    out: Dict = {s: value(fam, families, s) for s in ["whole", "both"] + ["%s:%s" % (k, f) for k in ("whole", "both", "twin") for f in FIELDS]}
    out["dep"] = interval(lambda fs: value(fam, fs, "dep"), families)
    out["dep"]["n"] = sum(fam[f]["dep"][1] for f in families if "dep" in fam[f])
    out["denominators"] = {s: sum(fam[f][s][1] for f in families if s in fam[f])
                           for s in ["%s:%s" % (k, f) for k in ("both", "twin") for f in FIELDS]}
    return out


def build(arms: Dict[str, str] = ARMS) -> Dict:
    out: Dict = {"tune": {}, "fit": {}, "protection": {}}
    fams = {}
    for split in ("tune", "fit"):
        for arm, run in arms.items():
            fam = units(load(os.path.join(run, "numbers_" + split, "numbers.jsonl")))
            out[split][arm] = arm_stats(fam)
            if split == "tune":
                fams[arm] = fam
    a, b = fams["A"], fams["B"]
    families = sorted(set(a) & set(b))
    if set(a) != set(b):
        raise SystemExit("the two arms were read on different families")
    out["B_minus_A"] = {s: interval(lambda fs, s=s: value(b, fs, s) - value(a, fs, s), families) for s in ("whole", "both")}
    B = out["tune"]["B"]
    out["conditions"] = {
        "whole_macro": out["B_minus_A"]["whole"]["value"] >= 0.10 and out["B_minus_A"]["whole"]["ci95"][0] > 0,
        "both_macro": out["B_minus_A"]["both"]["value"] >= 0.10 and out["B_minus_A"]["both"]["ci95"][0] > 0,
        "mass_speed": B["whole:mass"] >= 0.20 and B["whole:speed"] >= 0.20,
        "picture_dependence": B["dep"]["ci95"][0] > 0}
    out["continue_full_number_route"] = all(out["conditions"].values())
    for arm, run in arms.items():
        ev = os.path.join(run, "eval")
        rows = {}
        for p in _exp30.GUARD:
            m = _exp30.metric(_exp30.recs(_exp30.BASE["v03"], p), _exp30.recs(ev, p), p)
            rows[p] = {"difference": m["difference"], "ci95": m["ci95"]}
        out["protection"][arm] = {"panels": rows, "costs": [p for p, d in rows.items() if d["difference"] * 100 <= -3 and d["ci95"][1] < 0]}
    return out


def main():
    out = build()
    json.dump(out, open("runs/exp36-report.json", "w"), indent=1)
    for split in ("tune", "fit"):
        for arm in ("A", "B"):
            s = out[split][arm]
            print("%s %s whole %.3f (mass %.3f speed %.3f distance %.3f) | both %.3f | twin %s | dep %+.3f %s (n %d)" % (
                split, arm, s["whole"], s["whole:mass"], s["whole:speed"], s["whole:distance"], s["both"],
                {f: round(s["twin:" + f], 3) for f in FIELDS}, s["dep"]["value"], [round(x, 3) for x in s["dep"]["ci95"]], s["dep"]["n"]))
    for k, v in out["B_minus_A"].items():
        print("B - A %s %+.3f %s" % (k, v["value"], [round(x, 3) for x in v["ci95"]]))
    print("conditions %s -> continue the full-number route: %s" % (out["conditions"], out["continue_full_number_route"]))
    for arm, p in out["protection"].items():
        print("protection %s: %s costs %s" % (arm, {k: round(100 * v["difference"], 2) for k, v in p["panels"].items()}, p["costs"] or "none"))


if __name__ == "__main__":
    sys.exit(main())
