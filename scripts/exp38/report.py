"""exp38 readout (critic EXP38-READOUT-01; docs/experiments/2026-10-17-exp38-512-plan.md; rules written before running).
Arm H = 512 input / 256 visual tokens, arm C = 256 / 64, both 1,000 steps from exp37-L's end on the original
pictures. From the per-number records of scripts/exp36/readout.py on the ORIGINAL 600 tune pictures (each model at
its own input size), with exp36's statistics:

  - investment conditions for 512 (not a release standard): H - C whole-number macro >= +0.10 with lower bound > 0;
    H - C both-objects macro >= +0.10 with lower bound > 0; H's mass and speed whole-number accuracy each >= 0.20;
    H's value-changed twin real - wrong lower bound > 0. If C learns too and H's advantage misses them, that is "512
    not worth its cost here", not "the original-picture route failed";
  - reported: each model's 200 original fit pictures; exp37-L's fixed original-picture reading (0.113) as the
    starting point, not a control; CPU latency (P50 / P95, 200-question fixture, 4 threads); and, when present, the
    ScienceQA temperature / force true / fixed-wrong accuracies, dev16 and the 7 protection panels against v0.3.

    uv run python scripts/exp38/report.py
"""
import importlib.util
import json
import os
import sys

_spec = importlib.util.spec_from_file_location("exp36_report", os.path.join(os.path.dirname(__file__), "..", "exp36", "report.py"))
assert _spec is not None and _spec.loader is not None
r36 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r36)

RUNS = {"H": "runs/exp38-h", "C": "runs/exp38-c"}


def acc(path: str):
    if not os.path.exists(path):
        return None
    rs = [json.loads(x) for x in open(path) if x.strip()]
    return sum(r["correct"] for r in rs) / len(rs) if rs else None


def build(runs=RUNS) -> dict:
    out: dict = {"cells": {}, "report_only": {}}
    fams: dict = {}
    for arm, run in runs.items():
        for split in ("tune", "fit"):
            fam = r36.units(r36.load(os.path.join(run, "numbers_" + split, "numbers.jsonl")))
            out["cells"]["%s / %s" % (arm, split)] = r36.arm_stats(fam)
            fams[(arm, split)] = fam
    h, c = fams[("H", "tune")], fams[("C", "tune")]
    if set(h) != set(c):
        raise SystemExit("the arms were read on different families")
    families = sorted(h)
    out["H_minus_C"] = {k: r36.interval(lambda fs, k=k: r36.value(h, fs, k) - r36.value(c, fs, k), families) for k in ("whole", "both")}
    H = out["cells"]["H / tune"]
    out["conditions"] = {
        "whole_macro": out["H_minus_C"]["whole"]["value"] >= 0.10 and out["H_minus_C"]["whole"]["ci95"][0] > 0,
        "both_macro": out["H_minus_C"]["both"]["value"] >= 0.10 and out["H_minus_C"]["both"]["ci95"][0] > 0,
        "mass_speed": H["whole:mass"] >= 0.20 and H["whole:speed"] >= 0.20,
        "picture_dependence": H["dep"]["ci95"][0] > 0}
    out["invest_in_512"] = all(out["conditions"].values())
    for arm, run in runs.items():
        ro: dict = {}
        cpu = os.path.join(run, "cpu", "single.json")
        if os.path.exists(cpu):
            lat = json.load(open(cpu)).get("latency_ms") or {}
            ro["cpu_latency_ms"] = {k: lat.get(k) for k in ("p50", "p95")}
        ev = os.path.join(run, "eval")
        ro["sqa"] = {k: acc(os.path.join(ev, k + ".details.jsonl")) for k in ("sqa_temp", "sqa_temp_mism", "sqa_force", "sqa_force_mism")}
        d16 = os.path.join(ev, "dev16.details.jsonl")
        if os.path.exists(d16):
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "exp21"))
            from readout import classify, syn_index, syn_stat  # type: ignore
            idx = syn_index(r36.load(d16))
            ro["dev16"] = {w: syn_stat(idx, 16, w) for w in ("read", "cmp")}
            ro["dev16"]["read_class"] = classify(ro["dev16"]["read"])
        panels = {}
        for p in r36._exp30.GUARD:
            if os.path.exists(os.path.join(ev, p + ".details.jsonl")):
                m = r36._exp30.metric(r36._exp30.recs(r36._exp30.BASE["v03"], p), r36._exp30.recs(ev, p), p)
                panels[p] = {"difference": m["difference"], "ci95": m["ci95"]}
        ro["protection"] = panels
        ro["costs"] = [p for p, d in panels.items() if d["difference"] * 100 <= -3 and d["ci95"][1] < 0]
        out["report_only"][arm] = ro
    return out


def main():
    out = build()
    json.dump(out, open("runs/exp38-report.json", "w"), indent=1)
    for cell, st in out["cells"].items():
        print("%-9s whole %.3f (mass %.3f speed %.3f distance %.3f) | both %.3f | twin %.3f | dep %+.3f %s (n %d)" % (
            cell, st["whole"], st["whole:mass"], st["whole:speed"], st["whole:distance"], st["both"], st["twin"],
            st["dep"]["value"], [round(x, 3) for x in st["dep"]["ci95"]], st["dep"]["n"]))
    for k, v in out["H_minus_C"].items():
        print("H - C %s %+.3f %s" % (k, v["value"], [round(x, 3) for x in v["ci95"]]))
    print("conditions %s -> invest in 512: %s" % (out["conditions"], out["invest_in_512"]))
    for arm, ro in out["report_only"].items():
        print("%s report only: cpu %s | sqa %s | dev16 %s | protection %s costs %s" % (
            arm, ro.get("cpu_latency_ms"), {k: (round(v, 3) if v is not None else None) for k, v in ro["sqa"].items()},
            (round(ro["dev16"]["read"]["value"], 3), ro["dev16"]["read_class"]) if "dev16" in ro else None,
            {k: round(100 * v["difference"], 2) for k, v in ro["protection"].items()}, ro["costs"] or "none"))


if __name__ == "__main__":
    sys.exit(main())
