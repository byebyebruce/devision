"""exp37 readout (critic EXP37-READOUT-01; docs/experiments/2026-10-17-exp37-digit-size-plan.md; rules written before
running), from the per-number records of scripts/exp36/readout.py, with exp36's statistics:

  - main comparison: model L on L pictures against model S on S pictures, the same 600 tune families, paired by
    family (seed 0, 2,000 resamples), and exp36's four investment conditions: whole-number macro and both-objects
    macro differences each >= +0.10 with lower bound > 0; L's mass and speed whole-number accuracy each >= 0.20; L's
    value-changed twin real - wrong lower bound > 0. Passing says that training on and reading large digits rescues
    the task, not that the product input is fixed;
  - the 2 x 2 matrix (model S / L x pictures S / L), each model on the original tune pictures, each model's own fit
    slice: reported, to tell training adaptation from reading size;
  - the 7 protection panels of each model against v0.3, apart from the number endpoint.

    uv run python scripts/exp37/report.py
"""
import importlib.util
import json
import os
import sys

_spec = importlib.util.spec_from_file_location("exp36_report", os.path.join(os.path.dirname(__file__), "..", "exp36", "report.py"))
assert _spec is not None and _spec.loader is not None
r36 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r36)

RUNS = {"S": "runs/exp37-s", "L": "runs/exp37-l"}
INPUTS = ("tune_S", "tune_L", "tune_orig", "fit")


def build(runs=RUNS) -> dict:
    fams: dict = {}
    out: dict = {"cells": {}, "protection": {}}
    for m, run in runs.items():
        for inp in INPUTS:
            fam = r36.units(r36.load(os.path.join(run, "numbers_" + inp, "numbers.jsonl")))
            fams[(m, inp)] = fam
            out["cells"]["%s model / %s" % (m, inp)] = r36.arm_stats(fam)
    s, l = fams[("S", "tune_S")], fams[("L", "tune_L")]
    if set(s) != set(l):
        raise SystemExit("the main comparison is not on the same families")
    families = sorted(s)
    out["L_minus_S"] = {k: r36.interval(lambda fs, k=k: r36.value(l, fs, k) - r36.value(s, fs, k), families) for k in ("whole", "both")}
    L = out["cells"]["L model / tune_L"]
    out["conditions"] = {
        "whole_macro": out["L_minus_S"]["whole"]["value"] >= 0.10 and out["L_minus_S"]["whole"]["ci95"][0] > 0,
        "both_macro": out["L_minus_S"]["both"]["value"] >= 0.10 and out["L_minus_S"]["both"]["ci95"][0] > 0,
        "mass_speed": L["whole:mass"] >= 0.20 and L["whole:speed"] >= 0.20,
        "picture_dependence": L["dep"]["ci95"][0] > 0}
    out["large_digits_rescue"] = all(out["conditions"].values())
    for m, run in runs.items():
        ev = os.path.join(run, "eval")
        rows = {}
        for p in r36._exp30.GUARD:
            mm = r36._exp30.metric(r36._exp30.recs(r36._exp30.BASE["v03"], p), r36._exp30.recs(ev, p), p)
            rows[p] = {"difference": mm["difference"], "ci95": mm["ci95"]}
        out["protection"][m] = {"panels": rows, "costs": [p for p, d in rows.items() if d["difference"] * 100 <= -3 and d["ci95"][1] < 0]}
    return out


def main():
    out = build()
    json.dump(out, open("runs/exp37-report.json", "w"), indent=1)
    for cell, st in out["cells"].items():
        print("%-22s whole %.3f (mass %.3f speed %.3f distance %.3f) | both %.3f | twin %.3f | dep %+.3f %s (n %d)" % (
            cell, st["whole"], st["whole:mass"], st["whole:speed"], st["whole:distance"], st["both"], st["twin"],
            st["dep"]["value"], [round(x, 3) for x in st["dep"]["ci95"]], st["dep"]["n"]))
    for k, v in out["L_minus_S"].items():
        print("L/L - S/S %s %+.3f %s" % (k, v["value"], [round(x, 3) for x in v["ci95"]]))
    print("conditions %s -> large digits rescue: %s" % (out["conditions"], out["large_digits_rescue"]))
    for m, p in out["protection"].items():
        print("protection %s: %s costs %s" % (m, {k: round(100 * v["difference"], 2) for k, v in p["panels"].items()}, p["costs"] or "none"))


if __name__ == "__main__":
    sys.exit(main())
