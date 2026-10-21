"""exp38 C rescue readout (critic EXP38-MEM-03): arm C alone (256 input / 64 visual tokens, 1,000 steps from exp37-L's
end on the original pictures), after the resource-interrupted exp38 chain. No H, so no H - C and no four-condition
label: per field whole numbers, both objects where values differ, value-changed twins, picture dependence on the
value-changed twin numbers (exp36's statistics, intervals by family), on the original 600 tune and 200 fit pictures;
then whatever report-only items completed (status file): ScienceQA true / fixed-wrong, dev16, the 7 protection
panels, CPU latency. exp37-L's fixed original-picture reading (0.113) is the starting point, not a control.

    uv run python scripts/exp38/report_c.py
"""
import importlib.util
import json
import os
import sys

_spec = importlib.util.spec_from_file_location("exp38_report", os.path.join(os.path.dirname(__file__), "report.py"))
assert _spec is not None and _spec.loader is not None
r38 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r38)
r36 = r38.r36

RUN = "runs/exp38-c"
STATUS = "runs/exp38c-rescue-status.txt"


def build(run: str = RUN, status_path: str = STATUS) -> dict:
    st = r38.states(status_path)
    done = lambda n: st.get(n) == "completed"  # noqa: E731
    out: dict = {"status": st, "note": "arm C only, after the resource-interrupted exp38 chain; no H - C"}
    if not (done("readout:c:tune") and done("readout:c:fit")):
        out["outcome"] = "incomplete: C readouts missing"
        return out
    out["identity"] = r38.identity({"C": run})["C"]
    out["cells"] = {s: r36.arm_stats(r36.units(r36.load(os.path.join(run, "numbers_" + s, "numbers.jsonl")))) for s in ("tune", "fit")}
    sub: dict = {}
    ev = os.path.join(run, "eval")
    sub["sqa"] = {k: r38.acc(os.path.join(ev, k + ".details.jsonl")) if done("eval:c:" + k) else None
                  for k in ("sqa_temp", "sqa_temp_mism", "sqa_force", "sqa_force_mism")}
    if done("eval:c:dev16"):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "exp21"))
        from readout import classify, syn_index, syn_stat  # type: ignore
        idx = syn_index(r36.load(os.path.join(ev, "dev16.details.jsonl")))
        sub["dev16"] = {w: syn_stat(idx, 16, w) for w in ("read", "cmp")}
        sub["dev16"]["read_class"] = classify(sub["dev16"]["read"])
    panels = {}
    for p in r36._exp30.GUARD:
        if done("eval:c:" + p):
            m = r36._exp30.metric(r36._exp30.recs(r36._exp30.BASE["v03"], p), r36._exp30.recs(ev, p), p)
            panels[p] = {"difference": m["difference"], "ci95": m["ci95"]}
    sub["protection"] = panels
    sub["protection_complete"] = len(panels) == len(r38.PANELS)
    sub["costs"] = [p for p, d in panels.items() if d["difference"] * 100 <= -3 and d["ci95"][1] < 0]
    sub["costs_meaning"] = ("pre-written costs over all 7 panels" if sub["protection_complete"] else
                            "protection not completely evaluated: costs observed among the %d completed panels only" % len(panels))
    if done("cpu:c"):
        lat = json.load(open(os.path.join(run, "cpu", "single.json"))).get("latency_ms") or {}
        sub["cpu_latency_ms"] = {k: lat.get(k) for k in ("p50", "p95")}
    sub["not_completed"] = sorted(n for n, s in st.items() if s != "completed")
    out["report_only"] = sub
    out["outcome"] = "C readouts complete (no four-condition label without H)"
    return out


def main():
    out = build()
    json.dump(out, open("runs/exp38c-rescue-report.json", "w"), indent=1)
    print("outcome: " + out["outcome"])
    if "cells" not in out:
        return 0
    print("identity: %s" % out["identity"])
    for s, c in out["cells"].items():
        print("C / %-4s whole %.3f (mass %.3f speed %.3f distance %.3f) | both %.3f | twin %.3f | dep %+.3f %s (n %d)" % (
            s, c["whole"], c["whole:mass"], c["whole:speed"], c["whole:distance"], c["both"], c["twin"],
            c["dep"]["value"], [round(x, 3) for x in c["dep"]["ci95"]], c["dep"]["n"]))
        print("   both by field %s | twin by field %s | denominators %s" % (
            {f: round(c["both:" + f], 3) for f in ("mass", "speed", "distance")},
            {f: round(c["twin:" + f], 3) for f in ("mass", "speed", "distance")}, c["denominators"]))
    ro = out["report_only"]
    print("report only: sqa %s | dev16 %s | protection %s costs %s (%s) | cpu %s | not completed %s" % (
        {k: (round(v, 3) if v is not None else None) for k, v in ro["sqa"].items()},
        (round(ro["dev16"]["read"]["value"], 3), ro["dev16"]["read_class"]) if "dev16" in ro else None,
        {k: round(100 * v["difference"], 2) for k, v in ro["protection"].items()}, ro["costs"] or "none observed",
        ro["costs_meaning"], ro.get("cpu_latency_ms"), ro["not_completed"] or "none"))


if __name__ == "__main__":
    sys.exit(main())
