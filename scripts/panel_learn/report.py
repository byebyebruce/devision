"""EXP74 report, rules written before running (docs/experiments/2026-10-19-panel-learn-plan.md); read-only.

  d1  scripts/exp24/gate.py on runs/exp74-panel-d1/d1/align_report.json: masked-token accuracy with the right picture
      minus another picture on 1,000 held-out panel pictures >= 0.30 -> run D2 (exit 0), else stop (exit 10).
  d2  from D2's last epoch-end monitor and its exist / science probes:
      valid      -- the erased control stays invalid-for-reading (read < 0.50 and compare < 0.55, delivery.py's rule);
      fit        -- readA / readB and compare on the 200 training families (exp21 readout classify: pass = point >= 0.80
                    and lower >= 0.75; fail = point < 0.60 and upper < 0.65; else near);
      dev16      -- the same rule on the unseen families (development monitor; voided by an invalid control);
      kept       -- K text slice >= 0.95; the frozen reading step's states on the real questions: temperature >= 39/41,
                    magnets >= 0.458 (sealed 0.508 - 0.05); exist and science within 3 points of the sealed candidate;
      outcome    -- invalid (control) / A fit learned + kept / B fit learned, protection broken / C fit not learned
                    (fail) / D near threshold. No outcome starts anything.
"""
import glob
import importlib.util
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "exp21"))
from readout import classify, syn_index, syn_stat  # type: ignore  # noqa: E402

D1 = "runs/exp74-panel-d1/d1/align_report.json"
D2 = "runs/exp74-panel-d2"
SEALED = "runs/v04-k/eval"
LINES = {"k_text": 0.95, "rs_temp": 39 / 41, "rs_force": 0.458}


def gate():
    spec = importlib.util.spec_from_file_location("gate", os.path.join(os.path.dirname(__file__), "..", "exp24", "gate.py"))
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def jsonl(p):
    return [json.loads(x) for x in open(p) if x.strip()]


def accuracy(recs):
    return sum(bool(r["correct"]) for r in recs) / len(recs) if recs else None


def last_step(d2: str) -> str:
    steps = [p for p in glob.glob(os.path.join(d2, "d2", "monitor", "step*")) if re.search(r"step\d+$", p)]
    if not steps:
        raise SystemExit("no D2 monitor")
    return max(steps, key=lambda p: int(re.search(r"(\d+)$", p).group(1)))


def d2_report(d2: str = D2, sealed: str = SEALED):
    step = last_step(d2)
    rec = {n: jsonl(os.path.join(step, n + ".details.jsonl")) for n in ("val", "dev16", "erased", "k_text", "rs_temp", "rs_force",
                                                                        "rs_temp_mism", "rs_force_mism")}
    e = syn_index(rec["erased"])
    erased = {w: syn_stat(e, None, w) for w in ("read", "cmp")}
    valid = erased["read"]["value"] < 0.50 and erased["cmp"]["value"] < 0.55
    f, d = syn_index(rec["val"]), syn_index(rec["dev16"])
    fit = {w: syn_stat(f, 16, w) for w in ("read", "cmp")}
    dev = {w: syn_stat(d, 16, w) for w in ("read", "cmp")}
    for s in (fit, dev):
        for w in ("read", "cmp"):
            s[w + "_class"] = classify(s[w]) if valid else "void: erased control invalid"
    acc = {n: accuracy(rec[n]) for n in ("k_text", "rs_temp", "rs_force", "rs_temp_mism", "rs_force_mism")}
    probes = {}
    for n in ("exist", "science"):
        p, s = os.path.join(d2, "eval", n + ".json"), os.path.join(sealed, n + ".json")
        probes[n] = {"d2": json.load(open(p))["accuracy_all"] if os.path.exists(p) else None, "sealed": json.load(open(s))["accuracy_all"]}
    kept_parts = {n: acc[n] is not None and acc[n] >= LINES[n] for n in LINES}
    kept_parts.update({n: v["d2"] is not None and v["d2"] >= v["sealed"] - 0.03 for n, v in probes.items()})
    kept = all(kept_parts.values())
    classes = [fit["read_class"], fit["cmp_class"]]
    if not valid:
        outcome = "invalid: the erased control reads digits"
    elif all(c == "pass" for c in classes):
        outcome = "A: fit learned, protection kept" if kept else "B: fit learned, protection broken"
    elif any(c == "fail" for c in classes):
        outcome = "C: fit not learned"
    else:
        outcome = "D: near threshold"
    return {"monitor": step, "erased": dict(erased, valid=valid), "fit": fit, "dev16": dev, "accuracy": acc, "probes": probes,
            "kept": kept_parts, "outcome": outcome}


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if argv[:1] == ["d1"]:
        out = gate().decide(json.load(open(argv[1] if len(argv) > 1 else D1)))
        print(json.dumps(out, indent=1))
        return 0 if out["run_phase_b"] else 10
    out = d2_report(*(argv[1:3]))
    json.dump(out, open(os.path.join(argv[1] if len(argv) > 1 else D2, "report.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
