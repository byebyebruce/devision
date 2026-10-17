"""The v0.4 delivery evidence against v0.3 (docs/experiments/2026-10-17-v04-delivery-plan.md; critic V04-REL-R1-04),
from results already produced -- nothing refitted or rescored:

  1. primary: exp32's transfer readout (ScienceQA temperature + force dev, 68 : 114, accuracy and picture-gain
     changes with intervals) and its pre-written outcome label -- the label is exp32's candidate screen, not the
     whole set of release conditions;
  2. new ability kept: the candidate's dev16 synthetic reading and comparison, point estimate and interval
     resampled by family, each with the original rule (pass = point >= 0.80 and lower bound >= 0.75; fail = point
     < 0.60 and upper bound < 0.65); the erased-digit control of the same candidate must be valid (read < 0.50 and
     compare < 0.55) for that rule to count -- an invalid control voids the dev16 labels;
  3. protection: the 7 panels with intervals and the pre-written costs (exp32's report);
  4. the frozen identity (IDENTITY.json) the results belong to.

    uv run python scripts/release/delivery.py --report runs/exp32-fit/report.json \\
        --dev16 runs/exp32-fit/eval/dev16.details.jsonl --erased runs/exp32-fit/eval/syn_erased.details.jsonl \\
        --identity runs/exp32-fit/eval/IDENTITY.json --out release/v0.4/delivery_vs_v0.3.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "exp21"))
from readout import classify, syn_index, syn_stat  # type: ignore  # noqa: E402

from devision.train.compare import load_records  # noqa: E402


def build(report: dict, dev16: list, erased: list, identity: dict) -> dict:
    d, e = syn_index(dev16), syn_index(erased)
    out = {"identity": {"checkpoint": identity["checkpoint"], "model.safetensors": identity["files"].get("model.safetensors"),
                        "config.json": identity["files"].get("config.json"), "frozen": identity.get("frozen")},
           "primary": {"transfer": report["transfer"], "target_met": report["target_met"],
                       "exp32_outcome": report["outcome"],
                       "note": "exp32's pre-written candidate screen; not all release conditions"},
           "protection": {"panels": report["guard"], "costs": report["costs"]},
           "dev16": {w: syn_stat(d, 16, w) for w in ("read", "cmp")},
           "erased": {w: syn_stat(e, None, w) for w in ("read", "cmp")}}
    out["erased"]["valid"] = out["erased"]["read"]["value"] < 0.50 and out["erased"]["cmp"]["value"] < 0.55
    for w in ("read", "cmp"):
        out["dev16"][w + "_class"] = classify(out["dev16"][w]) if out["erased"]["valid"] else "void: erased control invalid"
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", required=True)
    ap.add_argument("--dev16", required=True)
    ap.add_argument("--erased", required=True)
    ap.add_argument("--identity", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = build(json.load(open(a.report)), load_records(a.dev16), load_records(a.erased), json.load(open(a.identity)))
    json.dump(out, open(a.out, "w"), indent=1)
    t = out["primary"]["transfer"]
    print("transfer accuracy %+.3f %s, gain %+.3f %s; exp32: %s" % (
        t["accuracy_change"]["value"], t["accuracy_change"]["ci95"], t["gain_change"]["value"], t["gain_change"]["ci95"],
        out["primary"]["exp32_outcome"]))
    for w in ("read", "cmp"):
        s = out["dev16"][w]
        print("dev16 %s %.3f %s -> %s" % (w, s["value"], s["ci95"], out["dev16"][w + "_class"]))
    print("erased read %.3f cmp %.3f valid %s; costs %s" % (out["erased"]["read"]["value"], out["erased"]["cmp"]["value"],
                                                           out["erased"]["valid"], out["protection"]["costs"] or "none"))


if __name__ == "__main__":
    main()
