"""The v0.4 delivery evidence against v0.3 (docs/experiments/2026-10-17-v04-delivery-plan.md; critic V04-REL-R1-04,
V04-REL-R2-04), computed only from candidate outputs that pass `check_eval.py check` under the frozen identity --
nothing refitted, the rules unchanged:

  1. primary: the transfer readout (exp21's readout.transfer: ScienceQA temperature + force dev, 68 : 114,
     accuracy and picture-gain changes with intervals), recomputed from the candidate's sealed sqa_* outputs and
     v0.3's exp27 records (content-checked, historical identity), next to the candidate's own screening label
     (critic V04-FALLBACK-01): --screen names the experiment, --screen-report its report and --screen-sha256 the
     weights that report was made from; the label is carried only when that sha256 equals the frozen identity's
     model.safetensors, otherwise it is marked unverifiable. A screening label is not the whole set of release
     conditions; a recomputed value that differs from the screening report is flagged;
  2. new ability kept: the candidate's dev16 synthetic reading and comparison, point estimate and interval
     resampled by family, each with the original rule (pass = point >= 0.80 and lower bound >= 0.75; fail = point
     < 0.60 and upper bound < 0.65); the erased-digit control of the same candidate must be valid (read < 0.50 and
     compare < 0.55) for that rule to count -- an invalid control voids the dev16 labels;
  3. protection: the 7 panels against v0.3 (a cost when <= -3 points with the upper bound < 0; ext 13-source
     macro, flip with its originals), from the sealed outputs of the evaluation YAML;
  4. the frozen identity (IDENTITY.json) all of it belongs to.

    uv run python scripts/release/delivery.py --config configs/v04-full-eval.yaml \\
        --extras configs/v04-extra-evidence.yaml --screen <experiment> --screen-report <report.json> \\
        --screen-sha256 <sha256 of the weights the report was made from> --out release/v0.4/delivery_vs_v0.3.json
"""
import argparse
import importlib.util
import json
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "exp21"))
from readout import classify, syn_index, syn_stat, transfer  # type: ignore  # noqa: E402

from devision.train.compare import load_records, paired_metric  # noqa: E402

_spec = importlib.util.spec_from_file_location("check_eval", os.path.join(os.path.dirname(__file__), "check_eval.py"))
assert _spec is not None and _spec.loader is not None
check_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_eval)

V03 = "runs/exp27-transfer/eval/v03"
GUARD = ("exist", "science", "ext", "count_wide", "count_narrow", "flip", "vsr")
EXT_SOURCES = json.load(open(os.path.join(os.path.dirname(__file__), "..", "r14b", "rules.json")))["ext_sources"]
SQA = ("sqa_temp", "sqa_force", "sqa_temp_mism", "sqa_force_mism")


def recs(path: str, name: str):
    rs = load_records(path)
    return [dict(r, image_id=r["image_id"].removesuffix(":flip")) for r in rs] if name == "flip" else rs


def screening(identity: dict, name: str, report: Optional[dict], weights_sha256: str) -> dict:
    """The candidate's screening label, bound to the weights it was made from."""
    model = identity["files"].get("model.safetensors")
    if report is None:
        return {"name": name, "status": "no screening report", "outcome": None}
    if not weights_sha256 or weights_sha256 != model:
        return {"name": name, "status": "unverifiable: the report's weights (%s) are not the frozen candidate (%s)"
                % (weights_sha256 or "not given", model), "outcome": None}
    return {"name": name, "status": "bound to the frozen weights", "outcome": report.get("outcome"),
            "transfer": report.get("transfer")}


def build(cand: dict, v03: dict, identity: dict, screen: dict) -> dict:
    """cand / v03: name -> records (SQA, GUARD; cand also dev16, syn_erased)."""
    t = transfer({k[4:]: v03[k] for k in SQA}, {k[4:]: cand[k] for k in SQA})
    guard, costs = {}, []
    for p in GUARD:
        m = paired_metric(v03[p], cand[p], "macro_accuracy", sources=EXT_SOURCES) if p == "ext" else paired_metric(v03[p], cand[p], "accuracy")
        guard[p] = {"difference": m["difference"], "ci95": m["ci95"]}
        if m["difference"] * 100 <= -3 and m["ci95"][1] < 0:
            costs.append(p)
    d, e = syn_index(cand["dev16"]), syn_index(cand["syn_erased"])
    same = (all(abs(t[k]["value"] - screen["transfer"][k]["value"]) < 1e-9 for k in ("accuracy_change", "gain_change"))
            if screen.get("transfer") else None)
    out = {"identity": {"checkpoint": identity["checkpoint"], "model.safetensors": identity["files"].get("model.safetensors"),
                        "config.json": identity["files"].get("config.json"), "frozen": identity.get("frozen")},
           "primary": {"transfer": t,
                       "target_met": t["accuracy_change"]["value"] >= 0.10 and t["accuracy_change"]["ci95"][0] > 0
                       and t["gain_change"]["ci95"][0] > 0,
                       "screening": {k: v for k, v in screen.items() if k != "transfer"}, "screening_report_matches": same,
                       "note": "recomputed from sealed outputs; a screening label is not all release conditions"},
           "protection": {"panels": guard, "costs": costs},
           "dev16": {w: syn_stat(d, 16, w) for w in ("read", "cmp")},
           "erased": {w: syn_stat(e, None, w) for w in ("read", "cmp")}}
    out["erased"]["valid"] = out["erased"]["read"]["value"] < 0.50 and out["erased"]["cmp"]["value"] < 0.55
    for w in ("read", "cmp"):
        out["dev16"][w + "_class"] = classify(out["dev16"][w]) if out["erased"]["valid"] else "void: erased control invalid"
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--extras", required=True)
    ap.add_argument("--screen", required=True, help="the experiment that screened the candidate")
    ap.add_argument("--screen-report", help="its report.json (outcome, transfer)")
    ap.add_argument("--screen-sha256", default="", help="sha256 of the model.safetensors that report was made from")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    paths = check_eval.verified(a.config, a.extras, list(SQA) + list(GUARD) + ["dev16", "syn_erased"])
    cand = {n: recs(p, n) for n, p in paths.items()}
    v03 = {n: recs(os.path.join(V03, n + ".details.jsonl"), n) for n in list(SQA) + list(GUARD)}
    plan = check_eval.Plan(a.config, a.extras)
    identity = json.load(open(plan.path("IDENTITY.json")))
    screen = screening(identity, a.screen, json.load(open(a.screen_report)) if a.screen_report else None, a.screen_sha256)
    out = build(cand, v03, identity, screen)
    json.dump(out, open(a.out, "w"), indent=1)
    t = out["primary"]["transfer"]
    sc = out["primary"]["screening"]
    print("transfer accuracy %+.3f %s, gain %+.3f %s; target met %s; screening %s: %s, %s (report matches: %s)" % (
        t["accuracy_change"]["value"], t["accuracy_change"]["ci95"], t["gain_change"]["value"], t["gain_change"]["ci95"],
        out["primary"]["target_met"], sc["name"], sc["status"], sc["outcome"], out["primary"]["screening_report_matches"]))
    for w in ("read", "cmp"):
        s = out["dev16"][w]
        print("dev16 %s %.3f %s -> %s" % (w, s["value"], s["ci95"], out["dev16"][w + "_class"]))
    print("erased read %.3f cmp %.3f valid %s; costs %s" % (out["erased"]["read"]["value"], out["erased"]["cmp"]["value"],
                                                           out["erased"]["valid"], out["protection"]["costs"] or "none"))


if __name__ == "__main__":
    main()
