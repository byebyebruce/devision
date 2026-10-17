"""Calibration evidence for a release candidate (critic V04-REL-R1-03): on the same frozen weights, raw
probabilities (all temperatures 1, `devision-eval --no-temperature`) against the candidate's stored temperatures
(`devision-eval`, fitted), accuracy / NLL / ECE overall and by option count. Each set is labelled with its use:
the calibration set (where the temperatures were fitted; its gain is in-sample) and sets that did not take part
in fitting them. No refit, nothing written to the checkpoint.

Inputs are named outputs that pass `check_eval.py check` under the frozen identity (critic V04-REL-R2-04): extras
of configs/v04-extra-evidence.yaml or planned sets of the evaluation YAML. The two runs of a set must hold the same
question copies -- an id repeated in the data (data/r13b/val.jsonl: 242 ids twice, with different options) is
matched copy by copy on its options and every copy is kept (critic V04-REL-R2-03).

    uv run python scripts/release/calibration.py --config configs/v04-full-eval.yaml \\
        --extras configs/v04-extra-evidence.yaml --out release/v0.4/calibration.json \\
        val:calibration_fit:calib_val_raw:calib_val_fitted bench_lv_scienceqa:monitoring:lv_scienceqa_raw:bench_lv_scienceqa
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List

import importlib.util

import numpy as np
from laya.common import ece_score

_spec = importlib.util.spec_from_file_location("check_eval", os.path.join(os.path.dirname(__file__), "check_eval.py"))
assert _spec is not None and _spec.loader is not None
check_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_eval)


def metrics(rs: List[Dict]) -> Dict:
    return {"n": len(rs), "accuracy": float(np.mean([r["correct"] for r in rs])), "nll": float(np.mean([r["nll"] for r in rs])),
            "ece": float(ece_score(np.array([r["p_prediction"] for r in rs]), np.array([r["correct"] for r in rs], dtype=float)))}


def copies(rs: List[Dict]) -> Dict:
    out: Dict = defaultdict(int)
    for r in rs:
        out[(r["sample_id"], r["qid"], tuple(r["options"]))] += 1
    return out


def compare(raw: List[Dict], fitted: List[Dict]) -> Dict:
    if copies(raw) != copies(fitted):
        raise SystemExit("the raw and fitted runs are not the same question copies")
    def split(rs):
        g = defaultdict(list)
        for r in rs:
            g["%s:%d" % (r["type"], len(r["options"]))].append(r)
        return g
    out = {"all": {"raw": metrics(raw), "fitted": metrics(fitted)}}
    a, b = split(raw), split(fitted)
    for k in sorted(a):
        out[k] = {"raw": metrics(a[k]), "fitted": metrics(b[k])}
    return out


def load(path: str) -> List[Dict]:
    return [json.loads(x) for x in open(path) if x.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sets", nargs="+", help="name:use:raw_output:fitted_output (output names, see the docstring)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--extras", required=True)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    specs = [x.split(":", 3) for x in a.sets]
    paths = check_eval.verified(a.config, a.extras, sorted({n for _, _, r, f in specs for n in (r, f)}))
    report = {}
    for name, use, raw_name, fitted_name in specs:
        raw, fitted = paths[raw_name], paths[fitted_name]
        for path, want in ((raw, "none (raw)"), (fitted, "fitted")):
            run = json.load(open(path.replace(".details.jsonl", ".json")))["run"]
            if run.get("temperature") != want:
                raise SystemExit("%s: temperature %r, expected %r" % (path, run.get("temperature"), want))
        cks = {json.load(open(p.replace(".details.jsonl", ".json")))["run"]["checkpoint"] for p in (raw, fitted)}
        if len(cks) != 1:
            raise SystemExit("%s: two checkpoints %s" % (name, cks))
        report[name] = {"use": use, "checkpoint": cks.pop(), "by": compare(load(raw), load(fitted))}
        for k, v in report[name]["by"].items():
            r, f = v["raw"], v["fitted"]
            print("%-20s %-16s %-11s n %5d | acc %.3f -> %.3f | nll %.3f -> %.3f | ece %.3f -> %.3f" % (
                name, use, k, r["n"], r["accuracy"], f["accuracy"], r["nll"], f["nll"], r["ece"], f["ece"]))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
