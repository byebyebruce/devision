"""Calibration evidence for a release candidate (critic V04-REL-R1-03): on the same frozen weights, raw
probabilities (all temperatures 1, `devision-eval --no-temperature`) against the candidate's stored temperatures
(`devision-eval`, fitted), accuracy / NLL / ECE overall and by option count. Each set is labelled with its use:
the calibration set (where the temperatures were fitted; its gain is in-sample) and sets that did not take part
in fitting them. No refit, nothing written to the checkpoint. Records of the two runs must be the same questions
of the same checkpoint.

    uv run python scripts/release/calibration.py --out release/v0.4/calibration.json \\
        val:calibration_fit:<raw.details.jsonl>:<fitted.details.jsonl> \\
        bench_lv_scienceqa:monitoring:<raw.details.jsonl>:<fitted.details.jsonl>
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np
from laya.common import ece_score


def metrics(rs: List[Dict]) -> Dict:
    return {"n": len(rs), "accuracy": float(np.mean([r["correct"] for r in rs])), "nll": float(np.mean([r["nll"] for r in rs])),
            "ece": float(ece_score(np.array([r["p_prediction"] for r in rs]), np.array([r["correct"] for r in rs], dtype=float)))}


def compare(raw: List[Dict], fitted: List[Dict]) -> Dict:
    ka = {(r["sample_id"], r["qid"]) for r in raw}
    kb = {(r["sample_id"], r["qid"]) for r in fitted}
    if ka != kb or len(ka) != len(raw):
        raise SystemExit("the raw and fitted runs are not the same questions")
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
    ap.add_argument("sets", nargs="+", help="name:use:raw_details:fitted_details")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    report = {}
    for spec in a.sets:
        name, use, raw, fitted = spec.split(":", 3)
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
