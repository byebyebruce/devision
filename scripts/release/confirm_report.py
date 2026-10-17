"""The v0.4 frozen confirmation readout (critic V04-REL-R1-05, V04-CONF-01 / 02): v0.3 and the locked candidate on
the same frozen questions, each subset ("kind": object-property, chart-reading) reported on its own -- never one
overall mean:

  - accuracy of each model on the true pictures;
  - picture gain = accuracy(true) - accuracy(fixed mismatched pictures), per model, and its change;
  - candidate - v0.3 accuracy change;
  - reversed-option sensitivity: the share of questions whose chosen option (by text) changes when the options are
    shown in reverse order.

Every interval is a 95% bootstrap over ORIGINAL pictures within the subset: the mismatched-picture files show
another picture, so a record's group is taken from its sample id in the true-picture data file (the frozen
mismatched file's image_id is the shown picture), never from the mismatched record's own image_id. No pass
threshold: small samples, descriptive evidence next to the 68 : 114 primary target. Evaluated once after the lock.

    uv run python scripts/release/confirm_report.py --data data/v04_confirm/confirm.jsonl \\
        --v03 <dir with confirm_property{,_mism,_rev}.details.jsonl> --cand <same for the candidate> --out <json>
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

from devision.train.compare import load_records

KINDS = ("none", "mism", "rev")


def groups_of(data: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """sample id -> original image id, sample id -> subset (kind), from the true-picture data file."""
    img, kind = {}, {}
    for line in open(data):
        if line.strip():
            s = json.loads(line)
            img[s["id"]], kind[s["id"]] = s["image_id"], s["kind"]
    return img, kind


def by_question(rs: List[Dict]) -> Dict[Tuple[str, str], Dict]:
    out = {}
    for r in rs:
        k = (r["sample_id"], r["qid"])
        if k in out:
            raise SystemExit("question %s twice" % (k,))
        out[k] = r
    return out


def stats(cells: Dict[str, List[Dict[str, float]]], resamples: int = 2000, seed: int = 0) -> Dict[str, Dict]:
    """cells: picture -> list of per-question dicts of values; mean of each value and its picture bootstrap."""
    pics = sorted(cells)
    names = sorted({k for v in cells.values() for q in v for k in q})
    def mean(sel, name):
        vals = [q[name] for p in sel for q in cells[p] if name in q]
        return sum(vals) / len(vals) if vals else float("nan")
    rng = random.Random(seed)
    boots = [[rng.choice(pics) for _ in pics] for _ in range(resamples)]
    out = {}
    for n in names:
        b = sorted(mean(s, n) for s in boots)
        out[n] = {"value": mean(pics, n), "ci95": [b[int(0.025 * resamples)], b[int(0.975 * resamples) - 1]],
                  "pictures": len(pics), "n": sum(1 for p in pics for q in cells[p] if n in q)}
    return out


def build(data: str, v03: Dict[str, List[Dict]], cand: Dict[str, List[Dict]]) -> Dict:
    """v03 / cand: "none" / "mism" / "rev" -> records."""
    img, kind = groups_of(data)
    m = {who: {k: by_question(recs[k]) for k in KINDS} for who, recs in (("v03", v03), ("cand", cand))}
    keys = set(m["v03"]["none"])
    for who in m:
        for k in KINDS:
            if set(m[who][k]) != keys:
                raise SystemExit("%s %s: not the same questions as v03 none" % (who, k))
    out = {}
    for sub in sorted(set(kind.values())):
        cells: Dict[str, List[Dict[str, float]]] = defaultdict(list)
        for key in sorted(keys):
            if kind[key[0]] != sub:
                continue
            q = {}
            for who in ("v03", "cand"):
                t, mi, rv = (m[who][k][key] for k in KINDS)
                q[who + "_acc"] = float(t["correct"])
                q[who + "_gain"] = float(t["correct"]) - float(mi["correct"])
                q[who + "_flip"] = float(t["prediction"] != rv["prediction"])
            q["acc_change"] = q["cand_acc"] - q["v03_acc"]
            q["gain_change"] = q["cand_gain"] - q["v03_gain"]
            cells[img[key[0]]].append(q)
        out[sub] = stats(dict(cells))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--v03", required=True, help="dir with <name>.details.jsonl, <name>_mism..., <name>_rev...")
    ap.add_argument("--cand", required=True)
    ap.add_argument("--name", default="confirm_property")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    def load(d):
        return {k: load_records("%s/%s%s.details.jsonl" % (d, a.name, "" if k == "none" else "_" + k)) for k in KINDS}
    out = build(a.data, load(a.v03), load(a.cand))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    for sub, st in out.items():
        print("%s (%d pictures)" % (sub, st["acc_change"]["pictures"]))
        for n in ("v03_acc", "cand_acc", "acc_change", "v03_gain", "cand_gain", "gain_change", "v03_flip", "cand_flip"):
            s = st[n]
            print("  %-12s %+.3f [%+.3f, %+.3f] (n %d)" % (n, s["value"], s["ci95"][0], s["ci95"][1], s["n"]))


if __name__ == "__main__":
    sys.exit(main())
