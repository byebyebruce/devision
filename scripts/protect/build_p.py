"""PROTECT-PREP-01 readout files (docs/experiments/2026-10-19-protect-replay-plan.md, C and the training val). CPU only,
loads no model, changes no existing file. Output data/protect_p/:

  P_K.jsonl          data/facts/tune_P.jsonl (EXP56's registered automatic tables) with EXP70's K block added to a
                     temperature state when its table's four values are all readable integers (scripts/exp70/build.py
                     k_block, computed from the state's own table only); otherwise -- an "unreadable" value or a decimal --
                     the state stays as it is (the live path's case), counted in the MANIFEST; magnet rows unchanged.
  slice_K_text.jsonl data/exp72/slice_K.jsonl with "input": "text" on every row (mixed_input needs the mark; EXP72-K's own
                     validation slice, scored exactly as before).

    uv run python scripts/protect/build_p.py [--out data/protect_p]
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
from collections import Counter
from typing import Dict

HERE = os.path.dirname(os.path.abspath(__file__))
TUNE_P, SLICE = "data/facts/tune_P.jsonl", "data/exp72/slice_K.jsonl"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, rel))
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


b70 = _load("exp70_build", "../exp70/build.py")


def sha(p: str) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def jsonl(p: str):
    return [json.loads(x) for x in open(p) if x.strip()]


def with_k_if_defined(r: Dict):
    """(row, reason): the row with the K block when its table defines K, else the row unchanged and why."""
    if r.get("kind") != "temp":
        return r, "magnets"
    try:
        return b70.with_k(r), "K"
    except (ValueError, SystemExit) as e:
        return r, "no K: %s" % ("unreadable value" if isinstance(e, ValueError) else "non-integer value")


def build(out: str) -> Dict:
    os.makedirs(out, exist_ok=False)
    reasons: Counter = Counter()
    rows = []
    for r in jsonl(TUNE_P):
        k, why = with_k_if_defined(r)
        reasons[why] += 1
        rows.append(k)
    with open(os.path.join(out, "P_K.jsonl"), "w") as fh:
        fh.write("".join(json.dumps(r) + "\n" for r in rows))
    with open(os.path.join(out, "slice_K_text.jsonl"), "w") as fh:
        fh.write("".join(json.dumps(dict(r, input="text")) + "\n" for r in jsonl(SLICE)))
    man = {"inputs": {p: sha(p) for p in (TUNE_P, SLICE, "scripts/exp70/build.py")}, "P_K_rows": dict(reasons),
           "sha256": {f: sha(os.path.join(out, f)) for f in ("P_K.jsonl", "slice_K_text.jsonl")}}
    with open(os.path.join(out, "MANIFEST.json"), "w") as fh:
        json.dump(man, fh, indent=1)
    return man


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/protect_p")
    print(json.dumps(build(ap.parse_args(argv).out), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
