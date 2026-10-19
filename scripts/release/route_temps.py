"""Write the routed temperatures into the calibrated candidate's config (critic V04-K-CALIB-RESULT-01 /
V04-ROUTE-TEMP-PREP-01; the routed calibration's branch 1). CPU only, loads no model. The only source is the registered
calibration run: its registry's route_calibration.json, recomputed from the registered records (route_calib.
analyse_registered) and equal to the written file, branch 1, the all-data per-domain temperatures; the candidate is the
one that calibration was registered for and has no reading/ yet (rebind binds after the config is final). Nothing else in
the config changes; a hand-typed constant is never a source.

    uv run python scripts/release/route_temps.py --calib runs/v04-k-calib --candidate runs/v04-k/cand
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from typing import Dict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def source_temperatures(calib: str) -> Dict[str, float]:
    """The registered branch-1 temperatures of a calibration run. The run's own evidence is checked by hash -- the
    registry's route_calibration.json, CALIB.json and records.jsonl; CALIB's rows file and the candidate config it was
    registered with -- and the statistics are recomputed from those registered records (route_calib.completeness and
    analyse) and must equal the written file. The code identity of that run is not re-checked against today's code:
    this change edits inference modules on purpose; the run's code was verified when it ran (V04-K-CALIB-RESULT-01).
    Refused unless branch 1 with two finite temperatures > 0."""
    reg = json.load(open(os.path.join(calib, "registry.json")))
    path = os.path.join(calib, "route_calibration.json")
    cdir = os.path.join(calib, "calib")
    checks = {path: reg.get("analyse", {}).get("route_calibration.json"),
              os.path.join(cdir, "CALIB.json"): reg.get("collect", {}).get("calib/CALIB.json"),
              os.path.join(cdir, "records.jsonl"): reg.get("collect", {}).get("calib/records.jsonl")}
    bad = [p for p, h in checks.items() if not h or not os.path.exists(p) or sha(p) != h]
    if bad:
        raise SystemExit("not the registered calibration evidence: %s" % bad)
    cal = json.load(open(os.path.join(cdir, "CALIB.json")))
    P = cal["paths"]
    if sha(P["rows"]) != cal["rows"] or sha(os.path.join(P["candidate"], "config.json")) != cal["candidate"]["config.json"]:
        raise SystemExit("the calibration's rows / candidate config are not the registered ones")
    rc = _load("route_temps_rc", "scripts/release/route_calib.py")
    recs = [json.loads(x) for x in open(os.path.join(cdir, "records.jsonl")) if x.strip()]
    rows = [json.loads(x) for x in open(P["rows"]) if x.strip()]
    if rc.completeness(recs, rows):
        raise SystemExit("the registered records are not complete")
    written = json.load(open(path))
    again = json.loads(json.dumps(rc.analyse(recs, json.load(open(os.path.join(P["candidate"], "config.json"))))))
    if {k: v for k, v in written.items() if k != "registration"} != again:
        raise SystemExit("route_calibration.json differs from a recomputation of the registered records")
    if not written["branch"].startswith("1"):
        raise SystemExit("the routed calibration is not branch 1 (%s): no routed temperature" % written["branch"][:40])
    temps = written.get("routed_temperature_all_data") or {}
    if set(temps) != {"temperature", "magnets"} or not all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in temps.values()):
        raise SystemExit("the registered routed temperatures are not two finite temperatures > 0: %r" % temps)
    return {k: float(v) for k, v in temps.items()}


def write(calib: str, candidate: str) -> Dict:
    reg_cand = json.load(open(os.path.join(calib, "calib", "CALIB.json")))["paths"]["candidate"]
    if os.path.realpath(reg_cand) != os.path.realpath(candidate):
        raise SystemExit("the calibration was registered for %s, not %s" % (reg_cand, candidate))
    if os.path.lexists(os.path.join(candidate, "reading")):
        raise SystemExit("the candidate already has reading/: write the routed temperatures before binding")
    cfg_path = os.path.join(candidate, "config.json")
    cfg = json.load(open(cfg_path))
    if cfg.get("temperature_by_route"):
        raise SystemExit("the candidate already has routed temperatures")
    temps = source_temperatures(calib)
    before = sha(cfg_path)
    cfg["temperature_by_route"] = temps
    from devision.model import ModelConfig
    ModelConfig.from_dict(cfg)                                     # the product's own validation
    with open(cfg_path, "w") as fh:
        json.dump(cfg, fh, indent=2)
    return {"temperature_by_route": temps, "config_before": before, "config_after": sha(cfg_path),
            "source": {"route_calibration.json": sha(os.path.join(calib, "route_calibration.json"))}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--candidate", required=True)
    a = ap.parse_args(argv)
    print(json.dumps(write(a.calib, a.candidate), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
