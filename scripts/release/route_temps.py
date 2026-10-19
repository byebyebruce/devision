"""Write the routed temperatures into the calibrated candidate's config (critic V04-K-CALIB-RESULT-01 /
V04-ROUTE-TEMP-PREP-01, ROUTE-TEMP-R1-01; the routed calibration's branch 1). CPU only, loads no model, fits nothing.
The only source is the accepted calibration run: its START / registry / final report are pinned (ACCEPTED, the run
V04-K-CALIB-RESULT-01 accepted), the registry's route_calibration.json / CALIB.json / records.jsonl / rl_calib
questions and the candidate's pre-write config are checked against it, and the report's two all-data per-domain
temperatures are read as written. Nothing else in the config changes; a hand-typed constant is never a source.

    uv run python scripts/release/route_temps.py --calib runs/v04-k-calib --candidate runs/v04-k/cand
"""
import argparse
import hashlib
import json
import math
import os
import sys
from typing import Dict

# runs/v04-k-calib as accepted in V04-K-CALIB-RESULT-01 (branch 1).
ACCEPTED = {"START.json": "2c02a535cb580dab0f1771b13147600e821326709f7152ac9ec05794b9c1a877",
            "registry.json": "5b35da89bb245cc680b118522eb66f3b3346858fd81515c232331005a7c001f6",
            "report.json": "1b54ff353ffe59b91fe503d4cbd4683e33965645655a025e4f33c854911b778b"}


def sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def source_temperatures(calib: str, pins: Dict[str, str] = ACCEPTED) -> Dict[str, float]:
    """The accepted run's branch-1 temperatures, read as written: START / registry / report must be the pinned files;
    route_calibration.json, CALIB.json, records.jsonl must be the registry's; CALIB's rows file and the rl_calib
    questions it registered must match. No statistic is recomputed (the result is accepted; the run's code was verified
    when it ran). Refused unless branch 1 with two finite temperatures > 0."""
    bad = [f for f, h in pins.items() if not os.path.exists(os.path.join(calib, f)) or sha(os.path.join(calib, f)) != h]
    if bad:
        raise SystemExit("not the accepted calibration run: %s differ from the pinned ones" % bad)
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
    rows = cal["paths"]["rows"]
    if not os.path.exists(rows) or sha(rows) != cal["rows"] or sha(rows) != reg.get("rl_calib", {}).get("questions.jsonl"):
        raise SystemExit("the calibration's rows are not the registered ones")
    written = json.load(open(path))
    if not str(written.get("branch", "")).startswith("1"):
        raise SystemExit("the routed calibration is not branch 1 (%s): no routed temperature" % str(written.get("branch"))[:40])
    temps = written.get("routed_temperature_all_data") or {}
    if set(temps) != {"temperature", "magnets"} or not all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in temps.values()):
        raise SystemExit("the registered routed temperatures are not two finite temperatures > 0: %r" % temps)
    return {k: float(v) for k, v in temps.items()}


def write(calib: str, candidate: str, pins: Dict[str, str] = ACCEPTED) -> Dict:
    temps = source_temperatures(calib, pins)
    cal = json.load(open(os.path.join(calib, "calib", "CALIB.json")))
    reg_cand = cal["paths"]["candidate"]
    if os.path.realpath(reg_cand) != os.path.realpath(candidate):
        raise SystemExit("the calibration was registered for %s, not %s" % (reg_cand, candidate))
    if os.path.lexists(os.path.join(candidate, "reading")):
        raise SystemExit("the candidate already has reading/: write the routed temperatures before binding")
    cfg_path = os.path.join(candidate, "config.json")
    cfg = json.load(open(cfg_path))
    if cfg.get("temperature_by_route"):
        raise SystemExit("the candidate already has routed temperatures")
    before = sha(cfg_path)
    reg = json.load(open(os.path.join(calib, "registry.json")))
    if before != cal["candidate"]["config.json"] or before != reg.get("calibrate", {}).get("config.json"):
        raise SystemExit("the candidate's config is not the one the calibration was registered with")
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
