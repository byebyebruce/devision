"""EXP76 device V preflight (DEVICE-V-IMPL-PREP-01), no model: the accepted EXP76 checks (R's files, EXP75's
evidence, the D2 data by its manifests, rlcd's 1,054-update plan) on the V config and its own fresh paths; the eight
V1 reference files by sha256; a boot cycle started after the EXP76 faults (the restart is the owner's; holding this
script is not a run permission). Writes runs/exp76-device-v-preflight.json and, when everything holds,
runs/exp76-device-v.START.json (code sha256, boot time, the gpuEvent files present at the start); exit 12 otherwise.
"""
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vcheck  # noqa: E402

NAME = "exp76-device-v"
CONFIG = "configs/%s.yaml" % NAME
OUT = "runs/%s-preflight.json" % NAME
START = "runs/%s.START.json" % NAME
FRESH = ("runs/" + NAME, "runs/%s.started" % NAME, START)
FAULT_END = 1792445418                     # EXP76-RUN-02's report, 2026-10-20 04:30:18: the last fault's run ended
DIAG = "/Library/Logs/DiagnosticReports"
CODE = (CONFIG, "scripts/device_v/run_v.sh", "scripts/device_v/preflight_v.py", "scripts/device_v/vcheck.py",
        "scripts/device_v/vrun.py", "scripts/device_v/v3.py", "scripts/device_v/report_v.py",
        "scripts/panel_decision/guarded.py", "scripts/panel_decision/preflight.py", "scripts/panel_learn/report.py",
        "src/devision/train/rlcd.py", "src/devision/train/pipeline.py", "src/devision/infer.py")


def base():
    spec = importlib.util.spec_from_file_location("pd_preflight", os.path.join(HERE, "..", "panel_decision", "preflight.py"))
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    setattr(m, "CONFIG", CONFIG)                      # the accepted checks, on the V config and its own paths
    setattr(m, "FRESH", FRESH)
    return m


def problems(boot=None) -> list:
    out = base().problems()
    out += ["V1 reference %s is missing or changed" % n for n, h in vcheck.REF.items()
            if not os.path.exists(os.path.join(vcheck.REF_DIR, n + ".details.jsonl"))
            or vcheck.sha(os.path.join(vcheck.REF_DIR, n + ".details.jsonl")) != h]
    out += vcheck.boot_problems(vcheck.boot_time() if boot is None else boot, FAULT_END)
    return out


def main() -> int:
    boot = vcheck.boot_time()
    bad = problems(boot)
    res = {"ok": not bad, "problems": bad}
    json.dump(res, open(OUT + ".tmp", "w"), indent=1)
    os.replace(OUT + ".tmp", OUT)
    if not bad:
        st = {"name": NAME, "boot": boot, "boot_str": vcheck.time_str(boot or 0), "reference": vcheck.REF,
              "gpu_events_present": sorted(f for f in os.listdir(DIAG) if f.startswith("gpuEvent")) if os.path.isdir(DIAG) else [],
              "code": {p: vcheck.sha(p) for p in CODE}, "preflight": vcheck.sha(OUT)}
        json.dump(st, open(START + ".tmp", "w"), indent=1)
        os.replace(START + ".tmp", START)
    print(json.dumps(res, indent=1))
    return 0 if not bad else 12


if __name__ == "__main__":
    sys.exit(main())
