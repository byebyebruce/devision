"""EXP76 r2 preflight (EXP76-RETRY-PREP-01), no model: everything the accepted EXP76 preflight checks (R's three files,
EXP75's evidence naming R completed and worth a decision diagnostic, the D2 data by its manifests, rlcd's plan = 1,054
updates) for the r2 config, its own fresh paths, and the retry-of bindings: the first attempt's started marker, log,
incomplete report, status / out, preflight and disposal records, and the MLM-REPRO-RUN-01 report must exist with
their recorded sha256 -- the retry never overwrites them. Writes runs/exp76-panel-decision-r2-preflight.json and, when
everything holds, runs/exp76-panel-decision-r2.START.json (retry_of, the r2 config / scripts / code sha256); exit 12
otherwise (nothing is trained).
"""
import hashlib
import importlib.util
import json
import os
import sys

NAME = "exp76-panel-decision-r2"
CONFIG = "configs/%s.yaml" % NAME
OUT = "runs/%s-preflight.json" % NAME
START = "runs/%s.START.json" % NAME
FRESH = ("runs/" + NAME, "runs/%s.started" % NAME, START)
RETRY_OF = {
    "runs/exp76-panel-decision.started": "ba61ee358aa696383669d1626a44602e73b4787ccf2a48695ab4e5bd35ea9c45",
    "runs/exp76-panel-decision/logs/d2.log": "60445a7b34bd6505595e6ea177c6607c16c446c53559c73ba3c1947694e79359",
    "runs/exp76-panel-decision/report.json": "bfadf709d15e24c79f67ac0457d43079e42abb7e4f2a6d6553700ed4729aeb8d",
    "runs/exp76-panel-decision-status.txt": "7126bd276e246c75046d0d1748d9d55e81c3de6f8dd962c1411825c6ca3cf0bb",
    "runs/exp76-panel-decision.out": "4080aa08150c23c74933bdf7bbe9a997f172970d6485eb54c1afece148191d22",
    "runs/exp76-preflight.json": "bcc2ccf8618bd488c2450b14bb2902a486481b83eace3a8b1f8183bf793fad74",
    "runs/exp76-disposal/before.txt": "084da41617e3d09280bb257b189dcaa6d9fb106871a582814e0c3506663bee02",
    "runs/exp76-disposal/actions.txt": "280e84f04bb2821f15741145a6af60a31152142e243fa6505fcc26cf5445b140",
    "runs/exp76-disposal/after.txt": "3bb39f6961515f3b79a8a32fbcd5125fb011a350f048295a105c092a307e4e28",
    "runs/exp76-disposal/d2.log.before": "60445a7b34bd6505595e6ea177c6607c16c446c53559c73ba3c1947694e79359",
    "runs/exp76-disposal/files.before": "74a9e151db3496eacbfba6d7eb827a2fb69a821ac4bc783c49f1a150d8e39e56",
    "runs/mlm-repro/report.json": "e27410676eec4c84439004777985b2f8febe602fc3d2a960fc2f0b1ac74dd36b"}
CODE = ("configs/%s.yaml" % NAME, "scripts/panel_decision/run_r2.sh", "scripts/panel_decision/preflight_r2.py", "scripts/panel_decision/guarded.py",
        "scripts/panel_decision/report_r2.py", "scripts/panel_decision/preflight.py", "scripts/panel_learn/report.py",
        "src/devision/train/rlcd.py", "src/devision/train/pipeline.py")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def base():
    spec = importlib.util.spec_from_file_location("pd_preflight", os.path.join(os.path.dirname(os.path.abspath(__file__)), "preflight.py"))
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    setattr(m, "CONFIG", CONFIG)                         # the accepted checks, on the r2 config and its own paths
    setattr(m, "FRESH", FRESH)
    return m


def problems() -> list:
    out = base().problems()
    out += ["retry-of evidence %s is missing or changed" % p for p, h in RETRY_OF.items() if not os.path.exists(p) or sha(p) != h]
    return out


def main() -> int:
    bad = problems()
    res = {"ok": not bad, "problems": bad}
    json.dump(res, open(OUT + ".tmp", "w"), indent=1)
    os.replace(OUT + ".tmp", OUT)
    if not bad:
        st = {"name": NAME, "retry_of": {"run": "runs/exp76-panel-decision", "evidence": RETRY_OF,
                                         "reason": "MPS command-buffer error at step 0 (EXP76-MPS-01); failed / incomplete"},
              "code": {p: sha(p) for p in CODE}, "preflight": sha(OUT)}
        json.dump(st, open(START + ".tmp", "w"), indent=1)
        os.replace(START + ".tmp", START)
    print(json.dumps(res, indent=1))
    return 0 if not bad else 12


if __name__ == "__main__":
    sys.exit(main())
