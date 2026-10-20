"""EXP76 preflight (PANEL-DECISION-PREP-01), no model: the start is EXP75's R end point and nothing else -- its
weights / config / training report by sha256, EXP75's accepted comparison naming R completed and worth a decision
diagnostic (N's invalid state is not read as anything); the accepted D2 data (train, fit / dev16 / erased, the sealed
read states) by their manifests; rlcd's plan for this config is 1,054 updates; EXP76's directories are fresh. Writes
runs/exp76-preflight.json; exit 12 when anything differs (nothing is trained).

    uv run python scripts/panel_decision/preflight.py
"""
import hashlib
import importlib.util
import json
import os
import sys

CONFIG = "configs/exp76-panel-decision.yaml"
OUT = "runs/exp76-preflight.json"
R = {"path": "runs/exp75-panel-r/r",
     "model.safetensors": "b7f471fc955756f91e8edde8147da76e8762d8228604548ba67321a5a28fb489",
     "config.json": "7fb24ad36c7076f71fa1b7cea634ed58ef62ce8414565cc54f2d4fddfe591bc8",
     "align_report.json": "2721ecd6b5baf24730184e1166becb82e13cb39904ab1248136d71276fe0c177"}
EXP75 = {"runs/panel-number/compare.json": "d69461b7de03fbdb927d5ff24e8a6a801382115fb1adf499e78451ed3a078d04",
         "runs/panel-number/report.json": "edb6d7bda3dfb9785d2e66daa1aa03e5d0602a39a1774b464c0db767dc274545",
         "runs/panel-number/r/numbers.jsonl": "1e68132ed272dff80cbbb975515fb1ce08ce753259daa067a51496b561b73d79"}
FRESH = ("runs/exp76-panel-decision", "runs/exp76-panel-decision.started")
PLANNED = 1054


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def problems() -> list:
    out = []
    for k in ("model.safetensors", "config.json", "align_report.json"):
        p = os.path.join(R["path"], k)
        if not os.path.exists(p) or sha(p) != R[k]:
            out.append("R %s is not the accepted one" % k)
    for p, h in EXP75.items():
        if not os.path.exists(p) or sha(p) != h:
            out.append("%s is not the accepted EXP75 evidence" % p)
    if not out:
        arm = json.load(open("runs/panel-number/compare.json"))["arms"]["r"]
        if arm.get("status") != "completed" or arm.get("worth_decision_diagnostic") is not True:
            out.append("EXP75 does not name R completed and worth a decision diagnostic")
    man = json.load(open("data/panel_learn/MANIFEST.json"))
    if sha("data/panel_learn/train.jsonl") != man["train_sha256"]:
        out.append("the D2 training data is not its manifest's")
    for n, e in json.load(open("data/panel_learn/eval_sets.MANIFEST.json")).items():
        if sha("data/panel_learn/%s.jsonl" % n) != e["sha256"]:
            out.append("data/panel_learn/%s.jsonl is not its manifest's" % n)
    for n, e in json.load(open("data/panel_learn/read_states.MANIFEST.json")).items():
        if sha("data/panel_learn/read_states_%s.jsonl" % n) != e["file_sha256"]:
            out.append("read_states_%s is not its manifest's" % n)
    import yaml
    st = yaml.safe_load(open(CONFIG))["stages"][0]
    if st["init"] != R["path"]:
        out.append("%s does not start from R" % CONFIG)
    spec = importlib.util.spec_from_file_location("pl_report", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "panel_learn", "report.py"))
    assert spec and spec.loader
    rep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rep)
    n = rep.planned_updates(CONFIG)
    if n != PLANNED:
        out.append("rlcd plans %d updates for %s, %d expected" % (n, CONFIG, PLANNED))
    out += ["%s exists (fresh only)" % p for p in FRESH if os.path.exists(p)]
    return out


def main() -> int:
    bad = problems()
    res = {"ok": not bad, "problems": bad}
    json.dump(res, open(OUT + ".tmp", "w"), indent=1)
    os.replace(OUT + ".tmp", OUT)
    print(json.dumps(res, indent=1))
    return 0 if not bad else 12


if __name__ == "__main__":
    sys.exit(main())
