"""The reading combination's identity hand-over (docs/experiments/2026-10-19-v04-reader-evidence-prep03.md 5.2; critic
V04-EVIDENCE-R3-02, V04-ENTRY-R1-03). CPU only, loads no tensor. Every record lives in <round>/reading_lineage/,
<round> being the candidate's parent directory (a round directory with run.json, so evaluations go to <round>/eval).

  source  check the source run against its pinned identity -- by default EXP71 (START / registry / report sha256,
          PINS) -- and that the given bundle and decision checkpoint are exactly the files its START records as
          bundles["on"] (reading.json, selector, head, corrector; EXP57's model.safetensors / config.json / tokenizer);
          write source.json. Another bundle or decision directory, however self-consistent, is refused.
  clone   copy the source decision checkpoint's inference files (config.json, model.safetensors, encoder, vision,
          tokenizer) to the new candidate directory; the source is read only. Ordinary calibration (devision-calibrate
          --in-place) and the release model_name are then set in the candidate, before any bundle exists
          (devision-calibrate refuses a checkpoint with reading/).
  bind    after the candidate's config is final: re-check the sources against source.json, check that the candidate's
          weights, tokenizer, encoder and vision are the source decision checkpoint's and its config.json differs only
          in ALLOWED keys, write <candidate>/reading/ from the source bundle's component files and lineage.json (with
          source.json's sha256). check_eval freeze binds both records; the release build checks them again.

    uv run python scripts/release/rebind.py source --run runs/exp71 --decision runs/exp57/... --bundle runs/exp71/bundle_on --round runs/v04-candidate
    uv run python scripts/release/rebind.py clone --round runs/v04-candidate --candidate runs/v04-candidate/stage
    uv run python scripts/release/rebind.py bind --candidate runs/v04-candidate/stage
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from typing import Dict, List, Optional

from devision.reading import check_rules, sha, tree_sha, write_bundle

FILES = ("config.json", "model.safetensors", "encoder", "vision", "tokenizer")
SAME = ("model.safetensors", "encoder", "vision", "tokenizer")
ALLOWED = {"temperature", "temperature_by_options", "model_name", "temperature_by_route"}
COMPONENTS = ("selector.pt", "head.pt", "corrector.pt")
DECISION = ("model.safetensors", "config.json", "tokenizer")
LINEAGE_DIR = "reading_lineage"
# EXP71 (EXP71-RESULT-01), whose bundles["on"] is the combination EXP73 guards
PINS = {"START.json": "da4c3cec8811f5847ba2cb5953952eece452074e3cefea174e01a9bbcea6b943",
        "registry.json": "dc93c7a1f969581c06b9c075fc9e8f5e28d8a449a658085d36beedd120ea6b07",
        "report.json": "6869596dff0f36fb31f7ff32d3a910c7e26291f92ef670b08a6b4ec3367d6eea"}


def digest(path: str) -> str:
    return tree_sha(path) if os.path.isdir(path) else sha(path)


def lineage_dir(candidate: str) -> str:
    return os.path.join(os.path.dirname(os.path.normpath(candidate)), LINEAGE_DIR)


def files_of(decision: str, bundle: str) -> Dict[str, str]:
    """What a START's bundles entry records: the bundle's files and the decision checkpoint's three."""
    out = {k: sha(os.path.join(bundle, k)) for k in ("reading.json",) + COMPONENTS if os.path.exists(os.path.join(bundle, k))}
    out.update({k: digest(os.path.join(decision, k)) for k in DECISION})
    return out


def check_source(rec: Dict) -> List[str]:
    """The source record's files are still the ones it names."""
    now = files_of(rec["decision"], rec["bundle"])
    return ["source %s changed since source.json" % k for k in sorted(set(now) | set(rec["bundle_on"])) if now.get(k) != rec["bundle_on"].get(k)]


def source(run: str, decision: str, bundle: str, round_dir: str, pins: Optional[Dict[str, str]] = None) -> Dict:
    pins = PINS if pins is None else pins
    problems = ["%s/%s is not the pinned source (%s)" % (run, k, v[:12]) for k, v in pins.items()
                if not os.path.exists(os.path.join(run, k)) or sha(os.path.join(run, k)) != v]
    if problems:
        raise SystemExit("refusing the source:\n  " + "\n  ".join(problems))
    want = json.load(open(os.path.join(run, "START.json")))["bundles"]["on"]
    got = files_of(decision, bundle)
    diff = sorted(k for k in set(want) | set(got) if want.get(k) != got.get(k))
    if diff:
        raise SystemExit("refusing the source: %s / %s are not the files %s/START.json records as bundles.on (%s)" % (bundle, decision, run, diff))
    rec = {"run": os.path.abspath(run), "pins": pins, "decision": os.path.abspath(decision), "bundle": os.path.abspath(bundle), "bundle_on": got}
    os.makedirs(os.path.join(round_dir, LINEAGE_DIR), exist_ok=True)
    path = os.path.join(round_dir, LINEAGE_DIR, "source.json")
    if os.path.exists(path):
        raise SystemExit("%s exists" % path)
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=1)
    return rec


def clone(round_dir: str, candidate: str) -> None:
    rec = json.load(open(os.path.join(round_dir, LINEAGE_DIR, "source.json")))
    bad = check_source(rec)
    if bad:
        raise SystemExit("refusing to clone:\n  " + "\n  ".join(bad))
    if os.path.normpath(os.path.dirname(os.path.normpath(candidate))) != os.path.normpath(round_dir) or os.path.exists(candidate):
        raise SystemExit("the candidate must be a new directory directly in %s" % round_dir)
    os.makedirs(candidate)
    for f in FILES:
        src, dst = os.path.join(rec["decision"], f), os.path.join(candidate, f)
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        elif subprocess.call(["cp", "-c", src, dst], stderr=subprocess.DEVNULL) != 0:   # APFS clone
            shutil.copyfile(src, dst)


def config_diff(a: Dict, b: Dict) -> List[str]:
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def bind(candidate: str) -> Dict:
    meta = lineage_dir(candidate)
    src_path = os.path.join(meta, "source.json")
    if not os.path.exists(src_path):
        raise SystemExit("refusing to bind: no %s (run `source` first)" % src_path)
    rec = json.load(open(src_path))
    decision, bundle = rec["decision"], rec["bundle"]
    ident = json.load(open(os.path.join(bundle, "reading.json")))
    problems = check_source(rec)
    problems += ["source bundle's rules differ from this code: %s" % x for x in check_rules(ident)]
    problems += ["candidate %s differs from the source decision checkpoint" % f for f in SAME
                 if digest(os.path.join(candidate, f)) != digest(os.path.join(decision, f))]
    src_cfg, cand_cfg = json.load(open(os.path.join(decision, "config.json"))), json.load(open(os.path.join(candidate, "config.json")))
    diff = config_diff(src_cfg, cand_cfg)
    problems += ["candidate config.json changes %r, which is not allowed" % k for k in diff if k not in ALLOWED]
    if os.path.lexists(os.path.join(candidate, "reading")):
        problems.append("candidate already has reading/")
    if problems:
        raise SystemExit("refusing to bind:\n  " + "\n  ".join(problems))
    comps = [c for c in COMPONENTS if c in ident["sha256"]]
    lineage = {"source_json_sha256": sha(src_path), "candidate": os.path.abspath(candidate),
               "components": {c: rec["bundle_on"][c] for c in comps}, "unchanged": {f: digest(os.path.join(decision, f)) for f in SAME},
               "config_changes": {k: {"source": src_cfg.get(k), "candidate": cand_cfg.get(k)} for k in diff}}
    new = write_bundle(os.path.join(candidate, "reading"), candidate, os.path.join(bundle, "selector.pt"), os.path.join(bundle, "head.pt"),
                       {"source_json_sha256": lineage["source_json_sha256"], "source_reading_json_sha256": rec["bundle_on"]["reading.json"]},
                       dims=ident["dims"], corrector_pt=os.path.join(bundle, "corrector.pt") if "corrector.pt" in comps else None)
    if {c: new["sha256"][c] for c in comps} != lineage["components"] or set(COMPONENTS) & set(new["sha256"]) != set(comps):
        shutil.rmtree(os.path.join(candidate, "reading"))
        raise SystemExit("the new bundle's components are not the source's")
    lineage["reading_json_sha256"] = sha(os.path.join(candidate, "reading", "reading.json"))
    with open(os.path.join(meta, "lineage.json"), "w") as fh:
        json.dump(lineage, fh, indent=1)
    return lineage


def lineage_problems(candidate: str, pins: Optional[Dict[str, str]] = None) -> List[str]:
    """The records a bound candidate must carry, checked against the candidate and the sources as they are now
    (used by check_eval freeze / preflight / seal / check and by the release build). `pins`: the source identity the
    record must name (default PINS, EXP71)."""
    meta = lineage_dir(candidate)
    paths = {n: os.path.join(meta, n) for n in ("source.json", "lineage.json")}
    missing = [n for n, p in paths.items() if not os.path.exists(p)]
    if missing:
        return ["%s: missing %s" % (meta, ", ".join(missing))]
    rec, lin = json.load(open(paths["source.json"])), json.load(open(paths["lineage.json"]))
    out = []
    if lin.get("source_json_sha256") != sha(paths["source.json"]):
        out.append("lineage.json names source.json %s, it is %s" % (lin.get("source_json_sha256"), sha(paths["source.json"])))
    if rec.get("pins") != (PINS if pins is None else pins):
        out.append("source.json pins %r, not the required source" % rec.get("pins"))
    out += check_source(rec)
    rj = os.path.join(candidate, "reading", "reading.json")
    if not os.path.exists(rj) or lin.get("reading_json_sha256") != sha(rj):
        out.append("candidate reading/reading.json is not the one lineage.json bound")
    out += ["candidate %s is not the source's" % f for f in SAME if digest(os.path.join(candidate, f)) != lin.get("unchanged", {}).get(f)]
    out += ["candidate reading/%s is not the source's" % c for c, v in lin.get("components", {}).items()
            if not os.path.exists(os.path.join(candidate, "reading", c)) or sha(os.path.join(candidate, "reading", c)) != v]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("source")
    for k in ("run", "decision", "bundle", "round"):
        s.add_argument("--" + k, required=True)
    c = sub.add_parser("clone")
    c.add_argument("--round", required=True)
    c.add_argument("--candidate", required=True)
    b = sub.add_parser("bind")
    b.add_argument("--candidate", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "source":
        print(json.dumps(source(a.run, a.decision, a.bundle, a.round), indent=1))
    elif a.cmd == "clone":
        clone(a.round, a.candidate)
    else:
        print(json.dumps(bind(a.candidate), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
