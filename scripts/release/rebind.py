"""The reading combination's identity hand-over (docs/experiments/2026-10-19-v04-reader-evidence-prep03.md 5.2; critic
V04-EVIDENCE-R3-02). CPU only, loads no tensor.

  clone  copy a decision checkpoint's inference files (config.json, model.safetensors, encoder, vision, tokenizer) to a
         new candidate directory; the source is read only. Ordinary calibration (devision-calibrate --in-place) and the
         release model_name are then set in the candidate, before any bundle exists (devision-calibrate refuses a
         checkpoint with reading/).
  bind   after the candidate's config is final: write <candidate>/reading/ from the source bundle's component files and
         <meta>/lineage.json, after checking that
           - the source bundle binds the source decision checkpoint (Reading's own hash checks) and its rules are this code's;
           - the candidate's weights, tokenizer, encoder and vision are byte-identical to the source decision checkpoint's;
           - the candidate's config.json differs from the source's only in ALLOWED keys (temperatures, model_name, the
             routed temperature if 5.3 adopts it);
         and that the new bundle carries the same component files. A changed weight, an extra config difference or a
         component that is not the source's is refused.

    uv run python scripts/release/rebind.py clone --source runs/exp57/... --candidate runs/v04-candidate
    uv run python scripts/release/rebind.py bind --source runs/exp57/... --bundle runs/exp71/bundle_on \
        --candidate runs/v04-candidate --meta runs/v04-candidate-meta
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from typing import Dict, List

from devision.reading import check_rules, sha, tree_sha, write_bundle

FILES = ("config.json", "model.safetensors", "encoder", "vision", "tokenizer")
SAME = ("model.safetensors", "encoder", "vision", "tokenizer")
ALLOWED = {"temperature", "temperature_by_options", "model_name", "temperature_by_route"}
COMPONENTS = ("selector.pt", "head.pt", "corrector.pt")


def digest(path: str) -> str:
    return tree_sha(path) if os.path.isdir(path) else sha(path)


def clone(source: str, candidate: str) -> None:
    if os.path.exists(candidate):
        raise SystemExit("%s exists" % candidate)
    if not os.path.exists(os.path.join(source, "config.json")):
        raise SystemExit("%s has no config.json (a reading bundle binds config.json)" % source)
    os.makedirs(candidate)
    for f in FILES:
        src, dst = os.path.join(source, f), os.path.join(candidate, f)
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        elif subprocess.call(["cp", "-c", src, dst], stderr=subprocess.DEVNULL) != 0:   # APFS clone
            shutil.copyfile(src, dst)


def config_diff(a: Dict, b: Dict) -> List[str]:
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def bind(source: str, bundle: str, candidate: str, meta: str) -> Dict:
    ident = json.load(open(os.path.join(bundle, "reading.json")))
    problems = []
    for k, path in (("model.safetensors", os.path.join(source, "model.safetensors")), ("config.json", os.path.join(source, "config.json")),
                    ("tokenizer", os.path.join(source, "tokenizer"))):
        if ident["sha256"].get(k) != digest(path):
            problems.append("source bundle does not bind the source decision checkpoint's %s" % k)
    comps = [c for c in COMPONENTS if c in ident["sha256"]]
    problems += ["source bundle's %s changed since it was bound" % c for c in comps if sha(os.path.join(bundle, c)) != ident["sha256"][c]]
    problems += ["source bundle's rules differ from this code: %s" % x for x in check_rules(ident)]
    problems += ["candidate %s differs from the source decision checkpoint" % f for f in SAME
                 if digest(os.path.join(candidate, f)) != digest(os.path.join(source, f))]
    diff = config_diff(json.load(open(os.path.join(source, "config.json"))), json.load(open(os.path.join(candidate, "config.json"))))
    problems += ["candidate config.json changes %r, which is not allowed" % k for k in diff if k not in ALLOWED]
    if os.path.exists(os.path.join(candidate, "reading")):
        problems.append("candidate already has reading/")
    if problems:
        raise SystemExit("refusing to bind:\n  " + "\n  ".join(problems))
    lineage = {"decision_source": os.path.abspath(source), "bundle_source": os.path.abspath(bundle),
               "source_reading_json_sha256": sha(os.path.join(bundle, "reading.json")),
               "components": {c: ident["sha256"][c] for c in comps}, "unchanged": {f: digest(os.path.join(source, f)) for f in SAME},
               "config_changes": {k: {"source": json.load(open(os.path.join(source, "config.json"))).get(k),
                                      "candidate": json.load(open(os.path.join(candidate, "config.json"))).get(k)} for k in diff}}
    new = write_bundle(os.path.join(candidate, "reading"), candidate, os.path.join(bundle, "selector.pt"), os.path.join(bundle, "head.pt"),
                       {"lineage": "see lineage.json", "source_reading_json_sha256": lineage["source_reading_json_sha256"]},
                       dims=ident["dims"], corrector_pt=os.path.join(bundle, "corrector.pt") if "corrector.pt" in comps else None)
    if {c: new["sha256"][c] for c in comps} != lineage["components"] or set(COMPONENTS) & set(new["sha256"]) != set(comps):
        shutil.rmtree(os.path.join(candidate, "reading"))
        raise SystemExit("the new bundle's components are not the source's")
    lineage["reading_json_sha256"] = sha(os.path.join(candidate, "reading", "reading.json"))
    os.makedirs(meta, exist_ok=True)
    with open(os.path.join(meta, "lineage.json"), "w") as fh:
        json.dump(lineage, fh, indent=1)
    return lineage


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("clone")
    c.add_argument("--source", required=True)
    c.add_argument("--candidate", required=True)
    b = sub.add_parser("bind")
    for k in ("source", "bundle", "candidate", "meta"):
        b.add_argument("--" + k, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "clone":
        clone(a.source, a.candidate)
    else:
        print(json.dumps(bind(a.source, a.bundle, a.candidate, a.meta), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
