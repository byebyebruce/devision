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

  Sources (critic V04-K-PREP-01): "livek" -- the release's (RELEASE_KIND): LIVE-K's run (START / registry / report pinned,
  13 states completed, gates live_P / transfer / protection true, every registered file unchanged) and its K bundle
  runs/livek/bundle_k (its reading.json is START's k_bundle, rules hash START's, the K rule present) with PROTECT's end
  (START's protect end) and EXP71's three components (START's exp71 bundle_on); "exp71" -- the earlier no-K source
  (bundles["on"] of EXP71's START), kept as it was. bind keeps the source bundle's K rule (and so its rules hash).

    uv run python scripts/release/rebind.py source --kind livek --run runs/livek --decision runs/protect-k/stage2 --bundle runs/livek/bundle_k --round runs/v04-candidate
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
# LIVE-K-RUN-01 (LIVE-K-RESULT-01): PROTECT's end + EXP71's components + the K rule -- the v0.4 source
PINS_LIVEK = {"START.json": "f9a10d6a7293722c3e04a9649ea092df3127ffaeefb11899eb79c28b75b57a32",
              "registry.json": "aa6e0e7544ff58d29c179ff27abb97533d9434469c2dd39ba0a1b22d1cf05d66",
              "report.json": "2e706f593fd1518823dc5a2df711e1bbfe23d85eaaca3e27611a25a84abfd560"}
SOURCES = {"exp71": PINS, "livek": PINS_LIVEK}
RELEASE_KIND = "livek"                 # what the release flow requires when no pins are given
LIVEK_GATES = ("live_P", "transfer", "protection")


def digest(path: str) -> str:
    return tree_sha(path) if os.path.isdir(path) else sha(path)


def lineage_dir(candidate: str) -> str:
    return os.path.join(os.path.dirname(os.path.normpath(candidate)), LINEAGE_DIR)


def files_of(decision: str, bundle: str) -> Dict[str, str]:
    """What a START's bundles entry records: the bundle's files and the decision checkpoint's three."""
    out = {k: sha(os.path.join(bundle, k)) for k in ("reading.json",) + COMPONENTS if os.path.exists(os.path.join(bundle, k))}
    out.update({k: digest(os.path.join(decision, k)) for k in DECISION})
    return out


def expected_files(run: str, bundle: str, kind: str) -> Dict[str, str]:
    """The files a source's START says it ran: EXP71's bundles.on; for LIVE-K its K bundle's reading.json (START's
    k_bundle), EXP71's three components and PROTECT's end as START bound them, the tokenizer the pinned reading.json binds."""
    st = json.load(open(os.path.join(run, "START.json")))
    if kind == "exp71":
        return dict(st["bundles"]["on"])
    want = {"reading.json": st["k_bundle"]["reading.json"], **{c: st["exp71"]["bundle_on"][c] for c in COMPONENTS},
            **{k: st["protect"]["end"][k] for k in ("model.safetensors", "config.json")}}
    rj = os.path.join(bundle, "reading.json")
    if os.path.exists(rj) and sha(rj) == want["reading.json"]:
        want["tokenizer"] = json.load(open(rj))["sha256"]["tokenizer"]
    return want


def run_problems(run: str, bundle: str, kind: str) -> List[str]:
    """LIVE-K's run as accepted: every state completed, the three gates true, every registered file unchanged, and its K
    bundle with the K rule and START's rules hash (EXP71's run keeps the earlier checks)."""
    if kind != "livek":
        return []
    out = []
    rep = json.load(open(os.path.join(run, "report.json")))
    if not rep.get("states") or any(v != "completed" for v in rep["states"].values()):
        out.append("%s: not every state completed" % run)
    if not all(rep.get("gates", {}).get(g) is True for g in LIVEK_GATES):
        out.append("%s: gates %s not all true" % (run, LIVEK_GATES))
    for files in json.load(open(os.path.join(run, "registry.json"))).values():
        out += ["%s: registered %s changed" % (run, f) for f, h in files.items()
                if not os.path.exists(os.path.join(run, f)) or sha(os.path.join(run, f)) != h]
    st = json.load(open(os.path.join(run, "START.json")))
    ident = json.load(open(os.path.join(bundle, "reading.json")))
    if "k_rule" not in ident or ident.get("rules_sha256") != st["k_bundle"]["rules_sha256"]:
        out.append("%s: the source bundle has no K rule or not START's rules hash" % bundle)
    out += ["%s: rules differ from this code: %s" % (bundle, x) for x in check_rules(ident, full=True)]
    return out


def verify_source(run: str, decision: str, bundle: str, pins: Optional[Dict[str, str]] = None, kind: str = "exp71") -> List[str]:
    """The one read-only check of a source (used when source.json is written and every time it is used): the run's
    pinned files (default SOURCES[kind]) have exactly their pinned sha256 on disk now, the run is the accepted one
    (run_problems), and the bundle and decision checkpoint are the files its START records (expected_files)."""
    if kind not in SOURCES:
        return ["unknown source kind %r" % kind]
    pins = SOURCES[kind] if pins is None else pins
    problems = ["%s/%s is not the pinned source (%s)" % (run, k, v[:12]) for k, v in pins.items()
                if not os.path.exists(os.path.join(run, k)) or sha(os.path.join(run, k)) != v]
    if problems:
        return problems
    want = expected_files(run, bundle, kind)
    got = files_of(decision, bundle)
    diff = sorted(k for k in set(want) | set(got) if want.get(k) != got.get(k))
    if diff:
        return ["%s / %s are not the files %s/START.json records%s (%s)" % (bundle, decision, run, " as bundles.on" if kind == "exp71" else "", diff)]
    return run_problems(run, bundle, kind)


def check_source(rec: Dict, pins: Optional[Dict[str, str]] = None) -> List[str]:
    """A source.json as it is consumed: written for the required pins, its run / bundle / decision verified again
    (`verify_source`, the pinned files' actual hashes), and its own bundle_on equal to that START's bundles.on (and so
    to the files now) -- a record replaced by another self-consistent source, or a changed pinned run, is refused."""
    kind = rec.get("kind", "exp71")
    if pins is None:                                     # the release flow: only its own source kind
        if kind != RELEASE_KIND:
            return ["source.json is a %r source, the release requires %r" % (kind, RELEASE_KIND)]
        pins = SOURCES[RELEASE_KIND]
    if rec.get("pins") != pins:
        return ["source.json pins %r, not the required source" % rec.get("pins")]
    out = verify_source(rec["run"], rec["decision"], rec["bundle"], pins, kind)
    if not out and rec.get("bundle_on") != expected_files(rec["run"], rec["bundle"], kind):
        out.append("source.json bundle_on is not what %s/START.json records" % rec["run"])
    return out


def source(run: str, decision: str, bundle: str, round_dir: str, pins: Optional[Dict[str, str]] = None, kind: str = "exp71") -> Dict:
    if kind not in SOURCES:
        raise SystemExit("unknown source kind %r" % kind)
    pins = SOURCES[kind] if pins is None else pins
    problems = verify_source(run, decision, bundle, pins, kind)
    if problems:
        raise SystemExit("refusing the source:\n  " + "\n  ".join(problems))
    got = files_of(decision, bundle)
    rec = {"kind": kind, "run": os.path.abspath(run), "pins": pins, "decision": os.path.abspath(decision), "bundle": os.path.abspath(bundle), "bundle_on": got}
    os.makedirs(os.path.join(round_dir, LINEAGE_DIR), exist_ok=True)
    path = os.path.join(round_dir, LINEAGE_DIR, "source.json")
    if os.path.exists(path):
        raise SystemExit("%s exists" % path)
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=1)
    return rec


def clone(round_dir: str, candidate: str, pins: Optional[Dict[str, str]] = None) -> None:
    rec = json.load(open(os.path.join(round_dir, LINEAGE_DIR, "source.json")))
    bad = check_source(rec, pins)
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


def bind(candidate: str, pins: Optional[Dict[str, str]] = None) -> Dict:
    meta = lineage_dir(candidate)
    src_path = os.path.join(meta, "source.json")
    if not os.path.exists(src_path):
        raise SystemExit("refusing to bind: no %s (run `source` first)" % src_path)
    rec = json.load(open(src_path))
    decision, bundle = rec["decision"], rec["bundle"]
    ident = json.load(open(os.path.join(bundle, "reading.json")))
    problems = check_source(rec, pins)
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
    keep_k = "k_rule" in ident                        # the source's K rule goes with it (V04-K-PREP-01)
    new = write_bundle(os.path.join(candidate, "reading"), candidate, os.path.join(bundle, "selector.pt"), os.path.join(bundle, "head.pt"),
                       {"source_json_sha256": lineage["source_json_sha256"], "source_reading_json_sha256": rec["bundle_on"]["reading.json"]},
                       dims=ident["dims"], corrector_pt=os.path.join(bundle, "corrector.pt") if "corrector.pt" in comps else None, k_rule=keep_k)
    if {c: new["sha256"][c] for c in comps} != lineage["components"] or set(COMPONENTS) & set(new["sha256"]) != set(comps) \
            or ("k_rule" in new) != keep_k or (keep_k and new["rules_sha256"] != ident["rules_sha256"]):
        shutil.rmtree(os.path.join(candidate, "reading"))
        raise SystemExit("the new bundle's components / rules are not the source's")
    lineage.update(k_rule=keep_k, rules_sha256=new["rules_sha256"])
    lineage["reading_json_sha256"] = sha(os.path.join(candidate, "reading", "reading.json"))
    with open(os.path.join(meta, "lineage.json"), "w") as fh:
        json.dump(lineage, fh, indent=1)
    return lineage


def temp_bundle(candidate: str, out: str, pins: Optional[Dict[str, str]] = None) -> Dict:
    """The temporary calibration bundle (prep03 5.2 step 4): the verified source's components and K rule bound to the
    candidate's CURRENT config (after ordinary calibration), written OUTSIDE the candidate -- for collecting the routed raw
    probabilities only; the final bundle is bound by `bind` after the config is final, and this one is never evidence."""
    rec = json.load(open(os.path.join(lineage_dir(candidate), "source.json")))
    bad = check_source(rec, pins)
    if bad:
        raise SystemExit("refusing the temporary bundle:\n  " + "\n  ".join(bad))
    if os.path.realpath(out).startswith(os.path.realpath(candidate) + os.sep):
        raise SystemExit("the temporary bundle must be outside the candidate")
    ident = json.load(open(os.path.join(rec["bundle"], "reading.json")))
    comps = [c for c in COMPONENTS if c in ident["sha256"]]
    new = write_bundle(out, candidate, os.path.join(rec["bundle"], "selector.pt"), os.path.join(rec["bundle"], "head.pt"),
                       {"temporary_calibration_bundle_of": os.path.abspath(candidate), "source_json_sha256": sha(os.path.join(lineage_dir(candidate), "source.json"))},
                       dims=ident["dims"], corrector_pt=os.path.join(rec["bundle"], "corrector.pt") if "corrector.pt" in comps else None,
                       k_rule="k_rule" in ident)
    if any(new["sha256"][c] != rec["bundle_on"][c] for c in comps) or ("k_rule" in ident and new["rules_sha256"] != ident["rules_sha256"]):
        raise SystemExit("the temporary bundle's components / rules are not the source's")
    return {"reading.json": sha(os.path.join(out, "reading.json")), "config.json": new["sha256"]["config.json"], "k_rule": "k_rule" in new}


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
    out += check_source(rec, pins)
    rj = os.path.join(candidate, "reading", "reading.json")
    if not os.path.exists(rj) or lin.get("reading_json_sha256") != sha(rj):
        out.append("candidate reading/reading.json is not the one lineage.json bound")
    out += ["candidate %s is not the source's" % f for f in SAME if digest(os.path.join(candidate, f)) != lin.get("unchanged", {}).get(f)]
    out += ["candidate reading/%s is not the source's" % c for c, v in lin.get("components", {}).items()
            if not os.path.exists(os.path.join(candidate, "reading", c)) or sha(os.path.join(candidate, "reading", c)) != v]
    if os.path.exists(rj) and os.path.exists(os.path.join(rec["bundle"], "reading.json")):
        cand, src = json.load(open(rj)), json.load(open(os.path.join(rec["bundle"], "reading.json")))
        if ("k_rule" in cand) != ("k_rule" in src) or lin.get("k_rule") != ("k_rule" in src) \
                or ("k_rule" in src and cand.get("rules_sha256") != src.get("rules_sha256")):
            out.append("candidate reading's K rule / rules hash is not the source's")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("source")
    for k in ("run", "decision", "bundle", "round"):
        s.add_argument("--" + k, required=True)
    s.add_argument("--kind", choices=sorted(SOURCES), default=RELEASE_KIND)
    c = sub.add_parser("clone")
    c.add_argument("--round", required=True)
    c.add_argument("--candidate", required=True)
    b = sub.add_parser("bind")
    b.add_argument("--candidate", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "source":
        print(json.dumps(source(a.run, a.decision, a.bundle, a.round, kind=a.kind), indent=1))
    elif a.cmd == "clone":
        clone(a.round, a.candidate)
    else:
        print(json.dumps(bind(a.candidate), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
