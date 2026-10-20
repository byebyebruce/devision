"""V04-K calibration run (critic V04-K-PREP-02-ACCEPT-01; docs/experiments/2026-10-19-v04-k-release-plan.md, PREP-02):
the LIVE-K combination's ordinary calibration (data/r13b/val.jsonl, 3,892 questions) and routed calibration (rl_calib,
1,200 pictures through the full reading entry with K, 5.3's four branches). Fixed source: LIVE-K (scripts/release/rebind.py
kind livek: PROTECT's end + EXP71's components + the K rule). No training; the source run and its files are read only.
candidate_eligible stays false; nothing is bound, frozen, confirmed or released here.

  start       model-free: the LIVE-K source verified, this code and the data hashed -> START.json; then (files only)
              rebind source --kind livek and clone into runs/v04-k/cand (a round directory with run.json).
  calibrate   MPS: devision-calibrate --in-place on the candidate (data/r13b/val.jsonl; progress per forward chunk), its
              report moved out of the candidate, model_name set to the release name; the candidate checked (weights the
              source's, config differing only in whitelisted keys).
  rl_calib    CPU: scripts/release/rl_calib.py writes data/rl_calib (300 + 300 families, 1,200 pictures, isolation and
              partition audited; refused unless complete).
  temp_bundle CPU: the temporary calibration bundle outside the candidate (rebind.temp_bundle).
  collect     MPS: route_calib collect -- identity checked and registered before loading, raw temperatures, every question
              read, the runtime code registered (progress per request).
  analyse     CPU: route_calib analyse on the registered calibration -> route_calibration.json.
  report      model-free: every stage's evidence registered and verified (missing -> incomplete, changed -> invalid
              before any number), the ordinary calibration before / after, the routed branch and what it allows next.

    uv run python scripts/calib/calib.py start --deadline EPOCH | calibrate | rl_calib | temp_bundle | collect | analyse | report
"""
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT, ROUND = "runs/v04-k-calib", "runs/v04-k"
VAL, VAL_QUESTIONS = "data/r13b/val.jsonl", 3892
LIVEK = {"run": "runs/livek", "decision": "runs/protect-k/stage2", "bundle": "runs/livek/bundle_k"}
RELEASE_NAME = "devision-v0.4"
RL_WANT = {"temp": 300, "force": 300}
PICTURES = 1200
FILES = ["scripts/calib/calib.py", "scripts/calib/run.sh", "scripts/release/rebind.py", "scripts/release/route_calib.py",
         "scripts/release/rl_calib.py", "scripts/data/real_layout.py", "src/devision/reading.py", "src/devision/infer.py",
         "src/devision/model.py", "src/devision/image.py", "src/devision/train/evaluate.py", "src/devision/train/rlcd.py",
         "src/devision/train/cli.py", "scripts/exp53/watch.py"]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rb = _load("calib_rebind", "scripts/release/rebind.py")
rc = _load("calib_route", "scripts/release/route_calib.py")
PINS = None                                     # None: the release's own LIVE-K pins (rebind.SOURCES["livek"]); fixtures set theirs


def sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path: str, obj: Any) -> None:
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1)


def register(out: str, stage: str, files: Dict[str, str]) -> None:
    path = os.path.join(out, "registry.json")
    reg = json.load(open(path)) if os.path.exists(path) else {}
    reg[stage] = files
    write_json(path, reg)


def registered(out: str, stage: str) -> Dict[str, str]:
    path = os.path.join(out, "registry.json")
    return (json.load(open(path)) if os.path.exists(path) else {}).get(stage, {})


def progress(out: str, stage: str, done: int, total: int) -> None:
    with open(os.path.join(out, "progress-%s.txt" % stage), "a") as fh:
        fh.write("%.3f %d %d\n" % (time.time(), done, total))


def cand() -> str:
    return os.path.join(ROUND, "cand")


def source_kwargs() -> Dict:
    return {"pins": PINS} if PINS is not None else {}


# ------------------------------------------------------------------ start

def stage_start(deadline: int, out: str = OUT) -> Dict:
    for p in (out, ROUND, "data/rl_calib"):
        if os.path.exists(p):
            raise SystemExit("%s exists (fresh only)" % p)
    bad = rb.verify_source(LIVEK["run"], LIVEK["decision"], LIVEK["bundle"], PINS, "livek")
    if bad:
        raise SystemExit("the LIVE-K source does not verify: %s" % bad)
    n_val = sum(len(json.loads(x)["questions"]) for x in open(VAL) if x.strip())
    if n_val != VAL_QUESTIONS:
        raise SystemExit("%s holds %d questions, not %d" % (VAL, n_val, VAL_QUESTIONS))
    os.makedirs(out)
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=ROOT).stdout.strip()  # noqa: E731
    st = {"git_head": git("rev-parse", "HEAD"), "dirty_files": [f for f in FILES if git("status", "--porcelain", "--", f)],
          "files": {f: sha(os.path.join(ROOT, f)) for f in FILES}, "source": {"kind": "livek", **LIVEK, "pins": PINS or rb.SOURCES["livek"]},
          "val": {"path": VAL, "sha256": sha(VAL), "questions": n_val}, "rl_calib": {"want": RL_WANT, "pictures": PICTURES},
          "release_name": RELEASE_NAME, "candidate_eligible": False, "started": time.strftime("%F %T"), "deadline": deadline,
          "deadline_text": time.strftime("%F %T", time.localtime(deadline))}
    write_json(os.path.join(out, "START.json"), st)
    os.makedirs(ROUND)
    write_json(os.path.join(ROUND, "run.json"), {"name": "v04-k", "purpose": "the LIVE-K combination's calibration (V04-K PREP-02)"})
    rb.source(LIVEK["run"], LIVEK["decision"], LIVEK["bundle"], ROUND, kind="livek", **source_kwargs())
    rb.clone(ROUND, cand(), PINS)
    register(out, "start", {"START.json": sha(os.path.join(out, "START.json")),
                            "source.json": sha(os.path.join(rb.lineage_dir(cand()), "source.json")),
                            "config.json": sha(os.path.join(cand(), "config.json"))})
    return st


def check_start(out: str = OUT, code: bool = True) -> Dict:
    st = json.load(open(os.path.join(out, "START.json")))
    if registered(out, "start").get("START.json") != sha(os.path.join(out, "START.json")):
        raise SystemExit("START.json is not the registered one")
    if code:
        bad = [f for f, h in st["files"].items() if sha(os.path.join(ROOT, f)) != h]
        if bad:
            raise SystemExit("code differs from START: %s" % bad)
    if sha(VAL) != st["val"]["sha256"]:
        raise SystemExit("%s changed" % VAL)
    rec = json.load(open(os.path.join(rb.lineage_dir(cand()), "source.json")))
    bad = rb.check_source(rec, PINS) + rb.candidate_problems(cand(), rec)
    if bad:
        raise SystemExit("source / candidate: %s" % bad)
    return st


# ------------------------------------------------------------------ calibrate

def stage_calibrate(out: str = OUT, device: str = "auto") -> Dict:
    from devision.train import cli, rlcd
    st = check_start(out)
    if registered(out, "calibrate"):
        raise SystemExit("already calibrated")
    if sha(os.path.join(cand(), "config.json")) != registered(out, "start")["config.json"]:
        raise SystemExit("the candidate's config changed before calibration")
    n_items, seen = st["val"]["questions"], [0]
    real = rlcd._forward

    def counted(model, batch, pixels, dev, *a, **k):            # a progress line per forward chunk (watch's stall rule)
        res = real(model, batch, pixels, dev, *a, **k)
        seen[0] += int(batch["input_ids"].shape[0])
        progress(out, "calibrate", min(seen[0], n_items), n_items)
        return res
    rlcd._forward = counted
    try:
        cli.calibrate_main(["--checkpoint", cand(), "--data", VAL, "--data-root", "data", "--in-place", "--device", device])
    finally:
        rlcd._forward = real
    shutil.move(os.path.join(cand(), "calibrate_report.json"), os.path.join(out, "calibrate_report.json"))
    cfg_path = os.path.join(cand(), "config.json")
    c = json.load(open(cfg_path))
    c["model_name"] = RELEASE_NAME
    write_json(cfg_path, c)
    rec = json.load(open(os.path.join(rb.lineage_dir(cand()), "source.json")))
    diff = rb.config_diff(json.load(open(os.path.join(rec["decision"], "config.json"))), c)
    bad = rb.candidate_problems(cand(), rec) + (["config changes %s beyond temperatures / name" % diff]
                                                if set(diff) - {"temperature", "temperature_by_options", "model_name"} else [])
    if bad:
        raise SystemExit("the calibrated candidate is not the source's: %s" % bad)
    register(out, "calibrate", {"calibrate_report.json": sha(os.path.join(out, "calibrate_report.json")), "config.json": sha(cfg_path)})
    return json.load(open(os.path.join(out, "calibrate_report.json")))["after"]


def build_rl_calib() -> Dict:
    rl = _load("calib_rl", "scripts/release/rl_calib.py")
    return rl.build("data", want=RL_WANT)


def stage_rl_calib(out: str = OUT) -> Dict:
    check_start(out)
    man = build_rl_calib()
    total = sum(RL_WANT.values())
    if man["families"] != RL_WANT or man["pictures"] != 2 * total or man["overlap_with_used"] or man["partition"]["calib_cases_on_confirm_side"]:
        raise SystemExit("rl_calib incomplete or not isolated: %s" % {k: man[k] for k in ("families", "pictures", "overlap_with_used")})
    register(out, "rl_calib", {"MANIFEST.json": sha("data/rl_calib/MANIFEST.json"), "questions.jsonl": sha("data/rl_calib/questions.jsonl")})
    return {k: man[k] for k in ("families", "pictures", "magnet_base_families", "magnet_view_axes")}


def bundle_dir() -> str:
    return os.path.join(ROUND, "meta", "calib_bundle")


def stage_temp_bundle(out: str = OUT) -> Dict:
    check_start(out)
    if registered(out, "calibrate").get("config.json") != sha(os.path.join(cand(), "config.json")):
        raise SystemExit("the candidate's config is not the calibrated one")
    tb = rb.temp_bundle(cand(), bundle_dir(), PINS)
    register(out, "temp_bundle", {"reading.json": sha(os.path.join(bundle_dir(), "reading.json"))})
    return tb


def stage_collect(out: str = OUT, device: str = "auto") -> Dict:
    from devision import reading as R
    from devision.train.evaluate import runtime_code
    check_start(out)
    for s in ("calibrate", "rl_calib", "temp_bundle"):
        if not registered(out, s):
            raise SystemExit("stage %s not done" % s)
    rows_path = "data/rl_calib/questions.jsonl"
    if sha(rows_path) != registered(out, "rl_calib")["questions.jsonl"]:
        raise SystemExit("rl_calib changed")
    ident = rc.identity(cand(), bundle_dir(), rows_path, "data", PINS, ROOT)          # before anything is loaded
    reading = R.Reading.load(cand(), bundle_dir(), device=device, full_rules=True)
    rows = [json.loads(x) for x in open(rows_path) if x.strip()]
    total = len(rows)
    n, real = [0], type(reading).decide

    def counted(self, *a, **k):
        res = real(self, *a, **k)
        n[0] += 1
        progress(out, "collect", min(n[0], total), total)
        return res
    type(reading).decide = counted
    try:
        recs = rc.collect(reading, rows, "data")
    finally:
        type(reading).decide = real
    calib = os.path.join(out, "calib")
    rc.register(calib, ident, recs, {"candidate": cand(), "bundle": bundle_dir(), "rows": rows_path, "data_root": "data"}, runtime_code(), ROOT)
    register(out, "collect", {"calib/CALIB.json": sha(os.path.join(calib, "CALIB.json")), "calib/records.jsonl": sha(os.path.join(calib, "records.jsonl"))})
    return {"records": len(recs)}


def stage_analyse(out: str = OUT) -> Dict:
    check_start(out)
    res = rc.analyse_registered(os.path.join(out, "calib"), PINS, ROOT)
    write_json(os.path.join(out, "route_calibration.json"), res)
    register(out, "analyse", {"route_calibration.json": sha(os.path.join(out, "route_calibration.json"))})
    return {"branch": res["branch"]}


# ------------------------------------------------------------------ report

STAGES = ("start", "calibrate", "rl_calib", "temp_bundle", "collect", "analyse")


def state_of(out: str, stage: str) -> str:
    files = registered(out, stage)
    if not files:
        return "missing"
    for f, h in files.items():
        p = os.path.join(out, f) if not f.endswith(("MANIFEST.json", "questions.jsonl")) else os.path.join("data", "rl_calib", f)
        if f == "config.json":
            p = os.path.join(cand(), f)
        if f == "source.json":
            p = os.path.join(rb.lineage_dir(cand()), f)
        if f == "reading.json":
            p = os.path.join(bundle_dir(), f)
        if not os.path.exists(p) or sha(p) != h:
            if f == "config.json" and stage == "start":
                continue                                          # calibration rewrites it by design
            return "invalid: %s changed" % f
    return "completed"


def report(out: str = OUT, finalize: str = "") -> Dict:
    res: Dict[str, Any] = {"states": {}, "candidate_eligible": False}
    if finalize:
        res["finalized"] = {"exit_codes": finalize, "note": "missing items are not low scores"}
    if not os.path.exists(os.path.join(out, "START.json")):
        res["branch"] = "incomplete: no START (nothing ran)"
        return write_report(out, res)
    try:
        check_start(out, code=False)
    except (SystemExit, OSError, ValueError, KeyError) as e:
        res["branch"] = "invalid: the run's identity does not verify (%s) -- no verdict" % e
        return write_report(out, res)
    S = res["states"]
    for s in STAGES:
        S[s] = state_of(out, s)
    if S["calibrate"] == "completed":
        r = json.load(open(os.path.join(out, "calibrate_report.json")))
        res["ordinary"] = {"before": r["before"], "after": r["after"], "items": r.get("items")}
    if S["analyse"] == "completed":
        try:
            again = rc.analyse_registered(os.path.join(out, "calib"), PINS, ROOT)        # recomputed, must equal what was written
            written = json.load(open(os.path.join(out, "route_calibration.json")))
            if json.loads(json.dumps(again)) != written:
                raise SystemExit("route_calibration.json differs from a recomputation")
            res["routed"] = {k: written[k] for k in ("branch", "pooled", "domains", "routed_temperatures_out_of_fold") if k in written}
            if "routed_temperature_all_data" in written:
                res["routed"]["routed_temperature_all_data"] = written["routed_temperature_all_data"]
        except (SystemExit, OSError, ValueError, KeyError) as e:
            S["analyse"] = "invalid: %s" % e
    bad = [k for k, v in S.items() if v != "completed"]
    if bad:
        kind = "invalid" if any(v.startswith("invalid") for v in S.values()) else "incomplete"
        res["branch"] = "%s: %s -- no calibration verdict" % (kind, bad)
    else:
        b = res["routed"]["branch"]
        nxt = {"1": "the temperature_by_route product change goes to review before it is written; then bind -> freeze",
               "2": "keep the ordinary temperatures; next (separately permitted) bind -> freeze",
               "3": "calibration fails on the routed path: the current temperatures are not releasable there; the next choice is ruled separately",
               "4": "routed-path calibration not validated: nothing released on it; ruled separately"}[b[0]]
        res["branch"] = "%s -- next: %s; not a candidate" % (b, nxt)
    return write_report(out, res)


def write_report(out: str, res: Dict) -> Dict:
    os.makedirs(out, exist_ok=True)
    write_json(os.path.join(out, "report.json"), res)
    text = json.dumps({k: res[k] for k in ("finalized", "states", "branch", "candidate_eligible") if k in res}, indent=1, default=str)
    open(os.path.join(out, "report.txt"), "w").write(text + "\n")
    print(text)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=list(STAGES) + ["report"])
    ap.add_argument("--deadline", type=int, default=0)
    ap.add_argument("--finalize", default="")
    a = ap.parse_args(argv)
    os.chdir(ROOT)
    fn = {"calibrate": stage_calibrate, "rl_calib": stage_rl_calib, "temp_bundle": stage_temp_bundle, "collect": stage_collect,
          "analyse": stage_analyse}
    if a.stage == "start":
        print(json.dumps({k: v for k, v in stage_start(a.deadline).items() if k != "files"}, indent=1, default=str))
    elif a.stage == "report":
        try:
            report(finalize=a.finalize)
        except Exception as e:
            os.makedirs(OUT, exist_ok=True)
            open(os.path.join(OUT, "report.txt"), "w").write("REPORT FAILED (invalid, no verdict): %s\n" % e)
            print("REPORT FAILED: %s" % e, file=sys.stderr)
            return 1
    else:
        print(json.dumps(fn[a.stage](), indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
