"""exp70: raw table vs one explicit K = m * v^2 block on the frozen EXP68-X (critic EXP70-PREP-01 / -PLAN-ACCEPT-01 /
-IMPL-PREP-01; docs/experiments/2026-10-19-exp70-intermediate-plan.md). Nothing is trained. EXP68-X's frozen end (2e5b272e /
87f82281), ordinary decide, image=None, raw temperature 1. candidate_eligible is always false.

  start    model-free: identities (EXP68's run / registry / X's FROZEN end and registered x_diag68, data/exp68, data/exp70,
           code) -> runs/exp70/START.json.
  decide   MPS, one load of X: first the 48 raw fidelity requests, which must reproduce EXP68's registered x_diag68 records
           (same prediction, every probability within 1e-4) -- otherwise everything on disk is registered and the stage
           stops before any other request; then diag68_K (480), train_raw (120), train_K (120). Every set is written and
           registered as soon as it is done.
  report   model-free: validity first (identities, the four sets and the reused raw records registered / bound / complete,
           fidelity recomputed); then per class raw vs K on the new diagnostic (accuracy, both views, near / large), the
           paired K - raw conflict macro and per conflict class (95 % percentile interval of 2,000 paired resamples of
           the combination families -- both views together -- seed 0: EXP68's conflict_difference / r41.interval), the
           training slice raw / K; the pre-written branch.

    uv run python scripts/exp70/exp70.py start --deadline EPOCH | decide | report [--finalize CODES]
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(__file__)
OUT, E68, X_RUN = "runs/exp70", "runs/exp68", "runs/exp68-x"
D70 = "data/exp70"
X_SHA = {"model.safetensors": "2e5b272e58a049e2633ad811ab28b2875006391acd86585078453eb54f154875",
         "config.json": "87f82281efd20b04d20f6ec626dad94913226e80b28acd5342bb4b06769e3f5e"}
E68_RUN = {"START.json": "94c7632b32e2bbb2dbd959e4a9018e551470e936eae2451ad2707b46a0e0dc8d",
           "report.json": "7178c2405ff32e256172a12b380b0a69376981e5d748d7bdfdafc3e1201932c6",
           "registry.json": "fe7a5225f26fc61053a2389da1c71d18b7b99b7b7bc489c0b3529a20228a5e93",
           "details_x_diag68.jsonl": "605b0e2e3ebbcea0e5bdf70a25b1fc88561a7e2e6e6ee76becc516ae1c6024ae"}
D70_SHA = {"diag68_K.jsonl": "065c11eaf06d34e3aa9d2c16cfa82ac944d84f0ed39a9c76b8edda67659dea2c",
           "fid_raw.jsonl": "c11ebabd26b9b25c9923244c801fdceab1bcddf54d540001a20982781578e8ca",
           "train_raw.jsonl": "1e85f1aa21dd5d257838e6f9a43801757a2418b56adcb11f057e16afbbd5cad5",
           "train_K.jsonl": "efb3c472403ac380d345ce1c9654d1376dd5835f13ed741ab851ec81f4652fc8"}
SETS = ("fid_raw", "diag68_K", "train_raw", "train_K")
FID_TOL = 1e-4
FILES = ["scripts/exp70/exp70.py", "scripts/exp70/build.py", "scripts/exp70/run.sh", "scripts/exp68/exp68.py", "scripts/exp68/build.py",
         "scripts/exp65/exp65.py", "scripts/exp57/exp57.py", "scripts/exp54/exp54.py", "src/devision/infer.py", "src/devision/model.py",
         "scripts/exp53/watch.py", "scripts/v04b/freeze.py"]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, rel))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


e68 = _load("exp68", "../exp68/exp68.py")
b70 = _load("exp70_build", "build.py")
e65, e57, e54, fz, r41 = e68.e65, e68.e57, e68.e54, e68.fz, e68.r41
sha, write_json, register, jsonl = e68.sha, e68.write_json, e68.register, e68.jsonl
CLASSES, CONFLICT = e68.CLASSES, e68.CONFLICT


# ------------------------------------------------------------------ identities

def bound_e68() -> Dict:
    bad = [f for f, h in E68_RUN.items() if sha(os.path.join(E68, f)) != h]
    if bad:
        raise SystemExit("EXP68's %s changed" % bad)
    for files in json.load(open(os.path.join(E68, "registry.json"))).values():
        for f, h in files.items():
            if sha(os.path.join(E68, f)) != h:
                raise SystemExit("EXP68's registered %s changed" % f)
    if fz.check(X_RUN, [])["weights"] != X_SHA or fz.weights(X_RUN) != X_SHA:
        raise SystemExit("EXP68-X's end is not 2e5b272e / 87f82281")
    return {"run": dict(E68_RUN), "x_end": dict(X_SHA)}


def bound_d70() -> Dict[str, str]:
    m = json.load(open(os.path.join(D70, "MANIFEST.json")))
    if m["sha256"] != D70_SHA or any(sha(os.path.join(D70, f)) != h for f, h in D70_SHA.items()):
        raise SystemExit("data/exp70 is not the accepted data")
    if m["sources"]["exp68_diag.jsonl"] != sha("data/exp68/diag.jsonl") or m["sources"]["exp68_train.jsonl"] != sha("data/exp68/train.jsonl"):
        raise SystemExit("data/exp70 was built on another data/exp68")
    return {"manifest": sha(os.path.join(D70, "MANIFEST.json")), **D70_SHA}


def stage_start(deadline: int, out: str = OUT) -> Dict:
    path = os.path.join(out, "START.json")
    if os.path.exists(path):
        raise SystemExit("%s exists (fresh only)" % path)
    from devision.reading import tree_sha
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    s = {"git_head": git("rev-parse", "HEAD"), "dirty_files": [f for f in FILES if git("status", "--porcelain", "--", f)],
         "files": {f: sha(f) for f in FILES}, "exp68": bound_e68(), "exp68_data": e68.bound_d68(), "data": bound_d70(),
         "tokenizer": tree_sha(os.path.join(X_RUN, "stage2", "tokenizer")),
         "requests": {n: len(jsonl(os.path.join(D70, n + ".jsonl"))) for n in SETS},
         "decide": {"image": None, "temperature": "none (raw)", "fidelity_tolerance": FID_TOL},
         "candidate_eligible": False, "started": time.strftime("%F %T"), "deadline": deadline, "deadline_text": time.strftime("%F %T", time.localtime(deadline))}
    os.makedirs(out, exist_ok=True)
    write_json(path, s)
    return s


def check_start(out: str = OUT, code: bool = True) -> Dict:
    from devision.reading import tree_sha
    st = json.load(open(os.path.join(out, "START.json")))
    if code:
        bad = [f for f, h in st["files"].items() if sha(f) != h]
        if bad:
            raise SystemExit("code differs from START: %s" % bad)
    if bound_e68() != st["exp68"] or e68.bound_d68() != st["exp68_data"] or bound_d70() != st["data"]:
        raise SystemExit("EXP68 / data/exp68 / data/exp70 differ from START")
    if tree_sha(os.path.join(X_RUN, "stage2", "tokenizer")) != st["tokenizer"]:
        raise SystemExit("EXP68-X's tokenizer differs from START")
    return st


# ------------------------------------------------------------------ decide

def inputs_of(name: str):
    rows = {r["id"]: r for r in jsonl(os.path.join(D70, name + ".jsonl"))}
    return rows, {rid: {"state": r["state_text"]} for rid, r in rows.items()}


def raw_registered() -> List[Dict]:
    """EXP68's registered x_diag68 records (X on the raw new diagnostic), after EXP68's own checks."""
    rows, inputs = e68.inputs_of("diag68")
    return e57.read_set(E68, "x_diag68", rows, X_SHA, inputs)


def fidelity(new: List[Dict], ref: List[Dict]) -> Dict:
    """The re-decided raw requests against the registered ones: same prediction, every probability within FID_TOL."""
    by = {(r["sample_id"], r["qid"]): r for r in ref}
    bad, worst = [], 0.0
    for r in new:
        o = by.get((r["sample_id"], r["qid"]))
        if o is None or o["prediction"] != r["prediction"] or set(o["probabilities"]) != set(r["probabilities"]):
            bad.append(r["sample_id"])
            continue
        d = max(abs(o["probabilities"][k] - r["probabilities"][k]) for k in r["probabilities"])
        worst = max(worst, d)
        if d > FID_TOL:
            bad.append(r["sample_id"])
    return {"n": len(new), "mismatches": bad, "max_abs_probability_difference": worst, "reproduced": not bad and len(new) > 0}


def write_fidelity(out: str, fid: Dict) -> None:
    write_json(os.path.join(out, "fidelity.json"), fid)
    register(out, "fidelity", {"fidelity.json": sha(os.path.join(out, "fidelity.json"))})


def stage_decide(out: str = OUT, device: str = "auto", decider=None) -> Dict:
    from devision.infer import Decider, best_device
    st = check_start(out)
    s_sha = sha(os.path.join(out, "START.json"))
    ref = raw_registered()
    dev = best_device() if device == "auto" else device
    if decider is None:
        if fz.weights(X_RUN) != X_SHA:                                       # before the load
            raise SystemExit("EXP68-X's weights are not START's")
        decider = Decider.load(os.path.join(X_RUN, "stage2"), device=dev)
    d = decider
    d.cfg.temperature, d.cfg.temperature_by_options = [1.0, 1.0, 1.0], {}
    total = sum(st["requests"].values())
    done, t0 = 0, time.time()
    for name in SETS:
        rows, inputs = inputs_of(name)
        recs = e54.decide_condition(d, rows, inputs, on_request=lambda n, b=done: e54.progress(out, "decide", b + n, total))
        done += len(rows)
        e57.write_set(out, name, recs, {"set": name, "model_of": "x", "start_sha256": s_sha, "inputs_sha256": e57.hash_inputs(inputs),
                                        "model": X_SHA, "temperature": "none (raw)", "image": None, "device": str(dev)})
        if name == "fid_raw":
            fid = fidelity(recs, ref)
            write_fidelity(out, fid)
            if not fid["reproduced"]:
                raise SystemExit("raw fidelity failed (%d of %d; max |dp| %.2e) -- no other request" % (len(fid["mismatches"]), fid["n"], fid["max_abs_probability_difference"]))
        print("%s done after %.1f s" % (name, time.time() - t0), flush=True)
    return {"seconds": time.time() - t0}


# ------------------------------------------------------------------ report

def class_difference(k: List[Dict], raw: List[Dict], rows: Dict[str, Dict], cls: str) -> Dict:
    """K - raw on one class, the combination families resampled (both views together; r41.interval, seed 0, 2,000)."""
    oc = {(r["sample_id"], r["qid"]): r for r in raw}
    fam: Dict[str, List[float]] = defaultdict(list)
    for r in k:
        row = rows[r["sample_id"]]
        if row["axis"] == cls:
            fam[row["family"]].append(float(r["correct"]) - float(oc[(r["sample_id"], r["qid"])]["correct"]))
    return r41.interval(lambda ks: sum(x for f in ks for x in fam[f]) / sum(len(fam[f]) for f in ks), sorted(fam), lambda ks: list(ks))


def by_class(recs: List[Dict], rows: Dict[str, Dict]) -> Dict[str, Optional[float]]:
    acc: Dict[str, List[bool]] = defaultdict(list)
    for r in recs:
        acc[rows[r["sample_id"]]["axis"]].append(bool(r["correct"]))
    return {c: (sum(acc[c]) / len(acc[c]) if acc[c] else None) for c in CLASSES}


def branch(res: Dict) -> str:
    d, kc = res["diag"]["k_minus_raw_conflict"], res["diag"]["k"]["accuracy"]
    lo, hi = d["ci95"]
    if lo > 0 and all((kc[c] or 0) >= 0.80 for c in CONFLICT):
        return ("the external intermediate quantity helps this fixed diagnostic's decision (K - raw lower bound > 0, both conflict "
                "classes >= 0.80 with K); not that the model learned multiplication / squaring; not a release capability")
    if lo > 0:
        return "partial help: K - raw lower bound > 0, but %s stays below 0.80 with K" % [c for c in CONFLICT if (kc[c] or 0) < 0.80]
    if hi < 0:
        return "the K block as given disturbs the decision (upper bound < 0; possibly out of distribution) -- no evidence against intermediate representations in general"
    return "not located: the interval covers 0; K's format not being used or the comparator not adapted remain possible -- no claim about the architecture"


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
    recs: Dict[str, List[Dict]] = {}
    try:
        recs["raw_diag68"] = raw_registered()
        res["states"]["raw_diag68"] = "completed (EXP68's registered x_diag68)"
    except (SystemExit, OSError, ValueError, KeyError) as e:
        res["states"]["raw_diag68"] = "invalid: %s" % e
    for name in SETS:
        rows, inputs = inputs_of(name)
        try:
            recs[name] = e57.read_set(out, name, rows, X_SHA, inputs)
            res["states"][name] = "completed"
        except (SystemExit, OSError, ValueError, KeyError) as e:
            res["states"][name] = e68.read_state(out, name, e)
    if "fid_raw" in recs and "raw_diag68" in recs:                          # recomputed here, never a stored flag
        res["fidelity"] = fidelity(recs["fid_raw"], recs["raw_diag68"])
        if not res["fidelity"]["reproduced"]:
            res["states"]["fidelity"] = "invalid: raw fidelity not reproduced"
    bad = [k for k, v in res["states"].items() if v.startswith("invalid")]
    miss = [k for k, v in res["states"].items() if v == "missing"]
    if bad or miss:
        res["validity"] = "invalid" if bad else "incomplete"
        res["branch"] = "%s: %s -- no verdict" % (res["validity"], bad or miss)
        return write_report(out, res)
    res["validity"] = "valid"
    drows = inputs_of("diag68_K")[0]
    raw_rows = e68.inputs_of("diag68")[0]
    if set(drows) != set(raw_rows):
        raise SystemExit("the K and raw diagnostics are not the same requests")
    res["diag"] = {"raw": e68.diag68_summary(recs["raw_diag68"], raw_rows), "k": e68.diag68_summary(recs["diag68_K"], drows),
                   "k_minus_raw_conflict": e65.conflict_difference(recs["diag68_K"], recs["raw_diag68"], drows),
                   "k_minus_raw_class": {c: class_difference(recs["diag68_K"], recs["raw_diag68"], drows, c) for c in CLASSES},
                   "interval": "95 % percentile interval of 2,000 paired resamples of the combination families (both views together), seed 0"}
    trows = inputs_of("train_raw")[0]
    res["train_slice"] = {"raw": by_class(recs["train_raw"], trows), "k": by_class(recs["train_K"], inputs_of("train_K")[0]), "underfit_note": None}
    low = [c for c in CONFLICT if (res["train_slice"]["raw"][c] or 0) < 0.90]
    res["train_slice"]["underfit_note"] = ("raw < 0.90 on its own training combinations for %s: under-fitting stays an alternative explanation" % low) if low else None
    res["branch"] = branch(res)
    res["scope"] = ("a fixed development diagnostic; the four raw numbers stay in the input, so a result does not locate one internal "
                    "defect uniquely; not a candidate")
    return write_report(out, res)


def write_report(out: str, res: Dict) -> Dict:
    os.makedirs(out, exist_ok=True)
    write_json(os.path.join(out, "report.json"), res)
    lines = [json.dumps({k: res[k] for k in ("finalized", "states", "validity", "fidelity", "branch", "candidate_eligible") if k in res}, indent=1, default=str)]
    if "diag" in res:
        lines.append("raw %s" % res["diag"]["raw"]["accuracy"])
        lines.append("K   %s" % res["diag"]["k"]["accuracy"])
        g = res["diag"]["k_minus_raw_conflict"]
        lines.append("K - raw conflict macro %+.3f [%+.3f, %+.3f]" % (g["value"], *g["ci95"]))
        lines.append("train slice raw %s / K %s" % (res["train_slice"]["raw"], res["train_slice"]["k"]))
    open(os.path.join(out, "report.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["start", "decide", "report"])
    ap.add_argument("--deadline", type=int, default=0)
    ap.add_argument("--finalize", default="")
    a = ap.parse_args(argv)
    if a.stage == "start":
        print(json.dumps({k: v for k, v in stage_start(a.deadline).items() if k != "files"}, indent=1))
    elif a.stage == "decide":
        print(json.dumps(stage_decide(), indent=1))
    else:
        try:
            report(finalize=a.finalize)
        except Exception as e:
            os.makedirs(OUT, exist_ok=True)
            open(os.path.join(OUT, "report.txt"), "w").write("REPORT FAILED (invalid, no verdict): %s\n" % e)
            print("REPORT FAILED: %s" % e, file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
