"""exp72: raw vs K arms trained from the same EXP68-X end with the same comparison-led plan (critic EXP72-PREP-01 /
-R1-01; docs/experiments/2026-10-19-exp72-k-supervision-plan.md). Both arms: EXP68-X's frozen end (2e5b272e / 87f82281) --
its decision head and merged earlier LoRA as they are, a fresh LoRA r16 (zero-initialised B) trained with that head --
EXP68-X's recipe, 475 updates, last step; the only difference is the K block on temperature states. Raw temperature 1.
candidate_eligible is always false.

  start     model-free: identities (EXP68 run / X's end, data/exp68, data/exp70, data/exp72 (MANIFEST, plan audit), EXP63 /
            EXP57 screens, the two YAMLs, code) -> runs/exp72/START.json.
  fid       MPS: EXP68-X on EXP70's 48 raw fidelity requests -- must reproduce EXP68's registered records (else stop).
  train_raw / train_k   MPS: devision-pipeline configs/exp72-<arm>.yaml.
  freeze_raw / freeze_k tensors only: 475 updates; the export against EXP68-X's actual end; FROZEN.json.
  decide    MPS, required sets first, one arm at a time: training slice (240), new diagnostic (480), G (600 requests; the
            1,800 numeric reading questions are the reading gate).
  (exist / science per arm: devision-eval, recorded under the arm's frozen identity -- required.)
  decide_fit  MPS, last and lowest priority: the old facts fit (400 requests) per arm, reported only.
  report    model-free: per arm the fit check, the diagnostic, G's reading 'all' (EXP63's metric, 1,800 questions),
            protection against EXP63; K - raw on the conflict classes; the pre-written route (route()).

    uv run python scripts/exp72/exp72.py start --deadline EPOCH | fid | train_<arm> | freeze_<arm> | decide | decide_fit | report
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
OUT = "runs/exp72"
ARMS = {"raw": "runs/exp72-raw", "k": "runs/exp72-k"}
YAML = {a: "configs/exp72-%s.yaml" % a for a in ARMS}
D72 = "data/exp72"
UPDATES = 475
SETS = {"raw": {"slice": "data/exp72/slice_raw.jsonl", "diag": "data/exp68/diag.jsonl", "G": "data/facts/tune_G.jsonl", "fit": "data/facts/fit.jsonl"},
        "k": {"slice": "data/exp72/slice_K.jsonl", "diag": "data/exp70/diag68_K.jsonl", "G": "data/exp72/G_K.jsonl", "fit": "data/exp72/fit_old_K.jsonl"}}
REQUIRED = ("slice", "diag", "G")
G_READING_QUESTIONS, G_LINE, FIT_LINE, CONFLICT_LINE, SCREEN_TOL = 1800, 0.98, 0.90, 0.80, -0.02
FILES = ["scripts/exp72/exp72.py", "scripts/exp72/build.py", "scripts/exp72/run.sh", YAML["raw"], YAML["k"], "scripts/exp70/exp70.py",
         "scripts/exp70/build.py", "scripts/exp68/exp68.py", "scripts/exp65/exp65.py", "scripts/exp63/exp63.py", "scripts/exp57/exp57.py",
         "scripts/exp54/exp54.py", "src/devision/train/rlcd.py", "src/devision/train/pipeline.py", "src/devision/infer.py",
         "src/devision/model.py", "scripts/exp53/watch.py", "scripts/v04b/freeze.py"]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, rel))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


e70 = _load("exp70", "../exp70/exp70.py")
e68, e65, e63, e57, e54, fz, r41 = e70.e68, e70.e65, e70.e65.e63, e70.e57, e70.e54, e70.fz, e70.r41
sha, write_json, register, jsonl = e70.sha, e70.write_json, e70.register, e70.jsonl
CLASSES, CONFLICT = e70.CLASSES, e70.CONFLICT
X_RUN, X_SHA = e70.X_RUN, e70.X_SHA


# ------------------------------------------------------------------ identities

def bound_d72() -> Dict[str, str]:
    m = json.load(open(os.path.join(D72, "MANIFEST.json")))
    bad = [f for f, h in m["sha256"].items() if sha(os.path.join(D72, f)) != h]
    if bad or not m["plan"]["plans_equal"] or m["plan"]["raw"]["updates"] != UPDATES:
        raise SystemExit("data/exp72 differs from its MANIFEST / plan audit: %s" % bad)
    src = {"exp68_train": "data/exp68/train.jsonl", "exp68_diag": "data/exp68/diag.jsonl", "exp70_diag68_K": "data/exp70/diag68_K.jsonl",
           "facts_fit": "data/facts/fit.jsonl", "facts_tune_G": "data/facts/tune_G.jsonl"}
    if any(m["sources"][k] != sha(p) for k, p in src.items()):
        raise SystemExit("data/exp72 was built on other sources")
    return {"manifest": sha(os.path.join(D72, "MANIFEST.json")), **m["sha256"]}


def bound_exp63() -> Dict:
    """EXP63's protection baseline as it is actually used (critic EXP72-R2-01): exp65.bound_e63 (EXP63's run files and
    registry, its frozen end, its frozen exist / science records), which must also equal the EXP63 record in EXP68's
    START -- whose own sha256 is pinned (exp70.E68_RUN) -- so an internally consistent replacement is refused."""
    b = e65.bound_e63()
    if b != json.load(open(os.path.join(e70.E68, "START.json")))["exp63"]:
        raise SystemExit("EXP63's baseline is not the one EXP68's START bound")
    return b


def frozen_state(run: str):
    """(state, weights) of an arm's frozen end (critic EXP72-R2-02): FROZEN.json or a checkpoint file absent -> missing;
    present but truncated, malformed or with other sha256 -> invalid (the reason kept); else completed."""
    if not os.path.exists(os.path.join(run, "FROZEN.json")):
        return "missing: no FROZEN.json in %s" % run, None
    absent = [f for f in ("model.safetensors", "config.json") if not os.path.exists(os.path.join(run, "stage2", f))]
    if absent:
        return "missing: %s absent in %s/stage2" % (absent, run), None
    try:
        return "completed", fz.check(run, [])["weights"]
    except (SystemExit, ValueError, KeyError, TypeError, OSError) as e:
        return "invalid: frozen identity of %s: %s" % (run, e), None


def check_yamls() -> None:
    import yaml
    y = {a: yaml.safe_load(open(YAML[a])) for a in ARMS}
    x = yaml.safe_load(open("configs/exp68-x.yaml"))["stages"][0]["params"]
    for a, c in y.items():
        st = c["stages"][0]
        want_data = "data/exp72/train_%s.jsonl" % ("K" if a == "k" else "raw")
        want_val = "data/exp72/slice_%s.jsonl" % ("K" if a == "k" else "raw")
        if c["name"] != "exp72-" + a or len(c["stages"]) != 1 or c.get("evaluate") or st["init"] != "runs/exp68-x/stage2" \
                or st["data"] != want_data or st["val"] != want_val or st["params"] != x:
            raise SystemExit("%s is not EXP68-X's arm on its data" % YAML[a])
    norm = lambda c: {**{k: v for k, v in c.items() if k not in ("name", "stages")},  # noqa: E731
                      "stage": {k: v for k, v in c["stages"][0].items() if k not in ("data", "val")}}
    if norm(y["raw"]) != norm(y["k"]):
        raise SystemExit("the arms differ in something else than name / data / val")


def check_pairs() -> None:
    """Every raw / K readout file pair: same ids, questions and golds (the K file differs only in temperature states)."""
    for n in ("slice", "diag", "G", "fit"):
        a = {r["id"]: r for r in jsonl(SETS["raw"][n])}
        b = {r["id"]: r for r in jsonl(SETS["k"][n])}
        if set(a) != set(b) or any(a[i]["questions"] != b[i]["questions"] or a[i]["gold"] != b[i]["gold"] for i in a):
            raise SystemExit("the raw / K %s sets are not the same questions" % n)


def stage_start(deadline: int, out: str = OUT) -> Dict:
    path = os.path.join(out, "START.json")
    if os.path.exists(path):
        raise SystemExit("%s exists (fresh only)" % path)
    for run in ARMS.values():
        if os.path.exists(run):
            raise SystemExit("%s exists (fresh only)" % run)
    check_yamls()
    check_pairs()
    from devision.reading import tree_sha
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    s = {"git_head": git("rev-parse", "HEAD"), "dirty_files": [f for f in FILES if git("status", "--porcelain", "--", f)],
         "files": {f: sha(f) for f in FILES}, "exp63": bound_exp63(), "exp68": e70.bound_e68(), "exp68_data": e68.bound_d68(), "exp70_data": e70.bound_d70(),
         "data": bound_d72(), "sets": {a: {n: sha(p) for n, p in d.items()} for a, d in SETS.items()},
         "tokenizer": tree_sha(os.path.join(X_RUN, "stage2", "tokenizer")), "exp57_screens": fz.check(e63.E57, list(e57.SCREENS))["evals"],
         "exp49_questions": sha(e54.QUESTIONS), "exp54_inputs_P": sha(os.path.join(e54.OUT, "inputs_P.jsonl")),
         "requests": {a: {n: len(jsonl(p)) for n, p in d.items()} for a, d in SETS.items()}, "fid": 48,
         "init": "EXP68-X's end as it is (decision head and merged LoRA); a fresh LoRA r16 (zero-initialised B) trained with that head",
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
    if bound_exp63() != st["exp63"]:
        raise SystemExit("EXP63's protection baseline differs from START")
    if e70.bound_e68() != st["exp68"] or e68.bound_d68() != st["exp68_data"] or e70.bound_d70() != st["exp70_data"] or bound_d72() != st["data"]:
        raise SystemExit("EXP68 / data/exp68 / data/exp70 / data/exp72 differ from START")
    if {a: {n: sha(p) for n, p in d.items()} for a, d in SETS.items()} != st["sets"]:
        raise SystemExit("a readout set differs from START")
    if tree_sha(os.path.join(X_RUN, "stage2", "tokenizer")) != st["tokenizer"] or sha(e54.QUESTIONS) != st["exp49_questions"] \
            or sha(os.path.join(e54.OUT, "inputs_P.jsonl")) != st["exp54_inputs_P"]:
        raise SystemExit("tokenizer / exp49 questions / EXP54 P table differ from START")
    return st


# ------------------------------------------------------------------ fid / train / freeze / decide

def stage_fid(out: str = OUT, decider=None) -> Dict:
    """EXP68-X on EXP70's 48 raw fidelity requests, before any training: the loaded start is the frozen one."""
    from devision.infer import Decider, best_device
    st = check_start(out)
    ref = e70.raw_registered()
    if decider is None:
        if fz.weights(X_RUN) != X_SHA:
            raise SystemExit("EXP68-X's weights are not START's")
        decider = Decider.load(os.path.join(X_RUN, "stage2"), device=best_device())
    decider.cfg.temperature, decider.cfg.temperature_by_options = [1.0, 1.0, 1.0], {}
    rows, inputs = e70.inputs_of("fid_raw")
    recs = e54.decide_condition(decider, rows, inputs)
    e57.write_set(out, "fid", recs, {"set": "fid", "model_of": "exp68-x", "start_sha256": sha(os.path.join(out, "START.json")),
                                     "inputs_sha256": e57.hash_inputs(inputs), "model": X_SHA, "temperature": "none (raw)", "image": None})
    fid = e70.fidelity(recs, ref)
    write_json(os.path.join(out, "fidelity.json"), fid)
    register(out, "fidelity", {"fidelity.json": sha(os.path.join(out, "fidelity.json"))})
    if not fid["reproduced"]:
        raise SystemExit("EXP68-X's start does not reproduce EXP68's registered records -- no training")
    return {"fidelity": fid, "requests": st["fid"]}


def stage_train(arm: str, out: str = OUT, poll: float = 5.0, cmd: Optional[List[str]] = None) -> int:
    check_start(out)
    fid = json.load(open(os.path.join(out, "fidelity.json")))
    if e54.check_registered(out, "fidelity").get("fidelity.json") != sha(os.path.join(out, "fidelity.json")) or not fid["reproduced"]:
        raise SystemExit("the start's fidelity is not established -- no training")
    log = os.path.join(ARMS[arm], "logs", "stage2.log")
    proc = subprocess.Popen(cmd or ["uv", "run", "devision-pipeline", YAML[arm]])
    seen = -1
    while True:
        rc = proc.poll()
        if os.path.exists(log):
            n = e57.progress_of(open(log, errors="replace").read().splitlines())
            if n is not None and n != seen:
                e54.progress(out, "train_" + arm, n, UPDATES)
                seen = n
        if rc is not None:
            return rc
        time.sleep(poll)


def stage_freeze(arm: str, out: str = OUT) -> Dict:
    from safetensors.torch import load_file
    check_start(out)
    run = ARMS[arm]
    rep = json.load(open(os.path.join(run, "stage2", "train_report.json")))
    if len(rep["losses"]) != UPDATES:
        raise SystemExit("arm %s ran %d updates" % (arm, len(rep["losses"])))
    if fz.weights(X_RUN) != X_SHA:
        raise SystemExit("EXP68-X changed")
    res = e57.export_check(load_file(os.path.join(X_RUN, "stage2", "model.safetensors")), load_file(os.path.join(run, "stage2", "model.safetensors")))
    res.update(updates=len(rep["losses"]), baseline=dict(X_SHA))
    write_json(os.path.join(run, "export_check.json"), res)
    if not res["ok"]:
        raise SystemExit("arm %s export check failed: %s" % (arm, res))
    res["frozen"] = fz.freeze(run)["weights"]
    return res


def inputs_of(arm: str, name: str):
    rows = {r["id"]: r for r in jsonl(SETS[arm][name])}
    return rows, {rid: {"state": r["state_text"]} for rid, r in rows.items()}


def decide_sets(out: str, names: List[str], label: str, deciders: Optional[Dict] = None) -> Dict:
    from devision.infer import Decider, best_device
    from devision.reading import tree_sha
    st = check_start(out)
    s_sha = sha(os.path.join(out, "START.json"))
    total = sum(len(jsonl(SETS[a][n])) for a in ARMS for n in names)
    done, t0 = 0, time.time()
    for arm, run in ARMS.items():
        weights = fz.check(run, [])["weights"]
        if deciders is None:
            if fz.weights(run) != weights or tree_sha(os.path.join(run, "stage2", "tokenizer")) != st["tokenizer"]:     # before any load
                raise SystemExit("%s: weights / tokenizer not the bound ones" % arm)
            d = Decider.load(os.path.join(run, "stage2"), device=best_device())
        else:
            d = deciders[arm]
        d.cfg.temperature, d.cfg.temperature_by_options = [1.0, 1.0, 1.0], {}
        for n in names:
            rows, inputs = inputs_of(arm, n)
            recs = e54.decide_condition(d, rows, inputs, on_request=lambda k, b=done: e54.progress(out, label, b + k, total))
            done += len(rows)
            name = "%s_%s" % (arm, n)
            e57.write_set(out, name, recs, {"set": name, "model_of": arm, "start_sha256": s_sha, "inputs_sha256": e57.hash_inputs(inputs),
                                            "model": weights, "temperature": "none (raw)", "image": None})
            print("%s done after %.1f s" % (name, time.time() - t0), flush=True)
        del d
    return {"seconds": time.time() - t0}


# ------------------------------------------------------------------ report

def g_reading(recs: List[Dict], rows: Dict[str, Dict], qs53: Dict, p_right: Dict) -> Dict:
    """G's numeric reading gate: EXP63's metric (exp57.reading_summary 'all') over exactly the 1,800 reading questions,
    each once (the 600 comparisons are not in the denominator); also the plain share right."""
    from devision.reading import TEMPLATES
    rd = [r for r in recs if rows[r["sample_id"]]["questions"][r["qid"]]["instructions"] in TEMPLATES]
    keys = {(r["sample_id"], r["qid"]) for r in rd}
    if len(rd) != G_READING_QUESTIONS or len(keys) != len(rd):
        raise SystemExit("G reading: %d records, %d unique (want %d)" % (len(rd), len(keys), G_READING_QUESTIONS))
    s = e57.reading_summary(recs, rows, qs53, p_right)
    return {"all": s["all"], "n": len(rd), "plain": sum(r["correct"] for r in rd) / len(rd), "fields": {f: s.get("all:" + f) for f in ("mass", "speed", "distance")}}


def by_class(recs: List[Dict], rows: Dict[str, Dict]) -> Dict[str, Optional[float]]:
    return e70.by_class(recs, rows)


def route(evidence: str, arms: Dict[str, Dict], k_minus_raw: Optional[Dict]) -> str:
    """The pre-written route. arms[a] = {"fit": {class: acc}, "diag": {class: acc}, "g_all": float, "screens_ok": bool}."""
    if evidence != "completed":
        return "%s: required evidence is %s -- no route (the computed metrics are reported as they are)" % (evidence, evidence)
    k, raw = arms["k"], arms["raw"]
    fitted = {a: all((x["fit"].get(c) or 0) >= FIT_LINE for c in CLASSES) for a, x in arms.items()}
    if not fitted["k"]:
        return "the K arm did not fit its own training combinations (a class < 0.90): generalisation not located; no added budget"
    if (k["g_all"] or 0) < G_LINE or not k["screens_ok"]:
        return ("stop: the K arm's numeric reading on G (< 0.98) or protection (exist / science below EXP63 - 0.02) fails -- "
                "whatever the comparisons; not continued")
    if k_minus_raw is None or not (k_minus_raw["ci95"][0] > 0):
        return ("no K gain over the raw arm (conflict macro lower bound <= 0)%s: stop this fixed-representation recipe"
                % ("" if fitted["raw"] else "; the raw arm did not fit either"))
    if all((k["diag"].get(c) or 0) >= CONFLICT_LINE for c in CONFLICT):
        return ("the K arm learns to use the given K: K - raw > 0 and both conflict classes >= 0.80 with reading and protection kept; "
                "next (separate permission) the live combination -- P (live tables) and the full old reading gate are not verified; "
                "K is computed outside the model; not a candidate")
    return "partial: K - raw > 0 but %s stays below 0.80 with K; no added budget" % [c for c in CONFLICT if (k["diag"].get(c) or 0) < CONFLICT_LINE]


def report(out: str = OUT, finalize: str = "") -> Dict:
    res: Dict[str, Any] = {"states": {}, "arms": {}, "candidate_eligible": False}
    if finalize:
        res["finalized"] = {"exit_codes": finalize, "note": "missing items are not low scores"}
    if not os.path.exists(os.path.join(out, "START.json")):
        res["route"] = "incomplete: no START (nothing ran)"
        return write_report(out, res)
    try:
        st = check_start(out, code=False)
    except (SystemExit, OSError, ValueError, KeyError) as e:
        res["route"] = "invalid: the run's identity does not verify (%s) -- no verdict" % e
        return write_report(out, res)
    try:
        fid = e70.fidelity(e57.read_set(out, "fid", *(lambda r: (r[0], X_SHA, r[1]))(e70.inputs_of("fid_raw"))), e70.raw_registered())
        res["fidelity"] = fid
        res["states"]["fid"] = "completed" if fid["reproduced"] else "invalid: the start's fidelity not reproduced"
    except (SystemExit, OSError, ValueError, KeyError) as e:
        res["states"]["fid"] = e68.read_state(out, "fid", e)
    qs53 = {q["key"]: q for q in jsonl(e54.QUESTIONS) if q["set"] == "rl_tune"}
    pin = e54.load_inputs(e54.OUT, "P")
    tune = {r["id"]: r for r in jsonl(e54.TUNE)}
    p_right = {k: {(o, qn): v for o, qn, v in pin[q["id"]]["table"]}[e54.TEMPLATES[tune[q["id"]]["questions"][q["qid"]]["instructions"]]] == q["target"]
               for k, q in qs53.items()}
    e63_screens = {n: e57.screen_side(e65.E63, n, e57.SCREENS[n])[0] for n in e57.SCREENS}
    recs: Dict[str, Dict[str, List[Dict]]] = {a: {} for a in ARMS}
    metrics: Dict[str, Dict] = {}
    for arm, run in ARMS.items():
        a: Dict[str, Any] = {}
        res["arms"][arm] = a
        state, weights = frozen_state(run)
        if state != "completed":
            res["states"][arm] = state                                        # the other arm is still computed
            continue
        for n in REQUIRED + ("fit",):
            rows, inputs = inputs_of(arm, n)
            try:
                recs[arm][n] = e57.read_set(out, "%s_%s" % (arm, n), rows, weights, inputs)
                res["states"]["%s_%s" % (arm, n)] = "completed"
            except (SystemExit, OSError, ValueError, KeyError) as e:
                res["states"]["%s_%s" % (arm, n)] = e68.read_state(out, "%s_%s" % (arm, n), e)
        if "slice" in recs[arm]:
            a["fit_slice"] = by_class(recs[arm]["slice"], inputs_of(arm, "slice")[0])
        if "diag" in recs[arm]:
            a["diag"] = e68.diag68_summary(recs[arm]["diag"], inputs_of(arm, "diag")[0])
        if "G" in recs[arm]:
            try:
                a["G_reading"] = g_reading(recs[arm]["G"], inputs_of(arm, "G")[0], qs53, p_right)
            except SystemExit as e:
                res["states"]["%s_G" % arm] = "invalid: %s" % e
        if "fit" in recs[arm]:
            frows = inputs_of(arm, "fit")[0]
            a["fit_old"] = {kind: e63.summary(e63.comparisons(recs[arm]["fit"], frows, kind), kind) for kind in ("temp", "force")}
        a["screens"] = {}
        for n in e57.SCREENS:
            try:
                sc = e57.screen(run, n, st["exp57_screens"][n], start_run=e63.E57)
                a["screens"][n] = {"new": sc["new"], "exp63": e63_screens[n], "difference": sc["new"] - e63_screens[n]}
                res["states"]["%s_%s" % (arm, n)] = "completed"
            except (SystemExit, OSError, ValueError, KeyError) as e:
                res["states"]["%s_%s" % (arm, n)] = "missing / invalid: %s" % e if os.path.exists(os.path.join(run, "eval", n + ".json")) else "missing"
        a["screens_ok"] = None if len(a["screens"]) < len(e57.SCREENS) else all(x["difference"] >= SCREEN_TOL for x in a["screens"].values())
        if all(x in a for x in ("fit_slice", "diag", "G_reading")) and a["screens_ok"] is not None:
            metrics[arm] = {"fit": a["fit_slice"], "diag": a["diag"]["accuracy"], "g_all": a["G_reading"]["all"], "screens_ok": a["screens_ok"]}
    kmr = None
    if all("diag" in recs[a] for a in ARMS):
        drows = inputs_of("k", "diag")[0]
        kmr = e65.conflict_difference(recs["k"]["diag"], recs["raw"]["diag"], drows)
        res["k_minus_raw"] = {"conflict_macro": kmr, "by_class": {c: e70.class_difference(recs["k"]["diag"], recs["raw"]["diag"], drows, c) for c in CLASSES},
                              "interval": "95 % percentile interval of 2,000 paired resamples of the combination families (both views together), seed 0"}
    required = ["fid"] + ["%s_%s" % (a, n) for a in ARMS for n in REQUIRED] + ["%s_%s" % (a, n) for a in ARMS for n in e57.SCREENS]
    states = [res["states"].get(k, "missing") for k in required] + [res["states"].get(a, "completed") for a in ARMS]
    evidence = "completed" if all(s == "completed" for s in states) else "invalid" if any(s.startswith("invalid") or s.startswith("missing / invalid") for s in states) else "incomplete"
    res["evidence"] = evidence
    if evidence == "completed" and len(metrics) != len(ARMS):
        evidence = "incomplete"
    res["evidence"] = evidence
    res["route"] = route(evidence, metrics, kmr)
    res["limits"] = ("P (EXP56's live tables) is not verified -- the full old reading gate is not shown, only G (gold tables); the "
                     "diagnostic is a repeatedly used development set; not a candidate")
    return write_report(out, res)


def write_report(out: str, res: Dict) -> Dict:
    os.makedirs(out, exist_ok=True)
    write_json(os.path.join(out, "report.json"), res)
    lines = [json.dumps({k: res[k] for k in ("finalized", "states", "evidence", "route", "candidate_eligible") if k in res}, indent=1, default=str)]
    for arm, a in res.get("arms", {}).items():
        lines.append("%s fit %s diag %s G %s screens_ok %s" % (arm, a.get("fit_slice"), (a.get("diag") or {}).get("accuracy"),
                                                              (a.get("G_reading") or {}).get("all"), a.get("screens_ok")))
    if "k_minus_raw" in res:
        g = res["k_minus_raw"]["conflict_macro"]
        lines.append("K - raw conflict macro %+.3f [%+.3f, %+.3f]" % (g["value"], *g["ci95"]))
    open(os.path.join(out, "report.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["start", "fid", "train_raw", "train_k", "freeze_raw", "freeze_k", "decide", "decide_fit", "report"])
    ap.add_argument("--deadline", type=int, default=0)
    ap.add_argument("--finalize", default="")
    a = ap.parse_args(argv)
    if a.stage == "start":
        print(json.dumps({k: v for k, v in stage_start(a.deadline).items() if k != "files"}, indent=1))
    elif a.stage == "fid":
        print(json.dumps(stage_fid(), indent=1))
    elif a.stage.startswith("train_"):
        return stage_train(a.stage.split("_", 1)[1])
    elif a.stage.startswith("freeze_"):
        print(json.dumps(stage_freeze(a.stage.split("_", 1)[1]), indent=1))
    elif a.stage == "decide":
        print(json.dumps(decide_sets(OUT, list(REQUIRED), "decide"), indent=1))
    elif a.stage == "decide_fit":
        print(json.dumps(decide_sets(OUT, ["fit"], "decide_fit"), indent=1))
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
