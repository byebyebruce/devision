"""exp65: equal-budget plain extra CE (C) vs rule-class weighted CE (W) from EXP63's end (critic EXP65-PREP-01;
docs/experiments/2026-10-19-exp65-weight-plan.md). Both arms: EXP63's frozen end (3e8443e9 / 70c302c4), a new LoRA r16 +
decision head, EXP63's recipe, data/facts/train.jsonl and its plan, 1,200 updates, last step; the only difference is
the per-question CE weight (W: data/exp65/weights.jsonl). Raw temperature 1 everywhere. Stages (fresh only):

  start    model-free: identities (EXP63 run / end / registered EXP57 start records, EXP57's screen records, data/facts,
           data/exp65, the two YAMLs, code) -> runs/exp65/START.json.
  train_c / train_w   MPS: devision-pipeline configs/exp65-<arm>.yaml (runs/exp65-<arm>).
  freeze_c / freeze_w tensors only: 1,200 updates; the export against EXP63's actual end; FROZEN.json.
  decide   MPS, one model at a time: EXP63's end on the diagnostic, then each arm on fit / G / P / the diagnostic.
  (exist / science per arm: devision-eval, recorded under the arm's frozen identity.)
  report   model-free: per arm EXP63's gates (against the registered EXP57 start records), the diagnostic by class,
           changes against EXP63, W - C, the pre-written practical condition and decision.

    uv run python scripts/exp65/exp65.py start --deadline EPOCH | train_c | freeze_c | train_w | freeze_w | decide | report
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
OUT, E63 = "runs/exp65", "runs/exp63"
ARMS = {"c": "runs/exp65-c", "w": "runs/exp65-w"}
YAML = {"c": "configs/exp65-c.yaml", "w": "configs/exp65-w.yaml"}
E63_SHA = {"model.safetensors": "3e8443e9143cfbd55686260e2a20af7c2bdb1b6d98de44c9122a6ed82b74660b",
           "config.json": "70c302c4bbafd49ef528dec08aef050b26fc26bb0034b229f5871e046f243805"}
E63_RUN = {"START.json": "a482ef6a8abd7d6367a42f6dbb57c56246fdc61a6d21a047ba1054c2d63db79d",
           "report.json": "3abc17108eaac1a245b74d14fe279ac693e372bb0f0a3c679950674b7fdba050",
           "registry.json": "ecb4ba67629df3549fd113ed86f212261b0287d8487577b03baff12a8ef5c802"}
D65 = "data/exp65"
CONDS = ("fit", "G", "P")
CONFLICT = ("opposite_mass_wins", "opposite_speed_wins")
FILES = ["scripts/exp65/exp65.py", "scripts/exp65/build.py", "scripts/exp65/run.sh", YAML["c"], YAML["w"], "scripts/exp63/exp63.py",
         "scripts/exp57/exp57.py", "scripts/facts/build.py", "src/devision/train/rlcd.py", "src/devision/train/pipeline.py",
         "src/devision/infer.py", "src/devision/model.py", "scripts/exp54/exp54.py", "scripts/exp53/watch.py", "scripts/v04b/freeze.py"]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, rel))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


e63 = _load("exp63", "../exp63/exp63.py")
b65 = _load("exp65_build", "build.py")
e57, e54, fz, r41 = e63.e57, e63.e54, e63.fz, e63.r41
sha, write_json, register, jsonl = e63.sha, e63.write_json, e63.register, e63.jsonl


# ------------------------------------------------------------------ identities

def bound_e63() -> Dict:
    bad = [f for f, h in E63_RUN.items() if sha(os.path.join(E63, f)) != h]
    if bad:
        raise SystemExit("EXP63's %s changed" % bad)
    for files in json.load(open(os.path.join(E63, "registry.json"))).values():
        for f, h in files.items():
            if sha(os.path.join(E63, f)) != h:
                raise SystemExit("EXP63's registered %s changed" % f)
    fr = fz.check(E63, list(e57.SCREENS))
    if fr["weights"] != E63_SHA:
        raise SystemExit("EXP63's end is not 3e8443e9 / 70c302c4")
    return {"run": dict(E63_RUN), "end": dict(E63_SHA), "screens": {n: fr["evals"][n] for n in e57.SCREENS}}


def bound_d65() -> Dict[str, str]:
    m = json.load(open(os.path.join(D65, "MANIFEST.json")))
    for f, h in m["sha256"].items():
        if sha(os.path.join(D65, f)) != h:
            raise SystemExit("data/exp65/%s differs from its MANIFEST" % f)
    if m["facts"]["train"] != e63.DATA_SHA["train.jsonl"] or m["facts"]["PLAN"] != sha(os.path.join(e63.FACTS, "PLAN.json")):
        raise SystemExit("data/exp65 was built on another data/facts")
    return {"manifest": sha(os.path.join(D65, "MANIFEST.json")), **m["sha256"]}


def check_yamls() -> None:
    import yaml
    c, w = yaml.safe_load(open(YAML["c"])), yaml.safe_load(open(YAML["w"]))
    pc, pw = c["stages"][0]["params"], w["stages"][0]["params"]
    want = {"epochs": 1, "objective": "ce", "text_only": True, "micro_batch": 8, "eval_batch": 8, "lora_r": 16, "lr_lora": 5e-5,
            "lr_head": 1e-5, "warmup": 100, "seed": 1, "amp": "off", "eval_every": 0, "save_every": 0}
    for y, p, name in ((c, pc, "exp65-c"), (w, pw, "exp65-w")):
        st = y["stages"][0]
        if y["name"] != name or len(y["stages"]) != 1 or st["init"] != "runs/exp63/stage2" or st["data"] != "data/facts/train.jsonl" \
                or st["val"] != "data/facts/fit.jsonl" or any(p.get(k) != v for k, v in want.items()) or y.get("evaluate"):
            raise SystemExit("%s is not the fixed arm" % name)
    if pc.get("question_weight_file") or pw.get("question_weight_file") != os.path.join(D65, "weights.jsonl") \
            or {k: v for k, v in pc.items()} != {k: v for k, v in pw.items() if k != "question_weight_file"}:
        raise SystemExit("the arms differ in something else than the weight file")


def stage_start(deadline: int, out: str = OUT) -> Dict:
    path = os.path.join(out, "START.json")
    if os.path.exists(path):
        raise SystemExit("%s exists (fresh only)" % path)
    for run in ARMS.values():
        if os.path.exists(run):
            raise SystemExit("%s exists (fresh only)" % run)
    e63_ = bound_e63()
    data = e63.bound_data()
    d65 = bound_d65()
    check_yamls()
    from devision.reading import tree_sha
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    n = {c: len(jsonl(os.path.join(e63.FACTS, e63.FILE_OF[c]))) for c in CONDS}
    diag_n = len(jsonl(os.path.join(D65, "diag.jsonl")))
    s = {"git_head": git("rev-parse", "HEAD"), "dirty_files": [f for f in FILES if git("status", "--porcelain", "--", f)],
         "files": {f: sha(f) for f in FILES}, "exp63": e63_, "data": data, "exp65_data": d65, "tokenizer": tree_sha(os.path.join(E63, "stage2", "tokenizer")),
         "exp57_screens": fz.check(e63.E57, list(e57.SCREENS))["evals"], "exp49_questions": sha(e54.QUESTIONS),
         "exp54_inputs_P": sha(os.path.join(e54.OUT, "inputs_P.jsonl")),
         "requests": {"exp63_diag": diag_n, "per_arm": {**n, "diag": diag_n}, "total": diag_n + 2 * (sum(n.values()) + diag_n)},
         "arms": {"c": "all weights 1 (no weight file)", "w": "data/exp65/weights.jsonl"}, "candidate_eligible": False,
         "started": time.strftime("%F %T"), "deadline": deadline, "deadline_text": time.strftime("%F %T", time.localtime(deadline))}
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
    if bound_e63() != st["exp63"] or e63.bound_data() != st["data"] or bound_d65() != st["exp65_data"]:
        raise SystemExit("EXP63 / data/facts / data/exp65 differ from START")
    if tree_sha(os.path.join(E63, "stage2", "tokenizer")) != st["tokenizer"] or sha(e54.QUESTIONS) != st["exp49_questions"] \
            or sha(os.path.join(e54.OUT, "inputs_P.jsonl")) != st["exp54_inputs_P"]:
        raise SystemExit("tokenizer / exp49 questions / EXP54 P table differ from START")
    return st


# ------------------------------------------------------------------ train / freeze

def stage_train(arm: str, out: str = OUT, poll: float = 5.0, cmd: Optional[List[str]] = None) -> int:
    check_start(out)
    log = os.path.join(ARMS[arm], "logs", "stage2.log")
    proc = subprocess.Popen(cmd or ["uv", "run", "devision-pipeline", YAML[arm]])
    seen = -1
    while True:
        rc = proc.poll()
        if os.path.exists(log):
            n = e57.progress_of(open(log, errors="replace").read().splitlines())
            if n is not None and n != seen:
                e54.progress(out, "train_" + arm, n, e63.UPDATES)
                seen = n
        if rc is not None:
            return rc
        time.sleep(poll)


def stage_freeze(arm: str, out: str = OUT) -> Dict:
    from safetensors.torch import load_file
    check_start(out)
    run = ARMS[arm]
    rep = json.load(open(os.path.join(run, "stage2", "train_report.json")))
    if len(rep["losses"]) != e63.UPDATES:
        raise SystemExit("arm %s ran %d updates" % (arm, len(rep["losses"])))
    if fz.weights(E63) != E63_SHA:
        raise SystemExit("EXP63 changed")
    res = e57.export_check(load_file(os.path.join(E63, "stage2", "model.safetensors")), load_file(os.path.join(run, "stage2", "model.safetensors")))
    res.update(updates=len(rep["losses"]), baseline=dict(E63_SHA))
    write_json(os.path.join(run, "export_check.json"), res)
    if not res["ok"]:
        raise SystemExit("arm %s export check failed: %s" % (arm, res))
    res["frozen"] = fz.freeze(run)["weights"]
    return res


# ------------------------------------------------------------------ decide

def inputs_of(cond: str):
    if cond == "diag":
        path = os.path.join(D65, "diag.jsonl")
        if sha(path) != bound_d65()["diag.jsonl"]:
            raise SystemExit("diag changed")
        rows = {r["id"]: r for r in jsonl(path)}
        return rows, {rid: {"state": r["state_text"]} for rid, r in rows.items()}
    return e63.set_inputs(cond)


def stage_decide(out: str = OUT, device: str = "auto") -> Dict:
    from devision.infer import Decider, best_device
    from devision.reading import tree_sha
    st = check_start(out)
    s_sha = sha(os.path.join(out, "START.json"))
    plan = [("exp63", E63, E63_SHA, ["diag"])] + [(arm, ARMS[arm], fz.check(ARMS[arm], [])["weights"], list(CONDS) + ["diag"]) for arm in ("c", "w")]
    total = sum(len(inputs_of(c)[0]) for _, _, _, cs in plan for c in cs)
    dev = best_device() if device == "auto" else device
    done, t0 = 0, time.time()
    for who, run, weights, conds in plan:
        if fz.weights(run) != weights or tree_sha(os.path.join(run, "stage2", "tokenizer")) != st["tokenizer"]:   # before any load
            raise SystemExit("%s: weights / tokenizer not the bound ones" % who)
        d = Decider.load(os.path.join(run, "stage2"), device=dev)
        d.cfg.temperature, d.cfg.temperature_by_options = [1.0, 1.0, 1.0], {}
        for c in conds:
            rows, inputs = inputs_of(c)
            recs = e54.decide_condition(d, rows, inputs, on_request=lambda n, b=done: e54.progress(out, "decide", b + n, total))
            done += len(rows)
            name = "%s_%s" % (who, c)
            e57.write_set(out, name, recs, {"set": name, "model_of": who, "start_sha256": s_sha, "inputs_sha256": e57.hash_inputs(inputs),
                                            "model": weights, "temperature": "none (raw)", "image": None, "device": dev})
            print("%s done after %.1f s" % (name, time.time() - t0), flush=True)
        del d
    return {"seconds": time.time() - t0}


# ------------------------------------------------------------------ report

def diag_summary(recs: List[Dict], rows: Dict[str, Dict]) -> Dict:
    by = defaultdict(list)
    for r in recs:
        by[rows[r["sample_id"]]["axis"]].append(bool(r["correct"]))
    out: Dict[str, Any] = {c: {"n": len(by[c]), "accuracy": sum(by[c]) / len(by[c]) if by[c] else None} for c in b65.CLASSES}
    vals = [out[c]["accuracy"] for c in CONFLICT]
    out["conflict_macro"] = None if None in vals else sum(vals) / 2
    return out


def conflict_difference(w: List[Dict], c: List[Dict], rows: Dict[str, Dict]) -> Dict:
    """W - C on the two conflict classes, equal-weight macro, the diagnostic's canonical cases resampled (both orders
    of a case together)."""
    oc = {(r["sample_id"], r["qid"]): r for r in c}
    fam: Dict[str, List[tuple]] = defaultdict(list)
    for r in w:
        row = rows[r["sample_id"]]
        if row["axis"] in CONFLICT:
            fam[row["family"]].append((row["axis"], float(r["correct"]) - float(oc[(r["sample_id"], r["qid"])]["correct"])))

    def fn(ks):
        parts = []
        for cl in CONFLICT:
            xs = [d for k in ks for a, d in fam[k] if a == cl]
            if not xs:
                return float("nan")
            parts.append(sum(xs) / len(xs))
        return sum(parts) / len(parts)
    return r41.interval(fn, sorted(fam), lambda ks: list(ks))


def arm_gates(res_arm: Dict) -> Dict:
    return e63.gates(res_arm)


def practical(g: Dict, diag: Optional[Dict], vs63: Dict) -> Any:
    if diag is None or any(v == "incomplete" for v in g.values()) or vs63.get("state") == "incomplete":
        return "incomplete"
    return bool(all(v is True for v in g.values()) and all((diag[c]["accuracy"] or 0) >= 0.80 for c in b65.CLASSES) and vs63["ok"])


def decision(pc: Any, pw: Any, attribution: Any) -> str:
    if "incomplete" in (pc, pw):
        return "incomplete: an arm's practical condition is missing -- no route decision"
    if pc and pw:
        return "both arms meet the practical condition; W preferred only with attribution: %s" % (
            "W (W - C conflict lower bound > 0)" if attribution is True else "C (no reliable W advantage: not credited to weighting)")
    if pc:
        return "C meets the practical condition, W does not: C (plain extra exposure suffices on this budget)"
    if pw:
        return ("W meets and the weighting is attributed: W" if attribution is True
                else "only W meets but the gain over C is not established: no route; stop this budget")
    return "neither arm meets the practical condition: stop this budget (no more epochs, no weight sweep)"


def report(out: str = OUT, finalize: str = "") -> Dict:
    res: Dict[str, Any] = {"states": {}, "arms": {}, "candidate_eligible": False}
    if finalize:
        res["finalized"] = {"exit_codes": finalize, "note": "missing items are not low scores"}
    if not os.path.exists(os.path.join(out, "START.json")):
        res["decision"] = "incomplete: no START (nothing ran)"
        return write_report(out, res)
    try:
        st = check_start(out, code=False)
    except (SystemExit, OSError, ValueError, KeyError) as e:
        res["decision"] = "invalid: the run's identity does not verify (%s) -- no verdict" % e
        return write_report(out, res)
    base = {}                                                                # EXP57 start records, registered in EXP63
    for c in CONDS:
        rows, inputs = e63.set_inputs(c)
        base[c] = (rows, e57.read_set(E63, "start_" + c, rows, e63.E57_SHA, inputs))
    e63_new = {c: e57.read_set(E63, c, base[c][0], E63_SHA, e63.set_inputs(c)[1]) for c in CONDS}
    drows, dinp = inputs_of("diag")
    diag_recs: Dict[str, List[Dict]] = {}
    try:
        diag_recs["exp63"] = e57.read_set(out, "exp63_diag", drows, E63_SHA, dinp)
        res["exp63_diag"] = diag_summary(diag_recs["exp63"], drows)
    except (SystemExit, OSError, ValueError, KeyError) as e:
        res["states"]["exp63_diag"] = "missing / invalid: %s" % e
    qs53 = {q["key"]: q for q in jsonl(e54.QUESTIONS) if q["set"] == "rl_tune"}
    pin = e54.load_inputs(e54.OUT, "P")
    tune = {r["id"]: r for r in jsonl(e54.TUNE)}
    p_right = {k: {(o, qn): v for o, qn, v in pin[q["id"]]["table"]}[e54.TEMPLATES[tune[q["id"]]["questions"][q["qid"]]["instructions"]]] == q["target"]
               for k, q in qs53.items()}
    e63_screens = {n: e57.screen_side(E63, n, e57.SCREENS[n])[0] for n in e57.SCREENS}
    for arm, run in ARMS.items():
        a: Dict[str, Any] = {"comparison": {}, "paired": {}, "reading": {}, "screens": {}, "vs_exp63": {}}
        res["arms"][arm] = a
        try:
            weights = fz.check(run, [])["weights"]
        except SystemExit as e:
            res["states"][arm] = "missing: %s" % e
            continue
        recs = {}
        for c in CONDS + ("diag",):
            try:
                rows, inputs = inputs_of(c)
                recs[c] = e57.read_set(out, "%s_%s" % (arm, c), rows, weights, inputs)
                res["states"]["%s_%s" % (arm, c)] = "completed"
            except (SystemExit, OSError, ValueError, KeyError) as e:
                res["states"]["%s_%s" % (arm, c)] = "missing / invalid: %s" % e
        for c in CONDS:
            if c not in recs:
                continue
            rows = base[c][0]
            for kind in ("temp", "force"):
                cs = e63.comparisons(recs[c], rows, kind)
                a["comparison"]["%s:%s" % (c, kind)] = e63.summary(cs, kind)
                if cs:
                    a["paired"]["%s:%s" % (c, kind)] = e57.paired_difference(cs, e63.comparisons(base[c][1], rows, kind))
                    a["vs_exp63"]["%s:%s" % (c, kind)] = e57.paired_difference(cs, e63.comparisons(e63_new[c], rows, kind))
        for c in ("G", "P"):
            if c in recs:
                a["reading"][c] = e57.reading_summary(recs[c], base[c][0], qs53, p_right)
        for n in e57.SCREENS:
            try:
                a["screens"][n] = e57.screen(run, n, st["exp57_screens"][n], start_run=e63.E57)
                new = a["screens"][n]["new"]
                a["vs_exp63"]["screen:" + n] = {"exp63": e63_screens[n], "difference": new - e63_screens[n]}
            except (SystemExit, OSError, ValueError, KeyError) as e:
                res["states"]["%s_%s" % (arm, n)] = "missing / invalid: %s" % e
        a["gates"] = arm_gates(a)
        if "diag" in recs:
            diag_recs[arm] = recs["diag"]
            a["diag"] = diag_summary(recs["diag"], drows)
        scr = [a["vs_exp63"].get("screen:" + n) for n in e57.SCREENS]
        vs = {"state": "incomplete"} if None in scr else {"ok": all(x["difference"] >= -0.02 for x in scr)}
        a["practical"] = practical(a["gates"], a.get("diag"), vs)
    attribution: Any = "incomplete"
    if "c" in diag_recs and "w" in diag_recs:
        d = conflict_difference(diag_recs["w"], diag_recs["c"], drows)
        res["w_minus_c_conflict"] = d
        attribution = bool(d["ci95"][0] > 0) if d["ci95"][0] == d["ci95"][0] else "incomplete"
    res["attribution"] = attribution
    res["decision"] = decision(res["arms"].get("c", {}).get("practical", "incomplete"), res["arms"].get("w", {}).get("practical", "incomplete"), attribution)
    res["exp63_outcome"] = "EXP63 stays 'fit not learned (fit_temp)'; this run does not rewrite it"
    return write_report(out, res)


def write_report(out: str, res: Dict) -> Dict:
    os.makedirs(out, exist_ok=True)
    write_json(os.path.join(out, "report.json"), res)
    lines = [json.dumps({k: res[k] for k in ("finalized", "states", "attribution", "decision", "exp63_outcome", "candidate_eligible") if k in res},
                        indent=1, default=str)]
    for arm, a in res.get("arms", {}).items():
        lines.append("arm %s gates %s practical %s diag %s" % (arm, a.get("gates"), a.get("practical"),
                                                                {c: v["accuracy"] for c, v in (a.get("diag") or {}).items() if isinstance(v, dict)}))
    if "exp63_diag" in res:
        lines.append("exp63 diag %s" % {c: v["accuracy"] for c, v in res["exp63_diag"].items() if isinstance(v, dict)})
    if "w_minus_c_conflict" in res:
        lines.append("W - C conflict macro %+.3f [%+.3f, %+.3f]" % (res["w_minus_c_conflict"]["value"], *res["w_minus_c_conflict"]["ci95"]))
    open(os.path.join(out, "report.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["start", "train_c", "freeze_c", "train_w", "freeze_w", "decide", "report"])
    ap.add_argument("--deadline", type=int, default=0)
    ap.add_argument("--finalize", default="")
    a = ap.parse_args(argv)
    if a.stage == "start":
        print(json.dumps({k: v for k, v in stage_start(a.deadline).items() if k != "files"}, indent=1))
    elif a.stage.startswith("train_"):
        return stage_train(a.stage[-1])
    elif a.stage.startswith("freeze_"):
        print(json.dumps(stage_freeze(a.stage[-1]), indent=1))
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
