"""exp68: new anti-shortcut temperature combinations, one plain-CE arm from EXP63's end (critic EXP68-PREP-01 /
-R1-ACCEPT-01; docs/experiments/2026-10-19-exp68-combinations-plan.md). The arm X: EXP63's frozen end (3e8443e9 /
70c302c4), a new LoRA r16 + decision head, EXP63's recipe (= EXP65-C's), data/exp68/train.jsonl and its real plan, 1,200
updates, last step, all weights 1. Reference: EXP65-C's frozen end (382e0a44 / cb7419ab) and its registered records in
runs/exp65 (reused, never rewritten); C is only read on the new diagnostic here. Raw temperature 1 everywhere.

  start    model-free: identities (EXP63 run / end, EXP65 run + C's frozen end / records / screens, data/facts, data/exp65,
           data/exp68 (train / diag / MANIFEST / PLAN), the YAML, code) -> runs/exp68/START.json.
  train    MPS: devision-pipeline configs/exp68-x.yaml (runs/exp68-x).
  freeze   tensors only: 1,200 updates; the export against EXP63's actual end; FROZEN.json.
  decide   MPS, one model at a time: X on fit / G / P / EXP65's diag / the new diag; EXP65-C on the new diag.
  (exist / science for X: devision-eval, recorded under X's frozen identity.)
  report   model-free: X's EXP63 gates (against the registered EXP57 start records), EXP65's diag by class (>= 0.80 each),
           the conflict macro gain over EXP65-C (case clusters, lower bound > 0), protection against EXP63 (>= -0.02);
           the new diag by class / both-views / near-large and paired with C (report only); the pre-written decision.

    uv run python scripts/exp68/exp68.py start --deadline EPOCH | train | freeze | decide | report [--finalize CODES]
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
OUT, RUN, E65, C_RUN = "runs/exp68", "runs/exp68-x", "runs/exp65", "runs/exp65-c"
YAML = "configs/exp68-x.yaml"
D68 = "data/exp68"
E65_RUN = {"START.json": "092537eecc1f296c9eae3d3ce5318cb264bacc67726a183cc067e0d9b9c544ef",
           "report.json": "23f8959a5000ac16c847a10a4bcaf8da267af1fa6cd6822cc4e24d1644a6c6da",
           "registry.json": "003e8abb6e5cb376a586ffca6a5414f831afda4cc9389c8b4ba197de8ee14fe4"}
C_SHA = {"model.safetensors": "382e0a44", "config.json": "cb7419ab"}               # prefixes; the full hashes come from C's FROZEN.json
FILES = ["scripts/exp68/exp68.py", "scripts/exp68/build.py", "scripts/exp68/run.sh", YAML, "scripts/exp65/exp65.py", "scripts/exp65/build.py",
         "scripts/exp63/exp63.py", "scripts/exp57/exp57.py", "scripts/facts/build.py", "src/devision/train/rlcd.py",
         "src/devision/train/pipeline.py", "src/devision/infer.py", "src/devision/model.py", "scripts/exp54/exp54.py", "scripts/exp53/watch.py",
         "scripts/v04b/freeze.py"]
X_CONDS = ("fit", "G", "P", "diag", "diag68")


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, rel))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


e65 = _load("exp65", "../exp65/exp65.py")
b68 = _load("exp68_build", "build.py")
e63, e57, e54, fz, r41 = e65.e63, e65.e57, e65.e54, e65.fz, e65.r41
sha, write_json, register, jsonl = e65.sha, e65.write_json, e65.register, e65.jsonl
CLASSES, CONFLICT = e65.b65.CLASSES, e65.CONFLICT


# ------------------------------------------------------------------ identities

def bound_e65() -> Dict:
    """EXP65's run (START / report / registry and every registered file) and C's frozen end (weights, screens)."""
    bad = [f for f, h in E65_RUN.items() if sha(os.path.join(E65, f)) != h]
    if bad:
        raise SystemExit("EXP65's %s changed" % bad)
    for files in json.load(open(os.path.join(E65, "registry.json"))).values():
        for f, h in files.items():
            if sha(os.path.join(E65, f)) != h:
                raise SystemExit("EXP65's registered %s changed" % f)
    fr = fz.check(C_RUN, list(e57.SCREENS))
    if any(not fr["weights"][k].startswith(v) for k, v in C_SHA.items()):
        raise SystemExit("EXP65-C's end is not 382e0a44 / cb7419ab")
    return {"run": dict(E65_RUN), "c_end": fr["weights"], "c_screens": {n: fr["evals"][n] for n in e57.SCREENS}}


def bound_d68() -> Dict[str, str]:
    """data/exp68: train / diag against the MANIFEST (and the accepted bytes), PLAN against the train file."""
    m = json.load(open(os.path.join(D68, "MANIFEST.json")))
    for f, h in m["sha256"].items():
        if sha(os.path.join(D68, f)) != h:
            raise SystemExit("data/exp68/%s differs from its MANIFEST" % f)
    if m["sha256"] != {"train.jsonl": "1798d1e836c7c6340f490c3cb5a0f5ca920844508f6418a43c501c6014018e92",
                       "diag.jsonl": "3a96c3f7dbb5f2b1602951d51bcd0461cab53743826909ed220bbe42b09c61b3"}:
        raise SystemExit("data/exp68 is not the accepted train / diag")
    p = json.load(open(os.path.join(D68, "PLAN.json")))
    if p["train_sha256"] != m["sha256"]["train.jsonl"] or p["updates"] != 1200 or p["split_units"] or not p["every_item_once"]:
        raise SystemExit("data/exp68 PLAN does not hold")
    if m["sources"]["facts_train"] != e63.DATA_SHA["train.jsonl"] or m["sources"]["exp65_diag"] != e65.bound_d65()["diag.jsonl"]:
        raise SystemExit("data/exp68 was built on other sources")
    return {"manifest": sha(os.path.join(D68, "MANIFEST.json")), "plan": sha(os.path.join(D68, "PLAN.json")), **m["sha256"]}


def check_yaml() -> None:
    import yaml
    y = yaml.safe_load(open(YAML))
    c = yaml.safe_load(open(e65.YAML["c"]))
    st, sc = y["stages"][0], c["stages"][0]
    if y["name"] != "exp68-x" or len(y["stages"]) != 1 or y.get("evaluate") or st["init"] != "runs/exp63/stage2" \
            or st["data"] != os.path.join(D68, "train.jsonl") or st["val"] != "data/facts/fit.jsonl" or st["params"] != sc["params"] \
            or y.get("device") != c.get("device") or y.get("common") != c.get("common"):
        raise SystemExit("configs/exp68-x.yaml is not EXP65-C's arm on data/exp68")


def stage_start(deadline: int, out: str = OUT) -> Dict:
    path = os.path.join(out, "START.json")
    if os.path.exists(path):
        raise SystemExit("%s exists (fresh only)" % path)
    if os.path.exists(RUN):
        raise SystemExit("%s exists (fresh only)" % RUN)
    check_yaml()
    from devision.reading import tree_sha
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    n = {c: len(e65.inputs_of(c)[0]) for c in ("fit", "G", "P", "diag")}
    n["diag68"] = len(jsonl(os.path.join(D68, "diag.jsonl")))
    s = {"git_head": git("rev-parse", "HEAD"), "dirty_files": [f for f in FILES if git("status", "--porcelain", "--", f)],
         "files": {f: sha(f) for f in FILES}, "exp63": e65.bound_e63(), "data": e63.bound_data(), "exp65_data": e65.bound_d65(),
         "exp65": bound_e65(), "exp68_data": bound_d68(), "tokenizer": tree_sha(os.path.join(e65.E63, "stage2", "tokenizer")),
         "exp57_screens": fz.check(e63.E57, list(e57.SCREENS))["evals"], "exp49_questions": sha(e54.QUESTIONS),
         "exp54_inputs_P": sha(os.path.join(e54.OUT, "inputs_P.jsonl")),
         "requests": {"x": n, "c_diag68": n["diag68"], "total": sum(n.values()) + n["diag68"]},
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
    if e65.bound_e63() != st["exp63"] or e63.bound_data() != st["data"] or e65.bound_d65() != st["exp65_data"]:
        raise SystemExit("EXP63 / data/facts / data/exp65 differ from START")
    if bound_e65() != st["exp65"] or bound_d68() != st["exp68_data"]:
        raise SystemExit("EXP65 / data/exp68 differ from START")
    if tree_sha(os.path.join(e65.E63, "stage2", "tokenizer")) != st["tokenizer"] or sha(e54.QUESTIONS) != st["exp49_questions"] \
            or sha(os.path.join(e54.OUT, "inputs_P.jsonl")) != st["exp54_inputs_P"]:
        raise SystemExit("tokenizer / exp49 questions / EXP54 P table differ from START")
    return st


# ------------------------------------------------------------------ train / freeze

def stage_train(out: str = OUT, poll: float = 5.0, cmd: Optional[List[str]] = None) -> int:
    check_start(out)
    log = os.path.join(RUN, "logs", "stage2.log")
    proc = subprocess.Popen(cmd or ["uv", "run", "devision-pipeline", YAML])
    seen = -1
    while True:
        rc = proc.poll()
        if os.path.exists(log):
            n = e57.progress_of(open(log, errors="replace").read().splitlines())
            if n is not None and n != seen:
                e54.progress(out, "train", n, e63.UPDATES)
                seen = n
        if rc is not None:
            return rc
        time.sleep(poll)


def stage_freeze(out: str = OUT) -> Dict:
    from safetensors.torch import load_file
    check_start(out)
    rep = json.load(open(os.path.join(RUN, "stage2", "train_report.json")))
    if len(rep["losses"]) != e63.UPDATES:
        raise SystemExit("X ran %d updates" % len(rep["losses"]))
    if fz.weights(e65.E63) != e65.E63_SHA:
        raise SystemExit("EXP63 changed")
    res = e57.export_check(load_file(os.path.join(e65.E63, "stage2", "model.safetensors")), load_file(os.path.join(RUN, "stage2", "model.safetensors")))
    res.update(updates=len(rep["losses"]), baseline=dict(e65.E63_SHA))
    write_json(os.path.join(RUN, "export_check.json"), res)
    if not res["ok"]:
        raise SystemExit("X export check failed: %s" % res)
    res["frozen"] = fz.freeze(RUN)["weights"]
    return res


# ------------------------------------------------------------------ decide

def inputs_of(cond: str):
    if cond == "diag68":
        path = os.path.join(D68, "diag.jsonl")
        if sha(path) != bound_d68()["diag.jsonl"]:
            raise SystemExit("data/exp68 diag changed")
        rows = {r["id"]: r for r in jsonl(path)}
        return rows, {rid: {"state": r["state_text"]} for rid, r in rows.items()}
    return e65.inputs_of(cond)


def stage_decide(out: str = OUT, device: str = "auto") -> Dict:
    from devision.infer import Decider, best_device
    from devision.reading import tree_sha
    st = check_start(out)
    s_sha = sha(os.path.join(out, "START.json"))
    plan = [("x", RUN, fz.check(RUN, [])["weights"], list(X_CONDS)), ("c", C_RUN, st["exp65"]["c_end"], ["diag68"])]
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

def diag68_summary(recs: List[Dict], rows: Dict[str, Dict]) -> Dict:
    """The new diag: accuracy per class and per class x stratum, and the share of combinations right in both orders."""
    by: Dict[str, List[bool]] = defaultdict(list)
    st: Dict[str, List[bool]] = defaultdict(list)
    fam: Dict[str, List[bool]] = defaultdict(list)
    for r in recs:
        row = rows[r["sample_id"]]
        by[row["axis"]].append(bool(r["correct"]))
        st["%s/%s" % (row["axis"], row["stratum"])].append(bool(r["correct"]))
        fam[row["family"]].append(bool(r["correct"]))
    both: Dict[str, List[bool]] = defaultdict(list)
    for f, v in fam.items():
        both[f.split(":")[0]].append(len(v) == 2 and all(v))
    def acc(v: List[bool]) -> Optional[float]:
        return sum(v) / len(v) if v else None
    conf = [acc(by[c]) for c in CONFLICT]
    return {"accuracy": {c: acc(by[c]) for c in CLASSES}, "both_views": {c: acc(both[c]) for c in CLASSES},
            "strata": {k: {"n": len(v), "accuracy": acc(v)} for k, v in sorted(st.items())},
            "conflict_macro": None if conf[0] is None or conf[1] is None else (conf[0] + conf[1]) / 2}


def practical(g: Dict, diag: Optional[Dict], gain: Any, vs63: Dict) -> Any:
    """EXP63's gates, EXP65's diag >= 0.80 in every class, the conflict gain over C (lower bound > 0), protection."""
    if diag is None or any(v == "incomplete" for v in g.values()) or vs63.get("state") == "incomplete" or gain == "incomplete":
        return "incomplete"
    return bool(all(v is True for v in g.values()) and all((diag[c]["accuracy"] or 0) >= 0.80 for c in CLASSES) and vs63["ok"] and gain is True)


def decision(p: Any) -> str:
    if p == "incomplete":
        return "incomplete: a required item is missing -- no route decision"
    if p:
        return "the new data recipe meets every pre-written condition: next (separate permission) live-transfer preparation; not a candidate"
    return ("the new data recipe does not meet the pre-written conditions: no more epochs / weights on this recipe; next look at an "
            "explicit numeric-combination representation or decomposed supervision")


def report(out: str = OUT, finalize: str = "") -> Dict:
    res: Dict[str, Any] = {"states": {}, "candidate_eligible": False}
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
    base = {}
    for c in e65.CONDS:
        rows, inputs = e63.set_inputs(c)
        base[c] = (rows, e57.read_set(e65.E63, "start_" + c, rows, e63.E57_SHA, inputs))
    e63_new = {c: e57.read_set(e65.E63, c, base[c][0], e65.E63_SHA, e63.set_inputs(c)[1]) for c in e65.CONDS}
    drows, dinp = inputs_of("diag")
    d68rows, d68inp = inputs_of("diag68")
    c_end = st["exp65"]["c_end"]
    c_recs: Dict[str, List[Dict]] = {}
    for name, rows_, inp_, out_ in (("diag", drows, dinp, E65), ("diag68", d68rows, d68inp, out)):   # C: EXP65's diag reused, the new one here
        try:
            c_recs[name] = e57.read_set(out_, "c_" + name, rows_, c_end, inp_)
            res["states"]["c_" + name] = "completed"
        except (SystemExit, OSError, ValueError, KeyError) as e:
            res["states"]["c_" + name] = "missing / invalid: %s" % e
    a: Dict[str, Any] = {"comparison": {}, "paired": {}, "reading": {}, "screens": {}, "vs_exp63": {}}
    res["x"] = a
    recs: Dict[str, List[Dict]] = {}
    try:
        weights = fz.check(RUN, [])["weights"]
    except SystemExit as e:
        res["states"]["x"] = "missing: %s" % e
        weights = None
    if weights is not None:
        for c in X_CONDS:
            try:
                rows, inputs = inputs_of(c)
                recs[c] = e57.read_set(out, "x_" + c, rows, weights, inputs)
                res["states"]["x_" + c] = "completed"
            except (SystemExit, OSError, ValueError, KeyError) as e:
                res["states"]["x_" + c] = "missing / invalid: %s" % e
    qs53 = {q["key"]: q for q in jsonl(e54.QUESTIONS) if q["set"] == "rl_tune"}
    pin = e54.load_inputs(e54.OUT, "P")
    tune = {r["id"]: r for r in jsonl(e54.TUNE)}
    p_right = {k: {(o, qn): v for o, qn, v in pin[q["id"]]["table"]}[e54.TEMPLATES[tune[q["id"]]["questions"][q["qid"]]["instructions"]]] == q["target"]
               for k, q in qs53.items()}
    e63_screens = {n: e57.screen_side(e65.E63, n, e57.SCREENS[n])[0] for n in e57.SCREENS}
    for c in e65.CONDS:
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
    if weights is not None:
        for n in e57.SCREENS:
            try:
                a["screens"][n] = e57.screen(RUN, n, st["exp57_screens"][n], start_run=e63.E57)
                a["vs_exp63"]["screen:" + n] = {"exp63": e63_screens[n], "difference": a["screens"][n]["new"] - e63_screens[n]}
            except (SystemExit, OSError, ValueError, KeyError) as e:
                res["states"]["x_" + n] = "missing / invalid: %s" % e
    a["gates"] = e65.arm_gates(a)
    if "diag" in recs:
        a["diag"] = e65.diag_summary(recs["diag"], drows)
    gain: Any = "incomplete"
    if "diag" in recs and "diag" in c_recs:
        d = e65.conflict_difference(recs["diag"], c_recs["diag"], drows)
        a["conflict_gain_over_c"] = d
        gain = bool(d["ci95"][0] > 0) if d["ci95"][0] == d["ci95"][0] else "incomplete"
    if "diag68" in recs:
        a["diag68"] = diag68_summary(recs["diag68"], d68rows)
        if "diag68" in c_recs:
            res["c_diag68"] = diag68_summary(c_recs["diag68"], d68rows)
            a["diag68_conflict_gain_over_c"] = e65.conflict_difference(recs["diag68"], c_recs["diag68"], d68rows)
    scr = [a["vs_exp63"].get("screen:" + n) for n in e57.SCREENS]
    vs = {"state": "incomplete"} if None in scr else {"ok": all(x["difference"] >= -0.02 for x in scr)}
    a["practical"] = practical(a["gates"], a.get("diag"), gain, vs)
    res["decision"] = decision(a["practical"])
    res["scope"] = "the whole new data recipe (class quotas and combination diversity changed together); not credited to diversity alone"
    return write_report(out, res)


def write_report(out: str, res: Dict) -> Dict:
    os.makedirs(out, exist_ok=True)
    write_json(os.path.join(out, "report.json"), res)
    lines = [json.dumps({k: res[k] for k in ("finalized", "states", "decision", "scope", "candidate_eligible") if k in res}, indent=1, default=str)]
    a = res.get("x") or {}
    if a:
        lines.append("x gates %s practical %s" % (a.get("gates"), a.get("practical")))
        if "diag" in a:
            lines.append("x diag %s" % {c: v["accuracy"] for c, v in a["diag"].items() if isinstance(v, dict)})
        if "conflict_gain_over_c" in a:
            g = a["conflict_gain_over_c"]
            lines.append("x - C conflict macro %+.3f [%+.3f, %+.3f]" % (g["value"], *g["ci95"]))
        if "diag68" in a:
            lines.append("x diag68 %s" % a["diag68"]["accuracy"])
    open(os.path.join(out, "report.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["start", "train", "freeze", "decide", "report"])
    ap.add_argument("--deadline", type=int, default=0)
    ap.add_argument("--finalize", default="")
    a = ap.parse_args(argv)
    if a.stage == "start":
        print(json.dumps({k: v for k, v in stage_start(a.deadline).items() if k != "files"}, indent=1))
    elif a.stage == "train":
        return stage_train()
    elif a.stage == "freeze":
        print(json.dumps(stage_freeze(), indent=1))
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
