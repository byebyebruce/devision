"""exp63: complete-fact decision CE on text (critic DECISION-FACTS-IMPL-01 A / B / C; docs/experiments/2026-10-19-decision-facts-prep.md).
Can EXP57's decision side learn both comparisons from complete correct facts while keeping its reading and old
abilities? Not a test of automatic magnet-size extraction; never a transfer claim. One arm: EXP57's end (962503fb /
09a5b447) -> configs/exp63.yaml (rlcd text_only CE, image=None, LoRA r16 + decision head, lr_lora 5e-5 / lr_head 1e-5,
warm-up 100, seed 1, micro / eval batch 8, one epoch = 1,200 updates of data/facts/train.jsonl, fit as val, fixed last
step). Raw temperature 1 in every evaluation. Stages (fresh only, runs/exp63):

  start       model-free: identities (EXP57 end + FROZEN + exist / science records, data/facts files / MANIFEST / PLAN,
              the facts sources, EXP56's live tables, exp49 questions, the YAML, this code) -> START.json.
  train       MPS: devision-pipeline configs/exp63.yaml, its log tailed into progress lines.
  freeze      tensors only: 1,200 updates; the export against EXP57's actual end (vision / projector / conditioning / word
              embeddings identical, changes only in the decision part); FROZEN.json.
  decide      MPS, the main conditions: the new end on fit / G / P, then EXP57 on fit / G / P (EXP57 had no records on these
              files: re-run under their option orders); each registered when complete.
  (exist / science: devision-eval on the frozen end, recorded.)
  decide_h    MPS: the new end on H only (report only; EXP57 not run on H).
  report      model-free: comparisons by semantic class (temperature A / B / same; magnets P1 / P2 / same, "weaker in Pair
              1" = P2) per kind, per row axis, both views of a family; new - EXP57 by case clusters; reading retention
              on the 1,800 field readings only; exist / science against EXP57's recorded results; gates and branch.

    uv run python scripts/exp63/exp63.py start --deadline EPOCH | train | freeze | decide | decide_h | report [--finalize CODES]
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(__file__)
OUT, E57 = "runs/exp63", "runs/exp57"
YAML = "configs/exp63.yaml"
FACTS = "data/facts"
E57_SHA = {"model.safetensors": "962503fbd5f41b6794e5b48a16fcd833b591b04b285d8f3afaa45f18d7c38131",
           "config.json": "09a5b44716b1b818df0372994b7581159625f4af7e267b6cf4e2ff4f12eb8d0d"}
DATA_SHA = {"train.jsonl": "38649b910feb2cab663c77cd43f5f2e9a2fca71653464995f91367b05f088042",
            "fit.jsonl": "295ee79730e6704b3927cf5ea505d7a5467f696488edd79ad752f41b163ba88d",
            "tune_G.jsonl": "f4a5c7ab6f4a86667efeb1295370b260763ab7c01b191770f60939db7795b078",
            "tune_H.jsonl": "58e4147b84ccf701a095ecbca730e7a7ad5d342f8fa55f86483872e6e2c0dc05",
            "tune_P.jsonl": "6b59b26bb9ac8834be7b6f3a4086e43f332cf5046ab8c2ea01a43c4e846c4e6b"}
UPDATES = 1200
MAIN = ("fit", "G", "P")                 # new end, then EXP57 ("start_" prefix), in this order
FILE_OF = {"fit": "fit.jsonl", "G": "tune_G.jsonl", "H": "tune_H.jsonl", "P": "tune_P.jsonl"}
FILES = ["scripts/exp63/exp63.py", "scripts/exp63/run.sh", YAML, "scripts/facts/build.py", "scripts/exp57/exp57.py",
         "src/devision/train/rlcd.py", "src/devision/train/pipeline.py", "src/devision/infer.py", "src/devision/model.py",
         "src/devision/reading.py", "scripts/exp54/exp54.py", "scripts/exp53/watch.py", "scripts/v04b/freeze.py"]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, rel))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


e57 = _load("exp57", "../exp57/exp57.py")
fb = _load("facts_build", "../facts/build.py")
e54, fz, r42, r41 = e57.e54, e57.fz, e57.r42, e57.r41
sha, write_json, register, jsonl = e57.sha, e57.write_json, e57.register, e57.jsonl


# ------------------------------------------------------------------ identities

def bound_data() -> Dict[str, Any]:
    bad = [f for f, h in DATA_SHA.items() if sha(os.path.join(FACTS, f)) != h]
    if bad:
        raise SystemExit("data/facts %s changed" % bad)
    m = json.load(open(os.path.join(FACTS, "MANIFEST.json")))
    if m["sha256"] != DATA_SHA:
        raise SystemExit("data/facts MANIFEST does not name these files")
    src = {"questions_train": "data/real_layout/questions_train.jsonl", "captions_train": "data/real_layout/captions_train.jsonl",
           "questions_tune": "data/real_layout/questions_tune.jsonl", "captions_tune": "data/real_layout/captions_tune.jsonl",
           "exp56_inputs_P": "runs/exp56/inputs_P.jsonl"}
    import hashlib
    if any(hashlib.sha256(open(p).read().encode()).hexdigest() != m["source"][k] for k, p in src.items()):
        raise SystemExit("the facts sources differ from data/facts MANIFEST")
    plan = json.load(open(os.path.join(FACTS, "PLAN.json")))
    want = {"items": 9600, "updates": UPDATES, "units": 4800, "split_units": 0, "every_item_once": True, "batch_sizes": {"8": UPDATES},
            "seed": 1, "micro_batch": 8}
    bad = {k: plan.get(k) for k, v in want.items() if plan.get(k) != v}
    if bad or any(v["state_dropped"] or v["options_not_distinct"] for v in plan["truncation"].values()):
        raise SystemExit("data/facts PLAN is not 1,200 x 8 over 4,800 whole units without truncation: %s" % bad)
    return {"files": dict(DATA_SHA), "manifest": sha(os.path.join(FACTS, "MANIFEST.json")), "plan": sha(os.path.join(FACTS, "PLAN.json"))}


def check_yaml(path: str = YAML) -> Dict:
    import yaml
    y = yaml.safe_load(open(path))
    st = y["stages"]
    p = st[0]["params"] if len(st) == 1 else {}
    want = {"epochs": 1, "objective": "ce", "text_only": True, "micro_batch": 8, "eval_batch": 8, "lora_r": 16, "lr_lora": 5e-5,
            "lr_head": 1e-5, "warmup": 100, "seed": 1, "amp": "off", "eval_every": 0, "save_every": 0}
    if y.get("name") != "exp63" or len(st) != 1 or st[0].get("init") != "runs/exp57/stage2" or st[0].get("data") != os.path.join(FACTS, "train.jsonl") \
            or st[0].get("val") != os.path.join(FACTS, "fit.jsonl") or any(p.get(k) != v for k, v in want.items()) or y.get("evaluate"):
        raise SystemExit("configs/exp63.yaml is not the fixed EXP63 arm")
    return y


def bound_start_model() -> Dict:
    fr = fz.check(E57, list(e57.SCREENS))
    if fr["weights"] != E57_SHA:
        raise SystemExit("EXP57's end is not 962503fb / 09a5b447")
    return fr


def stage_start(deadline: int, out: str = OUT) -> Dict:
    path = os.path.join(out, "START.json")
    if os.path.exists(path):
        raise SystemExit("%s exists (fresh only)" % path)
    fr = bound_start_model()
    data = bound_data()
    check_yaml()
    from devision.reading import tree_sha
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    rows = {c: len(jsonl(os.path.join(FACTS, f))) for c, f in FILE_OF.items()}
    s = {"git_head": git("rev-parse", "HEAD"), "dirty_files": [f for f in FILES if git("status", "--porcelain", "--", f)],
         "files": {f: sha(f) for f in FILES}, "start_model": {"run": E57, **E57_SHA, "tokenizer": tree_sha(os.path.join(E57, "stage2", "tokenizer"))},
         "start_screens": {n: fr["evals"][n] for n in e57.SCREENS}, "data": data, "exp49_questions": sha(e54.QUESTIONS),
         "exp54_inputs_P": sha(os.path.join(e54.OUT, "inputs_P.jsonl")), "screens": {n: {"path": p, "sha256": sha(p)} for n, p in e57.SCREENS.items()},
         "arm": {"init": "runs/exp57/stage2", "updates": UPDATES, "micro_batch": 8, "lora_r": 16, "lr_lora": 5e-5, "lr_head": 1e-5, "warmup": 100,
                 "seed": 1, "objective": "ce", "text_only": True, "trainable": "decision LoRA + decision head"},
         "temperature": "none (raw), in memory, every evaluation", "image": None,
         "requests": {"new": {c: rows[c] for c in ("fit", "G", "P", "H")}, "start": {c: rows[c] for c in MAIN},
                      "total": sum(rows[c] for c in ("fit", "G", "P", "H")) + sum(rows[c] for c in MAIN),
                      "questions": sum(sum(len(r["questions"]) for r in jsonl(os.path.join(FACTS, FILE_OF[c]))) for c in ("fit", "G", "P", "H"))
                      + sum(sum(len(r["questions"]) for r in jsonl(os.path.join(FACTS, FILE_OF[c]))) for c in MAIN)},
         "candidate_eligible": False, "started": time.strftime("%F %T"), "deadline": deadline,
         "deadline_text": time.strftime("%F %T", time.localtime(deadline))}
    os.makedirs(out, exist_ok=True)
    write_json(path, s)
    return s


def check_start(out: str = OUT, code: bool = True) -> Dict:
    st = json.load(open(os.path.join(out, "START.json")))
    if code:
        bad = [f for f, h in st["files"].items() if sha(f) != h]
        if bad:
            raise SystemExit("code differs from START: %s" % bad)
    if bound_data() != st["data"]:
        raise SystemExit("data/facts differs from START")
    if fz.weights(E57) != E57_SHA:
        raise SystemExit("EXP57's end changed")
    return st


# ------------------------------------------------------------------ train / freeze

def stage_train(out: str = OUT, poll: float = 5.0, cmd: Optional[List[str]] = None) -> int:
    check_start(out)
    log = os.path.join(out, "logs", "stage2.log")
    proc = subprocess.Popen(cmd or ["uv", "run", "devision-pipeline", YAML])
    seen = -1
    while True:
        rc = proc.poll()
        if os.path.exists(log):
            n = e57.progress_of(open(log, errors="replace").read().splitlines())
            if n is not None and n != seen:
                e54.progress(out, "train", n, UPDATES)
                seen = n
        if rc is not None:
            return rc
        time.sleep(poll)


def stage_freeze(out: str = OUT) -> Dict:
    from safetensors.torch import load_file
    check_start(out)
    rep = json.load(open(os.path.join(out, "stage2", "train_report.json")))
    if len(rep["losses"]) != UPDATES:
        raise SystemExit("training ran %d updates, not %d" % (len(rep["losses"]), UPDATES))
    if fz.weights(E57) != E57_SHA:
        raise SystemExit("EXP57 changed")
    res = e57.export_check(load_file(os.path.join(E57, "stage2", "model.safetensors")), load_file(os.path.join(out, "stage2", "model.safetensors")))
    res["updates"] = len(rep["losses"])
    res["baseline"] = dict(E57_SHA)
    write_json(os.path.join(out, "export_check.json"), res)
    if not res["ok"]:
        raise SystemExit("export check failed: %s" % res)
    res["frozen"] = fz.freeze(out)["weights"]
    return res


# ------------------------------------------------------------------ decide

def set_inputs(cond: str):
    """(rows, inputs) of a condition: the frozen data/facts file's rows, each row's own state."""
    path = os.path.join(FACTS, FILE_OF[cond])
    if sha(path) != DATA_SHA[FILE_OF[cond]]:
        raise SystemExit("%s changed" % path)
    rows = {r["id"]: r for r in jsonl(path)}
    return rows, {rid: {"state": r["state_text"]} for rid, r in rows.items()}


def run_sets(out: str, plan: List[tuple], device: str = "auto") -> Dict:
    """plan: [(model label, checkpoint run, weights, [conditions])], each model loaded after its weights check."""
    from devision.infer import Decider, best_device
    s_sha = sha(os.path.join(out, "START.json"))
    dev = best_device() if device == "auto" else device
    total = sum(len(set_inputs(c)[0]) for _, _, _, cs in plan for c in cs)
    done, t0 = 0, time.time()
    for who, run, weights, conds in plan:
        if fz.weights(run) != weights:                                   # before any load
            raise SystemExit("%s's weights / config are not the bound ones" % who)
        d = Decider.load(os.path.join(run, "stage2"), device=dev)
        d.cfg.temperature, d.cfg.temperature_by_options = [1.0, 1.0, 1.0], {}
        for c in conds:
            rows, inputs = set_inputs(c)
            recs = e54.decide_condition(d, rows, inputs, on_request=lambda n, b=done: e54.progress(out, "decide", b + n, total))
            done += len(rows)
            name = c if who == "new" else "start_" + c
            ident = {"set": name, "model_of": who, "start_sha256": s_sha, "inputs_sha256": e57.hash_inputs(inputs), "model": weights,
                     "temperature": "none (raw)", "image": None, "device": dev}
            e57.write_set(out, name, recs, ident)
            print("%s done after %.1f s" % (name, time.time() - t0), flush=True)
        del d
    return {"seconds": time.time() - t0}


def stage_decide(out: str = OUT) -> Dict:
    check_start(out)
    new = fz.check(out, [])["weights"]
    return run_sets(out, [("new", out, new, list(MAIN)), ("start", E57, E57_SHA, list(MAIN))])


def stage_decide_h(out: str = OUT) -> Dict:
    check_start(out)
    new = fz.check(out, [])["weights"]
    return run_sets(out, [("new", out, new, ["H"])])


# ------------------------------------------------------------------ report

def comparisons(recs: List[Dict], rows: Dict[str, Dict], kind: str) -> List[Dict]:
    """The comparison questions of `kind` ("temp" / "force") with the semantic gold / predicted class (facts build
    gold_class: temperature sample A / sample B / same; magnets P1 / P2 / same), family, case, row axis."""
    out = []
    for r in recs:
        row = rows[r["sample_id"]]
        if row.get("kind") != kind or e54.b40.field_of(row["questions"][r["qid"]]["instructions"]) != "cmp":
            continue
        out.append({"key": "%s|%s" % (r["sample_id"], r["qid"]), "family": row["family"], "case": row["case"], "axis": row.get("axis", ""),
                    "gold": fb.gold_class(kind, r["gold_answer"]), "prediction": fb.gold_class(kind, r["prediction"]), "right": bool(r["correct"])})
    return out


CLASSES = {"temp": ("sample A", "sample B", "same"), "force": ("P1", "P2", "same")}


def summary(cs: List[Dict], kind: str) -> Dict:
    if not cs:
        return {"n": 0}
    rate = lambda xs: sum(x["right"] for x in xs) / len(xs) if xs else None  # noqa: E731
    fam = defaultdict(list)
    for c in cs:
        fam[c["family"]].append(c["right"])
    two = [v for v in fam.values() if len(v) == 2]
    return {"n": len(cs), "accuracy": rate(cs), "recall": {g: rate([c for c in cs if c["gold"] == g]) for g in CLASSES[kind]},
            "gold": dict(Counter(c["gold"] for c in cs)), "predictions": dict(Counter(c["prediction"] for c in cs)),
            "by_axis": {a: {"n": len([c for c in cs if c["axis"] == a]), "accuracy": rate([c for c in cs if c["axis"] == a])}
                        for a in sorted({c["axis"] for c in cs})},
            "both_views_right": {"families": len(two), "rate": sum(all(v) for v in two) / len(two) if two else None}}


def screen(out: str, name: str, start_eval: Dict[str, str]) -> Dict:
    """EXP57-R1-01 / 02's screen check, with EXP57 (not EXP50-T) as the start."""
    return e57.screen(out, name, start_eval, start_run=E57)


def gates(res: Dict) -> Dict:
    c, p, g = res.get("comparison", {}), res.get("paired", {}), {}

    def line(s, acc, rec):
        if not s or not s.get("n"):
            return "incomplete"
        vals = [s["accuracy"]] + list(s["recall"].values())
        if any(v is None for v in vals):
            return "incomplete"
        return bool(s["accuracy"] >= acc and all(v >= rec for v in s["recall"].values()))
    for kind in ("temp", "force"):
        g["fit_" + kind] = line(c.get("fit:" + kind), 0.95, 0.90)
        base = line(c.get("G:" + kind), 0.80, 0.60)
        d = p.get("G:" + kind)
        g["G_" + kind] = base if base == "incomplete" or base is False else ("incomplete" if d is None else bool(d["ci95"][0] > 0))
    s, d = c.get("P:temp"), p.get("P:temp")
    g["P_temp"] = "incomplete" if not s or not s.get("n") or d is None else bool(s["accuracy"] >= 0.65 and d["ci95"][0] > 0)
    rd = res.get("reading", {})
    if "G" in rd and "P" in rd:
        p_use = e54.gates({"conditions": {"P": rd["P"]}, "differences": {}})["P_use"]
        g["reading"] = bool(rd["G"]["all"] >= 0.98 and rd["P"]["all"] >= 0.866 and p_use is True)
    else:
        g["reading"] = "incomplete"
    sc = res.get("screens", {})
    g["screens"] = "incomplete" if set(sc) != set(e57.SCREENS) else bool(all(v["difference"] >= -0.02 for v in sc.values()))
    return g


def branch(g: Dict) -> str:
    """EXP57's order, fit first; a missing item is incomplete, never a pass or a fail; never a candidate."""
    fit = [g["fit_temp"], g["fit_force"]]
    if False in fit:
        return "fit not learned (%s): constrains only this supervision / budget" % [k for k in ("fit_temp", "fit_force") if g[k] is False]
    if "incomplete" in fit:
        return "incomplete: fit missing -- missing is not a low score"
    gg = [g["G_temp"], g["G_force"]]
    if False in gg:
        return "fit learned, G-tune below its line (%s): consumption / rule generalisation not established" % [k for k in ("G_temp", "G_force") if g[k] is False]
    missing = [k for k in ("G_temp", "G_force", "P_temp", "reading", "screens") if g[k] == "incomplete"]
    if missing:
        return "incomplete: %s missing -- missing is not a low score" % missing
    if g["P_temp"] is False:
        return "G passes, temperature P does not: reading / comparison coupling"
    if g["reading"] is False or g["screens"] is False:
        return "G and P pass, protection fails: protection repair only, no release"
    return ("fit, G (both kinds), temperature P, reading retention and exist / science pass: next (separate permissions) an "
            "automatic magnet-size source and a live B1 check; H is a hybrid, never the product path; not a candidate")


def report(out: str = OUT, finalize: str = "") -> Dict:
    res: Dict[str, Any] = {"states": {}, "comparison": {}, "paired": {}, "reading": {}, "screens": {}, "candidate_eligible": False}
    if finalize:
        res["finalized"] = {"exit_codes": finalize, "note": "missing items are not low scores"}
    if not os.path.exists(os.path.join(out, "START.json")):
        res["branch"] = "incomplete: no START (nothing ran)"
        return write_report(out, res)
    try:
        st = check_start(out, code=False)
    except (SystemExit, OSError, ValueError, KeyError) as e:
        res["branch"] = "invalid: the run's identity does not verify (%s) -- no verdict" % e
        return write_report(out, res)
    if os.path.exists(os.path.join(out, "export_check.json")):
        res["export_check"] = json.load(open(os.path.join(out, "export_check.json")))
    new = None
    try:
        new = fz.check(out, [])["weights"]
    except SystemExit as e:
        res["states"]["frozen"] = "missing: %s" % e
    recs: Dict[str, List[Dict]] = {}
    rows_of: Dict[str, Dict] = {}
    for name in list(MAIN) + ["H"] + ["start_" + c for c in MAIN]:
        cond = name.replace("start_", "")
        try:
            if new is None:
                raise SystemExit("no frozen end")
            rows, inputs = set_inputs(cond)
            recs[name] = e57.read_set(out, name, rows, E57_SHA if name.startswith("start_") else new, inputs)
            rows_of[name] = rows
            res["states"][name] = "completed"
        except (SystemExit, OSError, ValueError, KeyError) as e:
            res["states"][name] = "missing / invalid: %s" % e
    for name, rs in recs.items():
        for kind in ("temp", "force"):
            res["comparison"]["%s:%s" % (name, kind)] = summary(comparisons(rs, rows_of[name], kind), kind)
    for cond in MAIN:
        if cond in recs and "start_" + cond in recs:
            for kind in ("temp", "force"):
                a, b = comparisons(recs[cond], rows_of[cond], kind), comparisons(recs["start_" + cond], rows_of[cond], kind)
                if a:
                    res["paired"]["%s:%s" % (cond, kind)] = e57.paired_difference(a, b)
    qs53 = {q["key"]: q for q in jsonl(e54.QUESTIONS) if q["set"] == "rl_tune"}
    pin = e54.load_inputs(e54.OUT, "P")
    tune = {r["id"]: r for r in jsonl(e54.TUNE)}
    p_right = {k: {(o, qn): v for o, qn, v in pin[q["id"]]["table"]}[e54.TEMPLATES[tune[q["id"]]["questions"][q["qid"]]["instructions"]]] == q["target"]
               for k, q in qs53.items()}
    for name in ("G", "P", "H", "start_G", "start_P"):
        if name in recs:
            res["reading"][name] = e57.reading_summary(recs[name], rows_of[name], qs53, p_right)
    for name in e57.SCREENS:
        try:
            res["screens"][name] = screen(out, name, st["start_screens"][name])
            res["states"][name] = "completed"
        except (SystemExit, OSError, ValueError, KeyError) as e:
            res["states"][name] = "missing / invalid: %s" % e
    res["gates"] = gates(res)
    res["branch"] = branch(res["gates"])
    res["main_complete"] = all(res["states"].get(k) == "completed" for k in list(MAIN) + ["start_" + c for c in MAIN] + list(e57.SCREENS))
    res["chain_complete"] = res["main_complete"] and res["states"].get("H") == "completed"
    return write_report(out, res)


def write_report(out: str, res: Dict) -> Dict:
    os.makedirs(out, exist_ok=True)
    write_json(os.path.join(out, "report.json"), res)
    lines = [json.dumps({k: res[k] for k in ("finalized", "states", "gates", "branch", "main_complete", "chain_complete", "candidate_eligible") if k in res},
                        indent=1, default=str)]
    for k, v in res.get("comparison", {}).items():
        if v.get("n"):
            lines.append("cmp %s acc %.3f recall %s axis %s both_views %s" % (k, v["accuracy"], {a: None if x is None else round(x, 3) for a, x in v["recall"].items()},
                                                                            {a: round(x["accuracy"], 3) for a, x in v["by_axis"].items()}, v["both_views_right"]))
    for k, v in res.get("paired", {}).items():
        lines.append("paired %s %+.3f [%+.3f, %+.3f]" % (k, v["value"], *v["ci95"]))
    for k, v in res.get("reading", {}).items():
        lines.append("reading %s all %.3f both %.3f twin %.3f read_right %s" % (k, v["all"], v["both"], v["twin"], v["read_right"]))
    for k, v in res.get("screens", {}).items():
        lines.append("screen %s new %.3f start %.3f diff %+.3f" % (k, v["new"], v["start"], v["difference"]))
    open(os.path.join(out, "report.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["start", "train", "freeze", "decide", "decide_h", "report"])
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
    elif a.stage == "decide_h":
        print(json.dumps(stage_decide_h(), indent=1))
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
