"""exp57: decision-side comparison CE on text (critic EXP57-PREP-01..03, DECISION-CMP-PREP-R1-01 / 02;
docs/experiments/2026-10-18-decision-cmp-plan.md). One arm, seed 1: EXP50-T (1a8e5439 / 99c6246e) -> configs/exp57.yaml
(rlcd text_only: image=None exactly as ordinary decide, only LoRA r16 and the decision head, gold CE, 1,000 updates of
micro batch 8 on data/cmp/train.jsonl, fixed end step, raw temperature 1 in every evaluation, nothing tuned on tune).
Stages (fresh only, runs/exp57):

  start   model-free: identities -- EXP50-T's frozen weights / config and its recorded exist / science, data/cmp (train,
          fit slice, fit rows, PLAN, MANIFEST) and the plan's 1,000 updates / 4,000 two-view units / exposure 1, EXP55's
          G inputs / details (START, registry), EXP56's P inputs / details (START, registry, report), questions /
          rl_tune, the YAML, this code -- and the per-stage request counts -> START.json.
  train   MPS: devision-pipeline configs/exp57.yaml, its log tailed into progress lines for the watch.
  freeze  model-free (tensors only): the training ran 1,000 updates; the exported weights against EXP50-T's, tensor by
          tensor: vision, projector, conditioning / local-view geometry and the word embeddings (the selector reads
          them) identical, the changed tensors only in the encoder layers (LoRA merged) and the decision head; then
          runs/exp57/FROZEN.json (scripts/v04b/freeze.py).
  decide  MPS: the frozen EXP57 checkpoint, raw temperature, image=None: fit (200 rows of the fit slice, gold tables),
          G (EXP55's inputs_G), P (EXP56's live inputs_P), then the frozen start EXP50-T on fit (the start's G / P
          records are EXP55's / EXP56's, reused after their checks). Each set registered as soon as it completes.
  (exist / science: devision-eval on the frozen checkpoint in run.sh, recorded with freeze.py record.)
  report  model-free: temperature comparison (overall, A / B / neither recall, per axis, both views of a family right,
          predictions) on fit / G / P with case-cluster intervals against the start; magnet comparisons reported apart;
          reading retention (exp54's statistics, read_right, P_use); exist / science against EXP50-T's recorded results;
          the pre-written gates and branch; --finalize after a failed / stopped stage (missing is not a low score).

    uv run python scripts/exp57/exp57.py start --deadline EPOCH | train | freeze | decide | report [--finalize CODES]
"""
import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional

HERE = os.path.dirname(__file__)
OUT, E55, E56 = "runs/exp57", "runs/exp55", "runs/exp56"
CMP = "data/cmp"
YAML = "configs/exp57.yaml"
FIT = os.path.join(CMP, "fit.jsonl")
DATA_SHA = {"train.jsonl": "b0f389da23454c7b9612797814e922bf185e0208c155abca668aa6fa13111174",
            "fit_slice.json": "7d3486262f02462e6e39773c987d4f4615690b5ca8328b45f6c78e5352f80a79",
            "PLAN.json": "c2288175a5804717a716a3156290cf0c2313468783483473f7a4e983c4bc759a",
            "MANIFEST.json": "d4ab423dba1abecfd291c2250e64f525b9fd148833954295897f81779e3f98eb"}
E55_SHA = {"START.json": "bf0595676f54e8decc666cebecf3693febce386a1be14209d8dd98526a644e12",
           "registry.json": "b185564f6b7ce5e7b58164d1694c2db4e99812f55bf7072166f28ee17efc9f6e",
           "inputs_G.jsonl": "a9c453c242cc8a9f4e0b8d55442d6dd6f32621eab19b7145c8db2969cebf91b2",
           "details_G.jsonl": "ad792b3ecbcd46a15c8fca8465f37c3d3bc3838622f6464bb427dd692d5515ba"}
E56_SHA = {"START.json": "ccdd23f28e20b0795db9d6ec12d3c9df68591d6616e7dd72173041af4046d2d5",
           "registry.json": "84c24f2e597e8a9ef2c950a28247849c45c5c04a098433035ceab88a1f4e582e",
           "report.json": "70d235e1a33029e97b8c55ddfe9f34af448108f84019ae776fb1fcdceeb319f6",
           "inputs_P.jsonl": "5792961f3d16972bc4e031e0493ddacfac9f322888b4b0f91208857322d987ac",
           "details_P.jsonl": "13b98a8628519704af0537e4653d9a76f71e770f5a7b5d11451b74b0ae48b1a5"}
UPDATES = 1000
SETS = ("fit", "G", "P")                      # the new checkpoint's decide sets, in order; then base_fit (EXP50-T)
SCREENS = {"exist": "data/r13b/panel_exist.jsonl", "science": "data/r13b/panel_science.jsonl"}
FROZEN_PREFIXES = ("vision.", "projector.", "condition.", "decision.encoder.embeddings.")
FILES = ["scripts/exp57/exp57.py", "scripts/exp57/run.sh", YAML, "src/devision/train/rlcd.py", "src/devision/train/pipeline.py",
         "src/devision/infer.py", "src/devision/model.py", "scripts/exp54/exp54.py", "scripts/exp53/watch.py", "scripts/v04b/freeze.py"]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, rel))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


e54 = _load("exp54", "../exp54/exp54.py")
fz = _load("v04b_freeze", "../v04b/freeze.py")
r42, r41, sha, write_json, register = e54.r42, e54.r41, e54.sha, e54.write_json, e54.register


def jsonl(path: str) -> List[Dict]:
    return [json.loads(x) for x in open(path) if x.strip()]


# ------------------------------------------------------------------ data (CPU prep and start)

def fit_rows(cmp: str = CMP) -> List[Dict]:
    """The fit slice's rows: every train.jsonl row (both views) of the 100 fixed families, in file order."""
    fams = set(json.load(open(os.path.join(cmp, "fit_slice.json")))["families"])
    return [r for r in jsonl(os.path.join(cmp, "train.jsonl")) if r["family"] in fams]


def write_fit(cmp: str = CMP) -> Dict:
    rows = fit_rows(cmp)
    path = os.path.join(cmp, "fit.jsonl")
    with open(path, "w") as fh:
        fh.write("".join(json.dumps(r) + "\n" for r in rows))
    return {"rows": len(rows), "families": len({r["family"] for r in rows}), "sha256": sha(path)}


def check_plan(plan: Dict) -> None:
    want = {"updates": UPDATES, "units": 4000, "unit_sizes": {"2": 4000}, "split_units": 0, "exposure_per_question": {"1": 8000},
            "every_item_once": True, "seed": 1, "micro_batch": 8, "batch_sizes": {"8": UPDATES}}
    bad = {k: plan.get(k) for k, v in want.items() if plan.get(k) != v}
    if bad:
        raise SystemExit("data/cmp/PLAN.json is not 1,000 updates of 4,000 whole two-view units, each question once: %s" % bad)


def check_yaml(path: str = YAML) -> Dict:
    import yaml
    y = yaml.safe_load(open(path))
    st = y["stages"]
    p = st[0]["params"] if len(st) == 1 else {}
    want = {"epochs": 1, "objective": "ce", "text_only": True, "micro_batch": 8, "lora_r": 16, "lr_lora": 5e-5, "lr_head": 1e-5,
            "warmup": 100, "seed": 1, "amp": "off"}
    if y.get("name") != "exp57" or len(st) != 1 or st[0].get("init") != "runs/exp50-t/stage2" or st[0].get("data") != os.path.join(CMP, "train.jsonl") \
            or st[0].get("val") != FIT or any(p.get(k) != v for k, v in want.items()) or y.get("evaluate"):
        raise SystemExit("configs/exp57.yaml is not the fixed EXP57 arm")
    return y


def bound(name: str, d: str, want: Dict[str, str]) -> Dict[str, str]:
    bad = [f for f, h in want.items() if sha(os.path.join(d, f)) != h]
    if bad:
        raise SystemExit("%s's %s changed" % (name, bad))
    for reg, files in json.load(open(os.path.join(d, "registry.json"))).items():
        for f, h in files.items():
            if sha(os.path.join(d, f)) != h:
                raise SystemExit("%s's registered %s changed" % (name, f))
    return dict(want)


def stage_start(deadline: int, out: str = OUT) -> Dict:
    path = os.path.join(out, "START.json")
    if os.path.exists(path):
        raise SystemExit("%s exists (fresh only)" % path)
    t = fz.check(e54.RUN, list(SCREENS))
    if t["weights"] != e54.T_SHA:
        raise SystemExit("EXP50-T is not its frozen 1a8e5439 / 99c6246e")
    data = bound_data()
    check_plan(json.load(open(os.path.join(CMP, "PLAN.json"))))
    fit = jsonl(FIT)
    if fit != fit_rows():
        raise SystemExit("data/cmp/fit.jsonl is not the fit slice's rows of train.jsonl")
    e55, e56 = bound("EXP55", E55, E55_SHA), bound("EXP56", E56, E56_SHA)
    st56 = json.load(open(os.path.join(E56, "START.json")))
    if sha(e54.QUESTIONS) != st56["questions_sha256"] or sha(e54.TUNE) != st56["tune_sha256"]:
        raise SystemExit("questions / rl_tune are not those EXP55 / EXP56 bound")
    check_yaml()
    rows = jsonl(e54.TUNE)
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    requests = {"fit": len(fit), "G": len(rows), "P": len(rows), "base_fit": len(fit)}
    s = {"git_head": git("rev-parse", "HEAD"), "dirty_files": [f for f in FILES if git("status", "--porcelain", "--", f)],
         "files": {f: sha(f) for f in FILES}, "start_model": {"run": e54.RUN, **e54.T_SHA},
         "start_screens": {n: t["evals"][n] for n in SCREENS}, "data": {**data, "fit.jsonl": sha(FIT)},
         "exp55": e55, "exp56": e56, "questions_sha256": sha(e54.QUESTIONS), "tune_sha256": sha(e54.TUNE),
         "screens": {n: {"path": p, "sha256": sha(p)} for n, p in SCREENS.items()},
         "arm": {"seed": 1, "updates": UPDATES, "micro_batch": 8, "lora_r": 16, "lr_lora": 5e-5, "lr_head": 1e-5, "warmup": 100,
                 "objective": "ce", "text_only": True, "trainable": "decision LoRA + decision head",
                 "frozen": "vision, projector, conditioning / local-view geometry, word embeddings (selector), selector, char head"},
         "temperature": "none (raw), in memory, every evaluation", "image": None,
         "requests": {**requests, "questions": {"fit": sum(len(r["questions"]) for r in fit), "tune": sum(len(r["questions"]) for r in rows)},
                      "exist": len(jsonl(SCREENS["exist"])), "science": len(jsonl(SCREENS["science"]))},
         "budget_minutes": {"train": "6-9", "freeze": "< 1", "decide": "about 5 (1,600 text requests)", "exist_science": "about 5",
                            "report": "< 1", "cap": 45, "report_reserve_s": 120},
         "started": time.strftime("%F %T"), "deadline": deadline, "deadline_text": time.strftime("%F %T", time.localtime(deadline))}
    os.makedirs(out, exist_ok=True)
    write_json(path, s)
    return s


def bound_data() -> Dict[str, str]:
    bad = [f for f, h in DATA_SHA.items() if sha(os.path.join(CMP, f)) != h]
    if bad:
        raise SystemExit("data/cmp %s changed" % bad)
    return dict(DATA_SHA)


def check_start(out: str = OUT, code: bool = True) -> Dict:
    st = json.load(open(os.path.join(out, "START.json")))
    if code:
        bad = [f for f, h in st["files"].items() if sha(f) != h]
        if bad:
            raise SystemExit("code differs from START: %s" % bad)
    if bound_data() != {k: v for k, v in st["data"].items() if k != "fit.jsonl"} or sha(FIT) != st["data"]["fit.jsonl"]:
        raise SystemExit("data/cmp differs from START")
    return st


# ------------------------------------------------------------------ train

STEP = re.compile(r"^(?:step (\d+) \{|epoch \d+ step (\d+)/(\d+) )")


def progress_of(log_lines: List[str]) -> Optional[int]:
    """The last training step a pipeline log shows: 'step N {...}' (an evaluation row) or 'epoch E step N/M ...'."""
    last = None
    for x in log_lines:
        m = STEP.match(x)
        if m:
            last = int(m.group(1) or m.group(2))
    return last


def stage_train(out: str = OUT, poll: float = 5.0, cmd: Optional[List[str]] = None) -> int:
    check_start(out)
    log = os.path.join(out, "logs", "stage2.log")
    proc = subprocess.Popen(cmd or ["uv", "run", "devision-pipeline", YAML])
    seen = -1
    while True:
        rc = proc.poll()
        if os.path.exists(log):
            n = progress_of(open(log, errors="replace").read().splitlines())
            if n is not None and n != seen:
                e54.progress(out, "train", n, UPDATES)
                seen = n
        if rc is not None:
            return rc
        time.sleep(poll)


# ------------------------------------------------------------------ freeze

def export_check(start_tensors: Dict, end_tensors: Dict) -> Dict:
    """Tensor by tensor: the same names, the frozen parts identical; which groups changed."""
    import torch
    if set(start_tensors) != set(end_tensors):
        raise SystemExit("the exported checkpoint has other tensors than EXP50-T: %s" % sorted(set(start_tensors) ^ set(end_tensors))[:5])
    changed = sorted(n for n in start_tensors if not torch.equal(start_tensors[n], end_tensors[n]))
    frozen = [n for n in start_tensors if n.startswith(FROZEN_PREFIXES)]
    moved = [n for n in changed if n.startswith(FROZEN_PREFIXES)]
    outside = [n for n in changed if not n.startswith("decision.")]
    groups = Counter("decision.encoder.layers" if n.startswith("decision.encoder.layers.") else n.split(".")[0] + "." + n.split(".")[1]
                     for n in changed)
    return {"tensors": len(start_tensors), "frozen_checked": len(frozen), "frozen_changed": moved, "changed_outside_decision": outside,
            "changed": len(changed), "changed_groups": dict(groups), "ok": bool(frozen and not moved and not outside and changed)}


def stage_freeze(out: str = OUT) -> Dict:
    from safetensors.torch import load_file
    check_start(out)
    rep = json.load(open(os.path.join(out, "stage2", "train_report.json")))
    if len(rep["losses"]) != UPDATES:
        raise SystemExit("training ran %d updates, not %d" % (len(rep["losses"]), UPDATES))
    if fz.weights(e54.RUN) != e54.T_SHA:
        raise SystemExit("EXP50-T changed")
    res = export_check(load_file(os.path.join(e54.RUN, "stage2", "model.safetensors")), load_file(os.path.join(out, "stage2", "model.safetensors")))
    res["updates"] = len(rep["losses"])
    write_json(os.path.join(out, "export_check.json"), res)
    if not res["ok"]:
        raise SystemExit("export check failed: %s" % res)
    res["frozen"] = fz.freeze(out)["weights"]
    return res


# ------------------------------------------------------------------ decide

def set_inputs(name: str, tune: Dict[str, Dict], fit: Dict[str, Dict]):
    """(rows, inputs) of a set: fit = the fit rows with their own state (gold tables); G / P = EXP55's / EXP56's
    registered inputs on the rl_tune rows."""
    if name in ("fit", "base_fit"):
        return fit, {rid: {"state": r["state_text"]} for rid, r in fit.items()}
    src = {"G": (E55, "inputs_G.jsonl", E55_SHA), "P": (E56, "inputs_P.jsonl", E56_SHA)}[name]
    path = os.path.join(src[0], src[1])
    if sha(path) != src[2][src[1]]:
        raise SystemExit("%s changed" % path)
    return tune, {r["id"]: r for r in jsonl(path)}


def write_set(out: str, name: str, recs: List[Dict], ident: Dict) -> None:
    f = "details_%s.jsonl" % name
    with open(os.path.join(out, f), "w") as fh:
        fh.write(json.dumps({"identity": ident}) + "\n")
        fh.write("".join(json.dumps(r) + "\n" for r in recs))
    register(out, "decide_" + name, {f: sha(os.path.join(out, f))})


def stage_decide(out: str = OUT, device: str = "auto") -> Dict:
    from devision.infer import Decider, best_device
    st = check_start(out)
    s_sha = sha(os.path.join(out, "START.json"))
    new = fz.check(out, [])["weights"]
    tune = {r["id"]: r for r in jsonl(e54.TUNE)}
    fit = {r["id"]: r for r in jsonl(FIT)}
    if sha(e54.TUNE) != st["tune_sha256"]:
        raise SystemExit("rl_tune differs from START")
    dev = best_device() if device == "auto" else device
    total = 2 * len(fit) + 2 * len(tune)
    done, t0 = 0, time.time()
    for who, ck, names in (("exp57", os.path.join(out, "stage2"), SETS), ("exp50-t", os.path.join(e54.RUN, "stage2"), ("base_fit",))):
        model = new if who == "exp57" else e54.T_SHA
        if fz.weights(os.path.dirname(ck)) != model:            # before any load
            raise SystemExit("%s's weights / config are not the bound ones" % who)
        d = Decider.load(ck, device=dev)
        d.cfg.temperature, d.cfg.temperature_by_options = [1.0, 1.0, 1.0], {}      # raw, in memory only
        for name in names:
            rows, inputs = set_inputs(name, tune, fit)
            recs = e54.decide_condition(d, rows, inputs, on_request=lambda n, b=done: e54.progress(out, "decide", b + n, total))
            done += len(rows)
            ident = {"set": name, "model_of": who, "start_sha256": s_sha, "inputs_sha256": hash_inputs(inputs), "model": model,
                     "temperature": "none (raw)", "image": None, "device": dev}
            write_set(out, name, recs, ident)
            print("%s (%s) done after %.1f s" % (name, who, time.time() - t0), flush=True)
        del d
    return {"seconds": time.time() - t0, "device": dev}


def hash_inputs(inputs: Dict[str, Dict]) -> str:
    import hashlib
    return hashlib.sha256(json.dumps({k: inputs[k]["state"] for k in sorted(inputs)}, sort_keys=True).encode()).hexdigest()


def read_set(out: str, name: str, rows: Dict[str, Dict], model: Dict[str, str], inputs: Dict[str, Dict]) -> List[Dict]:
    """A set's records after every check: registered and unchanged; identity (set, START, the exact input states,
    model, raw temperature, no image); every question of the rows once with the data's gold, options in order,
    prediction = argmax."""
    f = "details_%s.jsonl" % name
    if e54.check_registered(out, "decide_" + name).get(f) != sha(os.path.join(out, f)):
        raise SystemExit("%s not registered" % f)
    lines = jsonl(os.path.join(out, f))
    ident, recs = lines[0]["identity"], lines[1:]
    want = {"set": name, "start_sha256": sha(os.path.join(out, "START.json")), "inputs_sha256": hash_inputs(inputs), "model": model,
            "temperature": "none (raw)", "image": None}
    if any(ident.get(k) != v for k, v in want.items()):
        raise SystemExit("%s: identity does not bind" % f)
    got = r41.check_details("exp57/%s" % name, list(rows.values()), recs)
    for r in recs:
        if r["options"] != list(rows[r["sample_id"]]["questions"][r["qid"]]["criteria"]):
            raise SystemExit("%s %s: options differ" % (r["sample_id"], r["qid"]))
    return [got[(r["sample_id"], r["qid"])] | {"nll": r["nll"]} for r in recs]


# ------------------------------------------------------------------ report

def gold_class(answer: str) -> str:
    return "neither" if answer.startswith("neither") else answer


def comparisons(recs: List[Dict], rows: Dict[str, Dict], kind: str = "temp") -> List[Dict]:
    """The comparison questions of `kind` with family / case / axis / view, gold class and correctness."""
    out = []
    for r in recs:
        row = rows[r["sample_id"]]
        if row.get("kind") != kind or e54.b40.field_of(row["questions"][r["qid"]]["instructions"]) != "cmp":
            continue
        out.append({"key": "%s|%s" % (r["sample_id"], r["qid"]), "family": row["family"], "case": row["case"], "axis": row.get("axis", ""),
                    "gold": gold_class(r["gold_answer"]), "prediction": gold_class(r["prediction"]), "right": bool(r["correct"])})
    return out


def comparison_summary(cs: List[Dict]) -> Dict:
    if not cs:
        return {"n": 0}
    rate = lambda xs: sum(x["right"] for x in xs) / len(xs) if xs else None  # noqa: E731
    fam = defaultdict(list)
    for c in cs:
        fam[c["family"]].append(c["right"])
    two = [v for v in fam.values() if len(v) == 2]
    return {"n": len(cs), "accuracy": rate(cs),
            "recall": {g: rate([c for c in cs if c["gold"] == g]) for g in ("sample A", "sample B", "neither")},
            "gold": dict(Counter(c["gold"] for c in cs)), "predictions": dict(Counter(c["prediction"] for c in cs)),
            "by_axis": {a: {"n": len(xs), "accuracy": rate(xs)} for a in sorted({c["axis"] for c in cs})
                        for xs in [[c for c in cs if c["axis"] == a]]},
            "both_views_right": {"families": len(two), "rate": sum(all(v) for v in two) / len(two) if two else None}}


def paired_difference(new: List[Dict], old: List[Dict]) -> Dict:
    """new - old comparison accuracy on the same questions, case-cluster bootstrap 95% interval (exp42 clusters)."""
    o = {c["key"]: c for c in old}
    if set(o) != {c["key"] for c in new} or len(o) != len(new):
        raise SystemExit("new and start comparisons are not on the same questions")
    cl = r42.clusters(new)
    by_fam = defaultdict(list)
    for c in new:
        by_fam[c["family"]].append(c)
    keys = sorted(cl)
    expand = lambda ks: [c for k in ks for f in sorted(cl[k]) for c in by_fam[f]]  # noqa: E731
    fn = lambda cs: sum(c["right"] - o[c["key"]]["right"] for c in cs) / len(cs) if cs else float("nan")  # noqa: E731
    res = r41.interval(fn, keys, expand)
    res["clusters"] = len(keys)
    res["wrong_to_right"] = sum(c["right"] and not o[c["key"]]["right"] for c in new)
    res["right_to_wrong"] = sum(o[c["key"]]["right"] and not c["right"] for c in new)
    return res


def reading_summary(recs: List[Dict], rows: Dict[str, Dict], qs53: Dict, p_right: Dict[str, bool]) -> Dict:
    rr = e54.reading_records(recs, rows, qs53)
    s = e54.summary(e54.units(rr))
    s["nll"] = sum(r["nll"] for r in rr) / len(rr)
    ok = [r for r in rr if p_right[r["key"]]]
    s["read_right"] = {"n": len(ok), "accuracy": sum(r["right"] for r in ok) / len(ok) if ok else None}
    return s


def fit_reading(recs: List[Dict], rows: Dict[str, Dict]) -> Dict:
    rr = [r for r in recs if e54.b40.field_of(rows[r["sample_id"]]["questions"][r["qid"]]["instructions"]) != "cmp"]
    return {"n": len(rr), "accuracy": sum(r["correct"] for r in rr) / len(rr) if rr else None}


def screen(out: str, name: str, start_eval: Dict[str, str]) -> Dict:
    """The new checkpoint's recorded screen against EXP50-T's recorded one: accuracy, difference, image-bootstrap
    interval on the paired questions."""
    fz.check(out, [name])
    fz.check(e54.RUN, [name])
    if fz.check(e54.RUN, [name])["evals"][name] != start_eval:
        raise SystemExit("EXP50-T's %s is not the one START bound" % name)
    a = json.load(open(os.path.join(out, "eval", name + ".json")))
    b = json.load(open(os.path.join(e54.RUN, "eval", name + ".json")))
    for s in (a, b):
        if s["run"].get("temperature") != "none (raw)" or s["run"].get("control") != "none" or s["run"].get("data_sha256") != sha(SCREENS[name]):
            raise SystemExit("%s: not the raw / no-control evaluation of %s" % (name, SCREENS[name]))
    da = {(r["sample_id"], r["qid"]): r for r in jsonl(os.path.join(out, "eval", name + ".details.jsonl"))}
    db = {(r["sample_id"], r["qid"]): r for r in jsonl(os.path.join(e54.RUN, "eval", name + ".details.jsonl"))}
    if set(da) != set(db):
        raise SystemExit("%s: different questions" % name)
    by_img = defaultdict(list)
    for k, r in da.items():
        by_img[r["image_id"]].append(k)
    keys = sorted(by_img)
    expand = lambda ks: [q for k in ks for q in by_img[k]]  # noqa: E731
    fn = lambda qs: sum(da[q]["correct"] - db[q]["correct"] for q in qs) / len(qs)  # noqa: E731
    return {"new": a["accuracy_all"], "start": b["accuracy_all"], "difference": a["accuracy_all"] - b["accuracy_all"],
            "paired": r41.interval(fn, keys, expand), "n": len(da)}


def gates(res: Dict) -> Dict:
    c, g = res.get("comparison", {}), {}
    rec_ok = lambda s, m: all(v is not None and v >= m for v in s["recall"].values())  # noqa: E731
    f = c.get("fit")
    g["fit"] = "incomplete" if f is None else bool(f["accuracy"] >= 0.95 and rec_ok(f, 0.90))
    x, d = c.get("G"), res.get("paired", {}).get("G")
    g["G_tune"] = "incomplete" if x is None or d is None else bool(x["accuracy"] >= 0.80 and rec_ok(x, 0.60) and d["ci95"][0] > 0)
    x, d = c.get("P"), res.get("paired", {}).get("P")
    g["P_tune"] = "incomplete" if x is None or d is None else bool(x["accuracy"] >= 0.65 and d["ci95"][0] > 0)
    rd = res.get("reading", {})
    if "G" in rd and "P" in rd:
        p_use = e54.gates({"conditions": {"P": rd["P"]}, "differences": {}})["P_use"]
        g["reading"] = bool(rd["G"]["all"] >= 0.98 and rd["P"]["all"] >= 0.866 and p_use is True)
        g["reading_detail"] = {"G_all": rd["G"]["all"], "P_all": rd["P"]["all"], "P_use": p_use}
    else:
        g["reading"] = "incomplete"
    sc = res.get("screens", {})
    g["screens"] = "incomplete" if set(sc) != set(SCREENS) else bool(all(v["difference"] >= -0.02 for v in sc.values()))
    return g


def branch(g: Dict) -> str:
    order = ("fit", "G_tune", "P_tune", "reading", "screens")
    missing = [k for k in order if g.get(k) == "incomplete"]
    if g.get("fit") is False:
        return "fit not learned: constrains only this supervision / optimisation budget (no route-level conclusion)"
    if g.get("fit") is True and g.get("G_tune") is False:
        return "fit learned, G-tune below its line: generalisation / rule application not established"
    if missing:
        return "incomplete: %s missing -- G never stands in for P, missing is not a low score" % missing
    if g["P_tune"] is False:
        return "G passes, P does not: check how reading errors couple with the comparison"
    if g["reading"] is False or g["screens"] is False:
        return "G and P pass, protection fails (reading retention / exist / science): protection repair only, no release"
    return ("fit, G, P, reading retention and exist / science all pass: next, re-bind the reading components to this "
            "checkpoint and run the live path; not a release (seven protections, v0.3 transfer, full release checks)")


def report(out: str = OUT, finalize: str = "") -> Dict:
    st = json.load(open(os.path.join(out, "START.json"))) if os.path.exists(os.path.join(out, "START.json")) else None
    res: Dict = {"states": {}, "comparison": {}, "magnets": {}, "paired": {}, "reading": {}, "screens": {}}
    if st is None:
        res["states"]["start"] = {"state": "missing"}
    else:
        check_start(out, code=False)
        if sha(e54.QUESTIONS) != st["questions_sha256"] or sha(e54.TUNE) != st["tune_sha256"]:
            raise SystemExit("questions / rl_tune differ from START: no report")
    tune = {r["id"]: r for r in jsonl(e54.TUNE)}
    fit = {r["id"]: r for r in jsonl(FIT)}
    qs53 = {q["key"]: q for q in jsonl(e54.QUESTIONS) if q["set"] == "rl_tune"}
    pin = e54.load_inputs(e54.OUT, "P")                       # EXP53's frozen readings: the read-right set (1,470)
    p_right = {k: {(o, qn): v for o, qn, v in pin[q["id"]]["table"]}[e54.TEMPLATES[tune[q["id"]]["questions"][q["qid"]]["instructions"]]] == q["target"]
               for k, q in qs53.items()}
    for f in ("export_check.json",):
        if os.path.exists(os.path.join(out, f)):
            res["export_check"] = json.load(open(os.path.join(out, f)))
    recs: Dict[str, List[Dict]] = {}
    new = None
    try:
        new = fz.check(out, [])["weights"]
    except SystemExit as e:
        res["states"]["frozen"] = {"state": "missing", "reason": str(e)}
    for name in SETS + ("base_fit",):
        try:
            if new is None:
                raise SystemExit("no frozen checkpoint")
            rows, inputs = set_inputs(name, tune, fit)
            recs[name] = read_set(out, name, rows, e54.T_SHA if name == "base_fit" else new, inputs)
            res["states"][name] = {"state": "completed"}
        except (SystemExit, OSError, ValueError, KeyError) as e:
            res["states"][name] = {"state": "missing" if not os.path.exists(os.path.join(out, "details_%s.jsonl" % name)) else "invalid", "reason": str(e)}
    base = {"G": e54.read_condition(E55, "G", tune), "P": e54.read_condition(E56, "P", tune)}   # the start's frozen records
    rows_of = {"fit": fit, "base_fit": fit, "G": tune, "P": tune}
    for name, rs in recs.items():
        cs = comparisons(rs, rows_of[name])
        res["comparison"][name] = comparison_summary(cs)
        res["magnets"][name] = comparison_summary(comparisons(rs, rows_of[name], "force"))
        if name in ("G", "P"):
            res["paired"][name] = paired_difference(cs, comparisons(base[name], tune))
            res["reading"][name] = reading_summary(rs, tune, qs53, p_right)
        if name in ("fit", "base_fit"):
            res["reading"][name] = fit_reading(rs, fit)
    if "fit" in recs and "base_fit" in recs:
        res["paired"]["fit"] = paired_difference(comparisons(recs["fit"], fit), comparisons(recs["base_fit"], fit))
    for name, b in base.items():
        res["comparison"]["start_" + name] = comparison_summary(comparisons(b, tune))
        res["reading"]["start_" + name] = reading_summary(b, tune, qs53, p_right)
    for name in SCREENS:
        try:
            if st is None:
                raise SystemExit("no START")
            res["screens"][name] = screen(out, name, st["start_screens"][name])
            res["states"][name] = {"state": "completed"}
        except (SystemExit, OSError, ValueError, KeyError) as e:
            res["states"][name] = {"state": "missing", "reason": str(e)}
    res["gates"] = gates(res)
    res["branch"] = branch(res["gates"])
    if finalize:
        res["finalized"] = {"exit_codes": finalize, "note": "the run did not complete; missing items are not low scores"}
    write_json(os.path.join(out, "report.json"), res)
    lines = ["INCOMPLETE run: %s" % finalize] if finalize else []
    for k, v in res["states"].items():
        if v["state"] != "completed":
            lines.append("%s %s: %s" % (k, v["state"], v.get("reason", "")))
    for k, v in res["comparison"].items():
        if v.get("n"):
            lines.append("cmp %s acc %.3f recall %s both_views %s by_axis %s pred %s" % (
                k, v["accuracy"], {g: None if x is None else round(x, 3) for g, x in v["recall"].items()}, v["both_views_right"],
                {a: round(x["accuracy"], 3) for a, x in v["by_axis"].items()}, v["predictions"]))
    for k, v in res["paired"].items():
        lines.append("paired %s %+.3f [%+.3f, %+.3f] clusters %d w->r %d r->w %d" % (k, v["value"], *v["ci95"], v["clusters"], v["wrong_to_right"], v["right_to_wrong"]))
    for k, v in res["reading"].items():
        lines.append("reading %s %s" % (k, {x: (round(v[x], 3) if isinstance(v[x], float) else v[x]) for x in ("all", "both", "twin", "read_right", "accuracy") if x in v}))
    for k, v in res["screens"].items():
        lines.append("screen %s new %.3f start %.3f diff %+.3f paired [%+.3f, %+.3f]" % (k, v["new"], v["start"], v["difference"], *v["paired"]["ci95"]))
    lines.append("export %s" % res.get("export_check"))
    lines.append("gates %s\nbranch: %s" % (res["gates"], res["branch"]))
    open(os.path.join(out, "report.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["fit-rows", "start", "train", "freeze", "decide", "report"])
    ap.add_argument("--deadline", type=int, default=0)
    ap.add_argument("--finalize", default="")
    a = ap.parse_args(argv)
    if a.stage == "fit-rows":
        print(json.dumps(write_fit()))
    elif a.stage == "start":
        print(json.dumps({k: v for k, v in stage_start(a.deadline).items() if k not in ("files", "exp55", "exp56")}, indent=1))
    elif a.stage == "train":
        return stage_train()
    elif a.stage == "freeze":
        print(json.dumps(stage_freeze(), indent=1))
    elif a.stage == "decide":
        print(json.dumps(stage_decide(), indent=1))
    else:
        try:
            report(finalize=a.finalize)
        except (SystemExit, OSError, ValueError, KeyError) as e:
            msg = "REPORT FAILED: %s%s" % (e, " (after: %s)" % a.finalize if a.finalize else "")
            os.makedirs(OUT, exist_ok=True)
            open(os.path.join(OUT, "report.txt"), "w").write(msg + "\n")
            print(msg, file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
