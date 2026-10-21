"""exp58 (prepared, not run; critic LIVE-TRANSFER-PREP-01; docs/experiments/2026-10-18-exp58-live-transfer-plan.md): the
reading mode with EXP57's decision endpoint, live. Only after EXP57's own branch allows it and a separate model
permission; nothing here is chained to EXP57. NEW_SHA is EXP57's frozen end (962503fb / 09a5b447); the selector / head stay
EXP53's / EXP51's files (8f10f1fd / 0b548142). Modes: default (EXP57's pre-written branch passed -- it did not, so default
refuses) and explore-after-exp57-fit-fail (critic EXP58-EXPLORE-PREP-01: a disclosed post-hoc exploration bound to exactly
EXP57's frozen end, report e5a8565d and its gates fit false / the other four true; A + B1 only, never a candidate).

  start    model-free: EXP57 frozen (weights / config = NEW_SHA, its report's branch), EXP56's bundle components /
           pictures / reads / inputs_P, EXP57's details_P, the stage B data and v0.3 records, this code; a NEW reading
           bundle (devision.reading.write_bundle: the two component files copied, bound to EXP57's weights / config /
           tokenizer) -> START.json.
  live     stage A (MPS): devision.reading.Reading.load(EXP57, bundle) -- the product path -- on the 600 original
           rl_tune pictures, one request per picture with the row's questions, the actual picture passed as image=.
  fidelity model-free: the 1,800 live reads equal EXP56's (decoded string, validity, value), the 600 actual states equal
           EXP56's inputs_P, and the 2,400 answers equal EXP57's details_P (fixed EXP56 tables, same endpoint): same
           predictions, probabilities within 1e-5. Failure stops here (stage B does not start).
  transfer stage B1 (MPS): the same Reading object on dev_sqa temp / force and their fixed wrong-picture files (the
           wrong picture is the row's image in *_mism.jsonl), devision-eval's request grouping (32 per picture + text).
  guard    stage B2 (MPS): the seven protection panels through the same Reading object and grouping.
  report   model-free: exp21's 68:114 transfer (accuracy change and picture-gain change against v0.3's same-question
           records, the v0.4 target: >= +10 points, lower bound > 0; gain lower bound > 0), temp and force apart with the
           routed counts and error distribution; the seven panels against v0.3 (paired_metric; ext 13-source macro; a
           cost = <= -3 points with upper bound < 0), science's routed 20 apart. Missing items: incomplete.

Accuracy is argmax: v0.3's records used fitted temperatures, this run raw temperature 1 (EXP57-NOTE-01) -- per-question
temperature scaling does not move an argmax, so accuracies and picture gains compare; NLL / ECE are not compared.

    uv run python scripts/exp58/live.py start --deadline EPOCH | live | fidelity | transfer | guard | report [--finalize CODES]
"""
import argparse
import importlib.util
import io
import json
import os
import subprocess
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(__file__)
OUT, E56, E57 = "runs/exp58", "runs/exp56", "runs/exp57"
NEW_RUN = E57
# EXP57's frozen end (runs/exp57/FROZEN.json 709c2d01, verified by the critic 2026-10-18 22:45:30)
NEW_SHA: Optional[Dict[str, str]] = {"model.safetensors": "962503fbd5f41b6794e5b48a16fcd833b591b04b285d8f3afaa45f18d7c38131",
                                     "config.json": "09a5b44716b1b818df0372994b7581159625f4af7e267b6cf4e2ff4f12eb8d0d"}
DEFAULT, EXPLORE = "default", "explore-after-exp57-fit-fail"
# critic EXP58-EXPLORE-PREP-01 (EXP57 ruling 2026-10-18 22:45:30): a post-hoc transfer exploration allowed for exactly this
# EXP57 end -- its report (fit failed, the four other gates passed) is kept as it is; the exploration never makes EXP57 or
# the original investment rule pass, and its result is never a candidate (candidate_eligible false, B2 not permitted)
EXPLORE_ID = {"model": dict(NEW_SHA or {}), "report_sha256": "e5a8565d5dc13ae9c2f27aa66ef52c375b4d549e97b8455cf2b3b3dfa3b2e5b0",
              "gates": {"fit": False, "G_tune": True, "P_tune": True, "reading": True, "screens": True},
              "branch_prefix": "fit not learned"}
SELECTOR, SELECTOR_SHA = "runs/exp53/selector.pt", "8f10f1fd3de347dd7d5d2c375b7c4660ba97b75e65920e295132845d5cd2518a"
HEAD, HEAD_SHA = "runs/exp51/head_T.pt", "0b548142ed3697742ca9efaec7498e3aaa37d2872415d6c8c77f6f2ba9d0e3fa"
E56_SHA = {"START.json": "ccdd23f28e20b0795db9d6ec12d3c9df68591d6616e7dd72173041af4046d2d5",
           "registry.json": "84c24f2e597e8a9ef2c950a28247849c45c5c04a098433035ceab88a1f4e582e"}
V03 = "runs/exp27-transfer/eval/v03"
TRANSFER = {"sqa_temp": "data/exp21/dev_sqa_temp.jsonl", "sqa_force": "data/exp21/dev_sqa_force.jsonl",
            "sqa_temp_mism": "data/exp21/dev_sqa_temp_mism.jsonl", "sqa_force_mism": "data/exp21/dev_sqa_force_mism.jsonl"}
GUARD = {"exist": "data/r13b/panel_exist.jsonl", "science": "data/r13b/panel_science.jsonl", "ext": "data/ext/dev_ext.jsonl",
         "count_wide": "data/r13b/count_wide.jsonl", "count_narrow": "data/v11/dev_count_transfer.jsonl",
         "flip": "data/v6/dev_lr_pairs.jsonl", "vsr": "data/v5/dev_vsr.jsonl"}
A_PICTURES, A_READS, A_QUESTIONS = 600, 1800, 2400     # stage A's complete sets
MAX_QUESTIONS = 32                                  # devision-eval's default, as v0.3's records were made
FILES = ["scripts/exp58/live.py", "scripts/exp58/run.sh", "src/devision/reading.py", "src/devision/infer.py", "src/devision/model.py",
         "src/devision/image.py", "src/devision/train/evaluate.py", "scripts/exp56/exp56.py", "scripts/exp57/exp57.py",
         "scripts/exp21/readout.py", "scripts/exp53/watch.py"]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, rel))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


e57 = _load("exp57", "../exp57/exp57.py")
e56 = _load("exp56", "../exp56/exp56.py")
e54, fz, sha, write_json, register, jsonl = e57.e54, e57.fz, e57.sha, e57.write_json, e57.register, e57.jsonl


def new_sha() -> Dict[str, str]:
    if NEW_SHA is None:
        raise SystemExit("the EXP57 endpoint is a placeholder: fill NEW_SHA from runs/exp57/FROZEN.json after review")
    return NEW_SHA


# ------------------------------------------------------------------ one request path for every stage

def requests(samples: List[Dict]) -> List:
    """devision-eval's grouping (same picture file and text state, at most 32 questions, sample order), no control: the
    picture of a *_mism.jsonl row is its fixed wrong picture."""
    from devision.train.evaluate import _requests
    return _requests(samples, {}, "none", MAX_QUESTIONS)


def ask(reading, image_path: str, text: str, questions: Dict[str, Dict], data_root: str, pictures: Dict[str, str],
        diagnostics: Optional[Dict] = None) -> Dict:
    """One request through the product path: the actual picture bytes (checked against `pictures` when given) opened as
    an image and passed as image= to Reading.decide with the text state -- never inside state, never None. The bytes must
    hash to START's picture manifest (a picture outside it is refused)."""
    import hashlib

    from PIL import Image
    raw = open(os.path.join(data_root, image_path), "rb").read()
    if image_path not in pictures or hashlib.sha256(raw).hexdigest() != pictures[image_path]:
        raise SystemExit("%s differs from the bound picture" % image_path)
    with Image.open(io.BytesIO(raw)) as img:
        return reading.decide(text, questions, image=img.convert("RGB"), diagnostics=diagnostics)


def run_set(reading, samples: List[Dict], data_root: str, pictures: Dict[str, str], on_request=None) -> List[Dict]:
    """Every question of `samples` through `ask`, devision-eval's record fields (+ routed: the question went to the
    reading path), scored with evaluate.score_records."""
    from devision.reading import domain_of
    from devision.train.evaluate import _probabilities, score_records
    asked = {}
    for n, (image, text, items) in enumerate(requests(samples)):
        questions = {"%d:%s" % (i, qid): q for i, qid, q in items}
        res = ask(reading, image, text, questions, data_root, pictures)
        for key, (i, qid, q) in zip(questions, items):
            asked[(i, qid)] = (q, image, len(items), _probabilities(res["answers"][key]))
        if on_request:
            on_request(n + 1)
    recs = []
    for i, s in enumerate(samples):
        for qid in s["questions"]:
            q, image, size, probs = asked[(i, qid)]
            gold = s["gold"][qid]["probabilities"]
            pred = max(probs, key=probs.__getitem__)
            recs.append({"sample_id": s["id"], "qid": qid, "image_id": s["image_id"], "image_used": image, "source": s.get("source", ""),
                         "kind": s.get("kind", ""), "axis": s.get("axis", ""), "type": q["type"],
                         "options": list(q["criteria"]) if q["type"] == "choice" else ["false", "true"], "gold": gold,
                         "gold_answer": max(gold, key=gold.__getitem__), "probabilities": probs, "prediction": pred,
                         "control": "none", "questions_in_request": size, "routed": domain_of(q.get("instructions", "")) is not None})
    return score_records(recs)


def request_counts() -> Dict:
    """Requests / questions / routed questions per stage (model-free)."""
    from devision.reading import domain_of
    out: Dict[str, Any] = {"A_live": {"requests": len(jsonl(e54.TUNE)), "questions": sum(len(r["questions"]) for r in jsonl(e54.TUNE))}}
    for name, p in {**TRANSFER, **GUARD}.items():
        rows = jsonl(p)
        qs = [q for r in rows for q in r["questions"].values()]
        out[name] = {"requests": len(requests(rows)), "questions": len(qs),
                     "routed": dict(Counter(domain_of(q.get("instructions", "")) or "ordinary" for q in qs))}
    return out


# ------------------------------------------------------------------ start / stages

def picture_manifest(data_root: str = "data") -> Dict[str, str]:
    """Every picture any stage will open, by its path in the source files: stage A's 600 (which must be EXP56's frozen
    manifest entries) and the image field of every B row (true and fixed wrong pictures)."""
    e56pics = json.load(open(os.path.join(E56, "pictures.json")))
    pics = {}
    for r in jsonl(e54.TUNE):
        if r["image"] not in e56pics:
            raise SystemExit("%s is not in EXP56's picture manifest" % r["image"])
        pics[r["image"]] = e56pics[r["image"]]
    for p in {**TRANSFER, **GUARD}.values():
        for r in jsonl(p):
            pics.setdefault(r["image"], sha(os.path.join(data_root, r["image"])))
    return pics


def mode_record(mode: str) -> Dict:
    """The run's mode, checked against EXP57's report as it is now. default: EXP57's pre-written branch passed. explore:
    exactly EXPLORE_ID -- the frozen weights / config, the report's sha256, its five gates (fit false, the other four
    true) and its fit-first branch; anything else is refused (no general override)."""
    rp = os.path.join(NEW_RUN, "report.json")
    rep = json.load(open(rp))
    gates = {k: rep.get("gates", {}).get(k) for k in EXPLORE_ID["gates"]}
    if mode == DEFAULT:
        if not rep.get("branch", "").startswith("fit, G, P, reading retention and exist / science all pass"):
            raise SystemExit("EXP57's pre-written branch does not allow the live path: %s" % rep.get("branch"))
        return {"name": DEFAULT, "post_hoc": False, "exp57_report_sha256": sha(rp), "exp57_gates": gates, "exp57_branch": rep.get("branch")}
    if mode != EXPLORE:
        raise SystemExit("unknown mode %r" % mode)
    if new_sha() != EXPLORE_ID["model"] or fz.weights(NEW_RUN) != EXPLORE_ID["model"]:
        raise SystemExit("exploration is allowed only for EXP57's frozen 962503fb / 09a5b447")
    if sha(rp) != EXPLORE_ID["report_sha256"]:
        raise SystemExit("exploration is allowed only for EXP57's report e5a8565d (this one differs)")
    if gates != EXPLORE_ID["gates"] or not rep.get("branch", "").startswith(EXPLORE_ID["branch_prefix"]):
        raise SystemExit("exploration needs EXP57's exact gates %s / fit-first branch, got %s / %r" % (EXPLORE_ID["gates"], gates, rep.get("branch")))
    return {"name": EXPLORE, "post_hoc": True, "exp57_report_sha256": sha(rp), "exp57_gates": gates, "exp57_branch": rep["branch"],
            "purpose": "post-hoc transfer diagnosis after EXP57 failed its fit gate (critic EXP57 ruling 2026-10-18 22:45:30, "
                       "EXP58-EXPLORE-PREP-01): does the combination carry the measured in-domain comparison gain to the real "
                       "ScienceQA dev questions? EXP57 stays failed; not a candidate; B2 not permitted",
            "candidate_eligible": False}


def stage_start(deadline: int, out: str = OUT, data_root: str = "data", mode: str = DEFAULT) -> Dict:
    from devision.reading import write_bundle
    path = os.path.join(out, "START.json")
    if os.path.exists(path):
        raise SystemExit("%s exists (fresh only)" % path)
    want = new_sha()
    if fz.check(NEW_RUN, list(e57.SCREENS))["weights"] != want:
        raise SystemExit("EXP57 is not its frozen endpoint")
    mode_rec = mode_record(mode)
    e57.bound("EXP56", E56, E56_SHA)
    if sha(SELECTOR) != SELECTOR_SHA or sha(HEAD) != HEAD_SHA:
        raise SystemExit("the selector / head are not EXP53's / EXP51's files")
    st56 = json.load(open(os.path.join(E56, "START.json")))
    if sha(os.path.join(E56, "pictures.json")) != st56["pictures"]["pictures.json"]:
        raise SystemExit("EXP56's pictures.json changed")
    pics = picture_manifest(data_root)
    os.makedirs(out, exist_ok=True)
    write_json(os.path.join(out, "pictures.json"), pics)
    bundle = write_bundle(os.path.join(out, "bundle"), os.path.join(NEW_RUN, "stage2"), SELECTOR, HEAD,
                          {"selector": {"path": SELECTOR, "sha256": SELECTOR_SHA}, "head": {"path": HEAD, "sha256": HEAD_SHA}})
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    s = {"git_head": git("rev-parse", "HEAD"), "dirty_files": [f for f in FILES if git("status", "--porcelain", "--", f)],
         "files": {f: sha(f) for f in FILES}, "model": want, "mode": mode_rec, "exp57_report": sha(os.path.join(NEW_RUN, "report.json")),
         "exp57_details_P": sha(os.path.join(E57, "details_P.jsonl")),
         "bundle": {"reading.json": sha(os.path.join(out, "bundle", "reading.json")), **bundle["sha256"]},
         "pictures": {"pictures.json": sha(os.path.join(out, "pictures.json")), "count": len(pics),
                      "exp56_pictures_sha256": st56["pictures"]["pictures.json"]},
         "exp56": dict(E56_SHA), "data": {n: sha(p) for n, p in {**TRANSFER, **GUARD}.items()}, "tune_sha256": sha(e54.TUNE),
         "v03": {n: sha(os.path.join(V03, n + ".details.jsonl")) for n in {**TRANSFER, **GUARD}},
         "requests": request_counts(), "temperature": "none (raw), in memory", "max_questions": MAX_QUESTIONS,
         "started": time.strftime("%F %T"), "deadline": deadline, "deadline_text": time.strftime("%F %T", time.localtime(deadline))}
    write_json(path, s)
    return s


def check_run(out: str, data_root: str = "data", code: bool = True) -> Dict:
    """Before any load and in the report: START's combination as it is now -- code (optional), bundle files, the picture
    manifest (EXP56's contract: e56.check_combination), EXP57's weights / config = START's = NEW_SHA, the bundle's
    reading.json binding those weights / config, the source files, and (with data_root) every manifest picture's bytes."""
    st = json.load(open(os.path.join(out, "START.json")))
    e56.check_combination(out, st, code=code)
    if fz.weights(NEW_RUN) != st["model"] or st["model"] != new_sha():
        raise SystemExit("EXP57's weights / config are not START's")
    if mode_record(st["mode"]["name"]) != st["mode"]:
        raise SystemExit("EXP57's report / the run's mode no longer match START")
    rj = json.load(open(os.path.join(out, "bundle", "reading.json")))["sha256"]
    if {k: rj.get(k) for k in st["model"]} != st["model"]:
        raise SystemExit("the bundle does not bind START's weights / config")
    bad = [n for n, p in {**TRANSFER, **GUARD}.items() if sha(p) != st["data"][n]] + (["rl_tune"] if sha(e54.TUNE) != st["tune_sha256"] else [])
    if bad:
        raise SystemExit("source files differ from START: %s" % bad)
    if data_root:
        e56.check_pictures(out, data_root)
    return st


def load_reading(out: str, device: str = "auto", data_root: str = "data"):
    from devision.infer import best_device
    from devision.reading import Reading
    st = check_run(out, data_root)                                   # before the load
    r = Reading.load(os.path.join(NEW_RUN, "stage2"), os.path.join(out, "bundle"), device=best_device() if device == "auto" else device)
    r.decider.cfg.temperature, r.decider.cfg.temperature_by_options = [1.0, 1.0, 1.0], {}      # raw, in memory only
    return r, st, json.load(open(os.path.join(out, "pictures.json")))


def stage_live(out: str = OUT, data_root: str = "data") -> Dict:
    reading, st, pics = load_reading(out, data_root=data_root)
    return live_into(out, reading, st, pics, data_root)


def live_into(out: str, reading, st: Dict, pics: Dict[str, str], data_root: str) -> Dict:
    """Stage A's requests (one per rl_tune picture, the row's questions, the actual picture as image=) and its registered
    products: reads.jsonl, inputs_P.jsonl (the actual tables / states), details_P.jsonl bound to START / bundle / pictures."""
    rows = {r["id"]: r for r in jsonl(e54.TUNE)}
    recs, inputs, reads, t0 = [], {}, [], time.time()
    for n, rid in enumerate(sorted(rows)):
        diag: Dict = {}
        t = time.perf_counter()
        res = ask(reading, rows[rid]["image"], rows[rid].get("state_text", ""), rows[rid]["questions"], data_root, pics, diag)
        recs += e56.records(rows, rid, res)
        inputs[rid] = {"state": diag["state"], "table": diag["table"]}
        reads.append({"id": rid, "reads": diag["reads"], "read_seconds": diag["read_seconds"], "request_seconds": time.perf_counter() - t})
        e54.progress(out, "live", n + 1, len(rows))
    with open(os.path.join(out, "reads.jsonl"), "w") as fh:
        fh.write("".join(json.dumps(r) + "\n" for r in reads))
    e56.merge_register(out, "reads", {"reads.jsonl": sha(os.path.join(out, "reads.jsonl"))})
    e56.write_condition(out, "P", recs, sha(os.path.join(out, "START.json")), st["model"], "live", inputs,
                        {"reading_bundle": st["bundle"]["reading.json"], "pictures": st["pictures"]["pictures.json"]})
    return {"seconds": time.time() - t0}


def live_records(out: str, rows: Dict[str, Dict], st: Dict) -> List[Dict]:
    """The live details_P after every check: registered and unchanged; identity (condition P, this START, the registered
    inputs_P, START's model, raw temperature, START's bundle and pictures); every rl_tune question exactly once (gold,
    picture, valid probabilities, prediction = argmax, correct), options in order."""
    f = "details_P.jsonl"
    if e54.check_registered(out, "decide_P").get(f) != sha(os.path.join(out, f)):
        raise SystemExit("live details_P not registered")
    lines = jsonl(os.path.join(out, f))
    ident = lines[0]["identity"]
    want = {"condition": "P", "start_sha256": sha(os.path.join(out, "START.json")),
            "inputs_sha256": e54.check_registered(out, "inputs")["inputs_P.jsonl"], "model": st["model"], "temperature": "none (raw)",
            "reading_bundle": st["bundle"]["reading.json"], "pictures": st["pictures"]["pictures.json"]}
    if any(ident.get(k) != v for k, v in want.items()):
        raise SystemExit("live details_P: identity does not bind")
    got = e57.r41.check_details("exp58/live_P", list(rows.values()), lines[1:])
    for r in lines[1:]:
        if r["options"] != list(rows[r["sample_id"]]["questions"][r["qid"]]["criteria"]):
            raise SystemExit("live details_P %s %s: options differ" % (r["sample_id"], r["qid"]))
    return list(got.values())


def verify_fidelity(out: str, st: Dict) -> Dict:
    """Stage A's gate, recomputed from the registered evidence every time it is used (never a cached flag): the live reads
    pass EXP56's check_reads (registered, every picture once, the templates its questions call for, each value its
    decoded string's format, inputs_P = the reads serialised, details_P about START's bundle / pictures); the live
    details pass live_records; then reads = EXP56's 1,800, states = EXP56's 600 inputs_P, answers = EXP57's details_P
    (fixed EXP56 tables, same endpoint) with the same predictions and probabilities within 1e-5."""
    rows = {r["id"]: r for r in jsonl(e54.TUNE)}
    live_reads = e56.check_reads(out, rows, st)
    old_reads = e56.check_reads(E56, rows, json.load(open(os.path.join(E56, "START.json"))))
    key = lambda x: (x["decoded"], x["valid"], x["value"])  # noqa: E731
    n_reads = sum(len(v) for v in old_reads.values())
    same_reads = sum(key(live_reads[rid][q]) == key(x) for rid, d in old_reads.items() for q, x in d.items())
    si, oi = e54.load_inputs(out, "P"), e54.load_inputs(E56, "P")
    same_states = sum(si[rid]["state"] == oi[rid]["state"] for rid in oi) if set(si) == set(oi) else 0
    live = {(r["sample_id"], r["qid"]): r for r in live_records(out, rows, st)}
    if sha(os.path.join(E57, "details_P.jsonl")) != st["exp57_details_P"]:
        raise SystemExit("EXP57's details_P is not the one START bound")
    fixed = e57.read_set(E57, "P", rows, st["model"], e57.set_inputs("P", rows, {})[1])
    diff = max(abs(live[(r["sample_id"], r["qid"])]["probabilities"][k] - r["probabilities"][k]) for r in fixed for k in r["probabilities"])
    agree = all(live[(r["sample_id"], r["qid"])]["prediction"] == r["prediction"] for r in fixed)
    res = {"reads": n_reads, "same_reads": same_reads, "states": len(oi), "same_states": same_states, "questions": len(fixed),
           "same_predictions": agree, "max_probability_difference": diff, "start_sha256": sha(os.path.join(out, "START.json")),
           "evidence": {f: sha(os.path.join(out, f)) for f in ("reads.jsonl", "inputs_P.jsonl", "details_P.jsonl")}}
    res["reproduced"] = bool(n_reads == A_READS and same_reads == n_reads and len(oi) == A_PICTURES and same_states == A_PICTURES
                             and len(fixed) == A_QUESTIONS and len(live) == A_QUESTIONS and agree and diff <= 1e-5)
    return res


def fidelity(out: str = OUT) -> Dict:
    st = check_run(out, data_root="")
    res = verify_fidelity(out, st)
    write_json(os.path.join(out, "fidelity.json"), res)                  # a record; every consumer recomputes
    register(out, "fidelity", {"fidelity.json": sha(os.path.join(out, "fidelity.json"))})
    if not res["reproduced"]:
        raise SystemExit("stage A not reproduced: %s" % res)
    return res


def stage_sets(out: str, sets: Dict[str, str], label: str, data_root: str = "data") -> Dict:
    st0 = check_run(out, data_root="")
    if label == "guard" and st0["mode"]["name"] == EXPLORE:
        raise SystemExit("B2 (the seven panels) is not permitted in the exploration mode")
    if not verify_fidelity(out, st0)["reproduced"]:
        raise SystemExit("stage A not reproduced: stage B does not start")
    reading, st, pics = load_reading(out, data_root=data_root)
    s_sha = sha(os.path.join(out, "START.json"))
    total = sum(st["requests"][n]["requests"] for n in sets)
    done, t0 = 0, time.time()
    for name, path in sets.items():
        rows = jsonl(path)
        recs = run_set(reading, rows, data_root, pics, on_request=lambda k, b=done: e54.progress(out, label, b + k, total))
        done += st["requests"][name]["requests"]
        ident = {"set": name, "start_sha256": s_sha, "data_sha256": st["data"][name], "model": st["model"], "bundle": st["bundle"]["reading.json"],
                 "pictures": st["pictures"]["pictures.json"], "temperature": "none (raw)",
                 "path": "devision.reading.Reading.decide(image=actual picture)", "max_questions": MAX_QUESTIONS}
        e57.write_set(out, name, recs, ident)
        print("%s done after %.1f s" % (name, time.time() - t0), flush=True)
    return {"seconds": time.time() - t0}


# ------------------------------------------------------------------ report

def set_records(out: str, name: str, st: Dict) -> List[Dict]:
    """A set's records after every check: registered, identity bound, every question of the source file exactly once."""
    f = "details_%s.jsonl" % name
    if e54.check_registered(out, "decide_" + name).get(f) != sha(os.path.join(out, f)):
        raise SystemExit("%s not registered" % f)
    lines = jsonl(os.path.join(out, f))
    ident = lines[0]["identity"]
    want = {"set": name, "start_sha256": sha(os.path.join(out, "START.json")), "data_sha256": st["data"][name], "model": st["model"],
            "bundle": st["bundle"]["reading.json"], "pictures": st["pictures"]["pictures.json"], "temperature": "none (raw)"}
    if any(ident.get(k) != v for k, v in want.items()):
        raise SystemExit("%s: identity does not bind" % f)
    path = {**TRANSFER, **GUARD}[name]
    got = e57.r41.check_details("exp58/%s" % name, jsonl(path), lines[1:])
    return list(got.values())


def v03_records(name: str, st: Dict) -> List[Dict]:
    p = os.path.join(V03, name + ".details.jsonl")
    if sha(p) != st["v03"][name]:
        raise SystemExit("v0.3's %s changed" % name)
    rs = e57.r41.check_details("v03/%s" % name, jsonl({**TRANSFER, **GUARD}[name]), jsonl(p))
    return list(rs.values())


def errors(rs: List[Dict]) -> Dict:
    return {"routed": sum(bool(r.get("routed")) for r in rs), "wrong_by_gold_prediction": dict(Counter(
        "%s -> %s" % (r["gold_answer"], r["prediction"]) for r in rs if not r["correct"]))}


def write_report(out: str, res: Dict) -> Dict:
    os.makedirs(out, exist_ok=True)
    write_json(os.path.join(out, "report.json"), res)
    keys = ("finalized", "mode", "combination", "states", "fidelity_state", "transfer_target", "costs", "candidate_eligible", "branch")
    text = json.dumps({k: res[k] for k in keys if k in res}, indent=1, default=str)
    open(os.path.join(out, "report.txt"), "w").write(text + "\n")
    print(text)
    return res


def report(out: str = OUT, finalize: str = "") -> Dict:
    """Model-free. Without START, or when the combination / stage A evidence does not verify, the report says so
    (incomplete / invalid) and gives no transfer or candidate verdict; parts that do verify are still reported."""
    res: Dict[str, Any] = {"states": {}, "fidelity": None, "transfer": None, "guard": {}, "costs": [], "candidate_eligible": False}
    if finalize:
        res["finalized"] = {"exit_codes": finalize, "note": "the run did not complete; missing items are not low scores"}
    if not os.path.exists(os.path.join(out, "START.json")):
        res["combination"] = "incomplete: no START"
        res["branch"] = "incomplete: no START (nothing ran)"
        return write_report(out, res)
    try:
        st = check_run(out, data_root="", code=False)
        res["combination"] = "verified"
    except (SystemExit, OSError, ValueError, KeyError) as e:
        res["combination"] = "invalid: %s" % e
        try:
            res["mode"] = json.load(open(os.path.join(out, "START.json")))["mode"]
        except (OSError, ValueError, KeyError):
            pass
        res["branch"] = "invalid: the run's combination does not verify -- no verdict"
        return write_report(out, res)
    try:
        res["fidelity"] = verify_fidelity(out, st)
        res["fidelity_state"] = "verified" if res["fidelity"]["reproduced"] else "not reproduced"
    except (SystemExit, OSError, ValueError, KeyError) as e:
        res["fidelity"], res["fidelity_state"] = None, "missing / invalid: %s" % e
    sys.path.insert(0, os.path.join(HERE, "..", "exp21"))
    from readout import transfer  # type: ignore

    from devision.train.compare import paired_metric
    r32 = _load("exp32_report", "../exp32/report.py")
    recs = {}
    for name in list(TRANSFER) + list(GUARD):
        try:
            recs[name] = set_records(out, name, st)
            res["states"][name] = "completed"
        except (SystemExit, OSError, ValueError, KeyError) as e:
            res["states"][name] = "missing / invalid: %s" % e
    if all(n in recs for n in TRANSFER):
        try:
            t = transfer({k[4:]: v03_records(k, st) for k in TRANSFER}, {k[4:]: recs[k] for k in TRANSFER})
            res["transfer"] = t
            res["transfer_target"] = bool(t["accuracy_change"]["value"] >= 0.10 and t["accuracy_change"]["ci95"][0] > 0 and t["gain_change"]["ci95"][0] > 0)
            res["transfer_detail"] = {k: {"accuracy": sum(r["correct"] for r in recs[k]) / len(recs[k]), **errors(recs[k])} for k in TRANSFER}
        except (SystemExit, OSError, ValueError, KeyError) as e:
            res["states"]["transfer_baseline"] = "invalid: %s" % e
    for name in GUARD:
        if name not in recs:
            continue
        try:
            a, b = v03_records(name, st), recs[name]
            if name == "flip":
                a, b = [dict(r, image_id=r["image_id"].removesuffix(":flip")) for r in a], [dict(r, image_id=r["image_id"].removesuffix(":flip")) for r in b]
            m = paired_metric(a, b, "macro_accuracy", sources=r32.EXT_SOURCES) if name == "ext" else paired_metric(a, b, "accuracy")
        except (SystemExit, OSError, ValueError, KeyError) as e:
            res["states"][name] = "invalid baseline: %s" % e
            continue
        res["guard"][name] = {"difference": m["difference"], "ci95": m["ci95"], **errors(b)}
        if m["difference"] * 100 <= -3 and m["ci95"][1] < 0:
            res["costs"].append(name)
    if "science" in recs:
        routed = [r for r in recs["science"] if r["routed"]]
        res["science_routed"] = {"n": len(routed), "accuracy": sum(r["correct"] for r in routed) / len(routed) if routed else None}
    res["branch"] = branch(res)
    verdict(res, st["mode"])
    return write_report(out, res)


def verdict(res: Dict, mode: Dict) -> Dict:
    """Mode and eligibility on every report: the exploration keeps EXP57's failed fit and is never a candidate, whatever
    stage B measured; the default mode is eligible only on the final all-pass branch (still a development result)."""
    res["mode"] = mode
    if mode["name"] == EXPLORE:
        res["candidate_eligible"] = False
        res["exp57_gates"] = mode["exp57_gates"]
        res["branch"] = "post-hoc exploration (EXP57 fit failed, kept; not a candidate, no release): " + res["branch"]
    else:
        res["candidate_eligible"] = res["branch"].startswith("transfer target met on the monitored dev_sqa sets with no protection cost")
    return res


def branch(res: Dict) -> str:
    if not (res.get("fidelity") or {}).get("reproduced"):
        return "stage A not reproduced (or missing): the live path with EXP57 is not established; stage B not judged"
    if res.get("transfer") is None:
        return "incomplete: stage B1 (dev_sqa) missing"
    if not res["transfer_target"]:
        return "transfer target not met on dev_sqa (68:114): report temp / force apart; no protection verdict sought as a substitute"
    if len(res["guard"]) < len(GUARD):
        return "transfer target met; protection incomplete (%s missing): no candidate claim" % sorted(set(GUARD) - set(res["guard"]))
    if res["costs"]:
        return "transfer target met with protection costs %s: protection repair only, no release" % res["costs"]
    return ("transfer target met on the monitored dev_sqa sets with no protection cost: still a development result -- "
            "fresh confirmation, full calibration / evaluation, CPU / API / install checks remain before v0.4")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["counts", "start", "live", "fidelity", "transfer", "guard", "report"])
    ap.add_argument("--deadline", type=int, default=0)
    ap.add_argument("--finalize", default="")
    ap.add_argument("--mode", choices=[DEFAULT, EXPLORE], default=DEFAULT)
    a = ap.parse_args(argv)
    if a.stage == "counts":
        print(json.dumps(request_counts(), indent=1))
    elif a.stage == "start":
        print(json.dumps({k: v for k, v in stage_start(a.deadline, mode=a.mode).items() if k not in ("files", "v03", "data")}, indent=1))
    elif a.stage == "live":
        print(json.dumps(stage_live(), indent=1))
    elif a.stage == "fidelity":
        print(json.dumps(fidelity(), indent=1))
    elif a.stage == "transfer":
        print(json.dumps(stage_sets(OUT, TRANSFER, "transfer"), indent=1))
    elif a.stage == "guard":
        print(json.dumps(stage_sets(OUT, GUARD, "guard"), indent=1))
    else:
        try:
            report(finalize=a.finalize)
        except Exception as e:                      # the report never ends without a readable file
            os.makedirs(OUT, exist_ok=True)
            open(os.path.join(OUT, "report.txt"), "w").write("REPORT FAILED (invalid, no verdict): %s%s\n" % (e, " (after: %s)" % a.finalize if a.finalize else ""))
            print("REPORT FAILED: %s" % e, file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
