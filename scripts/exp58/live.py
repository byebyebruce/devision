"""exp58 (prepared, not run; critic LIVE-TRANSFER-PREP-01; docs/experiments/2026-10-18-exp58-live-transfer-plan.md): the
reading mode with EXP57's decision endpoint, live. Only after EXP57's own branch allows it and a separate model
permission; nothing here is chained to EXP57. The endpoint is a placeholder until EXP57 is frozen (NEW_SHA must be
filled from runs/exp57/FROZEN.json and reviewed); the selector / head stay EXP53's / EXP51's files (8f10f1fd / 0b548142).

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
NEW_SHA: Optional[Dict[str, str]] = None          # placeholder: filled from runs/exp57/FROZEN.json after EXP57's end, reviewed
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


def ask(reading, image_path: str, text: str, questions: Dict[str, Dict], data_root: str, pictures: Optional[Dict[str, str]] = None,
        diagnostics: Optional[Dict] = None) -> Dict:
    """One request through the product path: the actual picture bytes (checked against `pictures` when given) opened as
    an image and passed as image= to Reading.decide with the text state -- never inside state, never None."""
    import hashlib

    from PIL import Image
    raw = open(os.path.join(data_root, image_path), "rb").read()
    if pictures is not None and hashlib.sha256(raw).hexdigest() != pictures[image_path]:
        raise SystemExit("%s differs from the bound picture" % image_path)
    with Image.open(io.BytesIO(raw)) as img:
        return reading.decide(text, questions, image=img.convert("RGB"), diagnostics=diagnostics)


def run_set(reading, samples: List[Dict], data_root: str, on_request=None) -> List[Dict]:
    """Every question of `samples` through `ask`, devision-eval's record fields (+ routed: the question went to the
    reading path), scored with evaluate.score_records."""
    from devision.reading import domain_of
    from devision.train.evaluate import _probabilities, score_records
    asked = {}
    for n, (image, text, items) in enumerate(requests(samples)):
        questions = {"%d:%s" % (i, qid): q for i, qid, q in items}
        res = ask(reading, image, text, questions, data_root)
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

def stage_start(deadline: int, out: str = OUT) -> Dict:
    from devision.reading import write_bundle
    path = os.path.join(out, "START.json")
    if os.path.exists(path):
        raise SystemExit("%s exists (fresh only)" % path)
    want = new_sha()
    if fz.check(NEW_RUN, list(e57.SCREENS))["weights"] != want:
        raise SystemExit("EXP57 is not its frozen endpoint")
    rep = json.load(open(os.path.join(NEW_RUN, "report.json")))
    if not rep.get("branch", "").startswith("fit, G, P, reading retention and exist / science all pass"):
        raise SystemExit("EXP57's pre-written branch does not allow the live path: %s" % rep.get("branch"))
    e57.bound("EXP56", E56, E56_SHA)
    if sha(SELECTOR) != SELECTOR_SHA or sha(HEAD) != HEAD_SHA:
        raise SystemExit("the selector / head are not EXP53's / EXP51's files")
    st56 = json.load(open(os.path.join(E56, "START.json")))
    if sha(os.path.join(E56, "pictures.json")) != st56["pictures"]["pictures.json"]:
        raise SystemExit("EXP56's pictures.json changed")
    os.makedirs(out, exist_ok=True)
    bundle = write_bundle(os.path.join(out, "bundle"), os.path.join(NEW_RUN, "stage2"), SELECTOR, HEAD,
                          {"selector": {"path": SELECTOR, "sha256": SELECTOR_SHA}, "head": {"path": HEAD, "sha256": HEAD_SHA}})
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    s = {"git_head": git("rev-parse", "HEAD"), "dirty_files": [f for f in FILES if git("status", "--porcelain", "--", f)],
         "files": {f: sha(f) for f in FILES}, "model": want, "exp57_report": sha(os.path.join(NEW_RUN, "report.json")),
         "bundle": {"reading.json": sha(os.path.join(out, "bundle", "reading.json")), **bundle["sha256"]},
         "exp56": dict(E56_SHA, pictures=st56["pictures"]["pictures.json"]),
         "data": {n: sha(p) for n, p in {**TRANSFER, **GUARD}.items()}, "tune_sha256": sha(e54.TUNE),
         "v03": {n: sha(os.path.join(V03, n + ".details.jsonl")) for n in {**TRANSFER, **GUARD}},
         "requests": request_counts(), "temperature": "none (raw), in memory", "max_questions": MAX_QUESTIONS,
         "started": time.strftime("%F %T"), "deadline": deadline, "deadline_text": time.strftime("%F %T", time.localtime(deadline))}
    write_json(path, s)
    return s


def load_reading(out: str, device: str = "auto"):
    from devision.infer import best_device
    from devision.reading import Reading
    st = json.load(open(os.path.join(out, "START.json")))
    if fz.weights(NEW_RUN) != st["model"] or st["model"] != new_sha():
        raise SystemExit("EXP57's weights / config are not START's: refused before loading")
    e56.check_combination(out, st)
    r = Reading.load(os.path.join(NEW_RUN, "stage2"), os.path.join(out, "bundle"), device=best_device() if device == "auto" else device)
    r.decider.cfg.temperature, r.decider.cfg.temperature_by_options = [1.0, 1.0, 1.0], {}      # raw, in memory only
    return r, st


def stage_live(out: str = OUT, data_root: str = "data") -> Dict:
    reading, st = load_reading(out)
    pics = json.load(open(os.path.join(E56, "pictures.json")))
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
                        {"reading_bundle": st["bundle"]["reading.json"], "pictures": st["exp56"]["pictures"]})
    return {"seconds": time.time() - t0}


def fidelity(out: str = OUT) -> Dict:
    """Stage A's gate: live reads = EXP56's, states = EXP56's inputs_P, answers = EXP57's details_P (<= 1e-5)."""
    rows = {r["id"]: r for r in jsonl(e54.TUNE)}
    new = {r["id"]: {x["question"]: (x["decoded"], x["valid"], x["value"]) for x in r["reads"]} for r in jsonl(os.path.join(out, "reads.jsonl"))}
    old = {r["id"]: {x["question"]: (x["decoded"], x["valid"], x["value"]) for x in r["reads"]} for r in jsonl(os.path.join(E56, "reads.jsonl"))}
    n_reads = sum(len(v) for v in old.values())
    same_reads = sum(new.get(rid, {}).get(q) == v for rid, d in old.items() for q, v in d.items())
    si = e54.load_inputs(out, "P")
    oi = e54.load_inputs(E56, "P")
    same_states = sum(si.get(rid, {}).get("state") == oi[rid]["state"] for rid in oi)
    live = {(r["sample_id"], r["qid"]): r for r in jsonl(os.path.join(out, "details_P.jsonl"))[1:]}
    fixed = e57.read_set(E57, "P", rows, new_sha(), e57.set_inputs("P", rows, {})[1])
    diff = max(abs(live[(r["sample_id"], r["qid"])]["probabilities"][k] - r["probabilities"][k]) for r in fixed for k in r["probabilities"])
    agree = all(live[(r["sample_id"], r["qid"])]["prediction"] == r["prediction"] for r in fixed) and len(live) == len(fixed)
    res = {"reads": n_reads, "same_reads": same_reads, "states": len(oi), "same_states": same_states, "questions": len(fixed),
           "same_predictions": agree, "max_probability_difference": diff}
    res["reproduced"] = bool(n_reads == 1800 and same_reads == n_reads and same_states == len(oi) == 600 and agree and diff <= 1e-5)
    write_json(os.path.join(out, "fidelity.json"), res)
    if not res["reproduced"]:
        raise SystemExit("stage A not reproduced: %s" % res)
    return res


def stage_sets(out: str, sets: Dict[str, str], label: str, data_root: str = "data") -> Dict:
    if not json.load(open(os.path.join(out, "fidelity.json"))).get("reproduced"):
        raise SystemExit("stage A not reproduced: stage B does not start")
    reading, st = load_reading(out)
    s_sha = sha(os.path.join(out, "START.json"))
    total = sum(st["requests"][n]["requests"] for n in sets)
    done, t0 = 0, time.time()
    for name, path in sets.items():
        if sha(path) != st["data"][name]:
            raise SystemExit("%s differs from START" % path)
        rows = jsonl(path)
        recs = run_set(reading, rows, data_root, on_request=lambda k, b=done: e54.progress(out, label, b + k, total))
        done += st["requests"][name]["requests"]
        ident = {"set": name, "start_sha256": s_sha, "data_sha256": st["data"][name], "model": st["model"], "bundle": st["bundle"]["reading.json"],
                 "temperature": "none (raw)", "path": "devision.reading.Reading.decide(image=actual picture)", "max_questions": MAX_QUESTIONS}
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
            "bundle": st["bundle"]["reading.json"], "temperature": "none (raw)"}
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


def report(out: str = OUT, finalize: str = "") -> Dict:
    sys.path.insert(0, os.path.join(HERE, "..", "exp21"))
    from readout import transfer  # type: ignore

    from devision.train.compare import paired_metric
    r32 = _load("exp32_report", "../exp32/report.py")
    st = json.load(open(os.path.join(out, "START.json")))
    res: Dict[str, Any] = {"states": {}, "fidelity": None, "transfer": None, "guard": {}, "costs": []}
    fp = os.path.join(out, "fidelity.json")
    res["fidelity"] = json.load(open(fp)) if os.path.exists(fp) else {"state": "missing"}
    recs = {}
    for name in list(TRANSFER) + list(GUARD):
        try:
            recs[name] = set_records(out, name, st)
            res["states"][name] = "completed"
        except (SystemExit, OSError, ValueError, KeyError) as e:
            res["states"][name] = "missing: %s" % e
    if all(n in recs for n in TRANSFER):
        t = transfer({k[4:]: v03_records(k, st) for k in TRANSFER}, {k[4:]: recs[k] for k in TRANSFER})
        res["transfer"] = t
        res["transfer_target"] = bool(t["accuracy_change"]["value"] >= 0.10 and t["accuracy_change"]["ci95"][0] > 0 and t["gain_change"]["ci95"][0] > 0)
        res["transfer_detail"] = {k: {"accuracy": sum(r["correct"] for r in recs[k]) / len(recs[k]), **errors(recs[k])} for k in TRANSFER}
    for name in GUARD:
        if name not in recs:
            continue
        a, b = v03_records(name, st), recs[name]
        if name == "flip":
            a, b = [dict(r, image_id=r["image_id"].removesuffix(":flip")) for r in a], [dict(r, image_id=r["image_id"].removesuffix(":flip")) for r in b]
        m = paired_metric(a, b, "macro_accuracy", sources=r32.EXT_SOURCES) if name == "ext" else paired_metric(a, b, "accuracy")
        res["guard"][name] = {"difference": m["difference"], "ci95": m["ci95"], **errors(b)}
        if m["difference"] * 100 <= -3 and m["ci95"][1] < 0:
            res["costs"].append(name)
    if "science" in recs:
        routed = [r for r in recs["science"] if r["routed"]]
        res["science_routed"] = {"n": len(routed), "accuracy": sum(r["correct"] for r in routed) / len(routed) if routed else None}
    res["branch"] = branch(res)
    if finalize:
        res["finalized"] = {"exit_codes": finalize, "note": "missing items are not low scores"}
    write_json(os.path.join(out, "report.json"), res)
    print(json.dumps({k: res[k] for k in ("states", "transfer_target", "costs", "branch") if k in res}, indent=1))
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
    a = ap.parse_args(argv)
    if a.stage == "counts":
        print(json.dumps(request_counts(), indent=1))
    elif a.stage == "start":
        print(json.dumps({k: v for k, v in stage_start(a.deadline).items() if k not in ("files", "v03", "data")}, indent=1))
    elif a.stage == "live":
        print(json.dumps(stage_live(), indent=1))
    elif a.stage == "fidelity":
        print(json.dumps(fidelity(), indent=1))
    elif a.stage == "transfer":
        print(json.dumps(stage_sets(OUT, TRANSFER, "transfer"), indent=1))
    elif a.stage == "guard":
        print(json.dumps(stage_sets(OUT, GUARD, "guard"), indent=1))
    else:
        report(finalize=a.finalize)
    return 0


if __name__ == "__main__":
    sys.exit(main())
