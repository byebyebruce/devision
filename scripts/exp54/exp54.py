"""exp54: read values -> ordinary decide, zero training (critic EXP54-PREP-01..04; docs/experiments/2026-10-18-exp54-read-to-
decision-plan.md). Frozen: EXP50-T's decision weights (1a8e5439 / 99c6246e), EXP53's selector (8f10f1fd) and EXP51's head
(0b548142) -- their automatic readings are reused from EXP53's registered details_auto (2b8544f4), a component-combination
diagnostic, not a live end-to-end run. Per original rl_tune picture one fact table, appended to the row's own state_text,
then the row's questions (reading and comparison, original options and order) through Decider.decide with image=None and
raw temperature 1 (in memory, as devision-eval --no-temperature). Four conditions differ only in the table's values:

  P  EXP53's automatic reading of each listed value, strictly decoded; an invalid string is "unreadable" (nothing picked,
     repaired, or taken from the oracle or the options). Built from the picture's reading questions' TEXTS and the
     automatic predictions only -- the constructor never receives gold, boxes, family / case or options.
  D  the table P would give for the fixed twin picture (same family, other view): every value from the twin.
  E  every value "unreadable" (same rows).
  G  the gold numbers in the same template (an independent oracle control, built by its own function from the gold).

The rows come from the picture's reading questions through a fixed map of the six natural question templates (object,
quantity); values are unit-free numbers, the units stay in the original state / questions / options. Nothing compares
values with options or computes physics. Stages:

  start   model-free: identities (EXP50-T frozen, EXP53 START / registry / details_auto, questions, the tune set and its
          question source) and the four conditions' exact inputs -> runs/exp54/START.json, inputs_<C>.jsonl (registered).
  decide  MPS: the four conditions in the order P, G, E, D; one request per picture and condition (<= 5 questions);
          each condition's details written and registered as soon as it completes; progress lines for the watch.
  report  model-free: per condition exp42's statistics plus all-question and value-changed accuracy, NLL, read-right /
          read-wrong strata, donor following, comparison questions by kind / axis; the pre-written conditions and
          branch; --finalize after a failed / stopped stage.

    uv run python scripts/exp54/exp54.py start --deadline EPOCH | decide | report [--finalize CODES]
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional

HERE = os.path.dirname(__file__)
OUT = "runs/exp54"
TUNE = "data/real_layout/questions_tune.jsonl"
QUESTIONS = "data/exp49/questions.jsonl"
RUN, T_SHA = "runs/exp50-t", {"model.safetensors": "1a8e5439d5c4f64fd1dbce551f9c2df55152b2aea7cc3a93fc01eb5d364ca007",
                              "config.json": "99c6246e201c5c38c7f62c6831050792858b5aecf2b4be352c5cb7cbc3231fc6"}
E53 = "runs/exp53"
E53_DETAILS_SHA = "2b8544f47a067cad912e2df62bdc30e306a84f16035effc10a72ec60917c3e9e"
E53_START_SHA = "c640dd9a03cb1e3ec988690965e8839b0a069fcb21dec995adc070829a09c753"
CONDS = ("P", "G", "E", "D")
UNREADABLE = "unreadable"
FILES = ["scripts/exp54/exp54.py", "scripts/exp54/run.sh", "scripts/exp53/watch.py", "scripts/exp49/probe.py",
         "scripts/exp42/report.py", "scripts/exp41/report.py", "src/devision/infer.py", "src/devision/model.py"]
# the six natural question templates -> (object, quantity); the only source of a table row's names
TEMPLATES = {
    "What is the mass of each particle in sample A?": ("Sample A", "mass of each particle"),
    "What is the mass of each particle in sample B?": ("Sample B", "mass of each particle"),
    "What is the average particle speed in sample A?": ("Sample A", "average particle speed"),
    "What is the average particle speed in sample B?": ("Sample B", "average particle speed"),
    "What is the distance between the magnets in Pair 1?": ("Pair 1", "distance between the magnets"),
    "What is the distance between the magnets in Pair 2?": ("Pair 2", "distance between the magnets"),
}
ORDER = list(TEMPLATES)                       # the fixed row order


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, rel))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


p49 = _load("exp49_probe", "../exp49/probe.py")
r42 = _load("exp42_report", "../exp42/report.py")
r41, b40 = r42.r41, r42.b40


def sha(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, default=str)
    os.replace(tmp, path)


def register(out: str, name: str, files: Dict[str, str]) -> None:
    path = os.path.join(out, "registry.json")
    r = json.load(open(path)) if os.path.exists(path) else {}
    r[name] = files
    write_json(path, r)


def check_registered(out: str, name: str) -> Dict[str, str]:
    r = json.load(open(os.path.join(out, "registry.json")))
    if name not in r:
        raise SystemExit("%s not registered" % name)
    for f, h in r[name].items():
        if sha(os.path.join(out, f)) != h:
            raise SystemExit("%s changed after it was registered" % f)
    return r[name]


def progress(out: str, stage: str, done: int, total: int) -> None:
    with open(os.path.join(out, "progress-%s.txt" % stage), "a") as fh:
        fh.write("%.3f %d %d\n" % (time.time(), done, total))


# ------------------------------------------------------------------ tables

def rows_of(texts: List[str]) -> List[str]:
    """The table rows of a picture: its reading questions' template texts in the fixed order (unknown text: refused)."""
    bad = [t for t in texts if t not in TEMPLATES]
    if bad:
        raise SystemExit("not one of the six reading templates: %r" % bad[0])
    return [t for t in ORDER if t in texts]


def p_table(reading: List[Dict], preds: Dict[str, List[int]]) -> List[List[str]]:
    """P: [(object, quantity, value)] from the picture's reading questions ({"key", "text"}) and the automatic
    predictions only. A strictly invalid decode is UNREADABLE; nothing else is consulted."""
    by = {r["text"]: r["key"] for r in reading}
    out = []
    for t in rows_of(list(by)):
        d = p49.decode(preds[by[t]])
        out.append([*TEMPLATES[t], d["text"] if d["valid"] else UNREADABLE])
    return out


def g_table(reading: List[Dict], gold: Dict[str, str]) -> List[List[str]]:
    """G: the same template filled with the gold numbers (the oracle control, a separate constructor)."""
    by = {r["text"]: r["key"] for r in reading}
    return [[*TEMPLATES[t], gold[by[t]]] for t in rows_of(list(by))]


def e_table(table: List[List[str]]) -> List[List[str]]:
    return [[o, q, UNREADABLE] for o, q, _ in table]


def serialise(state_text: str, table: List[List[str]]) -> str:
    lines = ["%s: %s = %s" % (o, q, v) for o, q, v in table]
    return (state_text + "\n" if state_text else "") + "Values read from the picture (numbers only; units as in the question):\n" + "\n".join(lines)


def build_inputs(rows: Dict[str, Dict], qs53: List[Dict], preds: Dict[str, List[int]]) -> Dict[str, Dict[str, Dict]]:
    """{condition: {row id: {"state", "table", "twin_row"?}}} for the rl_tune rows. P / D from the automatic predictions
    and the questions' texts; G from the gold numbers (exp49 targets) by its own function; E from P's rows."""
    by_row: Dict[str, List[Dict]] = defaultdict(list)
    gold: Dict[str, str] = {}
    twin_row: Dict[str, str] = {}
    key_row = {}
    for q in qs53:
        if q["set"] == "rl_tune":
            key_row[q["key"]] = q["id"]
    for q in qs53:
        if q["set"] != "rl_tune":
            continue
        row = rows[q["id"]]
        by_row[q["id"]].append({"key": q["key"], "text": row["questions"][q["qid"]]["instructions"]})
        gold[q["key"]] = q["target"]
        if q["twin"]:
            tr = key_row[q["twin"]]
            if twin_row.setdefault(q["id"], tr) != tr:
                raise SystemExit("%s: its questions have twins on different pictures" % q["id"])
    out: Dict[str, Dict[str, Dict]] = {c: {} for c in CONDS}
    ptab = {rid: p_table(rd, preds) for rid, rd in by_row.items()}
    for rid, rd in by_row.items():
        st = rows[rid].get("state_text", "")
        g = g_table(rd, gold)
        if rid not in twin_row:
            raise SystemExit("%s has no twin picture" % rid)
        d = ptab[twin_row[rid]]
        if [r[:2] for r in d] != [r[:2] for r in ptab[rid]]:
            raise SystemExit("%s: the twin's table has other rows" % rid)
        out["P"][rid] = {"state": serialise(st, ptab[rid]), "table": ptab[rid]}
        out["G"][rid] = {"state": serialise(st, g), "table": g}
        out["E"][rid] = {"state": serialise(st, e_table(ptab[rid])), "table": e_table(ptab[rid])}
        out["D"][rid] = {"state": serialise(st, d), "table": d, "twin_row": twin_row[rid]}
    return out


# ------------------------------------------------------------------ start

def stage_start(deadline: int, out: str = OUT) -> Dict:
    path = os.path.join(out, "START.json")
    if os.path.exists(path):
        raise SystemExit("%s exists (fresh only)" % path)
    fz = _load("v04b_freeze", "../v04b/freeze.py")
    if fz.check(RUN, [])["weights"] != T_SHA:
        raise SystemExit("EXP50-T is not its frozen 1a8e5439 / 99c6246e")
    if sha(os.path.join(E53, "START.json")) != E53_START_SHA:
        raise SystemExit("EXP53's START changed")
    e53reg = json.load(open(os.path.join(E53, "registry.json")))
    det = os.path.join(E53, "details_auto.jsonl")
    if e53reg["readout_auto"]["details_auto.jsonl"] != E53_DETAILS_SHA or sha(det) != E53_DETAILS_SHA:
        raise SystemExit("EXP53's automatic details are not 2b8544f4 / not its registered ones")
    e53start = json.load(open(os.path.join(E53, "START.json")))
    if sha(QUESTIONS) != e53start["questions_sha256"] or sha(TUNE) != e53start["sources"]["rl_tune"]:
        raise SystemExit("questions / rl_tune are not those EXP53 bound")
    qs53 = [json.loads(x) for x in open(QUESTIONS) if x.strip()]
    pred, _, ident = p49.read_details(det, qs53)
    if ident.get("path") != "auto" or ident.get("start_sha256") != E53_START_SHA:
        raise SystemExit("details_auto's identity is not EXP53's automatic path")
    rows = {r["id"]: r for r in map(json.loads, open(TUNE))}
    preds = {q["key"]: p for q, p in zip(qs53, pred)}
    inputs = build_inputs(rows, qs53, preds)
    os.makedirs(out, exist_ok=True)
    files = {}
    for c in CONDS:
        f = "inputs_%s.jsonl" % c
        with open(os.path.join(out, f), "w") as fh:
            for rid in sorted(inputs[c]):
                fh.write(json.dumps({"id": rid, **inputs[c][rid]}) + "\n")
        files[f] = sha(os.path.join(out, f))
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    s = {"git_head": git("rev-parse", "HEAD"), "dirty_files": [f for f in FILES if git("status", "--porcelain", "--", f)],
         "files": {f: sha(f) for f in FILES}, "model": {"run": RUN, **T_SHA}, "temperature": "none (raw), in memory",
         "image": None, "max_questions_per_request": 5, "exp53": {"start_sha256": E53_START_SHA, "details_auto_sha256": E53_DETAILS_SHA},
         "questions_sha256": sha(QUESTIONS), "tune_sha256": sha(TUNE), "inputs": files, "conditions": list(CONDS),
         "started": time.strftime("%F %T"), "deadline": deadline, "deadline_text": time.strftime("%F %T", time.localtime(deadline))}
    write_json(path, s)
    register(out, "inputs", files)
    return s


def started(out: str = OUT) -> str:
    return sha(os.path.join(out, "START.json"))


def load_inputs(out: str, cond: str) -> Dict[str, Dict]:
    check_registered(out, "inputs")
    return {r["id"]: r for r in map(json.loads, open(os.path.join(out, "inputs_%s.jsonl" % cond)))}


# ------------------------------------------------------------------ decide

def decide_condition(decider, rows: Dict[str, Dict], inputs: Dict[str, Dict], on_request=None) -> List[Dict]:
    """One request per picture: state = the condition's text, the row's questions as they are; image=None."""
    from devision.train.evaluate import score_records
    recs = []
    for n, rid in enumerate(sorted(inputs)):
        row = rows[rid]
        res = decider.decide(state=inputs[rid]["state"], questions=row["questions"], image=None)
        for qid, q in row["questions"].items():
            probs = dict(res["answers"][qid]["probabilities"])
            gold = row["gold"][qid]["probabilities"]
            pred = max(probs, key=probs.__getitem__)
            recs.append({"sample_id": rid, "qid": qid, "image_id": row["image_id"], "kind": row.get("kind", ""), "axis": row.get("axis", ""),
                         "options": list(q["criteria"]), "gold": gold, "gold_answer": max(gold, key=gold.__getitem__),
                         "probabilities": probs, "prediction": pred})
        if on_request:
            on_request(n + 1)
    return score_records(recs)


def stage_decide(out: str = OUT, device: str = "auto") -> Dict:
    from devision.infer import Decider, best_device
    s_sha = started(out)
    rows = {r["id"]: r for r in map(json.loads, open(TUNE))}
    if sha(TUNE) != json.load(open(os.path.join(out, "START.json")))["tune_sha256"]:
        raise SystemExit("rl_tune differs from START")
    dev = best_device() if device == "auto" else device
    d = Decider.load(os.path.join(RUN, "stage2"), device=dev)
    d.cfg.temperature, d.cfg.temperature_by_options = [1.0, 1.0, 1.0], {}        # raw, in memory only
    total = len(CONDS) * len(rows)
    t0 = time.time()
    for k, c in enumerate(CONDS):
        inputs = load_inputs(out, c)
        recs = decide_condition(d, rows, inputs, on_request=lambda n, k=k: progress(out, "decide", k * len(rows) + n, total))
        f = "details_%s.jsonl" % c
        ident = {"condition": c, "start_sha256": s_sha, "inputs_sha256": sha(os.path.join(out, "inputs_%s.jsonl" % c)),
                 "model": T_SHA, "temperature": "none (raw)", "image": None, "device": dev}
        with open(os.path.join(out, f), "w") as fh:
            fh.write(json.dumps({"identity": ident}) + "\n")
            fh.write("".join(json.dumps(r) + "\n" for r in recs))
        register(out, "decide_" + c, {f: sha(os.path.join(out, f))})                # at once, this condition only
        print("condition %s done after %.1f s" % (c, time.time() - t0), flush=True)
    return {"seconds": time.time() - t0, "device": dev}


# ------------------------------------------------------------------ report

def read_condition(out: str, c: str, rows: Dict[str, Dict]) -> List[Dict]:
    """The condition's records after every check: registered and unchanged, its identity (condition, START, inputs,
    model, raw temperature, no image), every rl_tune question exactly once with the data's gold and picture, the
    options in order, finite probabilities summing to 1, prediction = argmax, correct consistent."""
    f = "details_%s.jsonl" % c
    if check_registered(out, "decide_" + c).get(f) != sha(os.path.join(out, f)):
        raise SystemExit("details not registered")
    lines = [json.loads(x) for x in open(os.path.join(out, f)) if x.strip()]
    ident, recs = lines[0]["identity"], lines[1:]
    want = {"condition": c, "start_sha256": started(out), "inputs_sha256": check_registered(out, "inputs")["inputs_%s.jsonl" % c],
            "model": T_SHA, "temperature": "none (raw)", "image": None}
    if any(ident.get(k) != v for k, v in want.items()):
        raise SystemExit("identity does not bind")
    got = r41.check_details("exp54/%s" % c, list(rows.values()), recs)
    for r in recs:
        row = rows[r["sample_id"]]
        if r["options"] != list(row["questions"][r["qid"]]["criteria"]) or set(r["probabilities"]) != set(r["options"]):
            raise SystemExit("%s %s: options differ" % (r["sample_id"], r["qid"]))
        if not all(math.isfinite(v) for v in r["probabilities"].values()) or abs(sum(r["probabilities"].values()) - 1) > 1e-4:
            raise SystemExit("%s %s: probabilities not finite / normalised" % (r["sample_id"], r["qid"]))
    return [got[(r["sample_id"], r["qid"])] | {"nll": r["nll"]} for r in recs]


def reading_records(recs: List[Dict], rows: Dict[str, Dict], qs53: Dict) -> List[Dict]:
    out = []
    for r in recs:
        row = rows[r["sample_id"]]
        q = row["questions"][r["qid"]]
        f = b40.field_of(q["instructions"])
        if f == "cmp":
            continue
        k = "rl_tune|%s|%s" % (r["sample_id"], r["qid"])
        out.append({"key": k, "family": row["family"], "case": row["case"], "field": f, "object": b40.object_of(q["instructions"]),
                    "view": row.get("view") or r["sample_id"].split(":")[2], "picture": row.get("orig_id", r["sample_id"]),
                    "gold": r["gold_answer"], "prediction": r["prediction"], "right": bool(r["correct"]), "nll": r["nll"],
                    "changed": qs53[k]["twin_changed"]})
    return out


def units(recs: List[Dict]):
    fam = r42.units(recs)
    for r in recs:
        fam[r["family"]]["all:" + r["field"]][0] += r["right"]
        fam[r["family"]]["all:" + r["field"]][1] += 1
        if r["changed"]:
            fam[r["family"]]["changed:" + r["field"]][0] += r["right"]
            fam[r["family"]]["changed:" + r["field"]][1] += 1
    return fam


STATS = ("all", "changed", "different", "both", "twin", "same_answer")


def summary(fam) -> Dict:
    fs = sorted(fam)
    out = {s: r42.value(fam, fs, s) for s in STATS}
    for s in STATS:
        for f in r42.FIELDS:
            out["%s:%s" % (s, f)] = r42.value(fam, fs, "%s:%s" % (s, f))
    return out


def report(out: str = OUT, finalize: str = "") -> Dict:
    rows = {r["id"]: r for r in map(json.loads, open(TUNE))}
    qs53 = {q["key"]: q for q in map(json.loads, open(QUESTIONS)) if q["set"] == "rl_tune"}
    res: Dict = {"states": {}, "conditions": {}, "differences": {}}
    recs, reading = {}, {}
    for c in CONDS:
        try:
            recs[c] = read_condition(out, c, rows)
            reading[c] = reading_records(recs[c], rows, qs53)
            res["states"][c] = {"state": "completed"}
        except (SystemExit, OSError, ValueError, KeyError) as e:
            res["states"][c] = {"state": "missing" if not os.path.exists(os.path.join(out, "details_%s.jsonl" % c)) else "invalid", "reason": str(e)}
    p_right = {}
    if os.path.exists(os.path.join(out, "inputs_P.jsonl")):
        pin = load_inputs(out, "P")
        for k, q in qs53.items():                 # was the frozen automatic reading of this question right?
            tab = {(o, qn): v for o, qn, v in pin[q["id"]]["table"]}
            t = rows[q["id"]]["questions"][q["qid"]]["instructions"]
            p_right[k] = tab[TEMPLATES[t]] == q["target"]
    fams = {c: units(r) for c, r in reading.items()}
    for c, fam in fams.items():
        rr = reading[c]
        s = summary(fam)
        s["nll"] = sum(r["nll"] for r in rr) / len(rr)
        if p_right:
            ok = [r for r in rr if p_right[r["key"]]]
            s["read_right"] = {"n": len(ok), "accuracy": sum(r["right"] for r in ok) / len(ok) if ok else None}
            s["read_wrong"] = {"n": len(rr) - len(ok), "accuracy": sum(r["right"] for r in rr if not p_right[r["key"]]) / max(1, len(rr) - len(ok))}
        cmp_ = defaultdict(lambda: [0, 0])
        for r in recs[c]:
            row = rows[r["sample_id"]]
            if b40.field_of(row["questions"][r["qid"]]["instructions"]) == "cmp":
                cmp_["%s|%s" % (row.get("kind", ""), row.get("axis", ""))][0] += r["correct"]
                cmp_["%s|%s" % (row.get("kind", ""), row.get("axis", ""))][1] += 1
        s["comparison"] = {k: {"right": v[0], "of": v[1]} for k, v in sorted(cmp_.items())}
        res["conditions"][c] = s
    if "D" in reading and "P" in reading:     # donor following: D's answer is the option carrying the twin's value
        follow, inside = [0, 0], 0
        for r in reading["D"]:
            q = qs53[r["key"]]
            if not q["twin_changed"]:
                continue
            tv = qs53[q["twin"]]["target"]
            opts = rows[q["id"]]["questions"][q["qid"]]["criteria"]
            carrying = [o for o in opts if o.split(" ")[0].replace(",", "") == tv]
            inside += bool(carrying)
            follow[0] += bool(carrying) and r["prediction"] in carrying
            follow[1] += 1
        res["donor"] = {"changed": follow[1], "twin_value_among_options": inside, "follows_twin_value": follow[0]}
    clusters = None
    for a, b, stats in (("P", "E", ("both", "different", "all")), ("P", "D", ("changed", "all")), ("G", "P", ("all", "both"))):
        if a in reading and b in reading:
            ca, cb = r42.clusters(reading[a]), r42.clusters(reading[b])
            if ca != cb:
                raise SystemExit("%s / %s on different questions" % (a, b))
            clusters = ca
            res["differences"]["%s-%s" % (a, b)] = {s: r42.diff(fams[a], fams[b], ca, s) for s in stats}
    res["clusters"] = len(clusters) if clusters else None
    res["gates"] = gates(res)
    res["branch"] = branch(res)
    if finalize:
        res["finalized"] = {"exit_codes": finalize, "note": "the run did not complete; missing conditions are not low scores"}
    write_json(os.path.join(out, "report.json"), res)
    lines = ["INCOMPLETE run: %s" % finalize] if finalize else []
    for c, st in res["states"].items():
        if st["state"] != "completed":
            lines.append("%s %s: %s" % (c, st["state"], st["reason"]))
    for c, s in res["conditions"].items():
        lines.append("%s all %.3f (%s) changed %.3f different %.3f both %.3f twin %.3f same_answer %.3f nll %.3f read_right %s read_wrong %s" % (
            c, s["all"], {f: round(s["all:" + f], 3) for f in r42.FIELDS}, s["changed"], s["different"], s["both"], s["twin"],
            s["same_answer"], s["nll"], s.get("read_right"), s.get("read_wrong")))
        lines.append("%s comparison %s" % (c, s["comparison"]))
    for k, v in res["differences"].items():
        lines.append("%s %s" % (k, {s: "%+.3f [%+.3f, %+.3f]" % (x["value"], *x["ci95"]) for s, x in v.items()}))
    lines.append("donor %s" % res.get("donor"))
    lines.append("gates %s\nbranch: %s" % (res["gates"], res["branch"]))
    open(os.path.join(out, "report.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return res


def gates(res: Dict) -> Dict:
    c, d = res["conditions"], res["differences"]
    g: Dict = {}
    if "G" in c:
        x = c["G"]
        g["G_text_use"] = bool(x["all"] >= 0.90 and x["both"] >= 0.80 and x["twin"] >= 0.80)
    else:
        g["G_text_use"] = "incomplete"
    if "P" in c:
        x = c["P"]
        g["P_use"] = bool(x["all"] >= 0.70 and all(x["all:" + f] >= 0.50 for f in r42.FIELDS) and x["both"] >= 0.50
                          and x["twin"] >= 0.50 and (x.get("read_right") or {}).get("accuracy") is not None
                          and x["read_right"]["accuracy"] >= 0.90)
    else:
        g["P_use"] = "incomplete"
    pe, pd = d.get("P-E"), d.get("P-D")
    g["P_minus_E"] = "incomplete" if pe is None else bool(pe["both"]["value"] >= 0.10 and pe["both"]["ci95"][0] > 0)
    g["P_minus_D"] = "incomplete" if pd is None else bool(pd["changed"]["value"] >= 0.20 and pd["changed"]["ci95"][0] > 0)
    return g


def branch(res: Dict) -> str:
    st, g = res["states"], res["gates"]
    missing = [c for c in ("P", "G", "E", "D") if st.get(c, {}).get("state") != "completed"]
    if missing:
        return "incomplete: %s missing / invalid -- G never stands in for P" % missing
    if g["G_text_use"] is not True:
        return "G below its line: the text interface / downstream consumption is not established (EXP53 not re-judged); check the text consumption first"
    if g["P_use"] is True and g["P_minus_E"] is True and g["P_minus_D"] is True:
        return "G and P pass with content dependence: next, automatic reading integrated into ordinary inference, then comparison / transfer / protection screens"
    return "G passes, P does not (or its content dependence does not): split by 'read right but chose wrong' vs 'read wrong' to choose interface adaptation or reading generalisation"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["start", "decide", "report"])
    ap.add_argument("--deadline", type=int, default=0)
    ap.add_argument("--finalize", default="")
    a = ap.parse_args(argv)
    if a.stage == "start":
        print(json.dumps(stage_start(a.deadline), indent=1))
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
