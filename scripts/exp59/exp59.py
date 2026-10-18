"""exp59: where the real-picture transfer breaks (critic EXP59-PREP-01; docs/experiments/2026-10-18-exp59-locate-plan.md). A
mechanism diagnosis after EXP58 (not a retry, not a confirmation): the same EXP57 end and EXP58 reading bundle on the 24
transcribed dev_sqa pictures (scripts/exp21/sqa_transcribed.json: 12 temperature, 12 magnets, chosen 2026-10-15 by
random.Random(0)), each with its one question, original options / order / state text, raw temperature 1:

  P        (24 requests) EXP58's request through the same Reading.decide with the actual picture as image=, now with
           diagnostics: boxes, decoded strings, validity, values, table, the full state, probabilities. Must first equal
           EXP58's records on these 24 questions (same predictions, probabilities within 1e-5); otherwise nothing else runs.
  G_num    (24) the same table template and order with the transcribed gold numbers (data/exp28/fields.jsonl, checked
           against the transcription), exp55's separators, ordinary decide(image=None). No rule, formula or option.
  G_facts  (12, magnets) G_num's state plus one line of non-number facts the picture shows (scripts/exp59/facts.json:
           magnets per pair, shape, relative size; never a force relation or an answer), ordinary decide(image=None).

Stages (fresh only, runs/exp59): start (model-free: identities, the 24 items, G_num / G_facts inputs written and
registered) -> run (MPS: P, the reproduction check, then G_num, G_facts; each registered as it completes) -> report
(model-free: fields exact / invalid, all-numbers-right pictures, unequal golds read as equal, per-question transitions and
paired differences, temperature and magnets apart; the pre-written branches). Gold numbers never reach the P path.

    uv run python scripts/exp59/exp59.py start --deadline EPOCH | run | report [--finalize CODES]
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from collections import Counter
from typing import Any, Dict, List

HERE = os.path.dirname(__file__)
OUT, E58 = "runs/exp59", "runs/exp58"
E58_SHA = {"START.json": "505516bd7bbd9055a83212118ab1e22632e86ba2e60fb8d4b6e15cc3a090b53e",
           "report.json": "d3ada400479e9f27c4a566d819e2094565dec46900087d37b637d1add8dee267",
           "registry.json": "ef0327d06da589388939c780b13fc230c01076aa1cae54e2bc67812a79bcd11e",
           "fidelity.json": "24ccc18e90e4123f60bc4d6fa9d199d780e7af5e895e463c18447fb3774eb10c"}
SOURCES = {"temp": ("sqa_temp", "data/exp21/dev_sqa_temp.jsonl"), "force": ("sqa_force", "data/exp21/dev_sqa_force.jsonl")}
TRANSCRIBED, FIELDS, FACTS = "scripts/exp21/sqa_transcribed.json", "data/exp28/fields.jsonl", "scripts/exp59/facts.json"
CONDS = ("P", "G_num", "G_facts")
FACTS_HEADER = "Also shown in the picture:"
FILES = ["scripts/exp59/exp59.py", "scripts/exp59/run.sh", FACTS, "scripts/exp58/live.py", "src/devision/reading.py",
         "src/devision/infer.py", "src/devision/model.py", "scripts/exp53/watch.py"]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, rel))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


e58 = _load("exp58_live", "../exp58/live.py")
e57, e54, sha, write_json, register, jsonl = e58.e57, e58.e54, e58.sha, e58.write_json, e58.register, e58.jsonl


# ------------------------------------------------------------------ the 24 items and their offline inputs

def items() -> List[Dict]:
    """The 24 transcribed questions in a fixed order (kind, then the source file's order), each with its index in the
    source file (EXP58's request key "<index>:<qid>") and its row."""
    ids = {k for k in json.load(open(TRANSCRIBED)) if not k.startswith("_")}
    out = []
    for kind, (name, path) in SOURCES.items():
        for i, r in enumerate(jsonl(path)):
            if r["id"] in ids:
                if len(r["questions"]) != 1:
                    raise SystemExit("%s: one question expected" % r["id"])
                qid = next(iter(r["questions"]))
                out.append({"kind": kind, "set": name, "index": i, "qid": qid, "key": "%d:%s" % (i, qid), "row": r})
    if len(out) != 24 or Counter(x["kind"] for x in out) != {"temp": 12, "force": 12}:
        raise SystemExit("the transcribed items are not 12 temperature + 12 magnets in the dev_sqa files")
    return out


def gold_numbers(its: List[Dict]) -> Dict[str, Dict[str, str]]:
    """{row id: {reading template: gold option text}} from data/exp28/fields.jsonl, for exactly the templates each
    question routes to; every gold option text must appear in the row's transcription."""
    from devision.reading import reads_for
    tr = json.load(open(TRANSCRIBED))
    gold: Dict[str, Dict[str, str]] = {}
    for f in jsonl(FIELDS):
        rid = f["id"].split(":", 1)[1].rsplit(":", 2)[0]
        q = f["questions"]["q"]
        g = max(f["gold"]["q"]["probabilities"].items(), key=lambda a: a[1])[0]
        gold.setdefault(rid, {})[q["instructions"]] = g
    out = {}
    for it in its:
        rid = it["row"]["id"]
        want = reads_for(it["row"]["questions"])
        if set(gold.get(rid, {})) != set(want):
            raise SystemExit("%s: field golds %s, the question reads %s" % (rid, sorted(gold.get(rid, {})), want))
        for t, g in gold[rid].items():
            if g not in tr[rid]:
                raise SystemExit("%s: gold %r is not in the transcription" % (rid, g))
        out[rid] = gold[rid]
    if sum(len(v) for v in out.values()) != 72:
        raise SystemExit("72 fields expected")
    return out


def value_of(option: str) -> str:
    """A gold option's number in the table's format ("1,300 m/s" -> "1,300", "42.8 mm" -> "42.8")."""
    from devision.reading import fmt
    return fmt(e54.number_of(option))


def g_inputs(its: List[Dict], gold: Dict[str, Dict[str, str]], facts: Dict[str, str]) -> Dict[str, Dict[str, Dict]]:
    """G_num / G_facts states, built offline: the question's templates in reading's fixed order, the gold numbers in the
    table's format, reading.serialise; G_facts appends one line of the picture's non-number facts (magnets only)."""
    from devision.reading import TEMPLATES, reads_for, serialise
    out: Dict[str, Dict[str, Dict]] = {"G_num": {}, "G_facts": {}}
    for it in its:
        r = it["row"]
        table = [[*TEMPLATES[t], value_of(gold[r["id"]][t])] for t in reads_for(r["questions"])]
        state = serialise(r.get("state_text", ""), table)
        out["G_num"][r["id"]] = {"state": state, "table": table}
        if it["kind"] == "force":
            out["G_facts"][r["id"]] = {"state": state + "\n%s %s" % (FACTS_HEADER, facts[r["id"]]), "table": table, "facts": facts[r["id"]]}
    return out


def load_facts(its: List[Dict]) -> Dict[str, str]:
    facts = {k: v for k, v in json.load(open(FACTS)).items() if not k.startswith("_")}
    want = {it["row"]["id"] for it in its if it["kind"] == "force"}
    if set(facts) != want:
        raise SystemExit("facts.json must describe exactly the 12 magnet pictures")
    banned = ("strong", "weak", "greater", "smaller force", "magnitude", "attract", "repel", "force")
    bad = [k for k, v in facts.items() if any(b in v.lower() for b in banned)]
    if bad:
        raise SystemExit("facts must not state a force relation: %s" % bad)
    return facts


# ------------------------------------------------------------------ start

def bound_e58() -> Dict[str, str]:
    bad = [f for f, h in E58_SHA.items() if sha(os.path.join(E58, f)) != h]
    if bad:
        raise SystemExit("EXP58's %s changed" % bad)
    for files in json.load(open(os.path.join(E58, "registry.json"))).values():
        for f, h in files.items():
            if sha(os.path.join(E58, f)) != h:
                raise SystemExit("EXP58's registered %s changed" % f)
    return dict(E58_SHA)


def stage_start(deadline: int, out: str = OUT, data_root: str = "data") -> Dict:
    path = os.path.join(out, "START.json")
    if os.path.exists(path):
        raise SystemExit("%s exists (fresh only)" % path)
    e58_files = bound_e58()
    st58 = e58.check_run(E58, data_root=data_root, code=False)          # EXP58's combination, model, bundle, pictures as now
    its = items()
    pics = json.load(open(os.path.join(E58, "pictures.json")))
    if any(it["row"]["image"] not in pics for it in its):
        raise SystemExit("a transcribed picture is not in EXP58's manifest")
    if any(sha(SOURCES[k][1]) != st58["data"][SOURCES[k][0]] for k in SOURCES):
        raise SystemExit("the dev_sqa files are not EXP58's")
    inputs = g_inputs(its, gold_numbers(its), load_facts(its))
    os.makedirs(out, exist_ok=True)
    files = {}
    for c in ("G_num", "G_facts"):
        f = "inputs_%s.jsonl" % c
        with open(os.path.join(out, f), "w") as fh:
            fh.write("".join(json.dumps({"id": rid, **v}) + "\n" for rid, v in sorted(inputs[c].items())))
        files[f] = sha(os.path.join(out, f))
    register(out, "inputs", files)
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    s = {"git_head": git("rev-parse", "HEAD"), "dirty_files": [f for f in FILES if git("status", "--porcelain", "--", f)],
         "files": {f: sha(f) for f in FILES}, "model": st58["model"], "bundle": st58["bundle"], "exp58": e58_files,
         "exp58_pictures": st58["pictures"], "items": [{k: it[k] for k in ("kind", "set", "index", "qid", "key")} | {"id": it["row"]["id"],
                                                         "image": it["row"]["image"], "picture_sha256": pics[it["row"]["image"]]} for it in its],
         "sources": {"transcribed": sha(TRANSCRIBED), "fields": sha(FIELDS), "facts": sha(FACTS),
                     **{SOURCES[k][0]: sha(SOURCES[k][1]) for k in SOURCES}},
         "inputs": files, "requests": {"P": 24, "G_num": 24, "G_facts": 12}, "temperature": "none (raw), in memory",
         "purpose": "mechanism diagnosis after EXP58 (critic EXP59-PREP-01): reading vs number consumption vs missing magnet facts; not a candidate",
         "started": time.strftime("%F %T"), "deadline": deadline, "deadline_text": time.strftime("%F %T", time.localtime(deadline))}
    write_json(path, s)
    return s


def check_start(out: str, data_root: str = "data", code: bool = True) -> Dict:
    st = json.load(open(os.path.join(out, "START.json")))
    if code:
        bad = [f for f, h in st["files"].items() if sha(f) != h]
        if bad:
            raise SystemExit("code differs from START: %s" % bad)
    if bound_e58() != st["exp58"]:
        raise SystemExit("EXP58's files differ from START")
    st58 = e58.check_run(E58, data_root=data_root, code=False)
    if st58["model"] != st["model"] or st58["bundle"] != st["bundle"]:
        raise SystemExit("EXP58's model / bundle differ from START")
    srcs = {"transcribed": sha(TRANSCRIBED), "fields": sha(FIELDS), "facts": sha(FACTS), **{SOURCES[k][0]: sha(SOURCES[k][1]) for k in SOURCES}}
    if srcs != st["sources"]:
        raise SystemExit("sources differ from START")
    if e54.check_registered(out, "inputs") != st["inputs"]:
        raise SystemExit("inputs differ from START")
    return st


# ------------------------------------------------------------------ run

def record(it: Dict, probs: Dict[str, float], extra: Dict) -> Dict:
    from devision.train.evaluate import score_records
    r = it["row"]
    gold = r["gold"][it["qid"]]["probabilities"]
    pred = max(probs, key=probs.__getitem__)
    return score_records([{"sample_id": r["id"], "qid": it["qid"], "image_id": r["image_id"], "kind": it["kind"],
                           "options": list(r["questions"][it["qid"]]["criteria"]), "gold": gold,
                           "gold_answer": max(gold, key=gold.__getitem__), "probabilities": probs, "prediction": pred, **extra}])[0]


def write_cond(out: str, c: str, recs: List[Dict], st: Dict) -> None:
    ident = {"condition": c, "start_sha256": sha(os.path.join(out, "START.json")), "model": st["model"],
             "bundle": st["bundle"]["reading.json"], "temperature": "none (raw)",
             "inputs_sha256": None if c == "P" else st["inputs"]["inputs_%s.jsonl" % c]}
    f = "details_%s.jsonl" % c
    with open(os.path.join(out, f), "w") as fh:
        fh.write(json.dumps({"identity": ident}) + "\n" + "".join(json.dumps(r) + "\n" for r in recs))
    register(out, "decide_" + c, {f: sha(os.path.join(out, f))})


def p_condition(reading, its: List[Dict], data_root: str, pics: Dict[str, str], on_request=None) -> List[Dict]:
    """EXP58's request for each item (the same key, picture, text state and question) with diagnostics kept."""
    from devision.train.evaluate import _probabilities
    recs = []
    for n, it in enumerate(its):
        r = it["row"]
        diag: Dict = {}
        res = e58.ask(reading, r["image"], r.get("state_text") or "", {it["key"]: r["questions"][it["qid"]]}, data_root, pics, diag)
        probs = _probabilities(res["answers"][it["key"]])            # as EXP58's run_set
        recs.append(record(it, probs, {"diagnostics": {k: diag.get(k) for k in ("reads", "table", "state")}}))
        if on_request:
            on_request(n + 1)
    return recs


def reproduction(p_recs: List[Dict], st58: Dict) -> Dict:
    """P against EXP58's checked records on the same 24 questions: same prediction, probabilities within 1e-5."""
    old = {}
    for name in ("sqa_temp", "sqa_force"):
        for r in e58.set_records(E58, name, st58):
            old[(r["sample_id"], r["qid"])] = r
    if len(p_recs) != 24 or any((r["sample_id"], r["qid"]) not in old for r in p_recs):
        return {"reproduced": False, "reason": "questions missing in EXP58's records"}
    pairs = [(r, old[(r["sample_id"], r["qid"])]) for r in p_recs]
    diff = max(abs(r["probabilities"][k] - o["probabilities"][k]) for r, o in pairs for k in r["probabilities"])
    same = all(r["prediction"] == o["prediction"] for r, o in pairs)
    return {"questions": len(pairs), "same_predictions": same, "max_probability_difference": diff, "reproduced": bool(same and diff <= 1e-5)}


def stage_run(out: str = OUT, data_root: str = "data") -> Dict:
    st = check_start(out, data_root)
    reading, st58, pics = e58.load_reading(E58, data_root=data_root)       # EXP58's checks before the load; raw temperature
    its = items()
    if [it["row"]["id"] for it in its] != [x["id"] for x in st["items"]]:
        raise SystemExit("items differ from START")
    t0 = time.time()
    prog = lambda base: (lambda n: e54.progress(out, "run", base + n, 60))  # noqa: E731
    p = p_condition(reading, its, data_root, pics, on_request=prog(0))
    write_cond(out, "P", p, st)
    check_diagnostics(p, its)                                             # invalid evidence stops before G
    rep = reproduction(p, st58)
    write_json(os.path.join(out, "reproduction.json"), rep)
    register(out, "reproduction", {"reproduction.json": sha(os.path.join(out, "reproduction.json"))})
    if not rep["reproduced"]:
        raise SystemExit("P does not reproduce EXP58 on the 24 questions: %s -- G_num / G_facts not run" % rep)
    by = {it["row"]["id"]: it for it in its}
    base = 24
    for c in ("G_num", "G_facts"):
        inp = e54.load_inputs(out, c)
        recs = []
        for n, rid in enumerate(sorted(inp, key=lambda k: [x["row"]["id"] for x in its].index(k))):
            it = by[rid]
            res = reading.decider.decide(inp[rid]["state"], {it["key"]: it["row"]["questions"][it["qid"]]}, image=None)
            recs.append(record(it, dict(res["answers"][it["key"]]["probabilities"]), {"state": inp[rid]["state"]}))
            e54.progress(out, "run", base + n + 1, 60)
        base += len(inp)
        write_cond(out, c, recs, st)
    return {"seconds": time.time() - t0}


# ------------------------------------------------------------------ report

def check_diagnostics(p_lines: List[Dict], its: List[Dict]) -> None:
    """P's intermediate evidence is the evidence actually consumed (critic EXP59-R1-01), else SystemExit (P invalid, no
    reading / consumption attribution): per question the reads are exactly reads_for's templates, in order, once each;
    each read has its fields and a 4-number box; valid -> the decoded string is a strict number and value = fmt(decoded),
    invalid -> value = "unreadable" (a real decoding failure, counted as a reading failure later, never repaired); the
    table is the reads in that order; the state is the row's own text + serialise(table)."""
    from devision.reading import TEMPLATES, UNREADABLE, VALID, fmt, reads_for, serialise
    by = {it["row"]["id"]: it for it in its}
    for r in p_lines:
        it = by[r["sample_id"]]
        d = r.get("diagnostics") or {}
        reads, table, state = d.get("reads"), d.get("table"), d.get("state")
        if not isinstance(reads, list) or not isinstance(table, list) or not isinstance(state, str):
            raise SystemExit("%s: diagnostics incomplete" % r["sample_id"])
        want = reads_for(it["row"]["questions"])
        if [x.get("question") for x in reads] != want:
            raise SystemExit("%s: reads %s, the question needs %s (each once, in order)" % (r["sample_id"], [x.get("question") for x in reads], want))
        for x in reads:
            box = x.get("box")
            if not all(k in x for k in ("decoded", "valid", "value")) or not (isinstance(box, list) and len(box) == 4
                                                                                and all(isinstance(v, (int, float)) for v in box)):
                raise SystemExit("%s: a read lacks its fields / box" % r["sample_id"])
            if x["valid"] is True:
                ok = bool(VALID.match(x["decoded"])) and x["value"] == fmt(x["decoded"])
            elif x["valid"] is False:
                ok = x["value"] == UNREADABLE
            else:
                ok = False
            if not ok:
                raise SystemExit("%s: a read's value is not its decoded string's format" % r["sample_id"])
        if table != [[*TEMPLATES[x["question"]], x["value"]] for x in reads]:
            raise SystemExit("%s: the table is not the reads" % r["sample_id"])
        if state != serialise(it["row"].get("state_text") or "", table):
            raise SystemExit("%s: the state is not the row's text + the table" % r["sample_id"])


def cond_records(out: str, c: str, st: Dict, its: List[Dict]) -> Dict[str, Dict]:
    f = "details_%s.jsonl" % c
    if e54.check_registered(out, "decide_" + c).get(f) != sha(os.path.join(out, f)):
        raise SystemExit("%s not registered" % f)
    lines = jsonl(os.path.join(out, f))
    ident = lines[0]["identity"]
    want = {"condition": c, "start_sha256": sha(os.path.join(out, "START.json")), "model": st["model"], "bundle": st["bundle"]["reading.json"],
            "temperature": "none (raw)", "inputs_sha256": None if c == "P" else st["inputs"]["inputs_%s.jsonl" % c]}
    if any(ident.get(k) != v for k, v in want.items()):
        raise SystemExit("%s: identity does not bind" % f)
    rows = [it["row"] for it in its if c != "G_facts" or it["kind"] == "force"]
    got = e57.r41.check_details("exp59/%s" % c, rows, lines[1:])
    if c == "P":
        check_diagnostics(lines[1:], its)
    else:
        inp = e54.load_inputs(out, c)
        if any(r["state"] != inp[r["sample_id"]]["state"] for r in lines[1:]):
            raise SystemExit("%s: a record's state is not its registered input" % f)
    return {k[0]: v for k, v in got.items()}


def reading_stats(p: Dict[str, Dict], gold: Dict[str, Dict[str, str]], its: List[Dict]) -> Dict:
    """Per kind: fields exact / invalid (72 in all), pictures with every needed number right, and pictures whose two
    gold values of a quantity differ but were read as equal."""
    from devision.reading import TEMPLATES
    out = {}
    for kind in ("temp", "force"):
        n = exact = invalid = pics_all = unequal = unequal_same = 0
        for it in its:
            if it["kind"] != kind:
                continue
            rid = it["row"]["id"]
            reads = {x["question"]: x for x in p[rid]["diagnostics"]["reads"]}
            ok_all = True
            for t, g in gold[rid].items():
                n += 1
                x = reads.get(t)
                invalid += not (x and x["valid"])
                hit = bool(x and x["valid"] and x["value"] == value_of(g))
                exact += hit
                ok_all &= hit
            pics_all += ok_all
            by_q: Dict[str, List[str]] = {}
            for t, g in gold[rid].items():
                by_q.setdefault(TEMPLATES[t][1], []).append(t)
            for q, ts in by_q.items():
                if len(ts) == 2 and value_of(gold[rid][ts[0]]) != value_of(gold[rid][ts[1]]):
                    unequal += 1
                    a, b = reads.get(ts[0]), reads.get(ts[1])
                    unequal_same += bool(a and b and a["value"] == b["value"])
        out[kind] = {"fields": n, "exact": exact, "invalid": invalid, "pictures_all_numbers_right": pics_all,
                     "unequal_gold_pairs": unequal, "unequal_read_as_equal": unequal_same}
    return out


def numbers_right(p: Dict, gold: Dict[str, str]) -> bool:
    """Every number the question needs was read, valid and equal to its gold (in the table's format)."""
    reads = {x["question"]: x for x in p["diagnostics"]["reads"]}
    return all(t in reads and reads[t]["valid"] and reads[t]["value"] == value_of(g) for t, g in gold.items())


def transitions(a: Dict[str, Dict], b: Dict[str, Dict], keys: List[str]) -> Dict:
    from devision.train.compare import paired_metric
    t = Counter("%s->%s" % ("R" if a[k]["correct"] else "W", "R" if b[k]["correct"] else "W") for k in keys)
    m = paired_metric([a[k] for k in keys], [b[k] for k in keys], "accuracy")
    return {"n": len(keys), "a_right": sum(a[k]["correct"] for k in keys), "b_right": sum(b[k]["correct"] for k in keys),
            "transitions": dict(t), "difference": m["difference"], "ci95": m["ci95"]}


def report(out: str = OUT, finalize: str = "") -> Dict:
    res: Dict[str, Any] = {"states": {}, "candidate_eligible": False}
    if finalize:
        res["finalized"] = {"exit_codes": finalize, "note": "the run did not complete; missing items are not low scores"}
    if not os.path.exists(os.path.join(out, "START.json")):
        res["branch"] = "incomplete: no START (nothing ran)"
        return write_report(out, res)
    try:
        st = check_start(out, data_root="", code=False)
    except (SystemExit, OSError, ValueError, KeyError) as e:
        res["branch"] = "invalid: the run's identity does not verify (%s) -- no verdict" % e
        return write_report(out, res)
    its = items()
    recs = {}
    for c in CONDS:
        try:
            recs[c] = cond_records(out, c, st, its)
            res["states"][c] = "completed"
        except (SystemExit, OSError, ValueError, KeyError) as e:
            res["states"][c] = "missing / invalid: %s" % e
    try:
        f = os.path.join(out, "reproduction.json")
        if e54.check_registered(out, "reproduction").get("reproduction.json") != sha(f):
            raise SystemExit("reproduction.json not registered")
        res["reproduction"] = reproduction(list(recs["P"].values()), e58.check_run(E58, data_root="", code=False)) if "P" in recs else None
    except (SystemExit, OSError, ValueError, KeyError) as e:
        res["reproduction"] = {"reproduced": False, "reason": str(e)}
    if not (res["reproduction"] or {}).get("reproduced"):
        res["branch"] = "invalid / incomplete: P does not reproduce EXP58 (or is missing) -- no reading / consumption verdict"
        return write_report(out, res)
    gold = gold_numbers(its)
    res["reading"] = reading_stats(recs["P"], gold, its)
    ids = {k: [it["row"]["id"] for it in its if it["kind"] == k] for k in ("temp", "force")}
    res["accuracy"] = {c: {k: {"right": sum(recs[c][i]["correct"] for i in ids[k] if i in recs[c]), "of": sum(i in recs[c] for i in ids[k])}
                           for k in ids} for c in recs}
    res["paired"] = {}
    if "G_num" in recs:
        for k in ids:
            res["paired"]["G_num-P:" + k] = transitions(recs["P"], recs["G_num"], ids[k])
    if "G_facts" in recs and "G_num" in recs:
        res["paired"]["G_facts-G_num:force"] = transitions(recs["G_num"], recs["G_facts"], ids["force"])
    res["per_question"] = [{"id": i, "kind": k, **{c: bool(recs[c][i]["correct"]) for c in recs if i in recs[c]},
                            "numbers_right": numbers_right(recs["P"][i], gold[i])} for k in ids for i in ids[k]]
    res["branch"] = branches(res)
    return write_report(out, res)


def branches(res: Dict) -> List[str]:
    """The pre-written readings (several can hold; 12 questions per kind: wide intervals, no difference is not absence)."""
    out = []
    pq = res["per_question"]
    # a condition missing / invalid for a question is undecided: only explicit True / False values count (EXP59-R1-02)
    wrong_numbers_fixed = [q for q in pq if not q["numbers_right"] and q.get("P") is False and q.get("G_num") is True]
    right_numbers_wrong = [q for q in pq if q["numbers_right"] and q.get("P") is False]
    gnum_wrong = [q for q in pq if q.get("G_num") is False]
    if wrong_numbers_fixed:
        out.append("reading-side repair has value: %d questions wrong with wrong numbers became right with gold numbers "
                   "(box placement decides the next selector / character check; not a sole-cause claim)" % len(wrong_numbers_fixed))
    if right_numbers_wrong or gnum_wrong:
        out.append("text consumption / rule problem stays: %d questions wrong with every number read right, %d wrong with gold "
                   "numbers -- same-kind supervision in training sources, not reading-only training" % (len(right_numbers_wrong), len(gnum_wrong)))
    mag = [q for q in pq if q["kind"] == "force"]
    fixed_by_facts = [q for q in mag if q.get("G_num") is False and q.get("G_facts") is True]
    still = [q for q in mag if q.get("G_facts") is False]
    if fixed_by_facts:
        out.append("magnets: %d wrong with gold numbers became right with the picture's facts -- the fact path to the decision first" % len(fixed_by_facts))
    if still:
        out.append("magnets: %d still wrong with the full facts -- consuming the complete facts is not established" % len(still))
    return out or ["no wrong question to localise"]


def write_report(out: str, res: Dict) -> Dict:
    os.makedirs(out, exist_ok=True)
    write_json(os.path.join(out, "report.json"), res)
    keys = ("finalized", "states", "reproduction", "reading", "accuracy", "paired", "branch", "candidate_eligible")
    text = json.dumps({k: res[k] for k in keys if k in res}, indent=1, default=str)
    open(os.path.join(out, "report.txt"), "w").write(text + "\n")
    print(text)
    return res


def preview() -> Dict:
    """Model-free: the 12 G_facts states (and the G_num ones) as they would be sent."""
    its = items()
    return g_inputs(its, gold_numbers(its), load_facts(its))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["preview", "start", "run", "report"])
    ap.add_argument("--deadline", type=int, default=0)
    ap.add_argument("--finalize", default="")
    a = ap.parse_args(argv)
    if a.stage == "preview":
        p = preview()
        for rid, v in p["G_facts"].items():
            print("%s\n%s\n" % (rid, v["state"]))
    elif a.stage == "start":
        print(json.dumps({k: v for k, v in stage_start(a.deadline).items() if k not in ("files", "items")}, indent=1))
    elif a.stage == "run":
        print(json.dumps(stage_run(), indent=1))
    else:
        try:
            report(finalize=a.finalize)
        except Exception as e:                       # never end without a readable file
            os.makedirs(OUT, exist_ok=True)
            open(os.path.join(OUT, "report.txt"), "w").write("REPORT FAILED (invalid, no verdict): %s\n" % e)
            print("REPORT FAILED: %s" % e, file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
