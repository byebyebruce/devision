"""exp56: the reading mode live on the 600 original rl_tune pictures (critic EXP56-PREP-01..03;
docs/experiments/2026-10-18-exp56-live-reading-plan.md). Zero training. Stages (fresh only, runs/exp56):

  start  model-free: identities -- EXP50-T's frozen weights / config, EXP53's selector.pt (8f10f1fd, its registry),
         EXP51's head_T.pt (0b548142, its registry), EXP53 / EXP55 / EXP54 frozen products used as the paired baseline,
         questions / rl_tune, this code -- then the reading bundle (runs/exp56/bundle: the two component files copied,
         reading.json binding them and the decision checkpoint's weights / config / tokenizer), START.json.
  live   MPS: EXP50-T and the bundle through devision.reading.Reading (the product path: picture -> selector -> head ->
         table -> ordinary decide with image=None). P: one request per picture with the row's questions as they are
         (reads, boxes, decoded strings, the actual state text, outputs and timings kept); then two diagnostic
         interventions on the live tables only -- E (every value unreadable) and D (the fixed twin picture's live
         table) -- through ordinary text decide. Each condition registered as soon as it completes.
  report model-free: live reads vs EXP53's automatic readings (same strings >= 99%, accuracy within 0.01); requests
         whose table text equals EXP55's P input: same predictions and probabilities within 1e-5 against EXP55's
         details; EXP54 / 55's gates with G from EXP55's frozen details (not re-run), the read-right set and donor
         counts as in EXP55; comparisons reported; timings.

    uv run python scripts/exp56/exp56.py start --deadline EPOCH | live | report [--finalize CODES]
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from typing import Dict, List

HERE = os.path.dirname(__file__)
OUT = "runs/exp56"
E53, E55 = "runs/exp53", "runs/exp55"
SELECTOR, SELECTOR_SHA = "runs/exp53/selector.pt", "8f10f1fd3de347dd7d5d2c375b7c4660ba97b75e65920e295132845d5cd2518a"
HEAD, HEAD_SHA = "runs/exp51/head_T.pt", "0b548142ed3697742ca9efaec7498e3aaa37d2872415d6c8c77f6f2ba9d0e3fa"
E55_START_SHA = "bf0595676f54e8decc666cebecf3693febce386a1be14209d8dd98526a644e12"
E53_PICTURES_SHA = "14366ad74e8d2dd8a4ff1c511bd85c990115725fd809707a656ae06607f5de8d"     # EXP53 START pictures_sha256
FILES = ["scripts/exp56/exp56.py", "scripts/exp56/run.sh", "src/devision/reading.py", "src/devision/infer.py", "src/devision/model.py",
         "src/devision/image.py", "scripts/exp54/exp54.py", "scripts/exp55/exp55.py", "scripts/exp53/watch.py"]
CONDS = ("P", "E", "D")


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, rel))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


e55 = _load("exp55", "../exp55/exp55.py")
e54 = e55.e54
r42, sha, write_json, register = e54.r42, e54.sha, e54.write_json, e54.register


def merge_register(out: str, name: str, files: Dict[str, str]) -> None:
    path = os.path.join(out, "registry.json")
    r = json.load(open(path)) if os.path.exists(path) else {}
    r[name] = {**r.get(name, {}), **files}
    write_json(path, r)


def stage_start(deadline: int, out: str = OUT) -> Dict:
    from devision.reading import write_bundle
    path = os.path.join(out, "START.json")
    if os.path.exists(path):
        raise SystemExit("%s exists (fresh only)" % path)
    if sha(SELECTOR) != SELECTOR_SHA or json.load(open(os.path.join(E53, "registry.json")))["selector"]["selector.pt"] != SELECTOR_SHA:
        raise SystemExit("EXP53's selector is not 8f10f1fd / not its registered one")
    if sha(HEAD) != HEAD_SHA or json.load(open("runs/exp51/registry.json"))["T"]["head_sha256"] != HEAD_SHA:
        raise SystemExit("EXP51's head is not 0b548142 / not its registered one")
    if sha(os.path.join(E55, "START.json")) != E55_START_SHA:
        raise SystemExit("EXP55's START changed")
    e55reg = json.load(open(os.path.join(E55, "registry.json")))
    for name, files in e55reg.items():
        for f, h in files.items():
            if sha(os.path.join(E55, f)) != h:
                raise SystemExit("EXP55's %s changed" % f)
    st55 = json.load(open(os.path.join(E55, "START.json")))
    model = e54.verified_model(st55)
    if sha(e54.QUESTIONS) != st55["questions_sha256"] or sha(e54.TUNE) != st55["tune_sha256"]:
        raise SystemExit("questions / rl_tune are not those EXP55 bound")
    e53det = os.path.join(E53, "details_auto.jsonl")
    if sha(e53det) != e54.E53_DETAILS_SHA:
        raise SystemExit("EXP53's automatic details changed")
    import hashlib
    qs_all = [json.loads(x) for x in open(e54.QUESTIONS) if x.strip()]
    allpics = {q["picture"]: sha(os.path.join("data", q["picture"])) for q in qs_all}
    if hashlib.sha256(json.dumps(allpics, sort_keys=True).encode()).hexdigest() != E53_PICTURES_SHA or \
            json.load(open(os.path.join(E53, "START.json")))["pictures_sha256"] != E53_PICTURES_SHA:
        raise SystemExit("the pictures are not those EXP53 bound (14366ad7)")
    rows0 = {r["id"]: r for r in map(json.loads, open(e54.TUNE))}
    pics = {r["image"]: allpics[r["image"]] for r in rows0.values()}
    os.makedirs(out, exist_ok=True)
    write_json(os.path.join(out, "pictures.json"), pics)
    bundle = write_bundle(os.path.join(out, "bundle"), os.path.join(e54.RUN, "stage2"), SELECTOR, HEAD,
                          {"selector": {"path": SELECTOR, "exp53_start": json.load(open(os.path.join(E53, "registry.json")))["selector"]},
                           "head": {"path": HEAD, "exp51_registry_T": json.load(open("runs/exp51/registry.json"))["T"]}})
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    s = {"git_head": git("rev-parse", "HEAD"), "dirty_files": [f for f in FILES if git("status", "--porcelain", "--", f)],
         "files": {f: sha(f) for f in FILES}, "model": model, "bundle": {"reading.json": sha(os.path.join(out, "bundle", "reading.json")), **bundle["sha256"]},
         "pictures": {"pictures.json": sha(os.path.join(out, "pictures.json")), "count": len(pics), "exp53_pictures_sha256": E53_PICTURES_SHA},
         "temperature": "none (raw), in memory", "questions_sha256": st55["questions_sha256"], "tune_sha256": st55["tune_sha256"],
         "baselines": {"exp53_details_auto": e54.E53_DETAILS_SHA, "exp55_start": E55_START_SHA, "exp55_registry": e55reg},
         "conditions": list(CONDS), "started": time.strftime("%F %T"), "deadline": deadline,
         "deadline_text": time.strftime("%F %T", time.localtime(deadline))}
    write_json(path, s)
    return s


def check_combination(out: str, st: Dict, code: bool = True) -> None:
    """The run's combination is START's: the code files (optional), the bundle files, the source pictures (critic
    EXP56-R1-01). Refused otherwise."""
    if code:
        bad = [f for f, h in st["files"].items() if sha(f) != h]
        if bad:
            raise SystemExit("code differs from START: %s" % bad)
    b = os.path.join(out, "bundle")
    for f in ("reading.json", "selector.pt", "head.pt"):
        if sha(os.path.join(b, f)) != st["bundle"][f]:
            raise SystemExit("bundle %s differs from START" % f)
    if sha(os.path.join(out, "pictures.json")) != st["pictures"]["pictures.json"]:
        raise SystemExit("pictures.json differs from START")


def check_pictures(out: str, data_root: str = "data") -> Dict[str, str]:
    pics = json.load(open(os.path.join(out, "pictures.json")))
    bad = [p for p, h in pics.items() if sha(os.path.join(data_root, p)) != h]
    if bad:
        raise SystemExit("%d pictures differ from START's, e.g. %s: refused before any forward" % (len(bad), bad[0]))
    return pics


def records(rows, rid, res) -> List[Dict]:
    from devision.train.evaluate import score_records
    row, out = rows[rid], []
    for qid, q in row["questions"].items():
        probs = dict(res["answers"][qid]["probabilities"])
        gold = row["gold"][qid]["probabilities"]
        out.append({"sample_id": rid, "qid": qid, "image_id": row["image_id"], "kind": row.get("kind", ""), "axis": row.get("axis", ""),
                    "options": list(q["criteria"]), "gold": gold, "gold_answer": max(gold, key=gold.__getitem__),
                    "probabilities": probs, "prediction": max(probs, key=probs.__getitem__)})
    return score_records(out)


def write_condition(out: str, c: str, recs: List[Dict], s_sha: str, model: Dict, dev: str, inputs: Dict[str, Dict], extra=None) -> None:
    f_in = "inputs_%s.jsonl" % c
    with open(os.path.join(out, f_in), "w") as fh:
        for rid in sorted(inputs):
            fh.write(json.dumps({"id": rid, **inputs[rid]}) + "\n")
    merge_register(out, "inputs", {f_in: sha(os.path.join(out, f_in))})
    ident = {"condition": c, "start_sha256": s_sha, "inputs_sha256": sha(os.path.join(out, f_in)), "model": model,
             "temperature": "none (raw)", "image": None, "device": dev, **(extra or {})}
    f = "details_%s.jsonl" % c
    with open(os.path.join(out, f), "w") as fh:
        fh.write(json.dumps({"identity": ident}) + "\n")
        fh.write("".join(json.dumps(r) + "\n" for r in recs))
    register(out, "decide_" + c, {f: sha(os.path.join(out, f))})


def stage_live(out: str = OUT, data_root: str = "data", device: str = "auto") -> Dict:
    from PIL import Image

    import hashlib
    import io

    from devision.infer import best_device
    from devision.reading import UNREADABLE, Reading, serialise
    s_sha = e54.started(out)
    st = json.load(open(os.path.join(out, "START.json")))
    model = e54.verified_model(st)                       # before any load
    check_combination(out, st)
    pics = check_pictures(out, data_root)
    rows = {r["id"]: r for r in map(json.loads, open(e54.TUNE))}
    if sha(e54.TUNE) != st["tune_sha256"] or sha(e54.QUESTIONS) != st["questions_sha256"]:
        raise SystemExit("rl_tune / questions differ from START")
    dev = best_device() if device == "auto" else device
    ck = os.path.join(e54.RUN, "stage2")
    reading = Reading.load(ck, os.path.join(out, "bundle"), device=dev)       # verified entry: loads its own Decider
    d = reading.decider
    d.cfg.temperature, d.cfg.temperature_by_options = [1.0, 1.0, 1.0], {}     # raw, in memory only
    total = 3 * len(rows)
    t0 = time.time()
    recs, inputs, reads = [], {}, []
    for n, rid in enumerate(sorted(rows)):
        row = rows[rid]
        diag: Dict = {}
        t = time.perf_counter()
        raw = open(os.path.join(data_root, row["image"]), "rb").read()
        if hashlib.sha256(raw).hexdigest() != pics[row["image"]]:
            raise SystemExit("%s changed during the run" % row["image"])
        with Image.open(io.BytesIO(raw)) as img:
            res = reading.decide(row.get("state_text", ""), row["questions"], image=img.convert("RGB"), diagnostics=diag)
        secs = time.perf_counter() - t
        recs += records(rows, rid, res)
        inputs[rid] = {"state": diag["state"], "table": diag["table"]}
        reads.append({"id": rid, "reads": diag["reads"], "read_seconds": diag["read_seconds"], "request_seconds": secs})
        e54.progress(out, "live", n + 1, total)
    with open(os.path.join(out, "reads.jsonl"), "w") as fh:
        fh.write("".join(json.dumps(r) + "\n" for r in reads))
    merge_register(out, "reads", {"reads.jsonl": sha(os.path.join(out, "reads.jsonl"))})
    write_condition(out, "P", recs, s_sha, model, dev, inputs,
                    {"reading_bundle": st["bundle"]["reading.json"], "pictures": st["pictures"]["pictures.json"]})
    print("P done after %.1f s" % (time.time() - t0), flush=True)
    qs53 = [json.loads(x) for x in open(e54.QUESTIONS)]
    twin_row = {}
    key_row = {q["key"]: q["id"] for q in qs53 if q["set"] == "rl_tune"}
    for q in qs53:
        if q["set"] == "rl_tune" and q["twin"]:
            twin_row[q["id"]] = key_row[q["twin"]]
    for k, c in enumerate(("E", "D")):
        cin = {}
        for rid in rows:
            st_text = rows[rid].get("state_text", "")
            table = [[o, q, UNREADABLE] for o, q, _ in inputs[rid]["table"]] if c == "E" else inputs[twin_row[rid]]["table"]
            cin[rid] = {"state": serialise(st_text, table), "table": table, **({"twin_row": twin_row[rid]} if c == "D" else {})}
        crecs = e54.decide_condition(d, rows, cin, on_request=lambda m, k=k: e54.progress(out, "live", (k + 1) * len(rows) + m, total))
        write_condition(out, c, crecs, s_sha, model, dev, cin)
        print("%s done after %.1f s" % (c, time.time() - t0), flush=True)
    return {"seconds": time.time() - t0, "device": dev}


# ------------------------------------------------------------------ report

def check_reads(out: str, rows: Dict[str, Dict], st: Dict) -> Dict[str, Dict[str, Dict]]:
    """The registered reads (critic EXP56-R1-01): unchanged since registration; every picture once; for each, exactly
    the reading templates its questions call for (reads_for), once each; each read's value is the strict decoding /
    separator format of its decoded string; inputs_P's table and state are exactly those reads serialised; the P
    details carry START's bundle and pictures. Returns {row id: {template: read}}; refuses otherwise."""
    from devision.reading import TEMPLATES, UNREADABLE, VALID, fmt, reads_for, serialise
    if e54.check_registered(out, "reads").get("reads.jsonl") != sha(os.path.join(out, "reads.jsonl")):
        raise SystemExit("reads.jsonl not registered")
    lines = [json.loads(x) for x in open(os.path.join(out, "reads.jsonl")) if x.strip()]
    ids = [r["id"] for r in lines]
    if len(ids) != len(set(ids)) or set(ids) != set(rows):
        raise SystemExit("reads: %d pictures for %d rows (duplicates or missing)" % (len(set(ids)), len(rows)))
    pin = e54.load_inputs(out, "P")
    out_reads: Dict[str, Dict[str, Dict]] = {}
    for r in lines:
        want = reads_for(rows[r["id"]]["questions"])
        got = [x["question"] for x in r["reads"]]
        if got != want:
            raise SystemExit("%s: reads %s, expected %s" % (r["id"], got, want))
        table = []
        for x in r["reads"]:
            if (x["valid"] and not VALID.match(x["decoded"])) or x["value"] != (fmt(x["decoded"]) if x["valid"] else UNREADABLE):
                raise SystemExit("%s: a read's value is not its decoded string's format" % r["id"])
            table.append([*TEMPLATES[x["question"]], x["value"]])
        if pin[r["id"]]["table"] != table or pin[r["id"]]["state"] != serialise(rows[r["id"]].get("state_text", ""), table):
            raise SystemExit("%s: inputs_P is not the reads serialised" % r["id"])
        out_reads[r["id"]] = {x["question"]: x for x in r["reads"]}
    ident = json.loads(open(os.path.join(out, "details_P.jsonl")).readline())["identity"]
    if ident.get("reading_bundle") != st["bundle"]["reading.json"] or ident.get("pictures") != st["pictures"]["pictures.json"]:
        raise SystemExit("P details are not about START's bundle / pictures")
    return out_reads


def read_fidelity(rows, live: Dict[str, Dict[str, Dict]]) -> Dict:
    """Live decoded strings vs EXP53's automatic readings, per reading question; accuracy of both against gold."""
    qs53 = [json.loads(x) for x in open(e54.QUESTIONS)]
    pred, _, _ = e54.p49.read_details(os.path.join(E53, "details_auto.jsonl"), qs53)
    e53 = {q["key"]: e54.p49.decode(p) for q, p in zip(qs53, pred)}
    same = right_live = right_53 = n = 0
    same_pictures = {}
    for q in qs53:
        if q["set"] != "rl_tune":
            continue
        t = rows[q["id"]]["questions"][q["qid"]]["instructions"]
        lv = live[q["id"]][t]
        a = e53[q["key"]]
        n += 1
        eq = lv["decoded"] == a["text"] and lv["valid"] == a["valid"]
        same += eq
        same_pictures[q["id"]] = same_pictures.get(q["id"], True) and eq
        right_live += lv["valid"] and lv["decoded"] == q["target"]
        right_53 += a["valid"] and a["text"] == q["target"]
    out = {"questions": n, "same_strings": same / n, "accuracy_live": right_live / n, "accuracy_exp53": right_53 / n,
           "pictures_all_same": sum(same_pictures.values())}
    out["reproduced"] = bool(n == 1800 and out["same_strings"] >= 0.99 and abs(out["accuracy_live"] - out["accuracy_exp53"]) <= 0.01)
    out["_same_pictures"] = sorted(k for k, v in same_pictures.items() if v)
    return out


def decision_fidelity(out: str, rows, same_pictures: List[str]) -> Dict:
    """critic EXP56-R1-02: every picture whose live readings all equal EXP53's must give EXP55's exact P state; on the
    requests whose state equals EXP55's, the same predictions and probabilities within 1e-5. No paired request, or a
    same-reading picture with another state, is not reproduced."""
    new_in = e54.load_inputs(out, "P")
    old_in = {r["id"]: r for r in map(json.loads, open(os.path.join(E55, "inputs_P.jsonl")))}
    same_req = {rid for rid in new_in if new_in[rid]["state"] == old_in[rid]["state"]}
    inconsistent = [rid for rid in same_pictures if rid not in same_req]
    new = e54.read_condition(out, "P", rows)
    old = e54.read_condition(E55, "P", rows)
    o = {(r["sample_id"], r["qid"]): r for r in old}
    sel = [r for r in new if r["sample_id"] in same_req]
    diff = max((abs(r["probabilities"][k] - o[(r["sample_id"], r["qid"])]["probabilities"][k]) for r in sel for k in r["probabilities"]), default=None)
    agree = bool(sel) and all(r["prediction"] == o[(r["sample_id"], r["qid"])]["prediction"] for r in sel)
    why = "no paired request" if not sel else "%d same-reading pictures with another state" % len(inconsistent) if inconsistent else ""
    return {"pictures_same_readings": len(same_pictures), "requests_same_table": len(same_req), "questions_paired": len(sel),
            "same_reading_other_state": len(inconsistent), "max_probability_difference": diff, "same_predictions": agree,
            "reproduced": bool(not why and agree and diff is not None and diff <= 1e-5), "reason": why}


def report(out: str = OUT, finalize: str = "") -> Dict:
    st = json.load(open(os.path.join(out, "START.json"))) if os.path.exists(os.path.join(out, "START.json")) else None
    if st is not None and (sha(e54.QUESTIONS) != st["questions_sha256"] or sha(e54.TUNE) != st["tune_sha256"]):
        raise SystemExit("questions / rl_tune differ from START: no report")
    rows = {r["id"]: r for r in map(json.loads, open(e54.TUNE))}
    qs53 = {q["key"]: q for q in map(json.loads, open(e54.QUESTIONS)) if q["set"] == "rl_tune"}
    res: Dict = {"states": {}, "conditions": {}, "differences": {}}
    recs, reading = {}, {}
    for c in CONDS:
        try:
            recs[c] = e54.read_condition(out, c, rows)
            reading[c] = e54.reading_records(recs[c], rows, qs53)
            res["states"][c] = {"state": "completed"}
        except (SystemExit, OSError, ValueError, KeyError) as e:
            res["states"][c] = {"state": "missing" if not os.path.exists(os.path.join(out, "details_%s.jsonl" % c)) else "invalid", "reason": str(e)}
    try:                                                   # G is EXP55's frozen result (not re-run)
        recs["G"] = e54.read_condition(E55, "G", rows)
        reading["G"] = e54.reading_records(recs["G"], rows, qs53)
        res["states"]["G"] = {"state": "completed", "source": "EXP55 (frozen, not re-run)"}
    except (SystemExit, OSError, ValueError, KeyError) as e:
        res["states"]["G"] = {"state": "invalid", "reason": str(e)}
    pin = e54.load_inputs(e55.E54_OUT, "P")              # EXP53's frozen readings: the read-right set (1,470)
    p_right = {k: {(o, qn): v for o, qn, v in pin[q["id"]]["table"]}[e54.TEMPLATES[rows[q["id"]]["questions"][q["qid"]]["instructions"]]] == q["target"]
               for k, q in qs53.items()}
    fams = {c: e54.units(r) for c, r in reading.items()}
    for c, fam in fams.items():
        rr = reading[c]
        s = e54.summary(fam)
        s["nll"] = sum(r["nll"] for r in rr) / len(rr)
        ok = [r for r in rr if p_right[r["key"]]]
        s["read_right"] = {"n": len(ok), "accuracy": sum(r["right"] for r in ok) / len(ok)}
        cmp_: Dict = {}
        for r in recs[c]:
            row = rows[r["sample_id"]]
            if e54.b40.field_of(row["questions"][r["qid"]]["instructions"]) == "cmp":
                k = "%s|%s" % (row.get("kind", ""), row.get("axis", ""))
                cmp_.setdefault(k, [0, 0])
                cmp_[k][0] += r["correct"]
                cmp_[k][1] += 1
        s["comparison"] = cmp_
        res["conditions"][c] = s
    if "D" in reading:
        res["donor"] = e54.donor_stats(reading["D"], rows, qs53, e55.plain_inputs(out, "D"))
    for a, b, stats in (("P", "E", ("both", "different", "all")), ("P", "D", ("changed", "all")), ("G", "P", ("all", "both"))):
        if a in reading and b in reading:
            res["differences"]["%s-%s" % (a, b)] = {s: r42.diff(fams[a], fams[b], r42.clusters(reading[a]), s) for s in stats}
    res["gates"] = e54.gates(res)
    try:
        if st is None:
            raise SystemExit("no START")
        check_combination(out, st, code=False)
        live = check_reads(out, rows, st)
        res["read_fidelity"] = read_fidelity(rows, live)
        res["decision_fidelity"] = decision_fidelity(out, rows, res["read_fidelity"].pop("_same_pictures"))
        reads = [json.loads(x) for x in open(os.path.join(out, "reads.jsonl"))]
        rs = sorted(r["request_seconds"] for r in reads)
        res["timing"] = {"requests": len(rs), "p50_s": rs[len(rs) // 2], "p95_s": rs[int(len(rs) * 0.95)],
                         "read_p50_s": sorted(r["read_seconds"] for r in reads)[len(rs) // 2]}
    except (SystemExit, OSError, ValueError, KeyError) as e:
        res["read_fidelity"] = res["decision_fidelity"] = {"state": "invalid / missing", "reason": str(e)}
    fid = res["read_fidelity"].get("reproduced") is True and res["decision_fidelity"].get("reproduced") is True
    base = e54.branch(res)
    if base.startswith("incomplete"):
        res["branch"] = base
    elif not fid:
        res["branch"] = "live path not reproduced (implementation / numeric path): no weights changed, no template tuned; " + base
    else:
        res["branch"] = "live path reproduced; " + base
    if finalize:
        res["finalized"] = {"exit_codes": finalize, "note": "the run did not complete; missing conditions are not low scores"}
    write_json(os.path.join(out, "report.json"), res)
    lines = ["INCOMPLETE run: %s" % finalize] if finalize else []
    for c, v in res["states"].items():
        lines.append("%s %s%s" % (c, v["state"], "" if v["state"] == "completed" else ": " + v["reason"]))
    for c, s in res["conditions"].items():
        lines.append("%s all %.3f (%s) changed %.3f both %.3f twin %.3f nll %.3f read_right %s comparison %s" % (
            c, s["all"], {f: round(s["all:" + f], 3) for f in r42.FIELDS}, s["changed"], s["both"], s["twin"], s["nll"], s["read_right"], s["comparison"]))
    for k, v in res["differences"].items():
        lines.append("%s %s" % (k, {s: "%+.3f [%+.3f, %+.3f]" % (x["value"], *x["ci95"]) for s, x in v.items()}))
    lines.append("donor %s" % res.get("donor"))
    lines.append("read fidelity %s\ndecision fidelity %s\ntiming %s" % (res["read_fidelity"], res["decision_fidelity"], res.get("timing")))
    lines.append("gates %s\nbranch: %s" % (res["gates"], res["branch"]))
    open(os.path.join(out, "report.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["start", "live", "report"])
    ap.add_argument("--deadline", type=int, default=0)
    ap.add_argument("--finalize", default="")
    a = ap.parse_args(argv)
    if a.stage == "start":
        print(json.dumps({k: v for k, v in stage_start(a.deadline).items() if k not in ("files", "baselines")}, indent=1))
    elif a.stage == "live":
        print(json.dumps(stage_live(), indent=1))
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
