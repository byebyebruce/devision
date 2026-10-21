"""exp60: can the frozen character head read the real pictures' numbers from the right field box? (critic EXP60-PREP-01;
docs/experiments/2026-10-18-exp60-oracle-box-plan.md). Everything frozen (EXP57 end 962503fb / 09a5b447, EXP58 bundle
70bd5aa8: vision input, full picture + 12 views, the intersected-patch / relative-geometry region rule, character head,
strict decoding, separators, ordinary decide, original questions / options, raw temperature 1); the same 24 transcribed
pictures as EXP59. Two conditions, 24 comparison requests and 72 field reads each:

  auto    EXP59's P request through Reading.decide (the product path, diagnostics kept); it must reproduce EXP59's P
          exactly (72 strings, 24 tables / states, predictions, probabilities within 1e-5), and this script's mirrored
          read (read_with, boxes from the selector) must give the same boxes and strings -- otherwise nothing else runs.
  oracle  read_with with the fixed oracle field boxes (scripts/exp60/oracle_boxes.json) in place of the selector's boxes,
          everything after the box unchanged; the table / state built exactly as Reading.decide does, ordinary
          decide(image=None). Gold numbers are used for scoring only, never as an input.

A change under oracle is the joint effect of the region chosen and its relative geometry, not of a coordinate or pixel
alone. Stages (fresh only, runs/exp60): start (model-free identities) -> run (MPS) -> report (model-free).

    uv run python scripts/exp60/exp60.py start --deadline EPOCH | run | report [--finalize CODES]
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(__file__)
OUT, E59 = "runs/exp60", "runs/exp59"
E59_SHA = {"START.json": "afb92f7eadcb90ea53cd1beacc85eaba300c54deeb2069c01e646fbf7d8b672d",
           "report.json": "844e41543b1e453f65e016d1257810854c850b161d81b8b18cb99c5672a4fd61",
           "registry.json": "9db91eb64cc974b77ce48bade51921b9632e2cd1b929d3798c0e56170c59c3fa"}
BOXES = os.path.join(HERE, "oracle_boxes.json")
CONDS = ("auto", "oracle")
FILES = ["scripts/exp60/exp60.py", "scripts/exp60/run.sh", "scripts/exp60/boxes.py", "scripts/exp60/oracle_boxes.json",
         "scripts/exp59/exp59.py", "scripts/exp58/live.py", "src/devision/reading.py", "src/devision/infer.py", "src/devision/model.py",
         "scripts/exp53/watch.py"]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, rel))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


e59 = _load("exp59", "../exp59/exp59.py")
e58, e57, e54, sha, write_json, register, jsonl = e59.e58, e59.e57, e59.e54, e59.sha, e59.write_json, e59.register, e59.jsonl


def field_of(template: str) -> str:
    return "mass" if "mass" in template else "speed" if "speed" in template else "distance"


# ------------------------------------------------------------------ the mirrored read (selector boxes or given boxes)

def read_with(reading, image, templates: List[str], boxes: Optional[Dict[str, List[float]]] = None) -> Dict:
    """Reading.read's steps unchanged (src/devision/reading.py Reading.read), except that with `boxes` the selector is not
    run and each template's box is the given one (degenerate false). Each read also carries its region summary (cells,
    views)."""
    import torch

    from devision.image import patch_geometry
    from devision.infer import _image_arg, image_tensor
    from devision.reading import GRID, TEMPLATES, UNREADABLE, decode, fmt, pixel_box, region
    with torch.no_grad():
        img = _image_arg(image)
        w, h = img.size
        m = reading.decider.model
        dev = next(m.parameters()).device
        kv = m.visual_inputs(image_tensor(img, reading.decider.cfg)[None].to(dev))[0][0]
        reads, regs = [], []
        if boxes is None:
            patches = kv[:256].half()[None].expand(len(templates), -1, -1)
            geom = patch_geometry((0, 0, w, h), w, h, GRID["size"], GRID["side"]).to(dev)[None].expand(len(templates), -1, -1)
            words, mask = reading._words(templates)
            frac = reading.selector(patches, geom, words, mask).float().cpu().tolist()
            given = [pixel_box(f, w, h) for f in frac]
        else:
            given = [(list(boxes[t]), False) for t in templates]
        for t, (box, bad) in zip(templates, given):
            reg = region(box, w, h)
            regs.append(reg)
            reads.append({"question": t, "box": box, "degenerate": bad, "cells": len(reg["index"]), "views": len(set(reg["view"]))})
        n = max(len(r["index"]) for r in regs)
        x = torch.zeros(len(regs), n, kv.shape[1], device=dev)
        msk = torch.zeros(len(regs), n, dtype=torch.bool, device=dev)
        rel = torch.zeros(len(regs), n, 2, dtype=torch.long, device=dev)
        gm = torch.zeros(len(regs), n, 4, device=dev)
        for i, r in enumerate(regs):
            k = len(r["index"])
            x[i, :k] = kv[torch.tensor(r["index"], device=dev)].half().float()
            msk[i, :k] = True
            rel[i, :k] = torch.tensor(r["rel"], device=dev)
            gm[i, :k] = torch.tensor(r["geom"], dtype=torch.float32, device=dev)
        ids = reading.head(x, msk, rel, gm).argmax(-1).cpu().tolist()
    table = []
    for rd, t, c in zip(reads, templates, ids):
        d = decode(c)
        rd.update(decoded=d["text"], valid=d["valid"], value=fmt(d["text"]) if d["valid"] else UNREADABLE)
        table.append([*TEMPLATES[t], rd["value"]])
    return {"table": table, "reads": reads}


# ------------------------------------------------------------------ start

def oracle_boxes(its: List[Dict]) -> Dict[str, Dict[str, List[float]]]:
    """The fixed boxes, checked: exactly the 24 pictures, each with exactly the templates its question reads, every box
    inside the picture with a positive size."""
    from devision.reading import reads_for
    b = {k: v for k, v in json.load(open(BOXES)).items() if not k.startswith("_")}
    if set(b) != {it["row"]["id"] for it in its}:
        raise SystemExit("oracle_boxes.json does not cover exactly the 24 pictures")
    out = {}
    for it in its:
        v = b[it["row"]["id"]]
        want = reads_for(it["row"]["questions"])
        if v["image"] != it["row"]["image"] or set(v["boxes"]) != set(want):
            raise SystemExit("%s: oracle boxes for %s, the question reads %s" % (it["row"]["id"], sorted(v["boxes"]), want))
        w, h = v["size"]
        for t, (x, y, bw, bh) in v["boxes"].items():
            if not (bw > 0 and bh > 0 and x >= 0 and y >= 0 and x + bw <= w and y + bh <= h):
                raise SystemExit("%s %s: box outside the picture" % (it["row"]["id"], t))
        out[it["row"]["id"]] = v["boxes"]
    return out


def bound_e59() -> Dict[str, str]:
    bad = [f for f, h in E59_SHA.items() if sha(os.path.join(E59, f)) != h]
    if bad:
        raise SystemExit("EXP59's %s changed" % bad)
    for files in json.load(open(os.path.join(E59, "registry.json"))).values():
        for f, h in files.items():
            if sha(os.path.join(E59, f)) != h:
                raise SystemExit("EXP59's registered %s changed" % f)
    return dict(E59_SHA)


def stage_start(deadline: int, out: str = OUT, data_root: str = "data") -> Dict:
    path = os.path.join(out, "START.json")
    if os.path.exists(path):
        raise SystemExit("%s exists (fresh only)" % path)
    e59_files = bound_e59()
    st59 = json.load(open(os.path.join(E59, "START.json")))
    st58 = e58.check_run(e59.E58, data_root=data_root, code=False)
    if st58["model"] != st59["model"] or st58["bundle"] != st59["bundle"]:
        raise SystemExit("EXP58's model / bundle are not EXP59's")
    its = e59.items()
    if [it["row"]["id"] for it in its] != [x["id"] for x in st59["items"]]:
        raise SystemExit("the 24 items are not EXP59's")
    oracle_boxes(its)
    e59.gold_numbers(its)
    os.makedirs(out, exist_ok=True)
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    s = {"git_head": git("rev-parse", "HEAD"), "dirty_files": [f for f in FILES if git("status", "--porcelain", "--", f)],
         "files": {f: sha(f) for f in FILES}, "model": st58["model"], "bundle": st58["bundle"], "exp59": e59_files,
         "oracle_boxes_sha256": sha(BOXES), "sources": st59["sources"], "items": st59["items"],
         "requests": {"auto": 24, "oracle": 24, "field_reads": {"auto_product": 72, "auto_mirror_check": 72, "oracle": 72}},
         "temperature": "none (raw), in memory", "candidate_eligible": False,
         "purpose": "oracle field boxes vs the selector's, all else frozen (critic EXP60-PREP-01): a diagnosis, not a candidate",
         "started": time.strftime("%F %T"), "deadline": deadline, "deadline_text": time.strftime("%F %T", time.localtime(deadline))}
    write_json(path, s)
    return s


def check_start(out: str, data_root: str = "data", code: bool = True) -> Dict:
    st = json.load(open(os.path.join(out, "START.json")))
    if code:
        bad = [f for f, h in st["files"].items() if sha(f) != h]
        if bad:
            raise SystemExit("code differs from START: %s" % bad)
    if bound_e59() != st["exp59"] or sha(BOXES) != st["oracle_boxes_sha256"]:
        raise SystemExit("EXP59's files / the oracle boxes differ from START")
    st58 = e58.check_run(e59.E58, data_root=data_root, code=False)
    if st58["model"] != st["model"] or st58["bundle"] != st["bundle"]:
        raise SystemExit("model / bundle differ from START")
    return st


# ------------------------------------------------------------------ run

def exp59_p() -> Dict[str, Dict]:
    st59 = json.load(open(os.path.join(E59, "START.json")))
    return e59.cond_records(E59, "P", st59, e59.items())


def same_reads(a: List[Dict], b: List[Dict]) -> bool:
    key = lambda x: (x["question"], [round(v, 6) for v in x["box"]], x["decoded"], x["valid"], x["value"])  # noqa: E731
    return [key(x) for x in a] == [key(x) for x in b]


def reproduces(auto: List[Dict], old: Dict[str, Dict]) -> Dict:
    """auto against EXP59's P: per question the same reads (template, box, string, validity, value), table, state,
    prediction, probabilities within 1e-5; and the mirror's reads equal the product's."""
    bad = []
    diff = 0.0
    for r in auto:
        o = old.get(r["sample_id"])
        if o is None:
            bad.append((r["sample_id"], "missing in EXP59"))
            continue
        d, od = r["diagnostics"], o["diagnostics"]
        if not same_reads(d["reads"], od["reads"]) or d["table"] != od["table"] or d["state"] != od["state"]:
            bad.append((r["sample_id"], "reads / table / state"))
        if not same_reads(r["mirror"]["reads"], d["reads"]) or r["mirror"]["table"] != d["table"]:
            bad.append((r["sample_id"], "mirror differs from the product read"))
        if r["prediction"] != o["prediction"]:
            bad.append((r["sample_id"], "prediction"))
        diff = max([diff] + [abs(r["probabilities"][k] - o["probabilities"][k]) for k in r["probabilities"]])
    ok = not bad and len(auto) == 24 and diff <= 1e-5
    return {"questions": len(auto), "problems": bad, "max_probability_difference": diff, "reproduced": bool(ok)}


def picture(data_root: str, path: str, pics: Dict[str, str]):
    """The picture's bytes checked against the bound manifest (as the P path's ask), opened as RGB."""
    import hashlib
    import io

    from PIL import Image
    raw = open(os.path.join(data_root, path), "rb").read()
    if path not in pics or hashlib.sha256(raw).hexdigest() != pics[path]:
        raise SystemExit("%s differs from the bound picture" % path)
    with Image.open(io.BytesIO(raw)) as img:
        return img.convert("RGB")


def write_cond(out: str, c: str, recs: List[Dict], st: Dict) -> None:
    ident = {"condition": c, "start_sha256": sha(os.path.join(out, "START.json")), "model": st["model"],
             "bundle": st["bundle"]["reading.json"], "temperature": "none (raw)",
             "boxes": "selector (product path)" if c == "auto" else "oracle_boxes.json " + st["oracle_boxes_sha256"]}
    f = "details_%s.jsonl" % c
    with open(os.path.join(out, f), "w") as fh:
        fh.write(json.dumps({"identity": ident}) + "\n" + "".join(json.dumps(r) + "\n" for r in recs))
    register(out, "decide_" + c, {f: sha(os.path.join(out, f))})


def stage_run(out: str = OUT, data_root: str = "data") -> Dict:
    from devision.reading import reads_for, serialise
    from devision.train.evaluate import _probabilities
    st = check_start(out, data_root)
    reading, st58, pics = e58.load_reading(e59.E58, data_root=data_root)       # checks before the load; raw temperature
    its = e59.items()
    boxes = oracle_boxes(its)
    t0 = time.time()
    auto = e59.p_condition(reading, its, data_root, pics, on_request=lambda n: e54.progress(out, "run", n, 48))
    for r, it in zip(auto, its):                                                  # the mirrored read, same picture bytes
        r["mirror"] = read_with(reading, picture(data_root, it["row"]["image"], pics), reads_for(it["row"]["questions"]))
    e59.check_diagnostics(auto, its)
    write_cond(out, "auto", auto, st)
    rep = reproduces(auto, exp59_p())
    write_json(os.path.join(out, "reproduction.json"), rep)
    register(out, "reproduction", {"reproduction.json": sha(os.path.join(out, "reproduction.json"))})
    if not rep["reproduced"]:
        raise SystemExit("auto does not reproduce EXP59: %s -- the oracle condition is not run" % rep)
    recs = []
    for n, it in enumerate(its):
        r = it["row"]
        rd = read_with(reading, picture(data_root, r["image"], pics), reads_for(r["questions"]), boxes[r["id"]])
        state = serialise(r.get("state_text") or "", rd["table"])
        res = reading.decider.decide(state, {it["key"]: r["questions"][it["qid"]]}, image=None)
        recs.append(e59.record(it, _probabilities(res["answers"][it["key"]]), {"diagnostics": {"reads": rd["reads"], "table": rd["table"], "state": state}}))
        e54.progress(out, "run", 24 + n + 1, 48)
    e59.check_diagnostics(recs, its)
    write_cond(out, "oracle", recs, st)
    return {"seconds": time.time() - t0}


# ------------------------------------------------------------------ report

def cond_records(out: str, c: str, st: Dict, its: List[Dict]) -> Dict[str, Dict]:
    f = "details_%s.jsonl" % c
    if e54.check_registered(out, "decide_" + c).get(f) != sha(os.path.join(out, f)):
        raise SystemExit("%s not registered" % f)
    lines = jsonl(os.path.join(out, f))
    ident = lines[0]["identity"]
    want = {"condition": c, "start_sha256": sha(os.path.join(out, "START.json")), "model": st["model"], "bundle": st["bundle"]["reading.json"],
            "temperature": "none (raw)"}
    if any(ident.get(k) != v for k, v in want.items()):
        raise SystemExit("%s: identity does not bind" % f)
    got = e57.r41.check_details("exp60/%s" % c, [it["row"] for it in its], lines[1:])
    e59.check_diagnostics(lines[1:], its)
    if c == "oracle":
        boxes = oracle_boxes(its)
        if any([x["box"] for x in r["diagnostics"]["reads"]] != [boxes[r["sample_id"]][x["question"]] for x in r["diagnostics"]["reads"]]
               for r in lines[1:]):
            raise SystemExit("oracle details: a box is not the fixed oracle box")
    return {k[0]: v for k, v in got.items()}


def field_hits(recs: Dict[str, Dict], gold: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, bool]]:
    return {rid: {x["question"]: bool(x["valid"] and x["value"] == e59.value_of(gold[rid][x["question"]])) for x in r["diagnostics"]["reads"]}
            for rid, r in recs.items()}


def speed_difference(a: Dict[str, Dict[str, bool]], b: Dict[str, Dict[str, bool]]) -> Dict:
    """oracle - auto speed exact share, the 12 temperature pictures resampled (both speeds of a picture together)."""
    pics = sorted(rid for rid in a if any(field_of(t) == "speed" for t in a[rid]))

    def fn(ks):
        cells = [(b[k][t], a[k][t]) for k in ks for t in a[k] if field_of(t) == "speed"]
        return sum(x - y for x, y in cells) / len(cells)
    return e57.r41.interval(fn, pics, lambda ks: list(ks))


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
    its = e59.items()
    recs = {}
    for c in CONDS:
        try:
            recs[c] = cond_records(out, c, st, its)
            res["states"][c] = "completed"
        except (SystemExit, OSError, ValueError, KeyError) as e:
            res["states"][c] = "missing / invalid: %s" % e
    try:
        if "auto" not in recs:
            raise SystemExit("auto missing")
        f = os.path.join(out, "reproduction.json")
        if e54.check_registered(out, "reproduction").get("reproduction.json") != sha(f):
            raise SystemExit("reproduction.json not registered")
        lines = {r["sample_id"]: r for r in jsonl(os.path.join(out, "details_auto.jsonl"))[1:]}
        res["reproduction"] = reproduces([lines[it["row"]["id"]] for it in its], exp59_p())
    except (SystemExit, OSError, ValueError, KeyError) as e:
        res["reproduction"] = {"reproduced": False, "reason": str(e)}
    if not res["reproduction"].get("reproduced"):
        res["branch"] = "invalid / incomplete: auto does not reproduce EXP59 (or is missing) -- no box verdict"
        return write_report(out, res)
    gold = e59.gold_numbers(its)
    hits = {c: field_hits(recs[c], gold) for c in recs}
    res["fields"] = {c: {f: {"exact": sum(v for rid in h for t, v in h[rid].items() if field_of(t) == f),
                             "of": sum(1 for rid in h for t in h[rid] if field_of(t) == f)} for f in ("mass", "speed", "distance")}
                     for c, h in hits.items()}
    kinds = {it["row"]["id"]: it["kind"] for it in its}
    res["pictures_all_right"] = {c: {k: sum(all(h[rid].values()) for rid in h if kinds[rid] == k) for k in ("temp", "force")} for c, h in hits.items()}
    res["accuracy"] = {c: {k: {"right": sum(recs[c][rid]["correct"] for rid in recs[c] if kinds[rid] == k), "of": 12} for k in ("temp", "force")}
                       for c in recs}
    if "oracle" in recs:
        res["speed_oracle_minus_auto"] = speed_difference(hits["auto"], hits["oracle"])
        res["field_transitions"] = {f: dict(Counter("%s->%s" % ("R" if hits["auto"][rid][t] else "W", "R" if hits["oracle"][rid][t] else "W")
                                                    for rid in hits["auto"] for t in hits["auto"][rid] if field_of(t) == f))
                                    for f in ("mass", "speed", "distance")}
        res["paired"] = {k: e59.transitions(recs["auto"], recs["oracle"], [rid for rid in kinds if kinds[rid] == k]) for k in ("temp", "force")}
        res["boxes"] = {c: {"speed_box_bottom": [min(x["box"][1] + x["box"][3] for r in recs[c].values() for x in r["diagnostics"]["reads"]
                                                     if field_of(x["question"]) == "speed"),
                                                 max(x["box"][1] + x["box"][3] for r in recs[c].values() for x in r["diagnostics"]["reads"]
                                                     if field_of(x["question"]) == "speed")]} for c in recs}
    res["branch"] = branch(res)
    return write_report(out, res)


def branch(res: Dict) -> str:
    """The pre-written direction (EXP60-PREP-01 item 4); none of them is a candidate."""
    if "speed_oracle_minus_auto" not in res:
        return "incomplete: the oracle condition is missing / invalid -- no box verdict"
    sp = res["fields"]["oracle"]["speed"]["exact"]
    lo = res["speed_oracle_minus_auto"]["ci95"][0]
    if sp >= 18 and lo > 0:
        return "speed read from the right box (oracle %d/24, improvement lower bound > 0): selector supervision first (layout coverage from training sources)" % sp
    if sp <= 12:
        return "speed still not read with the right box (oracle %d/24): cross-layout supervision of the region reading first, not selector-only training" % sp
    return "partial rescue (oracle %d/24, or improvement lower bound <= 0): recorded only, no long training follows from it" % sp


def write_report(out: str, res: Dict) -> Dict:
    os.makedirs(out, exist_ok=True)
    write_json(os.path.join(out, "report.json"), res)
    keys = ("finalized", "states", "reproduction", "fields", "pictures_all_right", "speed_oracle_minus_auto", "accuracy", "paired", "branch", "candidate_eligible")
    text = json.dumps({k: res[k] for k in keys if k in res}, indent=1, default=str)
    open(os.path.join(out, "report.txt"), "w").write(text + "\n")
    print(text)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["start", "run", "report"])
    ap.add_argument("--deadline", type=int, default=0)
    ap.add_argument("--finalize", default="")
    a = ap.parse_args(argv)
    if a.stage == "start":
        print(json.dumps({k: v for k, v in stage_start(a.deadline).items() if k not in ("files", "items")}, indent=1))
    elif a.stage == "run":
        print(json.dumps(stage_run(), indent=1))
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
