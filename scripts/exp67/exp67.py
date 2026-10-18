"""exp67: separate the selected-token package from the continuous box-relative geometry (critic EXP67-PREP-01;
docs/experiments/2026-10-19-exp67-interface-plan.md). A post-hoc mechanism diagnostic with oracle boxes -- not a product
path, not a confirmation. Nothing is trained; values, pictures and components are fixed: EXP57's vision (962503fb),
the n12 / 0.25 local views, EXP64's head 0e275bdf, the 24 transcribed dev pictures / 72 fields, strict decoding.

Boxes: A = the c92 auto boxes EXP64 registered (runs/exp64/dev24_auto_new.jsonl), O = EXP60's frozen oracle boxes
(scripts/exp60/oracle_boxes.json). Factor S = the selected tokens' index / order / mask with that token set's own discrete
row / col (region(box_S)); factor G = the box the continuous geom is taken against:
    geom = [(cx - bx) / bw, (cy - by) / bh, pw / bw, ph / bh]
from the same view's original absolute patch geometry of each S token (no new selection, no rel change, no clamp).
Conditions AA, AO, OA, OO (first letter S, second G); AA / OO are region()'s own output, the mixed ones replace geom only.

  start   model-free: code, EXP64 (run, head, registered dev24 reads), EXP60 (oracle boxes), EXP58 bundle + EXP57 end, the
          24 pictures, both box sets.
  run     MPS: each picture encoded once, the four conditions; per picture AA must reproduce EXP64's dev24_auto_new and
          OO its dev24_oracle_new (decoded / valid / value) -- else stop with what is on disk; per condition the
          per-field reads, token index / rel / geom hashes and geom ranges, the box, raw logits (EOS included).
  report  model-free: per field and condition correct counts and transitions; AO - AA, OO - OA (geometry), OA - AA,
          OO - AO (S package), interaction, each over the field's 12 pictures resampled; the pre-written branch.

    uv run python scripts/exp67/exp67.py start --deadline EPOCH | run | report [--finalize CODES]
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
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(__file__)
OUT, E64 = "runs/exp67", "runs/exp64"
CONDS = ("AA", "AO", "OA", "OO")
FIELDS = ("mass", "speed", "distance")
REF = {"AA": "dev24_auto_new", "OO": "dev24_oracle_new"}
FILES = ["scripts/exp67/exp67.py", "scripts/exp67/run.sh", "scripts/exp64/exp64.py", "scripts/exp61/exp61.py", "scripts/exp60/exp60.py",
         "scripts/exp59/exp59.py", "scripts/exp58/live.py", "scripts/exp49/probe.py", "src/devision/reading.py", "src/devision/infer.py",
         "src/devision/model.py", "src/devision/image.py", "scripts/exp53/watch.py"]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, rel))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


e64 = _load("exp64", "../exp64/exp64.py")
e61, e60, e59, e58, e54, p49 = e64.e61, e64.e60, e64.e59, e64.e58, e64.e54, e64.p49
sha, write_json, register, jsonl = e61.sha, e61.write_json, e61.register, e61.jsonl
E64_SHA = {"START.json": "5ac4da33924376e19e0e450d157cccdf204cc60680f9969df7fae300e8b6d8ad",
           "registry.json": "d743f3e290db25d45091cd27fde3335974d3123e7cdf0411ad7ae31674f5dff3",
           "report.json": "ae61798133c40deef6c728790e4dc9f451dcf1fc3224aa86f807e6163bda3bdf",
           "head.pt": "0e275bdf3b47bff9b8786160b090b79ae903734d784c828d5de65e61c6a2d7c3"}
HEAD_SHA = E64_SHA["head.pt"]
C92 = e64.E62_SHA["selector.pt"]


# ------------------------------------------------------------------ the interface

def geom_against(reg: Dict, box, w: int, h: int, n: int = 12, overlap: float = 0.25, size: int = 256, side: int = 16) -> List[List[float]]:
    """The continuous geom of reg's tokens against `box`: each token's own view's absolute patch geometry, the same
    arithmetic as devision.reading.region (so against reg's own box it is bit-identical), no clamp."""
    from devision.image import patch_geometry, view_rects
    bx, by, bw, bh = box
    views = [(0, 0, w, h)] + view_rects(w, h, n, overlap)
    cache: Dict[int, List] = {}
    out = []
    for idx, k in zip(reg["index"], reg["view"]):
        if k not in cache:
            cache[k] = patch_geometry(views[k], w, h, size, side).tolist()
        cx, cy, pw, ph = cache[k][idx - k * side * side]
        out.append([(cx * w - bx) / bw, (cy * h - by) / bh, pw * w / bw, ph * h / bh])
    return out


def condition(regs: Dict[str, Dict], boxes: Dict[str, List[float]], c: str, w: int, h: int) -> Dict:
    """Condition c = S + G: S's index / view / rel, G's box for the continuous geom (AA / OO: region()'s own output)."""
    s, g = c[0], c[1]
    reg = regs[s]
    if s == g:
        return {"index": list(reg["index"]), "view": list(reg["view"]), "rel": [list(r) for r in reg["rel"]], "geom": [list(x) for x in reg["geom"]]}
    return {"index": list(reg["index"]), "view": list(reg["view"]), "rel": [list(r) for r in reg["rel"]], "geom": geom_against(reg, boxes[g], w, h)}


def head_logits(head, kv, conds: List[Dict]):
    """exp64.read_heads' tensor build on given regions (token values = kv[index] in half, then float; rel; geom in
    float32; padding masked) -> logits [n, SLOTS, CLASSES]."""
    import torch
    dev = kv.device
    n = max(len(r["index"]) for r in conds)
    x = torch.zeros(len(conds), n, kv.shape[1], device=dev)
    msk = torch.zeros(len(conds), n, dtype=torch.bool, device=dev)
    rel = torch.zeros(len(conds), n, 2, dtype=torch.long, device=dev)
    gm = torch.zeros(len(conds), n, 4, device=dev)
    for i, r in enumerate(conds):
        k = len(r["index"])
        x[i, :k] = kv[torch.tensor(r["index"], device=dev)].half().float()
        msk[i, :k] = True
        rel[i, :k] = torch.tensor(r["rel"], device=dev)
        gm[i, :k] = torch.tensor(r["geom"], dtype=torch.float32, device=dev)
    with torch.no_grad():
        return head(x, msk, rel, gm)


def reads_of(logits, templates: List[str], boxes: List) -> List[Dict]:
    from devision.reading import UNREADABLE, decode, fmt
    out = []
    for t, b, ids in zip(templates, boxes, logits.argmax(-1).cpu().tolist()):
        d = decode(ids)
        out.append({"question": t, "box": b, "decoded": d["text"], "valid": d["valid"], "value": fmt(d["text"]) if d["valid"] else UNREADABLE})
    return out


def digest(xs) -> str:
    return hashlib.sha256(json.dumps(xs).encode()).hexdigest()


def summary(c: Dict) -> Dict:
    """What a condition fed the head: hashes of index / rel / geom, the token count and the geom ranges (tokens outside
    the G box keep their coordinates; counted, not clipped)."""
    g = c["geom"]
    centre_out = sum(1 for x in g if not (0.0 <= x[0] <= 1.0 and 0.0 <= x[1] <= 1.0))
    return {"tokens": len(c["index"]), "index_sha256": digest(c["index"]), "rel_sha256": digest(c["rel"]), "geom_sha256": digest(g),
            "geom_min": [min(x[j] for x in g) for j in range(4)], "geom_max": [max(x[j] for x in g) for j in range(4)],
            "centres_outside_box": centre_out, "finite": all(math.isfinite(v) for x in g for v in x)}


# ------------------------------------------------------------------ identities

def bound_e64() -> Dict[str, str]:
    bad = [f for f, h in E64_SHA.items() if sha(os.path.join(E64, f)) != h]
    if bad:
        raise SystemExit("EXP64's %s changed" % bad)
    reg = json.load(open(os.path.join(E64, "registry.json")))
    for files in reg.values():
        for f, h in files.items():
            if sha(os.path.join(E64, f)) != h:
                raise SystemExit("EXP64's registered %s changed" % f)
    return dict(E64_SHA)


def bound_gold() -> Dict[str, str]:
    """The gold sources the report scores with (exp59's TRANSCRIBED / FIELDS): their actual sha256 must equal the sources
    EXP60's START froze (critic EXP67-R1-02); EXP60's START itself is bound by e61.bound_e60."""
    e61.bound_e60()
    frozen = json.load(open(os.path.join(e60.OUT, "START.json")))["sources"]
    now = {"transcribed": sha(e59.TRANSCRIBED), "fields": sha(e59.FIELDS)}
    bad = [k for k, v in now.items() if frozen.get(k) != v]
    if bad:
        raise SystemExit("gold source %s differs from EXP60's START" % bad)
    return now


def reference(c: str) -> Dict[str, Dict[str, Dict]]:
    """EXP64's registered dev24 reads with this head (AA: c92 auto boxes; OO: EXP60's oracle boxes)."""
    name = REF[c]
    f = os.path.join(E64, name + ".jsonl")
    if e54.check_registered(E64, "reads_" + name).get(name + ".jsonl") != sha(f):
        raise SystemExit("EXP64's %s is not registered" % name)
    lines = jsonl(f)
    if lines[0]["identity"].get("head_sha256") != HEAD_SHA:
        raise SystemExit("EXP64's %s is not this head" % name)
    if c == "AA" and lines[0]["identity"].get("selector_sha256") != C92:
        raise SystemExit("EXP64's dev24_auto_new is not c92")
    return {x["id"]: {r["question"]: r for r in x["reads"]} for x in lines[1:]}


def boxes() -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    """{"A": c92 auto boxes (EXP64 registered), "O": EXP60 oracle boxes}, per picture and template."""
    a = {rid: {t: list(r["box"]) for t, r in v.items()} for rid, v in reference("AA").items()}
    o = {k: v["boxes"] for k, v in json.load(open(e60.BOXES)).items() if not k.startswith("_")}
    for rid, v in a.items():
        if set(v) != set(o.get(rid, {})):
            raise SystemExit("%s: the two box sets name other templates" % rid)
    oo = reference("OO")
    for rid, v in oo.items():
        for t, r in v.items():
            if [float(x) for x in r["box"]] != [float(x) for x in o[rid][t]]:
                raise SystemExit("%s %s: EXP64's oracle read used another box" % (rid, t))
    return {"A": a, "O": o}


def stage_start(deadline: int, out: str = OUT, data_root: str = "data") -> Dict:
    path = os.path.join(out, "START.json")
    if os.path.exists(path):
        raise SystemExit("%s exists (fresh only)" % path)
    st58 = e58.check_run(e59.E58, data_root=data_root, code=False)
    bx = boxes()
    its = e59.items()
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    s = {"git_head": git("rev-parse", "HEAD"), "dirty_files": [f for f in FILES if git("status", "--porcelain", "--", f)],
         "files": {f: sha(f) for f in FILES}, "exp64": bound_e64(), "exp60": e61.bound_e60(), "gold": bound_gold(), "model": st58["model"], "bundle": st58["bundle"],
         "pictures": {it["row"]["image"]: sha(os.path.join(data_root, it["row"]["image"])) for it in its},
         "boxes": {"A": digest(bx["A"]), "O": digest(bx["O"]), "A_source": "runs/exp64/dev24_auto_new.jsonl " + sha(os.path.join(E64, "dev24_auto_new.jsonl")),
                   "O_source": "%s %s" % (e60.BOXES, sha(e60.BOXES))},
         "head_sha256": HEAD_SHA, "conditions": list(CONDS), "requests": {"pictures": len(its), "fields": sum(len(v) for v in bx["A"].values())},
         "candidate_eligible": False, "started": time.strftime("%F %T"), "deadline": deadline,
         "deadline_text": time.strftime("%F %T", time.localtime(deadline))}
    os.makedirs(out, exist_ok=True)
    write_json(path, s)
    return s


def check_start(out: str, data_root: str = "data", code: bool = True) -> Dict:
    st = json.load(open(os.path.join(out, "START.json")))
    if code:
        bad = [f for f, h in st["files"].items() if sha(f) != h]
        if bad:
            raise SystemExit("code differs from START: %s" % bad)
    if bound_e64() != st["exp64"] or e61.bound_e60() != st["exp60"]:
        raise SystemExit("EXP64 / EXP60 differ from START")
    if bound_gold() != st["gold"]:
        raise SystemExit("the gold sources differ from START")
    bx = boxes()
    if digest(bx["A"]) != st["boxes"]["A"] or digest(bx["O"]) != st["boxes"]["O"]:
        raise SystemExit("the box sets differ from START")
    if data_root and any(sha(os.path.join(data_root, p)) != h for p, h in st["pictures"].items()):
        raise SystemExit("a picture differs from START")
    return st


# ------------------------------------------------------------------ run

def compare(got: List[Dict], ref: Dict[str, Dict]) -> List[Dict]:
    bad = []
    for r in got:
        e = ref.get(r["question"])
        if e is None or (e["decoded"], e["valid"], e["value"]) != (r["decoded"], r["valid"], r["value"]):
            bad.append({"question": r["question"], "expected": None if e is None else [e["decoded"], e["valid"], e["value"]],
                        "actual": [r["decoded"], r["valid"], r["value"]]})
    if len(got) != len(ref):
        bad.append({"question": None, "expected": len(ref), "actual": len(got)})
    return bad


def write_conditions(out: str, results: Dict[str, List[Dict]], st_sha: str, status: str) -> None:
    for c in CONDS:
        f = "cond_%s.jsonl" % c
        ident = {"start_sha256": st_sha, "condition": c, "selection_box": c[0], "geom_box": c[1], "head_sha256": HEAD_SHA, "status": status}
        with open(os.path.join(out, f), "w") as fh:
            fh.write(json.dumps({"identity": ident}) + "\n" + "".join(json.dumps(x) + "\n" for x in results[c]))
        register(out, "cond_" + c, {f: sha(os.path.join(out, f))})


def stage_run(out: str = OUT, data_root: str = "data") -> Dict:
    import torch

    from devision.reading import CLASSES, SLOTS, reads_for, region
    st = check_start(out, data_root)
    if sha(os.path.join(E64, "head.pt")) != HEAD_SHA:
        raise SystemExit("EXP64's head changed")
    refs = {c: reference(c) for c in REF}
    bx = boxes()
    reading, st58, pics58 = e58.load_reading(e59.E58, data_root=data_root)      # EXP57 + EXP58 bundle, verified
    dev = next(reading.decider.model.parameters()).device
    reading.head.load_state_dict(torch.load(os.path.join(E64, "head.pt"), map_location="cpu"), strict=True)
    head = reading.head.to(dev).eval()
    m, cfg = reading.decider.model, reading.decider.cfg
    st_sha = sha(os.path.join(out, "START.json"))
    results: Dict[str, List[Dict]] = {c: [] for c in CONDS}
    repro: Dict[str, Any] = {c: {"pictures": 0, "mismatches": []} for c in REF}
    its = e59.items()
    for n, it in enumerate(its):
        row = it["row"]
        img = e60.picture(data_root, row["image"], pics58)
        w, h = img.size
        kv = e64.encode(m, cfg, img)
        ts = reads_for(row["questions"])
        per: Dict[str, List[Any]] = {c: [] for c in CONDS}
        for t in ts:
            b = {"A": bx["A"][row["id"]][t], "O": bx["O"][row["id"]][t]}
            regs = {s: region(b[s], w, h) for s in "AO"}
            for c in CONDS:
                per[c].append((condition(regs, b, c, w, h), b[c[0]], b[c[1]]))
        for c in CONDS:
            conds = [x[0] for x in per[c]]
            lg = head_logits(head, kv, conds)
            if tuple(lg.shape) != (len(ts), SLOTS, CLASSES) or not bool(torch.isfinite(lg).all()):
                results[c].append({"id": row["id"], "reads": [], "error": "logits shape %s, finite %s" % (list(lg.shape), bool(torch.isfinite(lg).all()))})
                write_conditions(out, results, st_sha, "stopped: non-finite or misshaped logits in %s %s" % (c, row["id"]))
                raise SystemExit("%s %s: logits shape %s / non-finite -- stopped (not a wrong answer)" % (c, row["id"], list(lg.shape)))
            rd = reads_of(lg, ts, [x[1] for x in per[c]])
            for r, x, lgi in zip(rd, per[c], lg.float().cpu().tolist()):
                r.update({"geom_box": x[2], "inputs": summary(x[0]), "logits": lgi})
            results[c].append({"id": row["id"], "reads": rd})
        for c in REF:
            bad = compare(results[c][-1]["reads"], refs[c][row["id"]])
            repro[c]["pictures"] += 1
            repro[c]["mismatches"] += [dict(b_, picture=row["id"]) for b_ in bad]
        if any(repro[c]["mismatches"] for c in REF):
            write_conditions(out, results, st_sha, "stopped: a same-source condition does not reproduce EXP64")
            write_json(os.path.join(out, "repro.json"), repro)
            register(out, "repro", {"repro.json": sha(os.path.join(out, "repro.json"))})
            raise SystemExit("%s: AA / OO do not reproduce EXP64 -- stopped" % row["id"])
        e54.progress(out, "run", n + 1, len(its))
    write_conditions(out, results, st_sha, "completed")
    for c in REF:
        repro[c]["reproduced"] = not repro[c]["mismatches"] and repro[c]["pictures"] == len(its)
    write_json(os.path.join(out, "repro.json"), repro)
    register(out, "repro", {"repro.json": sha(os.path.join(out, "repro.json"))})
    return {"pictures": len(its)}


# ------------------------------------------------------------------ report

def load_condition(out: str, c: str) -> Dict[str, Dict[str, Dict]]:
    f = os.path.join(out, "cond_%s.jsonl" % c)
    if e54.check_registered(out, "cond_" + c).get("cond_%s.jsonl" % c) != sha(f):
        raise SystemExit("cond_%s not registered" % c)
    lines = jsonl(f)
    want = {"start_sha256": sha(os.path.join(out, "START.json")), "condition": c, "selection_box": c[0], "geom_box": c[1],
            "head_sha256": HEAD_SHA, "status": "completed"}
    if any(lines[0]["identity"].get(k) != v for k, v in want.items()):
        raise SystemExit("cond_%s: identity does not bind / not completed" % c)
    from devision.reading import reads_for
    want_t = {it["row"]["id"]: reads_for(it["row"]["questions"]) for it in e59.items()}
    got: Dict[str, Dict[str, Dict]] = {}
    for x in lines[1:]:
        if x["id"] in got or x["id"] not in want_t:
            raise SystemExit("cond_%s: %s twice / unknown" % (c, x["id"]))
        e64.check_reads_list(x["reads"], want_t[x["id"]])
        if not all(r["inputs"]["finite"] for r in x["reads"]):
            raise SystemExit("cond_%s: non-finite geometry" % c)
        for r in x["reads"]:
            check_logits(r, "cond_%s %s" % (c, x["id"]))
        got[x["id"]] = {r["question"]: r for r in x["reads"]}
    if set(got) != set(want_t):
        raise SystemExit("cond_%s: pictures missing" % c)
    return got


def check_logits(r: Dict, where: str) -> None:
    """A read's stored logits: SLOTS x CLASSES, all finite, and its decoded / valid / value exactly what argmax + strict
    decode + fmt give on them (critic EXP67-R1-01)."""
    from devision.reading import CLASSES, SLOTS, UNREADABLE, decode, fmt
    lg = r.get("logits")
    if not isinstance(lg, list) or len(lg) != SLOTS or any(not isinstance(row, list) or len(row) != CLASSES for row in lg):
        raise SystemExit("%s %s: logits are not %d x %d" % (where, r.get("question"), SLOTS, CLASSES))
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for row in lg for v in row):
        raise SystemExit("%s %s: non-finite logits" % (where, r.get("question")))
    d = decode([max(range(CLASSES), key=row.__getitem__) for row in lg])
    want = (d["text"], d["valid"], fmt(d["text"]) if d["valid"] else UNREADABLE)
    if (r["decoded"], r["valid"], r["value"]) != want:
        raise SystemExit("%s %s: the read is not the decoding of its logits" % (where, r.get("question")))


def hits(reads: Dict[str, Dict[str, Dict]]) -> Dict[str, Dict[str, bool]]:
    gold = e59.gold_numbers(e59.items())
    return {rid: {t: bool(r["valid"] and r["value"] == e59.value_of(gold[rid][t])) for t, r in v.items()} for rid, v in reads.items()}


def gold_score(reads: Dict[str, Dict[str, Dict]]) -> Dict[str, Optional[float]]:
    """Descriptive only: mean log-probability of the gold character sequence (EOS included) per field, from the stored
    logits -- computed in the report, never an input."""
    gold = e59.gold_numbers(e59.items())
    acc: Dict[str, List[float]] = {f: [] for f in FIELDS}
    for rid, v in reads.items():
        for t, r in v.items():
            lab = p49.labels(e59.value_of(gold[rid][t]).replace(",", ""))
            s = 0.0
            for slot, y in enumerate(lab):
                if y < 0:
                    continue
                row = r["logits"][slot]
                mx = max(row)
                s += row[y] - (mx + math.log(sum(math.exp(z - mx) for z in row)))
            acc[e60.field_of(t)].append(s)
    return {f: (sum(v) / len(v) if v else None) for f, v in acc.items()}


def difference(a: Dict[str, Dict[str, bool]], b: Dict[str, Dict[str, bool]], field: str, c: Optional[Dict] = None, d: Optional[Dict] = None) -> Dict:
    """(b - a) on the field's share right, or with c / d the interaction (b - a) - (d - c); the field's pictures resampled
    (both objects of a picture together; r41's interval, seed 0, 2,000 resamples)."""
    pics = sorted(rid for rid in a if any(e60.field_of(t) == field for t in a[rid]))

    def share(hs, ks):
        cells = [hs[k][t] for k in ks for t in hs[k] if e60.field_of(t) == field]
        return sum(cells) / len(cells)

    def fn(ks):
        v = share(b, ks) - share(a, ks)
        if c is not None and d is not None:
            v -= share(d, ks) - share(c, ks)
        return v
    res = e60.e57.r41.interval(fn, pics, lambda ks: list(ks))
    res["pictures"] = len(pics)
    return res


def transitions(a: Dict[str, Dict[str, bool]], b: Dict[str, Dict[str, bool]], field: str) -> Dict[str, int]:
    out = {"right_right": 0, "right_wrong": 0, "wrong_right": 0, "wrong_wrong": 0}
    for rid in a:
        for t in a[rid]:
            if e60.field_of(t) == field:
                out["%s_%s" % ("right" if a[rid][t] else "wrong", "right" if b[rid][t] else "wrong")] += 1
    return out


def effects(hs: Dict[str, Dict[str, Dict[str, bool]]], field: str) -> Dict:
    return {"geom_AO_minus_AA": difference(hs["AA"], hs["AO"], field), "geom_OO_minus_OA": difference(hs["OA"], hs["OO"], field),
            "S_OA_minus_AA": difference(hs["AA"], hs["OA"], field), "S_OO_minus_AO": difference(hs["AO"], hs["OO"], field),
            "interaction": difference(hs["OA"], hs["OO"], field, hs["AA"], hs["AO"]), "OO_minus_AA": difference(hs["AA"], hs["OO"], field)}


def explain(e: Dict) -> str:
    """The pre-written reading per field: stable = both differences of a factor with lower bounds > 0."""
    lo = lambda k: e[k]["ci95"][0]  # noqa: E731
    geom = lo("geom_AO_minus_AA") > 0 and lo("geom_OO_minus_OA") > 0
    s = lo("S_OA_minus_AA") > 0 and lo("S_OO_minus_AO") > 0
    if geom and s:
        return "both: the continuous geometry and the S package each help (head geometry robustness and localisation)"
    if geom:
        return "continuous geometry: consider geometry-robust training of the head"
    if s:
        return "S package: prioritise localisation / selection"
    if lo("OO_minus_AA") > 0 and lo("interaction") > 0:
        return ("positive interaction: the gain depends on the other factor's setting; each single factor's effect stays "
                "uncertain / setting-dependent (not shown to be ineffective; simple effects reported in full)")
    return "not located: the mixed conditions do not separate the effect (mixed inputs may be out of distribution; no part excluded)"


def report(out: str = OUT, finalize: str = "") -> Dict:
    res: Dict[str, Any] = {"states": {}, "candidate_eligible": False, "kind": "post-hoc mechanism diagnostic with oracle boxes"}
    if finalize:
        res["finalized"] = {"exit_codes": finalize, "note": "missing items are not low scores"}
    if not os.path.exists(os.path.join(out, "START.json")):
        res["branch"] = "incomplete: no START (nothing ran)"
        return write_report(out, res)
    try:
        check_start(out, data_root="", code=False)
    except (SystemExit, OSError, ValueError, KeyError) as e:
        res["branch"] = "invalid: the run's identity does not verify (%s) -- no verdict" % e
        return write_report(out, res)
    reads: Dict[str, Dict] = {}
    for c in CONDS:
        if not os.path.exists(os.path.join(out, "cond_%s.jsonl" % c)):
            res["states"][c] = "missing"
            continue
        try:
            reads[c] = load_condition(out, c)
            res["states"][c] = "completed"
        except (SystemExit, OSError, ValueError, KeyError) as e:
            res["states"][c] = "invalid: %s" % e
    for c in REF:                                                            # recomputed, never a stored flag
        if c in reads:
            try:
                ref = reference(c)
                bad = [b for rid in ref for b in compare(list(reads[c][rid].values()), ref[rid])]
                res["states"]["repro_" + c] = "reproduced" if not bad else "invalid: %d mismatches with EXP64" % len(bad)
            except (SystemExit, KeyError) as e:
                res["states"]["repro_" + c] = "invalid: %s" % e
    bad = [k for k, v in res["states"].items() if v.startswith("invalid")]
    miss = [k for k, v in res["states"].items() if v == "missing"]
    if bad or miss:
        res["validity"] = "invalid" if bad else "incomplete"
        res["branch"] = "%s: %s -- no verdict" % (res["validity"], bad or miss)
        return write_report(out, res)
    res["validity"] = "valid"
    hs = {c: hits(reads[c]) for c in CONDS}
    res["correct"] = {c: {f: sum(v for k in hs[c] for t, v in hs[c][k].items() if e60.field_of(t) == f) for f in FIELDS} for c in CONDS}
    res["transitions"] = {f: {"%s->%s" % (a, b): transitions(hs[a], hs[b], f) for a, b in (("AA", "AO"), ("OA", "OO"), ("AA", "OA"), ("AO", "OO"), ("AA", "OO"))}
                          for f in FIELDS}
    res["effects"] = {f: effects(hs, f) for f in FIELDS}
    res["explanation"] = {f: explain(res["effects"][f]) for f in FIELDS}
    res["inputs"] = {c: {"tokens": sum(r["inputs"]["tokens"] for v in reads[c].values() for r in v.values()),
                         "centres_outside_box": sum(r["inputs"]["centres_outside_box"] for v in reads[c].values() for r in v.values()),
                         "geom_min": [min(r["inputs"]["geom_min"][j] for v in reads[c].values() for r in v.values()) for j in range(4)],
                         "geom_max": [max(r["inputs"]["geom_max"][j] for v in reads[c].values() for r in v.values()) for j in range(4)]} for c in CONDS}
    res["gold_logprob"] = {c: gold_score(reads[c]) for c in CONDS}
    res["branch"] = ("diagnostic complete; speed: %s; distance (separately): %s -- a mechanism reading on 24 repeatedly used dev pictures, "
                     "not a product path or confirmation" % (res["explanation"]["speed"], res["explanation"]["distance"]))
    return write_report(out, res)


def write_report(out: str, res: Dict) -> Dict:
    os.makedirs(out, exist_ok=True)
    write_json(os.path.join(out, "report.json"), res)
    text = json.dumps({k: res[k] for k in ("finalized", "states", "validity", "correct", "explanation", "branch", "candidate_eligible") if k in res},
                      indent=1, default=str)
    for f, e in (res.get("effects") or {}).items():
        text += "\n%s %s" % (f, {k: (round(v["value"], 3), [round(x, 3) for x in v["ci95"]]) for k, v in e.items()})
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
        print(json.dumps({k: v for k, v in stage_start(a.deadline).items() if k not in ("files", "pictures")}, indent=1))
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
