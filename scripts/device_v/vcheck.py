"""EXP76 device V stage (DEVICE-V-IMPL-PREP-01; docs/experiments/2026-10-20-exp76-device-plan.md section 2): the pure
parts, no model and no process. The fixed V1 reference (EXP76-RUN-02's step 0, written before its GPU fault) and the
comparison against it; the 64 V3 questions; system GPU events (gpuEvent-*.ips) by their own timestamp; the boot
cycle; whether V3 may start; and the one outcome of a V record.
"""
import datetime
import hashlib
import json
import math
import os
import re
import subprocess
from typing import Any, Dict, List, Optional, Tuple

TOL = 1e-6                     # per question and option, absolute probability difference; predictions must be equal
REF_DIR = "runs/exp76-panel-decision-r2/d2/monitor/step0"
REF = {"val": "ac653bd98b85ffff634b3424740cc6e9af265070ec60b10a84ab1217f528e71f",
       "dev16": "faa21ab2295305d590b8dad0c68b3cd650053d2453ee336a5c0ac4176fdc6274",
       "erased": "0019bfd9154fb3d82e6e20e354e0f9f6aa0b3ca8aa4c8c1e52cc1550a4b7fd61",
       "k_text": "fcf67a72d377200fb62f4f52d1108f6db38904612ffbc3a68b1ceb203efe2ee5",
       "rs_temp": "b96ebb04dd1b8859f6be50d68a0f7d07987478f660027c8e69879e4056596e1c",
       "rs_force": "4faf66b791b223004a1521fe0e05a9ae29ad19100eeaef96dab303b6d88025de",
       "rs_temp_mism": "a99603e0e852c9cda9d921f51caaf12036443f40bb0333dddc867c46151ce855",
       "rs_force_mism": "c5ba9ceafe242c7fa1fd5242df9537d2e17898a87b627bb54eb393b655033ed7"}
REF_TOTAL = 3562
FIRST_DIR = "runs/exp76-panel-decision/d2/monitor/step0"        # EXP76's first attempt: described, never a pass rule
DIVERGENT = [("exp21:d0165:v0:h16", "cmp"), ("exp21:d0165:v1:h16", "readA"), ("exp21:d0165:v1:h16", "readB"),
             ("exp21:d0165:v1:h16", "cmp"), ("exp21:d0166:v0:h16", "readA"), ("exp21:d0166:v0:h16", "readB"),
             ("exp21:d0166:v0:h16", "cmp"), ("exp21:d0166:v1:h16", "readA")]     # dev16 rows 992-999, not exempt
SAME = ("image_id", "image_used", "source", "kind", "axis", "type", "options", "gold", "gold_answer", "control")
EVENTS = ("BIF0 page fault", "MMU interrupt")
EXP76_BOOT = 1786082443          # kern.boottime of the boot cycle both EXP76 MPS faults happened in (2026-08-07 14:00:43)
BINS = (0.0, 1e-9, 1e-7, TOL, 1e-4, 1e-2, float("inf"))


def sha(p: str) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def rows(path: str) -> List[Dict[str, Any]]:
    return [json.loads(x) for x in open(path) if x.strip()]


def keyed(rs: List[Dict[str, Any]]) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], List[str]]:
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    bad = []
    for r in rs:
        k = (r["sample_id"], r["qid"])
        if k in out:
            bad.append("%s/%s twice" % k)
        out[k] = r
    return out, bad


def compare_rows(ref: List[Dict[str, Any]], got: List[Dict[str, Any]], tol: float = TOL) -> Dict[str, Any]:
    """Every reference question once in `got`, nothing else, same inputs / gold, each option's probability within
    `tol` and the same prediction. Returns counts, the largest difference, its distribution over BINS and the
    problems (empty = consistent)."""
    a, bad_a = keyed(ref)
    b, bad_b = keyed(got)
    problems = ["reference: " + x for x in bad_a] + ["new: " + x for x in bad_b]
    problems += ["%s/%s missing" % k for k in a if k not in b]
    problems += ["%s/%s not in the reference" % k for k in b if k not in a]
    diffs: Dict[Tuple[str, str], float] = {}
    for k in a:
        if k not in b:
            continue
        x, y = a[k], b[k]
        if any(x.get(f) != y.get(f) for f in SAME):
            problems.append("%s/%s inputs or gold differ" % k)
        px, py = x["probabilities"], y["probabilities"]
        if set(px) != set(py):
            problems.append("%s/%s options differ" % k)
            continue
        bad = [o for o in px if not (isinstance(px[o], (int, float)) and isinstance(py[o], (int, float))
                                     and math.isfinite(px[o]) and math.isfinite(py[o]))]
        if bad:                                            # checked per option, before any max (R1-01)
            problems += ["%s/%s option %r probability not finite (%r -> %r)" % (k + (o, px[o], py[o])) for o in bad]
            diffs[k] = float("inf")
        else:
            d = max(abs(px[o] - py[o]) for o in px)
            diffs[k] = d
            if not d <= tol:
                problems.append("%s/%s probability differs by %.3g" % (k + (d,)))
        if x["prediction"] != y["prediction"]:
            problems.append("%s/%s prediction %r -> %r" % (k + (x["prediction"], y["prediction"])))
    hist = {"<=%g" % BINS[i + 1] if i + 2 < len(BINS) else ">%g" % BINS[-2]: 0 for i in range(len(BINS) - 1)}
    labels = list(hist)
    for d in diffs.values():
        i = next((j for j in range(len(BINS) - 1) if d <= BINS[j + 1]), len(BINS) - 2)
        hist[labels[i]] += 1
    return {"n": len(a), "compared": len(diffs), "largest": max(diffs.values(), default=0.0), "distribution": hist,
            "diffs": diffs, "problems": problems}


def step0_consistency(ref_dir: str, new_dir: str, ref: Optional[Dict[str, str]] = None, tol: float = TOL,
                      first_dir: Optional[str] = None) -> Dict[str, Any]:
    """V1: the new step 0 against the fixed reference, file by file. A changed / missing reference file is a refusal
    ("reference"), not a comparison. The eight divergent dev16 questions are listed on their own and judged like
    every other question. The first attempt, when given, is only described (largest difference per set)."""
    ref = REF if ref is None else ref
    out: Dict[str, Any] = {"tolerance": tol, "sets": {}, "problems": []}
    for name, h in ref.items():
        p = os.path.join(ref_dir, name + ".details.jsonl")
        if not os.path.exists(p) or sha(p) != h:
            out["problems"].append("reference %s is missing or changed" % name)
    if out["problems"]:
        out.update(ok=False, refused=True)
        return out
    total = 0
    for name in ref:
        p = os.path.join(new_dir, name + ".details.jsonl")
        if not os.path.exists(p):
            out["problems"].append("%s: no new step 0 file" % name)
            continue
        c = compare_rows(rows(os.path.join(ref_dir, name + ".details.jsonl")), rows(p), tol)
        total += c["n"]
        if name == "dev16":
            out["divergent8"] = [{"sample_id": s, "qid": q, "difference": c["diffs"].get((s, q))} for s, q in DIVERGENT]
        c.pop("diffs")
        out["problems"] += ["%s: %s" % (name, x) for x in c["problems"]]
        out["sets"][name] = c
        if first_dir and os.path.exists(os.path.join(first_dir, name + ".details.jsonl")):
            f = compare_rows(rows(os.path.join(first_dir, name + ".details.jsonl")), rows(p), tol)
            c["first_attempt_described"] = {"largest": f["largest"], "differing": len(f["problems"])}
    if total != REF_TOTAL and ref is REF:
        out["problems"].append("%d reference questions, %d expected" % (total, REF_TOTAL))
    out.update(ok=not out["problems"], refused=False, questions=total)
    return out


def v3_questions(dev16_rows: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """The 64 fixed V3 questions: the eight divergent ones by name, then the first 56 others by (sample_id, qid)."""
    keys = {(r["sample_id"], r["qid"]) for r in dev16_rows}
    missing = [k for k in DIVERGENT if k not in keys]
    if missing:
        raise ValueError("divergent questions not in dev16: %s" % missing)
    return list(DIVERGENT) + sorted(k for k in keys if k not in set(DIVERGENT))[:56]


def ips_event(path: str) -> Dict[str, Any]:
    """One gpuEvent-*.ips: its own timestamp (the first line's header) and restart reason. Unreadable -> time None."""
    out: Dict[str, Any] = {"file": os.path.basename(path), "time": None, "reason": None}
    try:
        text = open(path, errors="replace").read()
        head = json.loads(text.split("\n", 1)[0])
        out["time"] = datetime.datetime.strptime(head["timestamp"], "%Y-%m-%d %H:%M:%S.%f %z").timestamp()
    except (OSError, ValueError, KeyError):
        return out
    m = re.search(r'"restart_reason_desc"\s*:\s*"([^"]*)"', text)
    out["reason"] = m.group(1) if m else None
    return out


def events_in_window(diag_dir: str, start: float, end: float) -> List[Dict[str, Any]]:
    """gpuEvent files whose own timestamp lies in [start, end] with a BIF0 page fault / MMU interrupt reason; a file
    that cannot be read but was modified in the window counts too (it cannot be shown to be outside or harmless)."""
    hits = []
    start = float(int(start))              # ips timestamps have whole seconds: the start's own second counts (fails safe)
    if not os.path.isdir(diag_dir):
        return hits
    for f in sorted(os.listdir(diag_dir)):
        if not f.startswith("gpuEvent"):
            continue
        p = os.path.join(diag_dir, f)
        e = ips_event(p)
        if e["time"] is None:
            try:
                mt = os.path.getmtime(p)
            except OSError:
                continue
            if start <= mt <= end:
                hits.append(dict(e, unreadable=True))
        elif start <= e["time"] <= end and (e["reason"] is None or any(x in e["reason"] for x in EVENTS)):
            hits.append(e)
    return hits


def boot_time(text: Optional[str] = None) -> Optional[int]:
    """kern.boottime seconds (macOS `sysctl -n kern.boottime`: "{ sec = 1786082443, usec = ... } ...")."""
    if text is None:
        text = subprocess.run(["/usr/sbin/sysctl", "-n", "kern.boottime"], capture_output=True, text=True).stdout
    m = re.search(r"sec\s*=\s*(\d+)", text)
    return int(m.group(1)) if m else None


def boot_problems(boot: Optional[int], fault_end: float) -> List[str]:
    """The V stage runs only in a boot cycle started after EXP76's faults (the restart is the owner's)."""
    if boot is None:
        return ["the boot time cannot be read"]
    if boot == EXP76_BOOT or boot <= fault_end:
        return ["still the boot cycle of the EXP76 faults (booted %s): a restart by the owner comes first"
                % time_str(boot)]
    return []


def v3_allowed(stage: Dict[str, Any], now: float, deadline: float, v3_reserve: float, report_reserve: float) -> List[str]:
    """V3 starts only after the planned stop, a passed V1, a confirmed cleanup, no GPU / system event, and with its
    own reserve plus the report's still left."""
    why = []
    end = stage.get("end") or {}
    if end.get("why") != "planned_stop" or not end.get("initiated") or end.get("gone_before_term"):
        why.append("the training process did not end by the planned stop")
    if not (stage.get("v1") or {}).get("ok"):
        why.append("V1 did not pass")
    if not stage.get("cleanup_confirmed"):
        why.append("cleanup not confirmed")
    if stage.get("gpu_error") or stage.get("system_events") or stage.get("exception"):
        why.append("a GPU error or system GPU event")
    if deadline - now < v3_reserve + report_reserve:
        why.append("%.0f s left, %.0f needed" % (deadline - now, v3_reserve + report_reserve))
    return why


def outcome(rec: Optional[Dict[str, Any]], events: List[Dict[str, Any]]) -> str:
    """The one outcome of a V record (`events`: system GPU events found in the whole window, the report's own look
    included). Explicit failures first; then pass needs every piece of evidence present (R1-02): the window, V1 ok,
    the training stop initiated by the supervisor after >= 250 updates with its cleanup confirmed, V3 consistent with
    its cleanup confirmed. A missing piece is incomplete, never pass."""
    if not rec:
        return "incomplete: no V record (the V orchestration did not finish)"
    tr = rec.get("training") or {}
    v3 = rec.get("v3") or {}
    if events or tr.get("system_events") or v3.get("system_events"):
        return "execution-unverified: a system GPU event (BIF0 / MMU) inside the V window"
    for part in (tr, v3):
        if part and part.get("cleanup_confirmed") is False:
            return "failed: cleanup not confirmed -- no further model may start"
    if tr.get("gpu_error") or v3.get("gpu_error"):
        return "failed: GPU execution error"
    if tr.get("exception") or v3.get("exception") or rec.get("exception"):
        return "failed: the V supervisor raised (%s)" % (tr.get("exception") or v3.get("exception") or rec.get("exception"))
    if (tr.get("v1") or {}).get("refused"):
        return "refused: the V1 reference is missing or changed"
    if tr.get("v1") and not tr["v1"].get("ok"):
        return "consistency not established (V1)"
    end = tr.get("end") or {}
    why = end.get("why")
    if why == "nonfinite":
        return "failed: a non-finite training value"
    if why == "limit":
        return "incomplete: the limit came before the planned stop"
    if why != "planned_stop" or not end.get("initiated") or end.get("gone_before_term"):
        return "failed: the training process ended by itself (exit %s) before the planned stop" % tr.get("child_returncode")
    if (tr.get("updates") or 0) < 250:
        return "failed: planned stop recorded with fewer than 250 updates"
    if not rec.get("start") or not rec.get("deadline"):
        return "incomplete: no V window recorded"
    if (tr.get("v1") or {}).get("ok") is not True:
        return "incomplete: no V1 result recorded"
    if tr.get("cleanup_confirmed") is not True:
        return "incomplete: the training cleanup is not recorded as confirmed"
    if not v3:
        return "incomplete: V3 not started (%s)" % "; ".join(rec.get("v3_not_started") or ["no reason recorded"])
    if (v3.get("end") or {}).get("why") == "limit":
        return "incomplete: V3 stopped at its limit"
    res = v3.get("result") or {}
    if v3.get("exit") != 0 or not res:
        return "failed: V3 ended with exit %s" % v3.get("exit")
    if not res.get("ok"):
        return "consistency not established (V3 save / load)"
    if v3.get("cleanup_confirmed") is not True:
        return "incomplete: the V3 cleanup is not recorded as confirmed"
    return "pass"


def time_str(t: float) -> str:
    return datetime.datetime.fromtimestamp(t).strftime("%F %T")
