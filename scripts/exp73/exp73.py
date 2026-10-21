"""exp73: the seven protection panels through EXP71-on's complete reading combination (critic EXP71-RESULT-01 /
EXP73-PREP-01; docs/experiments/2026-10-19-exp73-guard-plan.md). A risk assessment approved separately after EXP71 met its
development transfer target -- EXP58's exploration ban on B2 is not touched (that run and report stay as they are). Fixed:
EXP71's frozen bundle_on (EXP57 decision end 962503fb / 09a5b447, selector c92b7b7e, head 0e275bdf, corrector 199b5367,
local views 12 / 0.25 / T, the current routing and table format), loaded through the verified entry Reading.load.
EXP71's A + B1 evidence is reused, not re-run. Raw temperature 1. candidate_eligible is always false.

  start   model-free: identities (EXP71's run -- START / report / registry and every registered file -- and its bundle_on,
          EXP57 frozen + the exploration record, the seven panel files, v0.3's records, the picture manifest = EXP58's bound
          entries for these panels) -> START.json.
  guard   MPS: Reading.load(EXP57, runs/exp71/bundle_on); every question of exist / science / ext / count_wide /
          count_narrow / flip / vsr through Reading.decide (the actual picture as image=), EXP58's request grouping;
          answers and per-request reads / tables / states registered per panel.
  report  model-free: per panel against v0.3 (EXP58's rules: paired_metric; ext the 13-source macro; flip with the original
          and flipped pictures resampled together; a cost = difference <= -3 points with upper bound < 0); science's 20
          routed questions (9 temperature, 11 magnets) and 557 ordinary ones described apart; every panel complete and bound
          or no verdict.

    uv run python scripts/exp73/exp73.py counts | start --deadline EPOCH | guard | report [--finalize CODES]
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List

HERE = os.path.dirname(__file__)
OUT, E71 = "runs/exp73", "runs/exp71"
E71_RUN = {"START.json": "da4c3cec8811f5847ba2cb5953952eece452074e3cefea174e01a9bbcea6b943",
           "report.json": "6869596dff0f36fb31f7ff32d3a910c7e26291f92ef670b08a6b4ec3367d6eea",
           "registry.json": "dc93c7a1f969581c06b9c075fc9e8f5e28d8a449a658085d36beedd120ea6b07"}
BUNDLE = os.path.join(E71, "bundle_on")
COST_POINTS = -3
FILES = ["scripts/exp73/exp73.py", "scripts/exp73/run.sh", "scripts/exp71/exp71.py", "scripts/exp58/live.py", "scripts/exp32/report.py",
         "src/devision/reading.py", "src/devision/infer.py", "src/devision/model.py", "src/devision/image.py", "src/devision/train/evaluate.py",
         "src/devision/train/compare.py", "scripts/exp53/watch.py"]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, rel))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


e71 = _load("exp71", "../exp71/exp71.py")
e58, e57, e54 = e71.e58, e71.e57, e71.e54
sha, write_json, register, jsonl = e71.sha, e71.write_json, e71.register, e71.jsonl
GUARD = e58.GUARD


# ------------------------------------------------------------------ identities

def bound_e71() -> Dict:
    """EXP71's run and its bundle_on, as EXP71's START bound them (the combination EXP71-on actually used)."""
    run = e71.bound_files(E71, E71_RUN, "EXP71")
    st71 = json.load(open(os.path.join(E71, "START.json")))
    b = st71["bundles"]["on"]
    if sha(os.path.join(BUNDLE, "reading.json")) != b["reading.json"]:
        raise SystemExit("EXP71's bundle_on/reading.json changed")
    for f in ("selector.pt", "head.pt", "corrector.pt"):
        if sha(os.path.join(BUNDLE, f)) != b[f]:
            raise SystemExit("EXP71's bundle_on/%s changed" % f)
    if b["corrector.pt"] != e71.CORR_SHA or b["selector.pt"] != e71.C92 or b["head.pt"] != e71.HEAD_SHA:
        raise SystemExit("bundle_on is not c92 + EXP64's head + EXP69's corrector")
    return {"run": run, "bundle_on": dict(b)}


def pictures(data_root: str = "data") -> Dict[str, str]:
    """Every picture the seven panels open, with EXP58's bound manifest entries (bytes checked with data_root)."""
    if sha(os.path.join(e71.E58R, "pictures.json")) != e71.E58_RUN["pictures.json"]:
        raise SystemExit("EXP58's picture manifest changed")
    m58 = json.load(open(os.path.join(e71.E58R, "pictures.json")))
    need = {r["image"] for p in GUARD.values() for r in jsonl(p)}
    missing = sorted(need - set(m58))
    if missing:
        raise SystemExit("pictures outside EXP58's manifest: %s" % missing[:3])
    pics = {p: m58[p] for p in sorted(need)}
    if data_root:
        bad = [p for p, h in pics.items() if sha(os.path.join(data_root, p)) != h]
        if bad:
            raise SystemExit("%d pictures differ from EXP58's manifest, e.g. %s" % (len(bad), bad[0]))
    return pics


def request_counts() -> Dict:
    from collections import Counter

    from devision.reading import domain_of
    out = {}
    for n, p in GUARD.items():
        rows = jsonl(p)
        qs = [q for r in rows for q in r["questions"].values()]
        out[n] = {"requests": len(e58.requests(rows)), "questions": len(qs),
                  "routed": dict(Counter(domain_of(q.get("instructions", "")) or "ordinary" for q in qs))}
    return out


def identities() -> Dict:
    return {"exp71": bound_e71(), "mode": e58.mode_record(e58.EXPLORE), "model": dict(e58.NEW_SHA or {}), "exp58": e71.bound_files(e71.E58R, e71.E58_RUN, "EXP58"),
            "data": {n: sha(p) for n, p in GUARD.items()}, "v03": {n: sha(os.path.join(e58.V03, n + ".details.jsonl")) for n in GUARD}}


def stage_start(deadline: int, out: str = OUT, data_root: str = "data") -> Dict:
    path = os.path.join(out, "START.json")
    if os.path.exists(path):
        raise SystemExit("%s exists (fresh only)" % path)
    ident = identities()
    pics = pictures(data_root)
    os.makedirs(out, exist_ok=True)
    write_json(os.path.join(out, "pictures.json"), pics)
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    s = {"git_head": git("rev-parse", "HEAD"), "dirty_files": [f for f in FILES if git("status", "--porcelain", "--", f)],
         "files": {f: sha(f) for f in FILES}, **ident, "pictures": {"pictures.json": sha(os.path.join(out, "pictures.json")), "count": len(pics)},
         "requests": request_counts(), "temperature": "none (raw), in memory", "max_questions": e58.MAX_QUESTIONS,
         "purpose": "B2 risk assessment of EXP71-on, approved separately after EXP71's development transfer target (EXP73-PREP-01); "
                    "EXP58's exploration ban is untouched; not a candidate",
         "candidate_eligible": False, "started": time.strftime("%F %T"), "deadline": deadline, "deadline_text": time.strftime("%F %T", time.localtime(deadline))}
    write_json(path, s)
    return s


def check_start(out: str, data_root: str = "data", code: bool = True) -> Dict:
    st = json.load(open(os.path.join(out, "START.json")))
    if code:
        bad = [f for f, h in st["files"].items() if sha(f) != h]
        if bad:
            raise SystemExit("code differs from START: %s" % bad)
    for k, v in identities().items():
        if st[k] != v:
            raise SystemExit("%s differs from START" % k)
    if e58.fz.weights(e58.NEW_RUN) != st["model"]:
        raise SystemExit("EXP57's weights / config are not START's")
    if sha(os.path.join(out, "pictures.json")) != st["pictures"]["pictures.json"] or pictures("") != json.load(open(os.path.join(out, "pictures.json"))):
        raise SystemExit("the picture manifest differs from START")
    if data_root:
        pictures(data_root)
    return st


# ------------------------------------------------------------------ guard

def run_set(reading, samples: List[Dict], data_root: str, pics: Dict[str, str], on_request=None):
    """EXP58's run_set (options of a noul question = ["false", "true"]) with each request's reads / table / state kept;
    every question answered on its own row's picture. (EXP71's copy assumed choice questions only -- right for dev_sqa,
    not for these panels -- and is left as it ran.)"""
    from devision.reading import domain_of
    from devision.train.evaluate import _probabilities, score_records
    asked, reads = {}, []
    for n, (image, text, items) in enumerate(e58.requests(samples)):
        questions = {"%d:%s" % (i, qid): q for i, qid, q in items}
        diag: Dict = {}
        res = e58.ask(reading, image, text, questions, data_root, pics, diag)
        reads.append({"image": image, "questions": [[samples[i]["id"], qid] for i, qid, _ in items], "reads": diag.get("reads"),
                      "table": diag.get("table"), "state": diag.get("state")})
        for key, (i, qid, q) in zip(questions, items):
            asked[(i, qid)] = (q, image, len(items), _probabilities(res["answers"][key]))
        if on_request:
            on_request(n + 1)
    recs = []
    for i, s in enumerate(samples):
        for qid in s["questions"]:
            q, image, size, probs = asked[(i, qid)]
            gold = s["gold"][qid]["probabilities"]
            recs.append({"sample_id": s["id"], "qid": qid, "image_id": s["image_id"], "image_used": image, "source": s.get("source", ""),
                         "kind": s.get("kind", ""), "axis": s.get("axis", ""), "type": q["type"],
                         "options": list(q["criteria"]) if q["type"] == "choice" else ["false", "true"], "gold": gold,
                         "gold_answer": max(gold, key=gold.__getitem__), "probabilities": probs, "prediction": max(probs, key=probs.__getitem__),
                         "control": "none", "questions_in_request": size, "routed": domain_of(q.get("instructions", "")) is not None})
    return score_records(recs), reads


def stage_guard(out: str = OUT, data_root: str = "data", reading=None) -> Dict:
    from devision.infer import best_device
    from devision.reading import Reading
    st = check_start(out, data_root)                                              # before any load
    if reading is None:
        reading = Reading.load(os.path.join(e58.NEW_RUN, "stage2"), BUNDLE, device=best_device())
        if reading.corrector is None:
            raise SystemExit("bundle_on loaded without its corrector")
    reading.decider.cfg.temperature, reading.decider.cfg.temperature_by_options = [1.0, 1.0, 1.0], {}
    pics = json.load(open(os.path.join(out, "pictures.json")))
    s_sha = sha(os.path.join(out, "START.json"))
    total = sum(v["requests"] for v in st["requests"].values())
    done, t0 = 0, time.time()
    for name, path in GUARD.items():
        recs, reads = run_set(reading, jsonl(path), data_root, pics, on_request=lambda k, b=done: e54.progress(out, "guard", b + k, total))
        done += st["requests"][name]["requests"]
        ident = {"set": name, "start_sha256": s_sha, "data_sha256": st["data"][name], "model": st["model"],
                 "bundle": st["exp71"]["bundle_on"]["reading.json"], "pictures": st["pictures"]["pictures.json"], "temperature": "none (raw)",
                 "path": "devision.reading.Reading.decide(image=actual picture), EXP71-on"}
        e57.write_set(out, name, recs, ident)
        f = "reads_%s.jsonl" % name
        with open(os.path.join(out, f), "w") as fh:
            fh.write("".join(json.dumps(x) + "\n" for x in reads))
        register(out, "reads_" + name, {f: sha(os.path.join(out, f))})
        print("%s done after %.1f s" % (name, time.time() - t0), flush=True)
    return {"seconds": time.time() - t0}


# ------------------------------------------------------------------ report

def g_records(out: str, name: str, st: Dict) -> List[Dict]:
    """A panel after every check: registered, identity bound, every question of the source once, each answered on its
    own row's picture."""
    f = "details_%s.jsonl" % name
    if e54.check_registered(out, "decide_" + name).get(f) != sha(os.path.join(out, f)):
        raise SystemExit("%s not registered" % f)
    lines = jsonl(os.path.join(out, f))
    want = {"set": name, "start_sha256": sha(os.path.join(out, "START.json")), "data_sha256": st["data"][name], "model": st["model"],
            "bundle": st["exp71"]["bundle_on"]["reading.json"], "pictures": st["pictures"]["pictures.json"], "temperature": "none (raw)"}
    if any(lines[0]["identity"].get(k) != v for k, v in want.items()):
        raise SystemExit("%s: identity does not bind" % f)
    src = jsonl(GUARD[name])
    got = e57.r41.check_details("exp73/%s" % name, src, lines[1:])
    image_of = {(r["id"], qid): r["image"] for r in src for qid in r["questions"]}
    if any(r["image_used"] != image_of[(r["sample_id"], r["qid"])] for r in lines[1:]):
        raise SystemExit("%s: a question was answered on another picture than its row's" % f)
    rf = "reads_%s.jsonl" % name
    if e54.check_registered(out, "reads_" + name).get(rf) != sha(os.path.join(out, rf)):
        raise SystemExit("%s not registered" % rf)
    return list(got.values())


def cost(m: Dict) -> bool:
    return bool(m["difference"] * 100 <= COST_POINTS and m["ci95"][1] < 0)


def branch(res: Dict) -> str:
    bad = [k for k, v in res["states"].items() if not v.startswith("completed")]
    if bad:
        kind = "invalid" if any(v.startswith("invalid") for v in res["states"].values()) else "incomplete"
        return "%s: %s -- no protection verdict (missing / invalid panels are never a pass)" % (kind, bad)
    if res["costs"]:
        return "protection costs %s on EXP71-on: protection repair before anything else; not a candidate" % res["costs"]
    return ("no pre-written protection cost on the seven panels for EXP71-on (not a proof of equivalence): still a development "
            "result -- independent confirmation, full calibration / evaluation, CPU / API / install checks remain; not a candidate")


def report(out: str = OUT, finalize: str = "") -> Dict:
    res: Dict[str, Any] = {"states": {}, "guard": {}, "costs": [], "candidate_eligible": False}
    if finalize:
        res["finalized"] = {"exit_codes": finalize, "note": "missing items are not low scores"}
    if not os.path.exists(os.path.join(out, "START.json")):
        res["branch"] = "incomplete: no START (nothing ran)"
        return write_report(out, res)
    try:
        st = check_start(out, data_root="", code=False)
    except (SystemExit, OSError, ValueError, KeyError) as e:
        res["branch"] = "invalid: the run's identity does not verify (%s) -- no verdict" % e
        return write_report(out, res)
    from devision.train.compare import paired_metric
    r32 = _load("exp32_report", "../exp32/report.py")
    for name in GUARD:
        if not os.path.exists(os.path.join(out, "details_%s.jsonl" % name)):
            res["states"][name] = "missing"
            continue
        try:
            b = g_records(out, name, st)
            a = e58.v03_records(name, st)
            if name == "flip":
                a = [dict(r, image_id=r["image_id"].removesuffix(":flip")) for r in a]
                b = [dict(r, image_id=r["image_id"].removesuffix(":flip")) for r in b]
            m = paired_metric(a, b, "macro_accuracy", sources=r32.EXT_SOURCES) if name == "ext" else paired_metric(a, b, "accuracy")
        except (SystemExit, OSError, ValueError, KeyError) as e:
            res["states"][name] = "invalid: %s" % e
            continue
        res["states"][name] = "completed"
        res["guard"][name] = {"difference": m["difference"], "ci95": m["ci95"], "cost": cost(m), **e58.errors(b)}
        if cost(m):
            res["costs"].append(name)
        if name == "science":
            routed = [r for r in b if r["routed"]]
            ordinary = [r for r in b if not r["routed"]]
            res["science_slices"] = {"routed": {"n": len(routed), "accuracy": sum(r["correct"] for r in routed) / len(routed) if routed else None,
                                                "temperature": sum(1 for r in routed if r["kind"].startswith("temp") or "temperature" in r.get("source", "")) or None},
                                     "ordinary": {"n": len(ordinary), "accuracy": sum(r["correct"] for r in ordinary) / len(ordinary) if ordinary else None}}
    res["branch"] = branch(res)
    res["mode"] = st["mode"]
    return write_report(out, res)


def write_report(out: str, res: Dict) -> Dict:
    os.makedirs(out, exist_ok=True)
    write_json(os.path.join(out, "report.json"), res)
    text = json.dumps({k: res[k] for k in ("finalized", "states", "guard", "costs", "science_slices", "branch", "candidate_eligible") if k in res},
                      indent=1, default=str)
    open(os.path.join(out, "report.txt"), "w").write(text + "\n")
    print(text)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["counts", "start", "guard", "report"])
    ap.add_argument("--deadline", type=int, default=0)
    ap.add_argument("--finalize", default="")
    a = ap.parse_args(argv)
    if a.stage == "counts":
        print(json.dumps(request_counts(), indent=1))
    elif a.stage == "start":
        print(json.dumps({k: v for k, v in stage_start(a.deadline).items() if k != "files"}, indent=1, default=str))
    elif a.stage == "guard":
        print(json.dumps(stage_guard(), indent=1))
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
