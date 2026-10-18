"""exp50 supplement (critic EXP50-SUP-PREP-01): one bounded read of arm C's fixed end on the sets the 90-minute chain did
not complete -- only after the chain has ended, its report is on disk, and the critic gives a separate permission. Nothing
of the original run is touched (report, status, FROZEN, complete or half-written outputs); new files go to
runs/exp50-supplement/ with their own hashes.

  plan    preconditions and the missing list, written to runs/exp50-supplement/START.json:
          - the chain is over: runs/exp50.out has its "final report" line, runs/exp50-report.json / .txt exist, the
            deadline has passed (no model process is running: checked by supplement.sh);
          - C's training completed (status "train completed") and its frozen weights / config are the stage2 files
            now; T's and X's frozen records check; the arms' identities pass report.arm_identity;
          - missing = C's six sets, in the chain's order, whose last status is not "completed" (interrupted, failed,
            missing or absent) -- by status alone, never by any score. A set that completed and is recorded under
            C's FROZEN.json is reused as is; an interrupted one is read again whole (no half details are joined).
          - START binds the original report json / txt, the status files, T / C / X FROZEN.json, C's weights, each
            missing set's input sha, the code commit (52ff548) and the missing list. Nothing missing: exit 3, no run.
          T missing anything is refused (exit 1): only C is in scope.
  record SET   after devision-eval wrote runs/exp50-supplement/c/eval/SET.json: its summary must be C's original
          checkpoint path, temperature 1, no control, the set's input sha, code 52ff548; its details answer every
          question exactly once (exp41 check_details); C's weights still the frozen ones; then both files' hashes
          go into runs/exp50-supplement/MANIFEST.json.
  report  runs/exp50-supplement/report.{json,txt}: "original_90min" (the original report as written, cited by sha,
          its conditions copied) and "after_supplement" (the same readout -- question sets, statistics, conditions,
          cut slice of scripts/exp50/report.py -- with C's missing sets taken from the supplement, every source and
          hash listed); the original incomplete is kept, the after-supplement result is labelled as computed after
          the 90 minutes. A set still not recorded stays incomplete.

    uv run python scripts/exp50/supplement.py plan|record SET|report|data SET
"""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from typing import Dict, List

HERE = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("exp50_report", os.path.join(HERE, "report.py"))
assert _spec is not None and _spec.loader is not None
rep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rep)
r48, r42, r41, frozen = rep.r48, rep.r42, rep.r41, rep.frozen

OUT = "runs/exp50-supplement"
ORDER = ["fit_O", "fit_L", "rl_tune", "rl_tune_L", "exist", "science"]
DATA = {**rep.SETS, **rep.RISK}
CODE = "52ff548"
ORIG_REPORT, ORIG_TXT, ORIG_OUT, DEADLINE_FILE = "runs/exp50-report.json", "runs/exp50-report.txt", "runs/exp50.out", "runs/exp50.deadline"


def sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def status(run: str) -> Dict[str, str]:
    return r42.states(run + "-status.txt")


def missing_of(st: Dict[str, str]) -> List[str]:
    return [n for n in ORDER if st.get("eval:" + n) != "completed"]


def plan(runs: Dict[str, str] = rep.RUNS, out: str = OUT, now: float = 0.0) -> Dict:
    now = now or time.time()
    if not (os.path.exists(ORIG_REPORT) and os.path.exists(ORIG_TXT) and "final report" in open(ORIG_OUT).read()):
        raise SystemExit("the original chain has not written its final report")
    if now <= int(open(DEADLINE_FILE).read()):
        raise SystemExit("the original deadline has not passed")
    if os.path.exists(os.path.join(out, "START.json")):
        raise SystemExit("%s/START.json exists (one supplement only)" % out)
    man = json.load(open("data/exp42/MANIFEST.json"))
    qsha = r41.sha(rep.QUADS)
    for arm in ("T", "C"):
        why = rep.arm_identity(arm, runs[arm], man, qsha)
        if why:
            raise SystemExit("%s: %s" % (arm, why))
    if frozen.check(runs["X"], list(DATA))["weights"] != rep.X_SHA:
        raise SystemExit("EXP48-X is not its frozen identity")
    st = {a: status(runs[a]) for a in ("T", "C")}
    for a in ("T", "C"):
        if st[a].get("train") != "completed" or not os.path.exists(frozen.path(runs[a])):
            raise SystemExit("%s: training not completed or not frozen -- no resume, no retraining" % a)
    if missing_of(st["T"]):
        raise SystemExit("T is missing %s: only C is in scope" % missing_of(st["T"]))
    t_fr = frozen.check(runs["T"], ORDER)
    c_done = [n for n in ORDER if n not in missing_of(st["C"])]
    c_fr = frozen.check(runs["C"], c_done)              # weights = the stage2 files now; completed sets unchanged
    miss = missing_of(st["C"])
    start = {"missing": miss, "reused_C": c_done, "code": CODE, "git_head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
             "original": {"report.json": sha(ORIG_REPORT), "report.txt": sha(ORIG_TXT), "exp50.out": sha(ORIG_OUT),
                          "status_T": sha(runs["T"] + "-status.txt"), "status_C": sha(runs["C"] + "-status.txt"),
                          "FROZEN_T": sha(frozen.path(runs["T"])), "FROZEN_C": sha(frozen.path(runs["C"])),
                          "FROZEN_X": sha(frozen.path(runs["X"]))},
             "weights": {"T": t_fr["weights"], "C": c_fr["weights"], "X": rep.X_SHA},
             "inputs": {n: sha(DATA[n]) for n in miss}, "budget_minutes": 15, "planned": time.strftime("%F %T")}
    if miss:
        os.makedirs(os.path.join(out, "c", "eval"), exist_ok=True)
        json.dump(start, open(os.path.join(out, "START.json"), "w"), indent=1)
    return start


def record(name: str, runs: Dict[str, str] = rep.RUNS, out: str = OUT) -> Dict:
    start = json.load(open(os.path.join(out, "START.json")))
    if name not in start["missing"]:
        raise SystemExit("%s is not a missing set" % name)
    if frozen.weights(runs["C"]) != start["weights"]["C"]:
        raise SystemExit("C's weights or config changed")
    ev = os.path.join(out, "c", "eval")
    info = json.load(open(os.path.join(ev, name + ".json")))["run"]
    if os.path.normpath(info.get("checkpoint", "")) != os.path.normpath(os.path.join(runs["C"], "stage2")) \
            or info.get("temperature") != "none (raw)" or info.get("control") != "none" \
            or info.get("data_sha256") != start["inputs"][name] or not str(info.get("code", "")).startswith(CODE):
        raise SystemExit("%s: not C's end at temperature 1, no control, on the original input, code %s" % (name, CODE))
    r41.check_details("supplement/%s" % name, r42.load(DATA[name]), r42.load(os.path.join(ev, name + ".details.jsonl")))
    mpath = os.path.join(out, "MANIFEST.json")
    m = json.load(open(mpath)) if os.path.exists(mpath) else {"start": sha(os.path.join(out, "START.json")), "evals": {}}
    m["evals"][name] = {f: sha(os.path.join(ev, f)) for f in (name + ".json", name + ".details.jsonl")}
    json.dump(m, open(mpath, "w"), indent=1)
    return m


def sources(runs: Dict[str, str] = rep.RUNS, out: str = OUT) -> Dict:
    """(who, set) -> (directory of eval/, checkpoint, origin) for every set with a verified record."""
    start = json.load(open(os.path.join(out, "START.json")))
    for k, v in start["original"].items():                     # the original products are unchanged
        p = {"report.json": ORIG_REPORT, "report.txt": ORIG_TXT, "exp50.out": ORIG_OUT,
             "status_T": runs["T"] + "-status.txt", "status_C": runs["C"] + "-status.txt",
             "FROZEN_T": frozen.path(runs["T"]), "FROZEN_C": frozen.path(runs["C"]), "FROZEN_X": frozen.path(runs["X"])}[k]
        if sha(p) != v:
            raise SystemExit("original %s changed after the supplement started" % k)
    if frozen.weights(runs["C"]) != start["weights"]["C"]:
        raise SystemExit("C's weights or config changed")
    m = json.load(open(os.path.join(out, "MANIFEST.json"))) if os.path.exists(os.path.join(out, "MANIFEST.json")) else {"evals": {}}
    src: Dict = {}
    for who in ("X", "T"):
        frozen.check(runs[who], ORDER)
        for n in ORDER:
            src[(who, n)] = (runs[who], os.path.join(runs[who], "stage2"), "original")
    frozen.check(runs["C"], start["reused_C"])
    for n in start["reused_C"]:
        src[("C", n)] = (runs["C"], os.path.join(runs["C"], "stage2"), "original")
    for n, files in m["evals"].items():
        for f, h in files.items():
            if sha(os.path.join(out, "c", "eval", f)) != h:
                raise SystemExit("supplement %s changed after it was recorded" % f)
        src[("C", n)] = (os.path.join(out, "c"), os.path.join(runs["C"], "stage2"), "supplement")
    return src


def readout(src: Dict) -> Dict:
    """scripts/exp50/report.py's readout over the given sources (same functions, statistics and conditions)."""
    out: Dict = {"readout": {}, "differences": {}, "quads": {}, "risk": {}}
    recs: Dict = {}
    for (who, n), (d, ck, _) in src.items():
        if n in rep.SETS:
            recs[(who, n)] = r48.records(d, n, ck)
        else:
            info = json.load(open(os.path.join(d, "eval", n + ".json")))["run"]
            if os.path.normpath(info.get("checkpoint", "")) != os.path.normpath(ck) or info.get("temperature") != "none (raw)" \
                    or info.get("control") != "none" or info.get("data_sha256") != r41.sha(DATA[n]):
                raise SystemExit("%s/%s: not %s at temperature 1, no control" % (d, n, ck))
            got = r41.check_details("%s/%s" % (d, n), r42.load(DATA[n]), r42.load(os.path.join(d, "eval", n + ".details.jsonl")))
            out["risk"].setdefault(n, {})[who] = sum(g["correct"] for g in got.values()) / len(got)
    fams = {k: r42.units(v) for k, v in recs.items()}
    qfams = {k: r48.quad_units(k[1], v) for k, v in recs.items()}
    for (who, n), fam in fams.items():
        out["readout"].setdefault(n, {})[who] = r42.summary(fam)
        q = qfams[(who, n)]
        out["quads"].setdefault(n, {})[who] = {"macro": r48.quad_value(q["units"], sorted(q["units"])), "quads": q["quads"],
                                               "excluded": q["excluded"]}
    for n in rep.SETS:
        for a, b in (("T", "X"), ("C", "X"), ("T", "C")):
            if (a, n) in fams and (b, n) in fams:
                ca, cb = r42.clusters(recs[(a, n)]), r42.clusters(recs[(b, n)])
                if ca != cb:
                    raise SystemExit("%s: %s and %s on different questions" % (n, a, b))
                d = {s: r42.diff(fams[(a, n)], fams[(b, n)], ca, s) for s in ("different", "both", "twin")}
                expand = lambda ks, ca=ca: [f for k in ks for f in sorted(ca[k])]  # noqa: E731
                qa, qb = qfams[(a, n)]["units"], qfams[(b, n)]["units"]
                d["quad"] = r41.interval(lambda fs, qa=qa, qb=qb: r48.quad_value(qa, fs) - r48.quad_value(qb, fs), sorted(ca), expand)
                out["differences"]["%s|%s-%s" % (n, a, b)] = d
    out["cut_slice"] = rep.cut_slice(recs)
    out["conditions"] = rep.conditions(out)
    return out


def report(runs: Dict[str, str] = rep.RUNS, out: str = OUT) -> Dict:
    src = sources(runs, out)
    orig = json.load(open(ORIG_REPORT))
    after = readout(src)
    res = {"original_90min": {"report.json_sha256": sha(ORIG_REPORT), "conditions": orig.get("conditions"), "runs": orig.get("runs")},
           "after_supplement": dict(after, note="computed after the 90-minute chain, from C's fixed end; the original "
                                                "incomplete stands as the 90-minute result"),
           "sources": {"%s|%s" % k: {"dir": v[0], "origin": v[2]} for k, v in sorted(src.items())},
           "still_missing_C": [n for n in ORDER if ("C", n) not in src],
           "supplement_manifest_sha256": sha(os.path.join(out, "MANIFEST.json")) if os.path.exists(os.path.join(out, "MANIFEST.json")) else None}
    json.dump(res, open(os.path.join(out, "report.json"), "w"), indent=1, default=str)
    lines = ["original (90 minutes) conditions %s" % res["original_90min"]["conditions"],
             "after supplement conditions %s (still missing for C: %s)" % (after["conditions"], res["still_missing_C"])]
    for n, by in after["readout"].items():
        for who, s in by.items():
            lines.append("%-9s %s different %.3f both %.3f twin %.3f quads %.3f [%s]" % (
                n, who, s["different"], s["both"], s["twin"], after["quads"][n][who]["macro"], src[(who, n)][2]))
    for k, v in after["differences"].items():
        lines.append("%s %s" % (k, {s: "%+.3f [%+.3f, %+.3f]" % (x["value"], *x["ci95"]) for s, x in v.items()}))
    lines.append("cut slice %s; risk %s" % (after["cut_slice"], after["risk"]))
    open(os.path.join(out, "report.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return res


def main(argv=None) -> int:
    a = sys.argv[1:] if argv is None else argv
    if a[:1] == ["plan"]:
        s = plan()
        print(json.dumps(s, indent=1))
        return 0 if s["missing"] else 3
    if a[:1] == ["record"] and len(a) == 2:
        print(json.dumps(record(a[1]), indent=1))
        return 0
    if a[:1] == ["data"] and len(a) == 2:
        print(DATA[a[1]])
        return 0
    if a[:1] == ["report"]:
        report()
        return 0
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main())
