"""exp52 readout (critic EXP52-PREP-01; rules written before the run). N = number reader + digit loss weight 1, Z = the
same reader with weight 0, both from EXP50-T's frozen end; B = EXP50-T itself, whose frozen temperature-1 records (fit_O,
fit_L, rl_tune, rl_tune_L, exist, science) are reused after their frozen check, and whose ScienceQA transfer slices
are read once in runs/exp52-base. All decisions are ordinary decide (devision-eval, raw temperature 1, max 8 questions
per request), no oracle.

Main sets (exp42 statistics: different, both, twin, same answer; field-equal macros; case-linked clusters, 2,000
resamples, seed 0; EXP48 strict quads reported): fit_O / fit_L / rl_tune / rl_tune_L; differences N - B, Z - B, N - Z.
Conditions (pre-written, a short screen -- not a release):
  progress_A (A in N, Z): original rl_tune different >= 0.60, both >= 0.20, twin >= 0.20, with A - B lower bounds of both
    and twin > 0; quads, same answer, fields, fit and L reported apart, never a substitute;
  attribution (digit supervision): N - Z both on rl_tune >= +0.10 with lower bound > 0 and progress_N.
Number readout (scripts/exp52/readout.py; no oracle, EXP51's statistics on fit_O / rl_tune; donor = the same question on
the value-changed twin picture): reported with EXP51's investment thresholds as a reference only (a harder task: no box).
A positive readout with no decision progress is "the use by the decision is not connected yet"; an unlearned readout is
not a new verdict on local pixels. Content use: rl_tune accuracy with the reader's content and with it left out (a
fixed removal, dependence evidence only). ScienceQA temp / force transfer (true picture, fixed wrong picture) and exist /
science: reported, not a protection verdict. Anything missing: "incomplete".

    uv run python scripts/exp52/report.py
"""
import hashlib
import importlib.util
import json
import os
import sys
from typing import Dict, List

HERE = os.path.dirname(__file__)


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, rel))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


r48 = _load("exp48_report", "../exp48/report.py")
p49 = _load("exp49_probe", "../exp49/probe.py")
r42, r41, frozen = r48.r42, r48.r41, r48.frozen
SETS, RISK = r42.SETS, r42.RISK
SQA = {"sqa_temp": "data/exp21/dev_sqa_temp.jsonl", "sqa_force": "data/exp21/dev_sqa_force.jsonl",
       "sqa_temp_mism": "data/exp21/dev_sqa_temp_mism.jsonl", "sqa_force_mism": "data/exp21/dev_sqa_force_mism.jsonl"}
B_RUN, B_CKPT, BASE = "runs/exp50-t", "runs/exp50-t/stage2", "runs/exp52-base"
B_SHA = {"model.safetensors": "1a8e5439d5c4f64fd1dbce551f9c2df55152b2aea7cc3a93fc01eb5d364ca007",
         "config.json": "99c6246e201c5c38c7f62c6831050792858b5aecf2b4be352c5cb7cbc3231fc6"}
RUNS = {"N": "runs/exp52-n", "Z": "runs/exp52-z"}
WEIGHT = {"N": 1.0, "Z": 0.0}
NUMBERS, QUADS = "data/exp52/numbers.jsonl", "data/exp48/quads.jsonl"


def sha(p: str) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def arm_identity(arm: str, run: str) -> str:
    try:
        s = json.load(open(os.path.join(run, "START.json")))
        st = json.load(open(os.path.join(run, "run.json")))["config"]["stages"][0]
        tr = json.load(open(os.path.join(run, "stage2", "train_report.json")))
        cfg = json.load(open(os.path.join(run, "stage2", "config.json")))
    except (OSError, ValueError, KeyError) as e:
        return "incomplete: %s" % e
    par, num, lo = st.get("params") or {}, tr.get("number") or {}, tr.get("local") or {}
    if s.get("init") != B_CKPT or s.get("init_sha256") != B_SHA or s.get("number_weight") != WEIGHT[arm] \
            or s.get("numbers_sha256") != sha(NUMBERS) or s.get("data_sha256") != sha("data/exp42/train.jsonl"):
        return "wrong arm: START %s" % s
    if st.get("init") != B_CKPT or st.get("data") != "data/exp42/train.jsonl" or par.get("objective") != "ce" \
            or par.get("interaction_weight") != 1.0 or par.get("interaction_file") != QUADS or not par.get("number_reader") \
            or par.get("number_weight") != WEIGHT[arm] or par.get("number_file") != NUMBERS or par.get("vision_unfreeze_top") \
            or (par.get("local_views"), par.get("local_overlap"), par.get("local_source")) != (12, 0.25, "T"):
        return "wrong arm: run.json %s" % par
    if num.get("weight") != WEIGHT[arm] or num.get("file_sha256") != sha(NUMBERS) or lo.get("vision_trainable") != 0 \
            or (tr.get("interaction") or {}).get("weight") != 1.0 or not cfg.get("number_reader"):
        return "wrong arm: train_report %s %s" % (num, lo)
    return ""


def summary_ok(d: str, name: str, ckpt: str, data: str) -> None:
    info = json.load(open(os.path.join(d, "eval", name + ".json")))["run"]
    if os.path.normpath(info.get("checkpoint", "")) != os.path.normpath(ckpt) or info.get("temperature") != "none (raw)" \
            or info.get("control") != "none" or info.get("data_sha256") != sha(data):
        raise SystemExit("%s/%s: not %s at temperature 1, no control, on %s" % (d, name, ckpt, data))


def accuracy(d: str, name: str, ckpt: str, data: str) -> float:
    summary_ok(d, name, ckpt, data)
    got = r41.check_details("%s/%s" % (d, name), r42.load(data), r42.load(os.path.join(d, "eval", name + ".details.jsonl")))
    return sum(g["correct"] for g in got.values()) / len(got)


READOUT_SETS = {"fit_O": "data/exp42/fit_O.jsonl", "rl_tune": "data/real_layout/questions_tune.jsonl"}
READOUT_FILES = ("readout.json", "readout_details.jsonl", "readout_use.jsonl")


def readout_checked(run: str, weights: Dict):
    """(qs, pred, donor, use rows) of an arm's readout after every check (critic EXP52-R1-02): the three files recorded
    under the arm's frozen identity and unchanged; readout.json's two file sha256 and identity; both files' identity
    headers equal to it; the frozen weights, the question map and both source sets; the details EXP49-strict (every
    question once, targets, donor keys); the use file holding every expected question once, with the source question's
    options in order, its gold index and finite logits with / without the content. Any failure: SystemExit(reason)."""
    fr = frozen.check(run, ["readout"])                    # recorded, and the files unchanged since
    ev = os.path.join(run, "eval")
    info = json.load(open(os.path.join(ev, "readout.json")))
    ident = info["identity"]
    if info["details_sha256"] != sha(os.path.join(ev, "readout_details.jsonl")) or info["use_sha256"] != sha(os.path.join(ev, "readout_use.jsonl")):
        raise SystemExit("readout.json does not bind the details / use files")
    if {k: ident.get(k) for k in fr["weights"]} != fr["weights"] or fr["weights"] != weights:
        raise SystemExit("readout of other weights")
    if ident.get("questions_sha256") != sha("data/exp49/questions.jsonl") or ident.get("sets") != {n: sha(p) for n, p in READOUT_SETS.items()}:
        raise SystemExit("readout of other questions / source sets")
    heads = [json.loads(open(os.path.join(ev, f)).readline())["identity"] for f in READOUT_FILES[1:]]
    if any(h != ident for h in heads):
        raise SystemExit("a readout file's identity header differs from readout.json")
    qs = [q for q in map(json.loads, open("data/exp49/questions.jsonl")) if q["set"] in READOUT_SETS]
    pred, donor, _ = p49.read_details(os.path.join(ev, "readout_details.jsonl"), qs)
    for q, p in zip(qs, pred):
        if len(p) != p49.SLOTS or any(not 0 <= c < p49.CLASSES for c in p):
            raise SystemExit("%s: malformed character prediction" % q["key"])
    use = [json.loads(x) for x in open(os.path.join(ev, "readout_use.jsonl")) if x.strip()][1:]
    by = {}
    for u in use:
        if u["key"] in by:
            raise SystemExit("use: %s twice" % u["key"])
        by[u["key"]] = u
    if set(by) != {q["key"] for q in qs}:
        raise SystemExit("use: %d questions missing, %d unknown" % (len({q["key"] for q in qs} - set(by)), len(set(by) - {q["key"] for q in qs})))
    rows = {n: {r["id"]: r for r in map(json.loads, open(p))} for n, p in READOUT_SETS.items()}
    import math
    for q in qs:
        u, src = by[q["key"]], rows[q["set"]][q["id"]]
        opts = list(src["questions"][q["qid"]]["criteria"])
        probs = src["gold"][q["qid"]]["probabilities"]
        if u["options"] != opts or u["gold"] != opts.index(max(opts, key=lambda o: probs.get(o, 0.0))):
            raise SystemExit("use: %s options / gold differ from the source question" % q["key"])
        if len(u["on"]) != len(opts) or len(u["off"]) != len(opts) or not all(math.isfinite(v) for v in u["on"] + u["off"]):
            raise SystemExit("use: %s logits malformed" % q["key"])
    return qs, pred, donor, [by[q["key"]] for q in qs]


def readout_of(qs, pred, donor) -> Dict:
    """EXP51's number readout statistics on fit_O / rl_tune (no oracle; donor = the same question on the twin picture)."""
    saved = dict(p49.SET_NAME)
    p49.SET_NAME.clear()
    p49.SET_NAME.update({"fit_O": "fit_O", "rl_tune": "tune_O"})
    try:
        ro = p49.readout(qs, pred, donor, [p49.labels(q["target"]) for q in qs])
    finally:
        p49.SET_NAME.clear()
        p49.SET_NAME.update(saved)
    x = ro["tune_O"]
    ref = bool(x["exact"] >= 0.80 and all((x["exact:" + f]["value"] or 0) >= 0.60 for f in p49.FIELDS) and x["both"] >= 0.50
               and x["twin"] >= 0.50 and x["real_minus_donor"] >= 0.20 and x["real_minus_donor_ci"][0] > 0)
    return {"state": "completed", "readout": ro, "exp51_threshold_reference": ref, "note": "no oracle region: a harder task than EXP51"}


def use_of(qs, use) -> Dict:
    out: Dict = {}
    for setname in READOUT_SETS:
        sel = [u for q, u in zip(qs, use) if q["set"] == setname]
        on = [max(range(len(u["on"])), key=u["on"].__getitem__) for u in sel]
        off = [max(range(len(u["off"])), key=u["off"].__getitem__) for u in sel]
        out[setname] = {"n": len(sel), "accuracy_with": sum(o == u["gold"] for o, u in zip(on, sel)) / len(sel),
                        "accuracy_without": sum(o == u["gold"] for o, u in zip(off, sel)) / len(sel),
                        "changed": sum(a != b for a, b in zip(on, off)) / len(sel)}
    return out


def readout_evidence(run: str, weights: Dict, step_state) -> tuple:
    """(numbers, use) for the report: the checked readout, or {"state": missing / invalid, "reason"} for both -- never an
    exception that would drop the verified decisions."""
    if step_state != "completed":
        why = {"state": "missing", "reason": "readout step %s" % (step_state or "not run")}
        return why, why
    try:
        qs, pred, donor, use = readout_checked(run, weights)
        return readout_of(qs, pred, donor), {"state": "completed", **use_of(qs, use)}
    except (SystemExit, OSError, ValueError, KeyError, IndexError) as e:
        why = {"state": "invalid", "reason": str(e)}
        return why, why


def build() -> Dict:
    if r41.sha(QUADS) != json.load(open("data/exp48/MANIFEST.json"))["sha256"] or sha(NUMBERS) != json.load(open("data/exp52/MANIFEST.json"))["sha256"]:
        raise SystemExit("quads / numbers are not their MANIFEST's")
    if frozen.check(B_RUN, list(SETS) + list(RISK))["weights"] != B_SHA:
        raise SystemExit("EXP50-T is not its frozen identity")
    out: Dict = {"identity": {"B": B_SHA}, "runs": {}, "readout": {}, "quads": {}, "differences": {}, "risk": {}, "numbers": {},
                 "use": {}, "transfer": {}}
    src: Dict = {("B", n): (B_RUN, B_CKPT) for n in list(SETS) + list(RISK)}
    for arm, run in RUNS.items():
        why = arm_identity(arm, run) if os.path.exists(run) else "incomplete: not started"
        if why.startswith("wrong"):
            raise SystemExit("%s: %s" % (arm, why))
        st = r42.states(run + "-status.txt")
        if why or st.get("train") != "completed" or not os.path.exists(frozen.path(run)):
            out["runs"][arm] = why or "incomplete: training"
            continue
        names = [n for n in list(SETS) + list(RISK) + list(SQA) if st.get("eval:" + n) == "completed"]
        fr = frozen.check(run, names)
        out["identity"][arm] = fr["weights"]
        out["runs"][arm] = {"completed": names, "readout": st.get("readout"), "missing": [n for n in list(SETS) + list(RISK) + list(SQA) if n not in names]}
        for n in names:
            src[(arm, n)] = (run, os.path.join(run, "stage2"))
        out["numbers"][arm], out["use"][arm] = readout_evidence(run, fr["weights"],
                                                                "completed" if st.get("readout") == "completed" and st.get("readout_record") == "completed"
                                                                else st.get("readout_record") or st.get("readout"))
    recs: Dict = {}
    for (who, n), (d, ck) in src.items():
        if n in SETS:
            recs[(who, n)] = r48.records(d, n, ck)
        elif n in RISK:
            out["risk"].setdefault(n, {})[who] = accuracy(d, n, ck, RISK[n])
    base = json.load(open(os.path.join(BASE, "RECORDS.json"))) if os.path.exists(os.path.join(BASE, "RECORDS.json")) else {}
    for n, data in SQA.items():
        for who in ("N", "Z"):
            if (who, n) in src:
                out["transfer"].setdefault(n, {})[who] = accuracy(src[(who, n)][0], n, src[(who, n)][1], data)
        if n in base and all(sha(os.path.join(BASE, "eval", f)) == h for f, h in base[n].items()):
            out["transfer"].setdefault(n, {})["B"] = accuracy(BASE, n, B_CKPT, data)
    for n in ("temp", "force"):
        t, m = out["transfer"].get("sqa_" + n, {}), out["transfer"].get("sqa_%s_mism" % n, {})
        out["transfer"]["picture_gain_" + n] = {w: t[w] - m[w] for w in t if w in m}
    fams = {k: r42.units(v) for k, v in recs.items()}
    qfams = {k: r48.quad_units(k[1], v) for k, v in recs.items()}
    for (who, n), fam in fams.items():
        out["readout"].setdefault(n, {})[who] = r42.summary(fam)
        q = qfams[(who, n)]
        out["quads"].setdefault(n, {})[who] = {"macro": r48.quad_value(q["units"], sorted(q["units"])), "quads": q["quads"]}
    for n in SETS:
        for a, b in (("N", "B"), ("Z", "B"), ("N", "Z")):
            if (a, n) in fams and (b, n) in fams:
                ca, cb = r42.clusters(recs[(a, n)]), r42.clusters(recs[(b, n)])
                if ca != cb:
                    raise SystemExit("%s: %s and %s on different questions" % (n, a, b))
                d = {s: r42.diff(fams[(a, n)], fams[(b, n)], ca, s) for s in ("different", "both", "twin")}
                expand = lambda ks, ca=ca: [f for k in ks for f in sorted(ca[k])]  # noqa: E731
                qa, qb = qfams[(a, n)]["units"], qfams[(b, n)]["units"]
                d["quad"] = r41.interval(lambda fs, qa=qa, qb=qb: r48.quad_value(qa, fs) - r48.quad_value(qb, fs), sorted(ca), expand)
                out["differences"]["%s|%s-%s" % (n, a, b)] = d
    out["conditions"] = conditions(out)
    return out


def conditions(out: Dict) -> Dict:
    rd, d = out["readout"], out["differences"]
    res: Dict = {}
    for arm in ("N", "Z"):
        if not all(arm in rd.get(n, {}) and "B" in rd.get(n, {}) for n in SETS) or "rl_tune|%s-B" % arm not in d:
            res["progress_" + arm] = "incomplete"
            continue
        a, ab = rd["rl_tune"][arm], d["rl_tune|%s-B" % arm]
        res["progress_" + arm] = bool(a["different"] >= 0.60 and a["both"] >= 0.20 and ab["both"]["ci95"][0] > 0
                                      and a["twin"] >= 0.20 and ab["twin"]["ci95"][0] > 0)
    nz = d.get("rl_tune|N-Z")
    res["attribution"] = "incomplete" if nz is None or "incomplete" in (res["progress_N"], res["progress_Z"]) else \
        bool(nz["both"]["value"] >= 0.10 and nz["both"]["ci95"][0] > 0 and res["progress_N"] is True)
    notes: List[str] = []
    for arm, v in out["numbers"].items():
        if v.get("state") == "completed" and v["exp51_threshold_reference"] and res.get("progress_" + arm) is False:
            notes.append("%s: the number readout reaches EXP51's thresholds but the decision did not progress: the use by "
                         "the decision is not connected yet" % arm)
    res["notes"] = notes
    return res


def main() -> int:
    out = build()
    json.dump(out, open("runs/exp52-report.json", "w"), indent=1, default=str)
    print("identity %s\nruns %s" % (out["identity"], out["runs"]))
    for n, by in out["readout"].items():
        for who, s in by.items():
            print("%-9s %s different %.3f both %.3f twin %.3f same_answer %.3f quads %.3f" % (
                n, who, s["different"], s["both"], s["twin"], s["same_answer"], out["quads"][n][who]["macro"]))
    for k, v in out["differences"].items():
        print("%s %s" % (k, {s: "%+.3f [%+.3f, %+.3f]" % (x["value"], *x["ci95"]) for s, x in v.items()}))
    for arm, v in out["numbers"].items():
        if v.get("state") != "completed":
            print("numbers %s %s: %s" % (arm, v["state"], v["reason"]))
            continue
        x = v["readout"]["tune_O"]
        print("numbers %s tune_O exact %.3f %s both %.3f twin %.3f real-donor %+.3f %s (EXP51 thresholds as reference: %s)" % (
            arm, x["exact"], {f: round(x["exact:" + f]["value"] or 0, 3) for f in p49.FIELDS}, x["both"], x["twin"],
            x["real_minus_donor"], [round(c, 3) for c in x["real_minus_donor_ci"]], v["exp51_threshold_reference"]))
    print("use %s\ntransfer %s\nrisk %s\nconditions %s" % (out["use"], out["transfer"], out["risk"], out["conditions"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
