"""exp55: the same frozen text-interface diagnostic as EXP54 with ONE change -- thousands separators in the fact tables'
values (critic EXP55-PREP-01..03; docs/experiments/2026-10-18-exp55-format-plan.md). EXP54's registered inputs are the
immutable source; P / G / D values go through fmt() (1360 -> 1,360; 14 and 0.75 unchanged; "unreadable" unchanged; any
value not a strict number refused), E is copied byte for byte. Everything else -- EXP50-T, EXP53's readings, the 600
original pictures and their questions / options / gold / state text, row order, units, request grouping, image=None,
raw temperature 1 -- is EXP54's. EXP54's details are the frozen paired baseline (read, never rewritten).

  start   model-free: identities (EXP54 START / registry / inputs / details, EXP50-T's weights, questions, this code), the
          new inputs (registered) with their round-trip / change counts, and a truncation check with the local
          tokenizer (no weights) -- refused when any new request loses state tokens or option distinctness.
  decide  EXP54's decide stage on runs/exp55 (same code path: weights verified before loading, P -> G -> E -> D).
  report  EXP54's statistics, gates and donor counts on the new details -- with EXP54's frozen read-right set (EXP53's
          1,470) and separator-free value matching -- plus the paired comparison with EXP54 (G speed new - old with a
          case-cluster interval, slices, transitions, NLL) and the reproduction control (E and every unchanged request:
          same predictions and probabilities within 1e-5).

    uv run python scripts/exp55/exp55.py start --deadline EPOCH | decide | report [--finalize CODES]
"""
import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List

HERE = os.path.dirname(__file__)
OUT, E54_OUT = "runs/exp55", "runs/exp54"
E54_START_SHA = "1098e403ebb9eaf7e852d8779cf1ee366f00b335ff963ab23afa7cbc71914a1e"
NUMBER = re.compile(r"^[0-9]+(\.[0-9]+)?$")
FILES = ["scripts/exp55/exp55.py", "scripts/exp55/run.sh", "scripts/exp54/exp54.py", "scripts/exp53/watch.py",
         "scripts/exp49/probe.py", "scripts/exp42/report.py", "scripts/exp41/report.py", "src/devision/infer.py", "src/devision/model.py"]

_spec = importlib.util.spec_from_file_location("exp54", os.path.join(HERE, "..", "exp54", "exp54.py"))
assert _spec is not None and _spec.loader is not None
e54 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e54)
r42, sha, write_json = e54.r42, e54.sha, e54.write_json
CONDS, UNREADABLE = e54.CONDS, e54.UNREADABLE


def fmt(value: str) -> str:
    """Thousands separators in the integer part, counted from the right; the fraction kept character for character;
    UNREADABLE unchanged. Only the value string is an input; anything not a strict number is refused."""
    if value == UNREADABLE:
        return value
    if not NUMBER.match(value):
        raise ValueError("not a strict number: %r" % value)
    whole, dot, frac = value.partition(".")
    groups = []
    while len(whole) > 3:
        groups.insert(0, whole[-3:])
        whole = whole[:-3]
    out = ",".join([whole] + groups) + dot + frac
    if out.replace(",", "") != value:
        raise ValueError("not reversible: %r -> %r" % (value, out))
    return out


def unfmt(value: str) -> str:
    return value.replace(",", "")


def new_inputs(old: Dict[str, Dict[str, Dict]], rows: Dict[str, Dict]) -> tuple:
    """(new inputs per condition, change counts): P / G / D tables reformatted value by value and their state rebuilt
    with EXP54's serialise (checked to reproduce the old state first); E copied unchanged."""
    out: Dict[str, Dict[str, Dict]] = {}
    counts = {}
    for c in CONDS:
        out[c] = {}
        req, vals = 0, 0
        for rid, inp in old[c].items():
            if c == "E":
                out[c][rid] = inp
                continue
            st = rows[rid].get("state_text", "")
            if e54.serialise(st, inp["table"]) != inp["state"]:
                raise SystemExit("%s %s: EXP54's state is not serialise(state_text, table)" % (c, rid))
            table = [[o, q, fmt(v)] for o, q, v in inp["table"]]
            if [unfmt(v) for _, _, v in table] != [v for _, _, v in inp["table"]]:
                raise SystemExit("%s %s: formatting is not reversible" % (c, rid))
            new = dict(inp, table=table, state=e54.serialise(st, table))
            n = sum(a[2] != b[2] for a, b in zip(table, inp["table"]))
            vals += n
            req += new["state"] != inp["state"]
            out[c][rid] = new
        counts[c] = {"requests_changed": req, "values_changed": vals, "requests": len(old[c])}
    return out, counts


def truncation(tok, cfg, inputs: Dict[str, Dict], rows: Dict[str, Dict]) -> Dict:
    """Per condition: questions whose state loses tokens, questions whose options are not all distinct (EXP50-T's
    tokenizer and sequence budget, no weights)."""
    from laya.common import build_sequence

    from devision.infer import _split_state, serialize_state
    n = {"questions": 0, "state_dropped": 0, "options_not_distinct": 0, "max_state_tokens": 0}
    for rid, inp in inputs.items():
        text_state, _ = _split_state(inp["state"])
        for q in rows[rid]["questions"].values():
            lq = {"t": q["type"], "ins": serialize_state(q["instructions"]), "crit": q.get("criteria")}
            res: Any = build_sequence(tok, text_state, lq, cfg.max_len, cfg.head_max_len, return_stats=True, return_truncation_stats=True)
            markers, stats, trunc = res[1], res[2], res[3]
            n["questions"] += 1
            n["state_dropped"] += trunc["state_tokens_dropped"] > 0
            n["options_not_distinct"] += stats.get("options_distinct", len(markers)) < len(markers)
            n["max_state_tokens"] = max(n["max_state_tokens"], trunc["state_tokens"])
    return n


def e54_bound() -> Dict:
    """EXP54's START, registry, inputs and details as registered (refused otherwise)."""
    if sha(os.path.join(E54_OUT, "START.json")) != E54_START_SHA:
        raise SystemExit("EXP54's START changed")
    reg = json.load(open(os.path.join(E54_OUT, "registry.json")))
    files = {}
    for name in ["inputs"] + ["decide_" + c for c in CONDS]:
        for f, h in reg[name].items():
            if sha(os.path.join(E54_OUT, f)) != h:
                raise SystemExit("EXP54's %s changed" % f)
            files[f] = h
    return files


def stage_start(deadline: int, out: str = OUT, check_tokens: bool = True) -> Dict:
    path = os.path.join(out, "START.json")
    if os.path.exists(path):
        raise SystemExit("%s exists (fresh only)" % path)
    e54_files = e54_bound()
    st54 = json.load(open(os.path.join(E54_OUT, "START.json")))
    model = e54.verified_model(st54)
    if sha(e54.QUESTIONS) != st54["questions_sha256"] or sha(e54.TUNE) != st54["tune_sha256"]:
        raise SystemExit("questions / rl_tune are not those EXP54 bound")
    rows = {r["id"]: r for r in map(json.loads, open(e54.TUNE))}
    old = {c: e54.load_inputs(E54_OUT, c) for c in CONDS}
    new, counts = new_inputs(old, rows)
    trunc = {}
    if check_tokens:
        from transformers import AutoTokenizer

        from devision.model import ModelConfig
        tok = AutoTokenizer.from_pretrained(os.path.join(e54.RUN, "stage2", "tokenizer"))
        cfg = ModelConfig.from_dict(json.load(open(os.path.join(e54.RUN, "stage2", "config.json"))))
        trunc = {c: {"exp55": truncation(tok, cfg, new[c], rows), "exp54": truncation(tok, cfg, old[c], rows)} for c in CONDS}
        bad = {c: v["exp55"] for c, v in trunc.items() if v["exp55"]["state_dropped"] or v["exp55"]["options_not_distinct"]}
        if bad:
            raise SystemExit("new inputs lose state tokens / option distinctness: %s" % bad)
    os.makedirs(out, exist_ok=True)
    files = {}
    for c in CONDS:
        f = "inputs_%s.jsonl" % c
        with open(os.path.join(out, f), "w") as fh:
            for rid in sorted(new[c]):
                rec = {"id": rid, **{k: v for k, v in new[c][rid].items() if k != "id"}}
                fh.write(json.dumps(rec) + "\n")
        files[f] = sha(os.path.join(out, f))
    if files["inputs_E.jsonl"] != e54_files["inputs_E.jsonl"]:
        raise SystemExit("E is not byte-identical to EXP54's")
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    s = {"git_head": git("rev-parse", "HEAD"), "dirty_files": [f for f in FILES if git("status", "--porcelain", "--", f)],
         "files": {f: sha(f) for f in FILES}, "model": model, "temperature": "none (raw), in memory", "image": None,
         "max_questions_per_request": 5, "exp54": {"start_sha256": E54_START_SHA, "files": e54_files},
         "questions_sha256": st54["questions_sha256"], "tune_sha256": st54["tune_sha256"], "inputs": files, "changes": counts,
         "truncation": trunc, "format": "fmt(): thousands separators in the integer part; fraction and 'unreadable' unchanged",
         "conditions": list(CONDS), "started": time.strftime("%F %T"), "deadline": deadline,
         "deadline_text": time.strftime("%F %T", time.localtime(deadline))}
    write_json(path, s)
    e54.register(out, "inputs", files)
    return s


# ------------------------------------------------------------------ report

def plain_inputs(out: str, c: str) -> Dict[str, Dict]:
    """The condition's registered inputs with the separators removed (for value matching in the report only)."""
    inp = e54.load_inputs(out, c)
    return {rid: dict(v, table=[[o, q, unfmt(x)] for o, q, x in v["table"]]) for rid, v in inp.items()}


def paired(new: List[Dict], old: List[Dict], rows: Dict[str, Dict], changed_requests: set) -> Dict:
    """Reproduction control on requests whose input is unchanged, transitions and NLL; both lists question-aligned."""
    key = lambda r: (r["sample_id"], r["qid"])  # noqa: E731
    o = {key(r): r for r in old}
    if set(o) != {key(r) for r in new} or len(o) != len(new):
        raise SystemExit("new and EXP54 details are not on the same questions")
    same = [r for r in new if r["sample_id"] not in changed_requests]
    maxdiff = max((abs(r["probabilities"][k] - o[key(r)]["probabilities"][k]) for r in same for k in r["probabilities"]), default=0.0)
    agree = all(r["prediction"] == o[key(r)]["prediction"] for r in same)
    t = {"wrong_to_right": 0, "right_to_wrong": 0}
    for r in new:
        a, b = o[key(r)]["correct"], r["correct"]
        t["wrong_to_right"] += (not a) and b
        t["right_to_wrong"] += a and not b
    return {"unchanged_questions": len(same), "max_probability_difference": maxdiff, "same_predictions": agree,
            "reproduced": bool(agree and maxdiff <= 1e-5), "transitions": t,
            "nll_new": sum(r["nll"] for r in new) / len(new), "nll_old": sum(r["nll"] for r in old) / len(old)}


def speed_comparison(new_reading: List[Dict], old_reading: List[Dict], rows, changed_requests: set) -> Dict:
    """G (or P) speed questions, new - old accuracy with a case-cluster interval, and its slices."""
    sp_new = [r for r in new_reading if r["field"] == "speed"]
    sp_old = [r for r in old_reading if r["field"] == "speed"]
    fn, fo = e54.units(sp_new), e54.units(sp_old)
    cl = r42.clusters(sp_new)
    if cl != r42.clusters(sp_old):
        raise SystemExit("speed questions differ between EXP55 and EXP54")
    d = r42.diff(fn, fo, cl, "all:speed")
    o = {r["key"]: r for r in sp_old}

    def acc(rs):
        return {"n": len(rs), "new": sum(r["right"] for r in rs) / len(rs) if rs else None,
                "old": sum(o[r["key"]]["right"] for r in rs) / len(rs) if rs else None}
    return {"n": len(sp_new), "new_minus_old": d, "gold_with_comma": acc([r for r in sp_new if "," in r["gold"]]),
            "gold_without_comma": acc([r for r in sp_new if "," not in r["gold"]]),
            "request_changed": acc([r for r in sp_new if r["key"].split("|")[1] in changed_requests]),
            "request_unchanged": acc([r for r in sp_new if r["key"].split("|")[1] not in changed_requests])}


def final_branch(base: str, reproduced: bool) -> str:
    """EXP54's pre-written branch, prefixed when the reproduction control failed (no single-variable attribution)."""
    if reproduced or base.startswith("incomplete"):
        return base
    return "control not reproduced: no single-variable format attribution; " + base


def report(out: str = OUT, finalize: str = "") -> Dict:
    st = json.load(open(os.path.join(out, "START.json"))) if os.path.exists(os.path.join(out, "START.json")) else None
    if st is not None and (sha(e54.QUESTIONS) != st["questions_sha256"] or sha(e54.TUNE) != st["tune_sha256"]):
        raise SystemExit("questions / rl_tune differ from START: no report")
    if st is not None and e54_bound() != st["exp54"]["files"]:
        raise SystemExit("EXP54's paired files differ from those START bound")
    rows = {r["id"]: r for r in map(json.loads, open(e54.TUNE))}
    qs53 = {q["key"]: q for q in map(json.loads, open(e54.QUESTIONS)) if q["set"] == "rl_tune"}
    res: Dict = {"states": {}, "conditions": {}, "differences": {}, "paired": {}}
    recs, reading, old_recs, old_reading = {}, {}, {}, {}
    for c in CONDS:
        try:
            recs[c] = e54.read_condition(out, c, rows)
            reading[c] = e54.reading_records(recs[c], rows, qs53)
            res["states"][c] = {"state": "completed"}
        except (SystemExit, OSError, ValueError, KeyError) as e:
            res["states"][c] = {"state": "missing" if not os.path.exists(os.path.join(out, "details_%s.jsonl" % c)) else "invalid", "reason": str(e)}
            continue
        old_recs[c] = e54.read_condition(E54_OUT, c, rows)
        old_reading[c] = e54.reading_records(old_recs[c], rows, qs53)
    pin = e54.load_inputs(E54_OUT, "P")            # EXP54's (= EXP53's) frozen readings: the read-right set (1,470)
    p_right = {}
    for k, q in qs53.items():
        tab = {(o, qn): v for o, qn, v in pin[q["id"]]["table"]}
        p_right[k] = tab[e54.TEMPLATES[rows[q["id"]]["questions"][q["qid"]]["instructions"]]] == q["target"]
    changed = {c: {rid for rid, v in e54.load_inputs(out, c).items() if v["state"] != e54.load_inputs(E54_OUT, c)[rid]["state"]}
               for c in CONDS if st is not None}
    fams = {c: e54.units(r) for c, r in reading.items()}
    for c, fam in fams.items():
        rr = reading[c]
        s = e54.summary(fam)
        s["nll"] = sum(r["nll"] for r in rr) / len(rr)
        ok = [r for r in rr if p_right[r["key"]]]
        s["read_right"] = {"n": len(ok), "accuracy": sum(r["right"] for r in ok) / len(ok)}
        s["read_wrong"] = {"n": len(rr) - len(ok), "accuracy": sum(r["right"] for r in rr if not p_right[r["key"]]) / max(1, len(rr) - len(ok))}
        res["conditions"][c] = s
        res["paired"][c] = paired(recs[c], old_recs[c], rows, changed.get(c, set()))
        if c in ("G", "P"):
            res["paired"][c]["speed"] = speed_comparison(reading[c], old_reading[c], rows, changed.get(c, set()))
        cmp_ = defaultdict(lambda: [0, 0])
        dist = defaultdict(lambda: defaultdict(int))
        for r in recs[c]:
            row = rows[r["sample_id"]]
            if e54.b40.field_of(row["questions"][r["qid"]]["instructions"]) == "cmp":
                k = "%s|%s" % (row.get("kind", ""), row.get("axis", ""))
                cmp_[k][0] += r["correct"]
                cmp_[k][1] += 1
                dist[row.get("kind", "")][r["prediction"]] += 1
        s["comparison"] = {k: {"right": v[0], "of": v[1]} for k, v in sorted(cmp_.items())}
        s["comparison_predictions"] = {k: dict(v) for k, v in dist.items()}
    if "D" in reading:
        res["donor"] = e54.donor_stats(reading["D"], rows, qs53, plain_inputs(out, "D"))
    for a, b, stats in (("P", "E", ("both", "different", "all")), ("P", "D", ("changed", "all")), ("G", "P", ("all", "both"))):
        if a in reading and b in reading:
            ca = r42.clusters(reading[a])
            res["differences"]["%s-%s" % (a, b)] = {s: r42.diff(fams[a], fams[b], ca, s) for s in stats}
    res["gates"] = e54.gates(res)
    rep_ok = all(res["paired"].get(c, {}).get("reproduced") for c in CONDS) and len(res["paired"]) == len(CONDS)
    res["reproduction_control"] = rep_ok
    res["branch"] = final_branch(e54.branch(res), rep_ok)
    g = res["paired"].get("G", {}).get("speed", {}).get("new_minus_old")
    res["format_effect_G_speed"] = None if g is None else {"value": g["value"], "ci95": g["ci95"], "supported": g["ci95"][0] > 0 and rep_ok}
    if finalize:
        res["finalized"] = {"exit_codes": finalize, "note": "the run did not complete; missing conditions are not low scores"}
    write_json(os.path.join(out, "report.json"), res)
    lines = ["INCOMPLETE run: %s" % finalize] if finalize else []
    for c, v in res["states"].items():
        if v["state"] != "completed":
            lines.append("%s %s: %s" % (c, v["state"], v["reason"]))
    for c, s in res["conditions"].items():
        lines.append("%s all %.3f (%s) changed %.3f both %.3f twin %.3f nll %.3f read_right %s" % (
            c, s["all"], {f: round(s["all:" + f], 3) for f in r42.FIELDS}, s["changed"], s["both"], s["twin"], s["nll"], s["read_right"]))
        p = res["paired"][c]
        lines.append("%s paired: reproduced %s (unchanged %d, max diff %.2e), transitions %s, nll %.3f -> %.3f%s" % (
            c, p["reproduced"], p["unchanged_questions"], p["max_probability_difference"], p["transitions"], p["nll_old"], p["nll_new"],
            " speed %s" % p["speed"] if "speed" in p else ""))
        lines.append("%s comparison %s" % (c, s["comparison"]))
    for k, v in res["differences"].items():
        lines.append("%s %s" % (k, {s: "%+.3f [%+.3f, %+.3f]" % (x["value"], *x["ci95"]) for s, x in v.items()}))
    lines.append("donor %s" % res.get("donor"))
    lines.append("format effect (G speed new - old) %s; reproduction control %s" % (res["format_effect_G_speed"], rep_ok))
    lines.append("gates %s\nbranch: %s" % (res["gates"], res["branch"]))
    open(os.path.join(out, "report.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["start", "decide", "report"])
    ap.add_argument("--deadline", type=int, default=0)
    ap.add_argument("--finalize", default="")
    a = ap.parse_args(argv)
    if a.stage == "start":
        print(json.dumps({k: v for k, v in stage_start(a.deadline).items() if k != "files"}, indent=1))
    elif a.stage == "decide":
        print(json.dumps(e54.stage_decide(OUT), indent=1))
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
