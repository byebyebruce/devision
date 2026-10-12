"""Apply the frozen rules (scripts/r13b/rules.json) of the v12 retrain plan to per-question records.

    # periodic stop rules: every monitor step of a run against its step 0 (v12 at temperature 1)
    uv run python scripts/r13b/judge.py monitor runs/r13b-v12-data/stage2
    # end-of-run requirements: candidate's devision-eval details against v12's (same calibration set)
    uv run python scripts/r13b/judge.py final runs/r13b-v12-data/eval runs/r13b-v12-base/stage2-recal \
        --bind-cf runs/r13b-v12-data/eval/compare/bind_cf.json
    # both arms passed: which one (9.1)
    uv run python scripts/r13b/judge.py pick runs/r13b-v12-data/eval runs/r13b-v12-s2/eval

Prints one line per rule with the numbers it used and writes the verdict next to the inputs as JSON. `monitor`
exits 3 when a stop rule is hit (the chain script stops that arm). Panels must match question for question
(devision.train.compare.paired_metric refuses otherwise).
"""
import argparse
import json
import os
import re
import sys

from devision.train.compare import load_records, paired_metric

RULES = os.path.join(os.path.dirname(__file__), "rules.json")


def holds(cond, r):
    d, (lo, hi) = r["difference"], r["ci95"]
    checks = {"difference_le": d <= cond.get("difference_le", 0), "difference_ge": d >= cond.get("difference_ge", 0),
              "difference_gt": d > cond.get("difference_gt", 0), "lo_gt": lo > cond.get("lo_gt", 0),
              "hi_lt": hi < cond.get("hi_lt", 0)}
    return all(checks[k] for k in cond)


def stat(rules, a_dir, b_dir, rule):
    a = load_records(os.path.join(a_dir, rule["set"] + ".details.jsonl"))
    b = load_records(os.path.join(b_dir, rule["set"] + ".details.jsonl"))
    return paired_metric(a, b, rule["metric"], sources=rules["ext_sources"] if rule["metric"] == "macro_accuracy" else None)


def fmt(r):
    return "%+.4f [%+.4f, %+.4f]  (%s %.4f -> %.4f)" % (r["difference"], r["ci95"][0], r["ci95"][1], r["metric"], r["a"], r["b"])


def monitor(rules, stage_dir):
    mon = os.path.join(stage_dir, "monitor")
    steps = sorted((int(m.group(1)), d) for d in os.listdir(mon) if (m := re.fullmatch(r"step(\d+)", d)))
    base = os.path.join(mon, "step0")
    out, streak, stop = [], {}, []
    for step, d in steps:
        if step == 0:
            continue
        for rule in rules["monitor"]:
            r = stat(rules, base, os.path.join(mon, d), rule)
            hit = holds(rule["stop_if"], r)
            streak[rule["id"]] = streak.get(rule["id"], 0) + 1 if hit else 0
            fired = streak[rule["id"]] >= rule.get("consecutive", 1)
            out.append({"step": step, "rule": rule["id"], **r, "hit": hit, "stop": fired})
            print("step %6d  %-20s %s%s" % (step, rule["id"], fmt(r), "  STOP" if fired else ("  (hit)" if hit else "")))
            if fired:
                stop.append((step, rule["id"]))
    json.dump({"rules": rules.get("_doc", ""), "checks": out, "stop": stop}, open(os.path.join(mon, "judge.json"), "w"), indent=1)
    return 3 if stop else 0


def final(rules, cand, base, bind_cf=None):
    out, ok = [], True
    for rule in rules["final"]:
        if "bind_cf" in rule:
            rep = json.load(open(bind_cf))
            r = dict(rep["models"][1]["vs_baseline"][rule["bind_cf"]], metric="bind_cf " + rule["bind_cf"],
                     a=rep["models"][0]["summary"][rule["bind_cf"]]["accuracy"],
                     b=rep["models"][1]["summary"][rule["bind_cf"]]["accuracy"])
        else:
            r = stat(rules, base, cand, rule)
        passed = holds(rule["require"], r) if "require" in rule else not holds(rule["fail_if"], r)
        ok &= passed
        out.append({"rule": rule["id"], **r, "pass": passed})
        print("%-20s %s  %s" % (rule["id"], fmt(r), "pass" if passed else "FAIL"))
    for rule in rules["report_only"]:
        r = stat(rules, base, cand, rule)
        out.append({"rule": "report:%s/%s" % (rule["set"], rule["metric"]), **r})
        print("%-20s %s" % ("report " + rule["set"], fmt(r)))
    print("verdict:", "pass" if ok else "fail")
    json.dump({"rules": rules.get("_doc", ""), "checks": out, "pass": ok}, open(os.path.join(cand, "judge_final.json"), "w"), indent=1)
    return 0 if ok else 1


def pick(rules, d_eval, s2_eval):
    rule = rules["pick"]
    r = stat(rules, d_eval, s2_eval, rule)
    choose = "s2" if holds(rule["choose_s2_if"], r) else "data"
    print("S2 - D %s -> %s" % (fmt(r), choose))
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rules", default=RULES, help="frozen rule file (default: the v12 retrain's)")
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("monitor"); m.add_argument("stage_dir")
    f = sub.add_parser("final"); f.add_argument("cand"); f.add_argument("base"); f.add_argument("--bind-cf", help="bind_report JSON, needed only by rules naming bind_cf")
    k = sub.add_parser("pick"); k.add_argument("d_eval"); k.add_argument("s2_eval")
    a = p.parse_args()
    rules = json.load(open(a.rules))
    if a.cmd == "monitor":
        sys.exit(monitor(rules, a.stage_dir))
    if a.cmd == "final":
        sys.exit(final(rules, a.cand, a.base, a.bind_cf))
    sys.exit(pick(rules, a.d_eval, a.s2_eval))


if __name__ == "__main__":
    main()
