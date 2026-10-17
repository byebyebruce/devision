"""exp28 readout (docs/experiments/2026-10-17-exp28-diagnose-plan.md, rules written before running). Models:
v03 (runs/r15-vision/stage2), exp24A (runs/exp24-align/stage1), exp26A (runs/exp26-align/stage1), exp26B (exp27's
step-5000 export). v03 / exp26B stage-set records are exp27's (runs/exp27-transfer/eval/{v03,exp26}); the rest
are exp28's (runs/exp28-diagnose/eval/<model>).

    uv run python scripts/exp28/report.py      # -> runs/exp28-diagnose/report.json, text on stdout
"""
import json
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "exp21"))
from readout import bootstrap, ratio, transfer  # type: ignore  # noqa: E402

from devision.infer import config_path  # noqa: E402
from devision.model import ModelConfig  # noqa: E402
from devision.train.compare import load_records, paired_metric  # noqa: E402
from laya.common import QTYPES  # noqa: E402

RUN, EXP27 = "runs/exp28-diagnose", "runs/exp27-transfer"
STAGES = ("v03", "exp24A", "exp26A", "exp26B")
GUARD = ("exist", "science", "ext", "count_wide", "count_narrow", "flip", "vsr")
EXT_SOURCES = json.load(open(os.path.join(os.path.dirname(__file__), "..", "r14b", "rules.json")))["ext_sources"]
CONTROL_MONITOR = "runs/exp26-read-compare/stage2/monitor/step5000/dev16.details.jsonl"
EXPORT = "runs/exp27-transfer/exp26-step5000"
TIE = 1e-3             # top-2 probability margin (temperature 1) below which a disagreement is a tie
CONTROL_MAX_MISS = 2   # positive control: at most 2 of 90 non-tie top-1 disagreements
FIELD_READS, FIELD_FAILS = 0.80, 0.60
RULE_RECOVERS, RULE_PLAIN_MAX, RULE_FAILS = 9, 4, 5   # out of 12
WORDING_DROP = 0.25


def path(stage, name):
    if name in ("fields", "force_rule", "equal_wording", "control30") or stage in ("exp24A", "exp26A"):
        return os.path.join(RUN, "eval", stage, name + ".details.jsonl")
    return os.path.join(EXP27, "eval", {"v03": "v03", "exp26B": "exp26"}[stage], name + ".details.jsonl")


def recs(stage, name):
    rs = load_records(path(stage, name))
    if name == "flip":
        rs = [dict(r, image_id=r["image_id"].removesuffix(":flip")) for r in rs]
    return rs


def is_same(option: str) -> bool:
    o = option.lower()
    return o.startswith("neither") or "equal" in o or "the same" in o


def share(rs, pred=lambda r: True):
    rs = [r for r in rs if pred(r)]
    return sum(is_same(r["prediction"]) for r in rs) / len(rs) if rs else None


def acc(rs):
    return sum(bool(r["correct"]) for r in rs) / len(rs) if rs else None


def stage_table() -> dict:
    out = {}
    for st in STAGES:
        row: dict = {k: acc(recs(st, k)) for k in ("sqa_temp", "sqa_force", "sqa_temp_mism", "sqa_force_mism", "sqa_conc", "sqa_repel")}
        row["same_share_temp"], row["same_share_conc"] = share(recs(st, "sqa_temp")), share(recs(st, "sqa_conc"))
        row["guard"] = {p: acc(recs(st, p)) for p in GUARD}
        if st != "v03":
            real = lambda s: {k: recs(s, "sqa_" + k) for k in ("temp", "force", "temp_mism", "force_mism")}  # noqa: E731
            t = transfer(real("v03"), real(st))
            row["vs_v03"] = {"accuracy_change": t["accuracy_change"], "gain_change": t["gain_change"]}
            costs, diffs = [], {}
            for p in GUARD:
                a, b = recs("v03", p), recs(st, p)
                m = paired_metric(a, b, "macro_accuracy", sources=EXT_SOURCES) if p == "ext" else paired_metric(a, b, "accuracy")
                diffs[p] = {"difference": m["difference"], "ci95": m["ci95"]}
                if m["difference"] * 100 <= -3 and m["ci95"][1] < 0:
                    costs.append(p)
            row["guard_vs_v03"], row["costs"] = diffs, costs
        out[st] = row
    first = lambda f: next((s for s in STAGES[1:] if f(out[s])), None)  # noqa: E731
    out["first_cost_stage"] = first(lambda r: len(r["costs"]) >= 2)
    out["first_same_stage"] = first(lambda r: max(r["same_share_temp"], r["same_share_conc"]) >= 0.5)
    return out


def control() -> dict:
    cfg = ModelConfig.from_dict(json.load(open(config_path(EXPORT))))
    mon = {(r["sample_id"], r["qid"]): r for r in load_records(CONTROL_MONITOR)}
    got = load_records(path("exp26B", "control30"))
    miss, ties, worst = [], 0, 0.0
    for r in got:
        m = mon[(r["sample_id"], r["qid"])]
        probs = r["probabilities"]
        t = cfg.temperature_for(QTYPES[r["type"]], len(probs))
        z = {k: t * math.log(max(v, 1e-12)) for k, v in probs.items()}       # decide's logits / t, times t
        mx = max(z.values())
        e = {k: math.exp(v - mx) for k, v in z.items()}
        total = sum(e.values())
        p1: dict = {k: v / total for k, v in e.items()}
        worst = max(worst, max(abs(p1[k] - m["probabilities"][k]) for k in p1))
        top = sorted(m["probabilities"].values(), reverse=True)
        if max(p1, key=lambda k: p1[k]) != m["prediction"]:
            if top[0] - top[1] < TIE:
                ties += 1
            else:
                miss.append(r["sample_id"] + "/" + r["qid"])
    return {"questions": len(got), "disagreements": len(miss), "ties": ties, "max_prob_diff_t1": worst,
            "examples": miss[:5], "pass": len(miss) <= CONTROL_MAX_MISS}


def fields(stage) -> dict:
    rs = recs(stage, "fields")
    out = {}
    for kind in ("temp-mass", "temp-speed", "force-distance"):
        sel = [r for r in rs if r["source"] == "exp28-" + kind]
        other = 0
        for r in sel:
            if not r["correct"]:
                other += _other_value(r)
        out[kind] = {"n": len(sel), "accuracy": acc(sel), "wrong": sum(not r["correct"] for r in sel),
                     "wrong_other_object": other}
    temp = [r for r in rs if r["source"].startswith("exp28-temp")]
    out["temp_all"] = {"n": len(temp), "accuracy": acc(temp)}
    return out


_OTHER = {}


def _other_value(r) -> int:
    """1 when the wrong pick is the other object's value of the same field (from the field set itself)."""
    if not _OTHER:
        for s in map(json.loads, open("data/exp28/fields.jsonl")):
            sid, field, obj = s["id"].rsplit(":", 2)
            _OTHER[(sid, field, obj)] = next(k for k, v in s["gold"]["q"]["probabilities"].items() if v == 1.0)
    sid, field, obj = r["sample_id"].rsplit(":", 2)
    alt = {"A": "B", "B": "A", "1": "2", "2": "1"}[obj]
    return int(_OTHER.get((sid, field, alt)) == r["prediction"])


def rule(stage) -> dict:
    plain = [r for r in recs(stage, "sqa_text") if r["sample_id"] in _force_ids()]
    ruled = recs(stage, "force_rule")
    return {"plain": sum(r["correct"] for r in plain), "rule": sum(r["correct"] for r in ruled), "n": len(ruled)}


def _force_ids():
    return {json.loads(x)["id"].removeprefix("exp28-rule:") for x in open("data/exp28/force_rule.jsonl")}


def wording(stage) -> dict:
    rs = recs(stage, "equal_wording")
    by = defaultdict(dict)   # (question, order) -> wording -> record
    for r in rs:
        base, w, o = r["sample_id"].rsplit(":", 2)
        by[(base, o)][w] = r
    out = {}
    for w in ("w0", "w1"):
        sel = [r for r in rs if r["sample_id"].rsplit(":", 2)[1] == w]
        out[w] = {"accuracy": acc(sel), "same_share": share(sel),
                  "by_order": {o: acc([r for r in sel if r["sample_id"].endswith(":" + o)]) for o in ("o0", "o1", "o2")}}
    units = defaultdict(lambda: [(0.0, 0.0), (0.0, 0.0)])
    for (base, o), pair in by.items():
        img = pair["w0"]["image_id"]
        c = units[img]
        c[0] = (c[0][0] + float(pair["w0"]["correct"]), c[0][1] + 1)
        c[1] = (c[1][0] + float(pair["w1"]["correct"]), c[1][1] + 1)
    out["accuracy_change"] = bootstrap(dict(units), lambda c: c[1][0] / c[1][1] - c[0][0] / c[0][1])
    out["same_drop"] = out["w0"]["same_share"] - out["w1"]["same_share"]
    out["template_dependence"] = out["same_drop"] >= WORDING_DROP and out["accuracy_change"]["ci95"][0] > 0
    return out


def decide(out: dict) -> dict:
    """The next fix, by the priority written before running: pipeline, temperature extraction, magnet rule,
    wording; old-ability replay placement always goes with it."""
    f, r, w = out["fields"]["exp26B"]["temp_all"]["accuracy"], out["rule"]["exp26B"], out["wording"]["exp26B"]
    if not out["control"]["pass"]:
        nxt = "fix the export / inference path first: the positive control does not reproduce step 5000"
    elif f <= FIELD_FAILS:
        nxt = "real-layout reading / attribute-binding bridge (temperature fields not read from real pictures)"
    elif r["rule"] >= RULE_RECOVERS and r["plain"] <= RULE_PLAIN_MAX:
        nxt = "relation-rule and direction training for force (recovers with the general rule)"
    elif w["template_dependence"]:
        nxt = "answer-wording consistency (the 'same' option's wording changes the answers)"
    elif f >= FIELD_READS:
        nxt = "field-to-decision combination on real layouts (fields are read, the question is not answered)"
    else:
        nxt = "uncertain: no branch is clearly indicated; report the evidence"
    stage = out["stages"]["first_cost_stage"]
    replay = ("replay real tasks in the alignment stage (old abilities already drop after %s)" % stage
              if stage in ("exp24A", "exp26A") else
              "replay real tasks in the decision stage B (old abilities drop only after B)" if stage == "exp26B"
              else "no stage shows >= 2 costs")
    magnet = ("minimal text-task diagnosis for force first (the rule does not help)" if r["rule"] <= RULE_FAILS else None)
    return {"next_fix": nxt, "replay": replay, "force_note": magnet}


def build() -> dict:
    out = {"stages": stage_table(), "control": control(),
           "fields": {s: fields(s) for s in ("v03", "exp26B")}, "rule": {s: rule(s) for s in ("v03", "exp26B")},
           "wording": {s: wording(s) for s in ("v03", "exp26B")}}
    out["decision"] = decide(out)
    return out


def render(out: dict) -> str:
    f = lambda x: "  -  " if x is None else "%.3f" % x  # noqa: E731
    lines = ["stage            temp  force  temp_m force_m conc  repel same_t same_c | " + " ".join("%-6s" % p[:6] for p in GUARD)]
    for st in STAGES:
        r = out["stages"][st]
        lines.append("%-15s %s %s %s %s %s %s %s %s | %s" % (
            st, f(r["sqa_temp"]), f(r["sqa_force"]), f(r["sqa_temp_mism"]), f(r["sqa_force_mism"]), f(r["sqa_conc"]),
            f(r["sqa_repel"]), f(r["same_share_temp"]), f(r["same_share_conc"]), " ".join(f(r["guard"][p]) + " " for p in GUARD)))
    for st in STAGES[1:]:
        r = out["stages"][st]
        lines.append("%s vs v0.3: transfer accuracy %+.3f [%+.3f, %+.3f], gain %+.3f; costs %s" % (
            st, r["vs_v03"]["accuracy_change"]["value"], *r["vs_v03"]["accuracy_change"]["ci95"],
            r["vs_v03"]["gain_change"]["value"], r["costs"] or "none"))
    lines.append("first stage with >= 2 costs: %s | first stage choosing 'same' >= 0.5: %s" % (
        out["stages"]["first_cost_stage"], out["stages"]["first_same_stage"]))
    c = out["control"]
    lines.append("control30: %d questions, %d non-tie disagreements, %d ties, max |p| diff at T=1 %.4f -> %s" % (
        c["questions"], c["disagreements"], c["ties"], c["max_prob_diff_t1"], "pass" if c["pass"] else "FAIL"))
    for s in ("v03", "exp26B"):
        fd = out["fields"][s]
        lines.append("fields %-6s " % s + " | ".join("%s %s (%d wrong, %d other object)" % (
            k, f(fd[k]["accuracy"]), fd[k]["wrong"], fd[k]["wrong_other_object"]) for k in ("temp-mass", "temp-speed", "force-distance"))
            + " | temperature all %s" % f(fd["temp_all"]["accuracy"]))
        r = out["rule"][s]
        lines.append("force rule %-6s plain text %d / %d, with the rule %d / %d" % (s, r["plain"], r["n"], r["rule"], r["n"]))
        w = out["wording"][s]
        lines.append("wording %-6s original acc %s same %s | rewrite acc %s same %s | change %+.3f [%+.3f, %+.3f] | by order %s / %s" % (
            s, f(w["w0"]["accuracy"]), f(w["w0"]["same_share"]), f(w["w1"]["accuracy"]), f(w["w1"]["same_share"]),
            w["accuracy_change"]["value"], *w["accuracy_change"]["ci95"],
            ",".join(f(v) for v in w["w0"]["by_order"].values()), ",".join(f(v) for v in w["w1"]["by_order"].values())))
    d = out["decision"]
    lines += ["next fix: " + d["next_fix"], "replay: " + d["replay"]] + (["force: " + d["force_note"]] if d["force_note"] else [])
    return "\n".join(lines)


def main():
    out = build()
    json.dump(out, open(os.path.join(RUN, "report.json"), "w"), indent=1)
    print(render(out))


if __name__ == "__main__":
    main()
