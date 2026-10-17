"""exp32 readout (docs/experiments/2026-10-17-exp32-bridge-plan.md, rules written before running): the bridge
checkpoint (runs/exp32-fit/stage2, devision-eval with its fitted temperatures) against v0.3 on the same questions.

  - primary (the v0.4 delivery target, docs/experiments/2026-10-17-v04-delivery-plan.md): ScienceQA temperature +
    force dev accuracy change, 68 : 114, >= +10 points with the lower bound > 0, and the picture-gain change lower
    bound > 0 (exp21's readout.transfer, v0.3 in place of step 0);
  - protection: the 7 panels (a cost when <= -3 points with the upper bound < 0; ext 13-source macro, flip with
    its originals) -- no cost found is not equivalence;
  - report: the real-layout tune and confirm questions (reading by field, the ScienceQA-form question) against
    v0.3; dev16 synthetic reading (exp26's pass rule) as a regression check; concentration / attract-repel;
    the transcribed / blank questions.

    uv run python scripts/exp32/report.py
"""
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "exp21"))
from readout import classify, syn_index, syn_stat, transfer  # type: ignore  # noqa: E402

from devision.train.compare import load_records, paired_metric  # noqa: E402

RUN = "runs/exp32-fit"
V03 = "runs/exp27-transfer/eval/v03"
GUARD = ("exist", "science", "ext", "count_wide", "count_narrow", "flip", "vsr")
EXT_SOURCES = json.load(open(os.path.join(os.path.dirname(__file__), "..", "r14b", "rules.json")))["ext_sources"]


def recs(d, name):
    rs = load_records(os.path.join(d, name + ".details.jsonl"))
    return [dict(r, image_id=r["image_id"].removesuffix(":flip")) for r in rs] if name == "flip" else rs


def by_kind(rs, data_path):
    kinds = {}
    for s in map(json.loads, open(data_path)):
        for qid, q in s["questions"].items():
            ins = q["instructions"]
            kinds[(s["id"], qid)] = ("mass" if "mass" in ins else "speed" if "speed" in ins else
                                     "distance" if "distance" in ins else s["kind"] + "-question")
    out = defaultdict(list)
    for r in rs:
        out[kinds[(r["sample_id"], r["qid"])]].append(r["correct"])
    return {k: sum(v) / len(v) for k, v in out.items()}


def build(run: str = RUN) -> dict:
    ev = os.path.join(run, "eval")
    real = lambda d: {k: recs(d, "sqa_" + k) for k in ("temp", "force", "temp_mism", "force_mism")}  # noqa: E731
    out: dict = {"transfer": transfer(real(V03), real(ev))}
    t = out["transfer"]
    out["target_met"] = (t["accuracy_change"]["value"] >= 0.10 and t["accuracy_change"]["ci95"][0] > 0
                         and t["gain_change"]["ci95"][0] > 0)
    out["guard"], out["costs"] = {}, []
    for p in GUARD:
        a, b = recs(V03, p), recs(ev, p)
        m = paired_metric(a, b, "macro_accuracy", sources=EXT_SOURCES) if p == "ext" else paired_metric(a, b, "accuracy")
        out["guard"][p] = {"difference": m["difference"], "ci95": m["ci95"]}
        if m["difference"] * 100 <= -3 and m["ci95"][1] < 0:
            out["costs"].append(p)
    out["real_layout"] = {}
    for split in ("rl_tune", "rl_confirm"):
        a, b = recs(os.path.join(run, "eval_v03"), split), recs(ev, split)
        path = "data/real_layout/questions_%s.jsonl" % split.split("_")[1]
        out["real_layout"][split] = {"v03": by_kind(a, path), "exp32": by_kind(b, path),
                                     "change": paired_metric(a, b, "accuracy")}
    d = syn_index(recs(ev, "dev16"))
    out["dev16"] = {"read": syn_stat(d, 16, "read"), "cmp": syn_stat(d, 16, "cmp")}
    out["dev16"]["read_class"] = classify(out["dev16"]["read"])
    out["report_only"] = {k: {"v03": sum(r["correct"] for r in recs(V03, k)) / len(recs(V03, k)),
                              "exp32": sum(r["correct"] for r in recs(ev, k)) / len(recs(ev, k))}
                          for k in ("sqa_conc", "sqa_repel", "sqa_text", "sqa_blank")}
    if out["target_met"] and not out["costs"]:
        out["outcome"] = "delivery target met with no pre-written protection cost: candidate for v0.4 validation"
    elif out["target_met"]:
        out["outcome"] = "delivery target met, protection costs remain"
    elif t["accuracy_change"]["ci95"][0] > 0:
        out["outcome"] = "improves on v0.3 but below the delivery target"
    else:
        out["outcome"] = "no improvement on v0.3 shown"
    return out


def main():
    out = build()
    json.dump(out, open(os.path.join(RUN, "report.json"), "w"), indent=1)
    t = out["transfer"]
    pct = lambda d: "%+.1f [%+.1f, %+.1f]" % (100 * d["value"], 100 * d["ci95"][0], 100 * d["ci95"][1])  # noqa: E731
    print("transfer vs v0.3 (68:114): accuracy %s, picture gain %s; temp %s, force %s" % (
        pct(t["accuracy_change"]), pct(t["gain_change"]), pct(t["temp"]["accuracy_change"]), pct(t["force"]["accuracy_change"])))
    for p, g in out["guard"].items():
        print("guard %-12s %+.2f [%+.2f, %+.2f]" % (p, 100 * g["difference"], 100 * g["ci95"][0], 100 * g["ci95"][1]))
    print("costs: %s" % (out["costs"] or "none"))
    for s, r in out["real_layout"].items():
        c = r["change"]
        print("%s: v0.3 %s | exp32 %s | change %+.1f [%+.1f, %+.1f]" % (
            s, {k: round(v, 3) for k, v in r["v03"].items()}, {k: round(v, 3) for k, v in r["exp32"].items()},
            100 * c["difference"], 100 * c["ci95"][0], 100 * c["ci95"][1]))
    print("dev16 read %.3f (%s), compare %.3f" % (out["dev16"]["read"]["value"], out["dev16"]["read_class"], out["dev16"]["cmp"]["value"]))
    print("report only: %s" % {k: {m: round(v, 3) for m, v in d.items()} for k, d in out["report_only"].items()})
    print("outcome: " + out["outcome"])


if __name__ == "__main__":
    main()
