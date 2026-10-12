"""Round 14 acceptance against v12 (docs/experiments/2026-10-12-r14-count.md, written before training): per set,
r14 - v12 per question (picture-resampled 95% intervals) for accuracy and calibrated NLL, then the three rules.

    uv run python scripts/r14/accept.py [--cand runs/r14-count/eval] [--base runs/v12-ext/eval]
"""
import argparse
import json
import os

from devision.train.compare import load_records, paired_metric

PRIMARY = ("test_count_fresh", "bench_countbench")
PROTECTED = ("test_vsr", "test_v7w", "bench_pope", "test_vqa_yesno", "test_vqa_choice", "test_gqa", "test_exist",
             "test_position", "test_relation", "test_size", "test_position_flip", "test_relation_flip",
             "test_count_fresh", "test_bind_gqa", "bench_lv_vqav2_yesno", "bench_lv_aokvqa", "bench_lv_scienceqa",
             "bench_lv_pope")
REPORT = ("bench_countbench", "bench_whatsup", "dev_mix")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cand", default="runs/r14-count/eval")
    p.add_argument("--base", default="runs/v12-ext/eval")
    a = p.parse_args()
    rows, out = [], {}
    for s in dict.fromkeys(PRIMARY + PROTECTED + REPORT):
        fa, fb = os.path.join(a.base, s + ".details.jsonl"), os.path.join(a.cand, s + ".details.jsonl")
        if not (os.path.exists(fa) and os.path.exists(fb)):
            print("%-22s missing (%s)" % (s, "v12" if not os.path.exists(fa) else "r14"))
            continue
        A, B = load_records(fa), load_records(fb)
        acc, nll = paired_metric(A, B, "accuracy"), paired_metric(A, B, "nll")
        lo, hi = acc["ci95"]
        verdict = "proven worse" if hi < 0 else ("not shown held" if lo <= -0.02 else "held")
        out[s] = {"accuracy": acc, "nll": nll, "protection": verdict}
        warn = nll["difference"] > 0.02 and nll["ci95"][0] > 0
        print("%-22s %.3f -> %.3f  %+.1f [%+.1f, %+.1f]  %-15s nll %+.3f [%+.3f, %+.3f]%s" % (
            s, acc["a"], acc["b"], 100 * acc["difference"], 100 * lo, 100 * hi, verdict if s in PROTECTED else "",
            nll["difference"], nll["ci95"][0], nll["ci95"][1], "  WARNING" if warn else ""))
        out[s]["nll_warning"] = warn
    primary = any(out.get(s, {}).get("accuracy", {}).get("ci95", [0])[0] > 0 for s in PRIMARY)
    protected = all(out[s]["protection"] == "held" for s in PROTECTED if s in out)
    warnings = [s for s, v in out.items() if v["nll_warning"]]
    print("\nprimary (a counting test up, lo > 0): %s" % primary)
    print("protection (every test / bench lo > -2): %s  %s" % (protected, [s for s in PROTECTED if s in out and out[s]["protection"] != "held"]))
    print("probability warnings: %s" % warnings)
    print("verdict: %s" % ("candidate for v0.3" if primary and protected and not warnings else "v12 stays by the rules; owner decides"))
    os.makedirs(os.path.join(a.cand, "compare"), exist_ok=True)
    json.dump({"sets": out, "primary": primary, "protection": protected, "warnings": warnings},
              open(os.path.join(a.cand, "compare", "accept_vs_v12.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
