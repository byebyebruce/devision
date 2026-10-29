"""r23-v05 acceptance against v0.4 (docs/experiments/2026-10-29-r23-v05-plan.md, written before training): per set,
candidate - v0.4 per question (picture-resampled 95%) for accuracy and NLL; the plan's rules 1-2 and the per-set
ECE / NLL list for rule 3. Writes <out>.json and <out>.md.

    uv run python scripts/v05/accept_r23.py [--cand runs/r23-v05/eval] [--base runs/r21-soft/eval] [--out release/v0.5/accept_vs_v0.4]
"""
import argparse
import json
import os

from devision.train.compare import load_records, paired_metric

NEW = ("test_chartqa", "test_gui_shop_state", "test_gui_state", "test_gui", "test_ego")
POPE = ("bench_pope", "bench_lv_pope")
OLD = ("test_vsr", "test_v7w", "test_vqa_yesno", "test_vqa_choice", "test_gqa", "test_exist", "test_position",
       "test_relation", "test_size", "test_position_flip", "test_relation_flip", "test_count_coco",
       "bench_lv_vqav2_yesno", "bench_lv_aokvqa", "bench_lv_scienceqa", "bench_whatsup", "bench_countbench",
       "bench_cogent_b", "bench_mmbench", "bench_cvbench", "bench_mme", "bench_naturalbench", "bench_realworld",
       "bench_egothink")
REPORT = ("dev_mix", "dev_sqa")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cand", default="runs/r23-v05/eval")
    p.add_argument("--base", default="runs/r21-soft/eval")
    p.add_argument("--out", default="release/v0.5/accept_vs_v0.4")
    a = p.parse_args()
    rows, out = [], {}
    for s in NEW + POPE + OLD + REPORT:
        A = load_records(os.path.join(a.base, s + ".details.jsonl"))
        B = load_records(os.path.join(a.cand, s + ".details.jsonl"))
        acc, nll = paired_metric(A, B, "accuracy"), paired_metric(A, B, "nll")
        sa, sb = (json.load(open(os.path.join(d, s + ".json"))) for d in (a.base, a.cand))
        lo, hi = acc["ci95"]
        if s in NEW:
            role, ok = "新用途", lo > 0
        elif s in POPE:
            role, ok = "POPE", acc["difference"] >= -0.01
        elif s in OLD:
            role, ok = "旧能力", hi >= 0
        else:
            role, ok = "报告", None
        out[s] = {"role": role, "pass": ok, "accuracy": acc, "nll": nll, "ece": [sa["ece"], sb["ece"]]}
        rows.append("| %s | %s | %d | %.3f | %.3f | %+.1f [%+.1f, %+.1f] | %+.3f [%+.3f, %+.3f] | %.3f → %.3f | %s |" % (
            s, role, len(B), acc["a"], acc["b"], 100 * acc["difference"], 100 * lo, 100 * hi, nll["difference"],
            nll["ci95"][0], nll["ci95"][1], sa["ece"], sb["ece"], {True: "过", False: "不过", None: ""}[ok]))
    r1 = all(out[s]["pass"] for s in NEW)
    r2 = all(out[s]["pass"] for s in POPE + OLD)
    worse = [s for s in NEW + POPE + OLD if out[s]["nll"]["difference"] > 0 and out[s]["ece"][1] > out[s]["ece"][0]]
    head = ["| 集合 | 用途 | 题数 | v0.4 | r23 | 准确率差 [95%] | NLL 差 [95%] | ECE | 判据 |", "|---|---|---|---|---|---|---|---|---|"]
    tail = ["", "- 判据 1（新用途下界 > 0）：%s" % ("过" if r1 else "不过"),
            "- 判据 2（旧能力无显著退步；POPE ≥ −1.0 点）：%s" % ("过" if r2 else "不过"),
            "- 判据 3 参考：NLL 与 ECE 都比 v0.4 差的集合：%s" % (", ".join(worse) or "无")]
    open(a.out + ".md", "w").write("\n".join(head + rows + tail) + "\n")
    json.dump({"sets": out, "rule1": r1, "rule2": r2, "nll_and_ece_worse": worse}, open(a.out + ".json", "w"), indent=1)
    print("\n".join(head + rows + tail))


if __name__ == "__main__":
    main()
