"""v0.4 acceptance (docs/experiments/2026-10-21-v04-eval-freeze.md): a candidate's eval/ against v0.3's, set by set,
paired by question, pictures resampled 2,000 times.

    uv run python scripts/v04/accept.py runs/r19-n2/eval [--base runs/v03/eval] [--out runs/r19-n2/eval/compare/accept_v04.md]

Prints / writes one table (accuracy and NLL difference with 95% intervals per set) and the four rules' verdicts.
"""
import argparse
import json
import os
import sys
from typing import Dict, List

from devision.train.compare import load_records, paired_metric

JUDGED = ["test_vsr", "test_v7w", "bench_pope", "test_vqa_yesno", "test_vqa_choice", "test_gqa", "test_exist",
          "test_position", "test_relation", "test_size", "test_position_flip", "test_relation_flip",
          "bench_lv_vqav2_yesno", "bench_lv_aokvqa", "bench_lv_scienceqa", "bench_whatsup", "bench_countbench",
          "bench_cogent_b", "test_count_coco", "bench_mmbench", "bench_cvbench", "bench_mme", "bench_naturalbench"]
REPORTED = ["dev_mix", "dev_sqa"]
WEAK = {"ScienceQA": ["bench_lv_scienceqa"], "VSR": ["test_vsr"], "counting": ["bench_countbench", "test_count_coco"],
        "A-OKVQA": ["bench_lv_aokvqa"], "V7W": ["test_v7w"]}


def verdict(rows: Dict[str, dict]) -> Dict[str, object]:
    judged = {s: r for s, r in rows.items() if s in JUDGED}
    worse = [s for s, r in judged.items() if r["acc"]["ci95"][1] < 0]
    up = [s for s, r in judged.items() if r["acc"]["difference"] > 0]
    weak_ok = {k: any(rows[s]["acc"]["ci95"][0] > 0 for s in sets if s in rows) for k, sets in WEAK.items()}
    nll_bad = [s for s, r in judged.items() if r["nll"]["difference"] > 0.02 and r["nll"]["ci95"][0] > 0]
    missing = [s for s in JUDGED if s not in rows]
    return {"missing": missing,
            "1_no_set_worse": {"pass": not worse and not missing, "worse": worse},
            "2_at_least_15_up": {"pass": len(up) >= 15, "up": len(up), "of": len(judged)},
            "3_weak_4_of_5": {"pass": sum(weak_ok.values()) >= 4, "by_ability": weak_ok},
            "4_nll_not_worse": {"pass": not nll_bad, "worse": nll_bad}}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cand")
    p.add_argument("--base", default="runs/v03/eval")
    p.add_argument("--out", default="")
    a = p.parse_args(argv)
    rows: Dict[str, dict] = {}
    for s in JUDGED + REPORTED:
        fa, fb = (os.path.join(d, s + ".details.jsonl") for d in (a.base, a.cand))
        if not (os.path.exists(fa) and os.path.exists(fb)):
            continue
        ra, rb = load_records(fa), load_records(fb)
        rows[s] = {"acc": paired_metric(ra, rb, "accuracy"), "nll": paired_metric(ra, rb, "nll")}
    lines = ["| 集合 | 判定 | 题数 | v0.3 | 候选 | 准确率差 [95%] | NLL 差 [95%] |", "|---|---|---|---|---|---|---|"]
    for s, r in rows.items():
        acc, nll = r["acc"], r["nll"]
        lines.append("| %s | %s | %d | %.3f | %.3f | %+.1f [%+.1f, %+.1f] | %+.3f [%+.3f, %+.3f] |" % (
            s, "是" if s in JUDGED else "报告", acc["questions"], acc["a"], acc["b"], 100 * acc["difference"],
            100 * acc["ci95"][0], 100 * acc["ci95"][1], nll["difference"], nll["ci95"][0], nll["ci95"][1]))
    v = verdict(rows)
    text = "\n".join(lines) + "\n\n```json\n" + json.dumps(v, indent=1, ensure_ascii=False) + "\n```\n"
    print(text)
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w") as f:
            f.write(text)
        with open(a.out.rsplit(".", 1)[0] + ".json", "w") as f:
            json.dump({"rows": rows, "verdict": v}, f, indent=1)
    sys.exit(0)


if __name__ == "__main__":
    main()
