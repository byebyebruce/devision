"""Detailed comparison of one evaluated checkpoint with laya-vision 201M, from per-question records only.

    uv run python scripts/lv_report.py runs/v8-vsr-v7w/eval [--prev runs/v7-sqa/eval]

Reads <eval>/bench_lv_{vqav2_yesno,aokvqa,scienceqa,pope}{,.mismatched}.details.jsonl and laya-vision's
published per-question rows (data/lv_bench/laya-vision-201m.*.details.jsonl, scripts/data/lv_bench.py).
Writes <eval>/compare/laya_vision.json and .md (also printed):
1. paired accuracy, ours - theirs, on all questions and on the ones our training never saw, 95% interval
   resampling pictures; with --prev, the previous checkpoint's difference on the same questions;
2. head to head: questions both get right, only we do, only they do, neither;
3. calibration on the same questions: NLL and ECE (15 bins on the top probability) for both;
4. what our answer owes to the picture: our accuracy with a mismatched picture (theirs is not published);
5. ScienceQA split by subject, by option count and on the questions that need the picture (same text,
   options and hint as another question but another answer);
6. POPE per split against their published accuracy (per-question rows are not published): our accuracy,
   precision, recall, F1 and yes ratio, on all questions and on pictures our training never saw.
"""
import argparse
import glob
import json
import math
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
from laya.common import ece_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sqa_slices import gold, text_key  # noqa: E402

SETS = ("vqav2_yesno", "aokvqa", "scienceqa")
# laya-vision 201M model card (thaitea/laya-vision): POPE accuracy per split, 3,000 questions each
POPE_PUBLISHED = {"pope-random": 0.836, "pope-popular": 0.819, "pope-adversarial": 0.777}


def read(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def by_id(rows: List[dict]) -> Dict[str, dict]:
    return {r["sample_id"]: r for r in rows}


def interval(ours: Dict[str, dict], theirs: Dict[str, dict], ids, resamples=2000, seed=0) -> Optional[dict]:
    pics: Dict[str, List[float]] = defaultdict(list)
    for i in ids:
        if i in ours and i in theirs:
            pics[ours[i]["image_id"]].append(float(ours[i]["correct"]) - float(theirs[i]["correct"]))
    v = list(pics.values())
    if not v:
        return None
    sums, counts = [sum(x) for x in v], [len(x) for x in v]
    rng = random.Random(seed)
    boot = sorted(sum(sums[k] for k in idx) / sum(counts[k] for k in idx)
                  for idx in ([rng.randrange(len(v)) for _ in v] for _ in range(resamples)))
    n = sum(counts)
    shared = [i for i in ids if i in ours and i in theirs]
    return {"questions": n, "ours": sum(float(ours[i]["correct"]) for i in shared) / n,
            "theirs": sum(float(theirs[i]["correct"]) for i in shared) / n,
            "difference": sum(sums) / n, "ci95": [boot[int(0.025 * resamples)], boot[int(0.975 * resamples) - 1]]}


def calibration(rows: List[dict]) -> dict:
    nll = [-math.log(max(r["probabilities"].get(r["gold_answer"], 0.0), 1e-12)) for r in rows]
    return {"nll": float(np.mean(nll)),
            "ece": float(ece_score(np.array([r["p_prediction"] for r in rows]), np.array([float(r["correct"]) for r in rows])))}


def head_to_head(ours: Dict[str, dict], theirs: Dict[str, dict], ids) -> dict:
    out = {"both": 0, "only_ours": 0, "only_theirs": 0, "neither": 0}
    for i in ids:
        a, b = bool(ours[i]["correct"]), bool(theirs[i]["correct"])
        out["both" if a and b else "only_ours" if a else "only_theirs" if b else "neither"] += 1
    return out


def pope_stats(rows: List[dict]) -> dict:
    tp = sum(r["gold_answer"] == "true" and r["prediction"] == "true" for r in rows)
    fp = sum(r["gold_answer"] == "false" and r["prediction"] == "true" for r in rows)
    fn = sum(r["gold_answer"] == "true" and r["prediction"] == "false" for r in rows)
    n = len(rows)
    p = tp / max(1, tp + fp)
    rc = tp / max(1, tp + fn)
    return {"n": n, "accuracy": sum(float(r["correct"]) for r in rows) / max(1, n), "precision": p, "recall": rc,
            "f1": 2 * p * rc / max(1e-12, p + rc), "yes_ratio": (tp + fp) / max(1, n)}


def sqa_groups(root: str) -> Dict[str, set]:
    bench = read(os.path.join(root, "lv_bench", "scienceqa.jsonl"))
    answers: Dict[tuple, set] = defaultdict(set)
    for s in bench:
        answers[text_key(s)].add(gold(s))
    subject: Dict[str, str] = {}
    val = glob.glob(os.path.join(root, "raw", "hf", "datasets--derek-thomas--ScienceQA", "snapshots", "*", "data",
                                 "validation-*.parquet"))
    if val:
        import pyarrow.parquet as pq
        subject = {"lv-scienceqa:val-%d" % i: r["subject"]
                   for i, r in enumerate(pq.read_table(val[0], columns=["subject"]).to_pylist())}
    out: Dict[str, set] = defaultdict(set)
    for s in bench:
        sid = s["id"]
        out["subject=" + subject.get(sid, "?")].add(sid)
        out["options=%d" % len(s["questions"]["q"]["criteria"])].add(sid)
        if len(answers[text_key(s)]) > 1:
            out["needs_picture"].add(sid)
            out["needs_picture&subject=" + subject.get(sid, "?")].add(sid)
    return out


def pct(x: Optional[float]) -> str:
    return "—" if x is None else "%.1f%%" % (100 * x)


def ci(d: Optional[dict]) -> str:
    return "—" if not d else "%+.1f [%+.1f, %+.1f]" % (100 * d["difference"], 100 * d["ci95"][0], 100 * d["ci95"][1])


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("eval_dir")
    p.add_argument("--prev", help="the previous checkpoint's eval directory, for its difference on the same questions")
    p.add_argument("--root", default="data")
    a = p.parse_args(argv)
    lv = os.path.join(a.root, "lv_bench")
    rep: dict = {"eval_dir": a.eval_dir, "prev": a.prev, "sets": {}}
    md = ["# %s 与 laya-vision 201M 的详细比较" % a.eval_dir, "",
          "差值 = 我们 − laya-vision，95% 区间按图片配对重采样；\"未见过\" = 去掉我们训练用过图片的题。", "",
          "## 1. 逐题配对准确率", "",
          "| 集合 | 范围 | 题数 | laya-vision | 我们 | 差值 [95%] |" + (" 上一版差值 |" if a.prev else ""),
          "|---|---|---|---|---|---|" + ("---|" if a.prev else "")]
    h2h_md, cal_md, mism_md = [], [], []
    for s in SETS:
        theirs = by_id(read(os.path.join(lv, "laya-vision-201m.%s.details.jsonl" % s)))
        ours = by_id(read(os.path.join(a.eval_dir, "bench_lv_%s.details.jsonl" % s)))
        prev = by_id(read(os.path.join(a.prev, "bench_lv_%s.details.jsonl" % s))) if a.prev else None
        unseen = {r["id"] for r in read(os.path.join(lv, "%s.unseen.jsonl" % s))}
        entry: dict = {}
        for scope, ids in (("all", set(theirs)), ("unseen", set(theirs) & unseen)):
            d, dp = interval(ours, theirs, ids), (interval(prev, theirs, ids) if prev else None)
            entry[scope] = {"ours_minus_theirs": d, "prev_minus_theirs": dp}
            if d:
                md.append("| %s | %s | %d | %s | %s | %s |%s" % (s, "全部" if scope == "all" else "未见过", d["questions"],
                          pct(d["theirs"]), pct(d["ours"]), ci(d), (" %s |" % ci(dp)) if a.prev else ""))
        shared = [i for i in theirs if i in ours]
        entry["head_to_head"] = h = head_to_head(ours, theirs, shared)
        h2h_md.append("| %s | %d | %d | %d | %d |" % (s, h["both"], h["only_ours"], h["only_theirs"], h["neither"]))
        co, ct = calibration([ours[i] for i in shared]), calibration([theirs[i] for i in shared])
        entry["calibration"] = {"ours": co, "theirs": ct}
        cal_md.append("| %s | %.3f | %.3f | %.3f | %.3f |" % (s, ct["nll"], co["nll"], ct["ece"], co["ece"]))
        mpath = os.path.join(a.eval_dir, "bench_lv_%s.mismatched.details.jsonl" % s)
        if os.path.exists(mpath):
            m = by_id(read(mpath))
            mi = [i for i in shared if i in m]
            acc_m = sum(float(m[i]["correct"]) for i in mi) / max(1, len(mi))
            acc_r = sum(float(ours[i]["correct"]) for i in mi) / max(1, len(mi))
            entry["ours_mismatched"] = {"real": acc_r, "mismatched": acc_m, "picture_gain": acc_r - acc_m}
            mism_md.append("| %s | %s | %s | %+.1f |" % (s, pct(acc_r), pct(acc_m), 100 * (acc_r - acc_m)))
        if s == "scienceqa":
            entry["slices"] = {}
            for g, ids in sorted(sqa_groups(a.root).items()):
                entry["slices"][g] = interval(ours, theirs, ids)
        rep["sets"][s] = entry
    md += ["", "## 2. 逐题对照（同一批题）", "", "| 集合 | 都对 | 只有我们对 | 只有对方对 | 都错 |", "|---|---|---|---|---|"] + h2h_md
    md += ["", "## 3. 校准（同一批题，ECE 按最高概率 15 档）", "", "| 集合 | 对方 NLL | 我们 NLL | 对方 ECE | 我们 ECE |",
           "|---|---|---|---|---|"] + cal_md
    md += ["", "## 4. 我们的分数有多少来自图片（对方未公开配错图结果）", "", "| 集合 | 真图 | 配错图 | 差 |", "|---|---|---|---|"] + mism_md
    md += ["", "## 5. ScienceQA 切片", "", "| 切片 | 题数 | laya-vision | 我们 | 差值 [95%] |", "|---|---|---|---|---|"]
    for g, d in rep["sets"]["scienceqa"]["slices"].items():
        if d:
            md.append("| %s | %d | %s | %s | %s |" % (g, d["questions"], pct(d["theirs"]), pct(d["ours"]), ci(d)))
    pope = read(os.path.join(a.eval_dir, "bench_lv_pope.details.jsonl"))
    unseen_pope = {r["id"] for r in read(os.path.join(lv, "pope.unseen.jsonl"))}
    rep["pope"] = {}
    md += ["", "## 6. POPE（对方只公开每档准确率，不能逐题配对）", "",
           "| 档 | laya-vision 公开 | 我们 | 我们（未见过的图） | precision | recall | F1 | 答\"是\"比例 |", "|---|---|---|---|---|---|---|---|"]
    for split, published in POPE_PUBLISHED.items():
        rows = [r for r in pope if r["source"] == split]
        st, su = pope_stats(rows), pope_stats([r for r in rows if r["sample_id"] in unseen_pope])
        rep["pope"][split] = {"published": published, "ours": st, "ours_unseen": su}
        md.append("| %s | %s | %s | %s | %.3f | %.3f | %.3f | %.3f |" % (
            split.split("-")[1], pct(published), pct(st["accuracy"]), pct(su["accuracy"]), st["precision"], st["recall"], st["f1"], st["yes_ratio"]))
    out = os.path.join(a.eval_dir, "compare")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "laya_vision.json"), "w") as f:
        json.dump(rep, f, indent=1)
    with open(os.path.join(out, "laya_vision.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
