"""Criterion 1 of the board-gap experiments (exp137 / exp138, pre-written in their docs): a family passes when its
dev_board_<family> paired difference against r23 has a 95% lower bound > 0 AND every mapped Decision Index benchmark
skill rises against r23 (point estimate). Also lists criterion 2 (old sets with a significant drop > 1 point against
r23). Prints a JSON report; with --passed prints only the comma-separated passing families (for the r24 chain).

  usage: judge_board.py RUN FAMILY[,FAMILY...] [--ref r23-v05] [--passed]
"""
import argparse
import json
import sys
from pathlib import Path

# family -> Decision Index benchmarks whose skill must rise; "any" lists need only one of them to rise
BOARD = {
    "caption": ("all", ["Winoground"]),
    "kie": ("all", ["KIE (CORD+FUNSD)"]),
    "infographic": ("all", ["InfographicVQA"]),
    "figure": ("all", ["CharXiv"]),
    "depth": ("all", ["CV-Bench", "BLINK"]),  # BLINK by accuracy: its skill is clipped at 0 for r23
    "som": ("all", ["Mind2Web"]),
    "drive": ("any", ["R-Bench-M", "RealWorldQA"]),
    "hateful": ("all", ["Moderation (Hateful Memes)"]),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("families")
    ap.add_argument("--ref", default="r23-v05")
    ap.add_argument("--passed", action="store_true")
    a = ap.parse_args()
    run, ref = a.run, a.ref
    cmp = Path("runs") / run / "eval" / "compare"
    b_ref = json.load(open(Path("runs/ext-decision-index") / ref / "vision-scores.json"))["benchmarks"]
    b_new = json.load(open(Path("runs/ext-decision-index") / run / "vision-scores.json"))["benchmarks"]
    report: dict = {"run": run, "ref": ref, "families": {}, "old_drops": []}
    for fam in a.families.split(","):
        d = json.load(open(cmp / ("dev_board_%s.vs-%s.txt" % (fam, ref))))["all"]
        mode, names = BOARD[fam]
        key = lambda n: "accuracy" if n == "BLINK" else "skill"  # noqa: E731
        moves = {n: b_new[n][key(n)] - b_ref[n][key(n)] for n in names}
        up = (all if mode == "all" else any)(v > 0 for v in moves.values())
        dev_ok = d["ci95"][0] > 0
        report["families"][fam] = {"dev_diff": d["difference"], "dev_ci95": d["ci95"], "dev_ok": dev_ok,
                                   "board_moves": moves, "board_ok": up, "pass": dev_ok and up}
    for f in sorted(cmp.glob("*.vs-%s.txt" % ref)):
        name = f.name.split(".vs-")[0]
        if name.startswith("dev_board_"):
            continue
        d = json.load(open(f))["all"]
        if d["ci95"][1] < 0 and d["difference"] < -0.01:
            report["old_drops"].append({"set": name, "diff": d["difference"], "ci95": d["ci95"]})
    if a.passed:
        print(",".join(f for f, v in report["families"].items() if v["pass"]))
    else:
        json.dump(report, sys.stdout, indent=1)
        print()


if __name__ == "__main__":
    main()
