"""exp24 phase B readout (docs/experiments/2026-10-16-exp24-read-align-plan.md): exp22's readout (labels and
thresholds unchanged) on runs/exp24-fit, plus the paired change against exp22 (same training data, order, seed and
settings; only the init differs): 16 px reading accuracy at step 2000, exp24 - exp22, on the same fit and dev16
questions, families resampled.

    uv run python scripts/exp24/report.py [exp24 run] [exp22 run]   # -> <exp24 run>/report.json, text on stdout
"""
import importlib.util
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "exp21"))
from readout import bootstrap, fmt_stat, parse  # type: ignore  # noqa: E402

from devision.train.compare import load_records  # noqa: E402

spec = importlib.util.spec_from_file_location("exp22_report", os.path.join(HERE, "..", "exp22", "report.py"))
assert spec is not None and spec.loader is not None
exp22 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp22)


def paired_read(a_path: str, b_path: str, resamples: int = 2000) -> dict:
    """b - a 16 px reading accuracy (readA + readB) on the same questions; families resampled."""
    def index(path):
        return {(r["sample_id"], r["qid"]): r for r in load_records(path) if r["qid"] in ("readA", "readB")}
    a, b = index(a_path), index(b_path)
    if a.keys() != b.keys():
        raise ValueError("the two runs hold different questions")
    units = defaultdict(lambda: [(0.0, 0.0), (0.0, 0.0)])
    for k, x in a.items():
        fam = parse(k[0])[0]
        c = units[fam]
        c[0] = (c[0][0] + float(x["correct"]), c[0][1] + 1)
        c[1] = (c[1][0] + float(b[k]["correct"]), c[1][1] + 1)
    res = bootstrap(dict(units), lambda c: c[1][0] / c[1][1] - c[0][0] / c[0][1], resamples)
    res["n"] = len(a)
    return res


def build(run: str, control: str) -> dict:
    out = exp22.build(os.path.join(run, "stage2", "monitor"))
    out["vs_exp22"] = {}
    for name in ("val", "dev16"):
        out["vs_exp22"][name] = paired_read(
            os.path.join(control, "stage2", "monitor", "step2000", name + ".details.jsonl"),
            os.path.join(run, "stage2", "monitor", "step2000", name + ".details.jsonl"))
    return out


def main(argv=None):
    args = argv if argv is not None else sys.argv[1:]
    run = args[0] if args else "runs/exp24-fit"
    control = args[1] if len(args) > 1 else "runs/exp22-fit"
    out = build(run, control)
    json.dump(out, open(os.path.join(run, "report.json"), "w"), indent=1)
    print(exp22.render(out))
    for name, label in (("val", "fit"), ("dev16", "dev16")):
        print("vs exp22, 16 px read %-5s %s" % (label, fmt_stat(out["vs_exp22"][name])))


if __name__ == "__main__":
    main()
