"""exp22 readout (docs/experiments/2026-10-16-exp22-fit-plan.md, rules written before running), from the
temperature-1 monitor records of runs/exp22-fit/stage2/monitor/step{0,500,...,2000}:

  - fit (= the val set): 200 training families at 16 px -- were the training pictures fitted;
  - dev16: 200 unseen families at 16 px -- does it generalize; dev: the same unseen families at every height;
  - erased / text: exp21's controls (validity; comparison without reading).

    uv run python scripts/exp22/report.py [run dir]     # -> <run>/report.json, text on stdout
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "exp21"))
from readout import FAIL, HEIGHTS, PASS, classify, fmt_stat, syn_index, syn_stat  # type: ignore  # noqa: E402

from devision.train.compare import load_records  # noqa: E402

RUN = "runs/exp22-fit"
STEPS = (0, 500, 1000, 1500, 2000)
WHAT = ("read", "cmp", "both_read", "cmp_given_both", "cf_read", "cf_cmp")


def idx(mon, step, name):
    return syn_index(load_records(os.path.join(mon, "step%d" % step, name + ".details.jsonl")))


def label(fit: str, dev16: str, valid: bool) -> str:
    if not valid:
        return "invalid: erased control"
    if fit == PASS and dev16 == PASS:
        return "fitted and generalizes at 16 px: the visual tokens carry the digits under focused training"
    if fit == PASS and dev16 == FAIL:
        return "fitted but does not generalize: the training pictures were memorized"
    if fit == FAIL:
        return "not fitted: even the training pictures at 16 px are not learned in this setting"
    return "uncertain"


def build(mon: str) -> dict:
    out: dict = {"trajectory": {}}
    for step in STEPS:
        if os.path.isdir(os.path.join(mon, "step%d" % step)):
            out["trajectory"][str(step)] = {n: syn_stat(idx(mon, step, n), 16, "read")
                                            for n in ("val", "dev16")}
    last = 2000
    fit, dev16, dev = idx(mon, last, "val"), idx(mon, last, "dev16"), idx(mon, last, "dev")
    out["fit"] = {w: syn_stat(fit, 16, w) for w in WHAT}
    out["dev16"] = {w: syn_stat(dev16, 16, w) for w in WHAT}
    out["dev"] = {str(h): {w: syn_stat(dev, h, w) for w in ("read", "cmp")} for h in HEIGHTS}
    out["erased"] = {w: syn_stat(idx(mon, last, "erased"), None, w) for w in ("read", "cmp")}
    out["text"] = {w: syn_stat(idx(mon, last, "text"), None, w) for w in ("read", "cmp")}
    out["valid"] = out["erased"]["read"]["value"] < 0.50 and out["erased"]["cmp"]["value"] < 0.55
    out["fit_class"], out["dev16_class"] = classify(out["fit"]["read"]), classify(out["dev16"]["read"])
    out["label"] = label(out["fit_class"], out["dev16_class"], out["valid"])
    return out


def render(out: dict) -> str:
    lines = ["16 px read, fit / dev16 by step:"]
    for step, t in out["trajectory"].items():
        lines.append("  step %-5s fit %s | dev16 %s" % (step, fmt_stat(t["val"]), fmt_stat(t["dev16"])))
    for name in ("fit", "dev16"):
        s = out[name]
        lines.append("%-5s read %s | cmp %s | both read %s | cmp|both %s | cf read %s | cf cmp %s" % (
            name, *(fmt_stat(s[w]) for w in WHAT)))
    lines.append("dev by height (read / cmp): " + "; ".join(
        "%s px %s / %s" % (h, fmt_stat(v["read"]), fmt_stat(v["cmp"])) for h, v in out["dev"].items()))
    lines.append("erased read %s cmp %s | text read %s cmp %s" % (
        fmt_stat(out["erased"]["read"]), fmt_stat(out["erased"]["cmp"]), fmt_stat(out["text"]["read"]),
        fmt_stat(out["text"]["cmp"])))
    lines.append("fit: %s, dev16: %s" % (out["fit_class"], out["dev16_class"]))
    lines.append("label: " + out["label"])
    return "\n".join(lines)


def main(argv=None):
    run = (argv if argv is not None else sys.argv[1:] or [RUN])[0]
    out = build(os.path.join(run, "stage2", "monitor"))
    json.dump(out, open(os.path.join(run, "report.json"), "w"), indent=1)
    print(render(out))


if __name__ == "__main__":
    main()
