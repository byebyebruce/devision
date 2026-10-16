"""exp25 readout (docs/experiments/2026-10-16-exp25-read-compare-plan.md, rules written before running), from the
temperature-1 monitor records of runs/exp25-read-compare/stage2/monitor/step{0,1000,...,5000}: exp22's readout at
step 5000 (reading labels and thresholds unchanged), plus comparison:

  - text comparison: exp21's text control (dev families, the values given as text; never a training family);
  - picture comparison: dev16's compare question; compare given both readings right is reported with its n.

    uv run python scripts/exp25/report.py [run dir]     # -> <run>/report.json, text on stdout
"""
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "exp21"))
from readout import FAIL, PASS, classify, fmt_stat  # type: ignore  # noqa: E402

spec = importlib.util.spec_from_file_location("exp22_report", os.path.join(HERE, "..", "exp22", "report.py"))
assert spec is not None and spec.loader is not None
exp22 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp22)

RUN = "runs/exp25-read-compare"
LAST = 5000
STEPS = (0, 1000, 2000, 3000, 4000, 5000)


def outcome(out: dict) -> str:
    """Three-state classes (pass / fail / near threshold); a near-threshold class never becomes a negative finding,
    and the labels are observations, not claims about internal steps (critic EXP25-I01)."""
    if not out["valid"]:
        return "invalid: erased control"
    read, pic, text = out["dev16_class"], classify(out["dev16"]["cmp"]), classify(out["text"]["cmp"])
    if read == PASS and pic == PASS:
        return "reading and comparison both pass on unseen 16 px families"
    if text == FAIL:
        return "text comparison fails: comparing two given numbers is not learned in this setting"
    if read == PASS and text == PASS and pic == FAIL:
        return ("reading and text comparison pass, picture comparison fails: applying both on the picture is "
                "still short")
    if text == PASS and read == FAIL:
        return "text comparison passes, reading fails"
    return "partial / uncertain: at least one class is near threshold (see the readings)"


def build(run: str) -> dict:
    out = exp22.build(os.path.join(run, "stage2", "monitor"), last=LAST, steps=STEPS)
    out["text_cmp_class"] = classify(out["text"]["cmp"])
    out["picture_cmp_class"] = classify(out["dev16"]["cmp"])
    out["outcome"] = outcome(out)
    return out


def main(argv=None):
    run = (argv if argv is not None else sys.argv[1:] or [RUN])[0]
    out = build(run)
    json.dump(out, open(os.path.join(run, "report.json"), "w"), indent=1)
    print(exp22.render(out))
    print("text compare %s: %s | picture compare (dev16) %s: %s | compare given both read (dev16) %s" % (
        fmt_stat(out["text"]["cmp"]), out["text_cmp_class"], fmt_stat(out["dev16"]["cmp"]), out["picture_cmp_class"],
        fmt_stat(out["dev16"]["cmp_given_both"])))
    print("outcome:", out["outcome"])


if __name__ == "__main__":
    main()
