"""exp21 precheck on r15 (docs/experiments/2026-10-15-exp21-reading-plan.md section 4.1; critic EXP21-R1-N01),
rules written before running. Reads devision-eval records in <dir>: dev_syn, dev_syn_erased, dev_syn_text,
sqa_text, sqa_blank. Exit 0 = train; exit 10 = do not train, with the reason:

  - leak: on the erased pictures (numbers not drawn) r15 reads >= 0.50 or compares >= 0.55 -- the synthetic data
    gives the answer away without the numbers; fix the generator;
  - already reads: read and compare both pass at 6 and 8 px -- training would not show anything new;
  - text comparison fails: given the two values as text, compare < 0.60 -- the question or number format is the
    problem, not reading; fix that first.

    uv run python scripts/exp21/precheck.py runs/exp21-reading/precheck
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from readout import PASS, accuracy, classify, syn_index, syn_stat  # noqa: E402

from devision.train.compare import load_records  # noqa: E402


def decide(d: str) -> dict:
    rec = {s: load_records(os.path.join(d, s + ".details.jsonl"))
           for s in ("dev_syn", "dev_syn_erased", "dev_syn_text", "sqa_text", "sqa_blank")}
    syn, erased, text = (syn_index(rec[s]) for s in ("dev_syn", "dev_syn_erased", "dev_syn_text"))
    out = {"syn": {str(h): {w: syn_stat(syn, h, w) for w in ("read", "cmp", "both_read")} for h in (6, 8, 11, 16)},
           "erased": {w: syn_stat(erased, None, w) for w in ("read", "cmp")},
           "text": {w: syn_stat(text, None, w) for w in ("read", "cmp")},
           "sqa_text": accuracy(rec["sqa_text"]), "sqa_blank": accuracy(rec["sqa_blank"])}
    small = [classify(out["syn"][h][w]) for h in ("6", "8") for w in ("read", "cmp")]
    if out["erased"]["read"]["value"] >= 0.50 or out["erased"]["cmp"]["value"] >= 0.55:
        out["decision"], out["train"] = "leak: answers recoverable without the numbers", False
    elif all(c == PASS for c in small):
        out["decision"], out["train"] = "already reads: read and compare pass at 6 and 8 px", False
    elif out["text"]["cmp"]["value"] < 0.60:
        out["decision"], out["train"] = "text comparison fails (< 0.60 with the values given as text)", False
    else:
        out["decision"], out["train"] = "train", True
    return out


def main():
    d = sys.argv[1]
    out = decide(d)
    json.dump(out, open(os.path.join(d, "precheck.json"), "w"), indent=1)
    f = lambda s: "%.3f [%.3f, %.3f] n=%d" % (s["value"], s["ci95"][0], s["ci95"][1], s["n"])  # noqa: E731
    for h in ("6", "8", "11", "16"):
        print("syn h%-2s read %s | cmp %s | both read %s" % (h, f(out["syn"][h]["read"]), f(out["syn"][h]["cmp"]),
                                                           f(out["syn"][h]["both_read"])))
    print("erased  read %s | cmp %s" % (f(out["erased"]["read"]), f(out["erased"]["cmp"])))
    print("text    read %s | cmp %s" % (f(out["text"]["read"]), f(out["text"]["cmp"])))
    print("ScienceQA 24 transcribed: with values %s, blank %s (report only)" % (f(out["sqa_text"]), f(out["sqa_blank"])))
    print("decision:", out["decision"])
    sys.exit(0 if out["train"] else 10)


if __name__ == "__main__":
    main()
