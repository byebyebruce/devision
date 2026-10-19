"""EXP76 device V3 (device plan section 2): save / load of R on the V device. Loads R, asks the 64 fixed dev16 questions
through decide (vcheck.v3_questions: the eight divergent ones by name, then the first 56 others), saves R with
Decider.save to a new directory, loads that, asks the same questions, and compares (per question and option
|difference| <= 1e-6, same prediction). Writes --out with the identities and the comparison; exit 0 when consistent,
1 otherwise. No training, nothing else evaluated. Only the V runner starts it; it is never run in tests.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vcheck  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="runs/exp75-panel-r/r")
    ap.add_argument("--data", default="data/panel_learn/dev16.jsonl")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if os.path.exists(a.save_dir):
        raise SystemExit("%s exists: V3 saves into a new directory only" % a.save_dir)
    from devision.infer import Decider
    from devision.train.evaluate import evaluate_records
    from devision.train.samples import read_decision_jsonl
    samples = read_decision_jsonl(a.data)
    want = vcheck.v3_questions([{"sample_id": s["id"], "qid": q} for s in samples for q in s["questions"]])
    keep = {}
    for s, q in want:
        keep.setdefault(s, []).append(q)
    asked = [dict(s, questions={q: s["questions"][q] for q in keep[s["id"]]}) for s in samples if s["id"] in keep]
    res = {"checkpoint": a.checkpoint, "start_identity": {f: vcheck.sha(os.path.join(a.checkpoint, f))
                                                           for f in ("model.safetensors", "config.json")},
           "questions": [list(k) for k in want], "device": a.device}
    before = evaluate_records(Decider.load(a.checkpoint, device=a.device), asked, a.data_root)
    Decider.load(a.checkpoint, device=a.device).save(a.save_dir)
    res["saved_identity"] = {f: vcheck.sha(os.path.join(a.save_dir, f)) for f in ("model.safetensors", "config.json")}
    after = evaluate_records(Decider.load(a.save_dir, device=a.device), asked, a.data_root)
    c = vcheck.compare_rows(before, after)
    c["diffs"] = {"%s/%s" % k: d for k, d in c["diffs"].items()}
    res.update(comparison=c, ok=not c["problems"] and c["n"] == 64)
    json.dump(res, open(a.out + ".tmp", "w"), indent=1)
    os.replace(a.out + ".tmp", a.out)
    print("V3 %s: %d questions, largest difference %.3g" % ("consistent" if res["ok"] else "NOT consistent", c["n"], c["largest"]))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
