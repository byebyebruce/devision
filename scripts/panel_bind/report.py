"""PANEL-BIND-PREP-01 start record and report (no model).

  start   writes <out>/START.json before any forward pass: each end point's model / config / tokenizer sha256, the cached
          MLM head file, the validation file and the sha256 of its 1,000 pictures' bytes, the code (probe / report /
          align), the wrong-picture map (row -> the next row of its batch of 16) and the reproduction references.
  report  per end point: status (completed / missing / invalid: reproduction outside tolerance or rows incomplete);
          from the "numbers" rows -- per picture mode: number-token accuracy, A / B exact, both exact on A != B,
          same prediction on both ends, A / B swapped, a wrong end equal to the other end's gold, A = B rows apart;
          paired by family (2,000 resamples, seed 0): true - wrong picture on both-exact (A != B), D1 - start and
          D1 - EXP26 on the same rows; the share of wrong pictures with the same quantity name.

Reproduction (random-mask mode, align._evaluate step for step) must match the logged validation within TOLERANCE on
both accuracies: start = D1's step 0, D1 = D1's step 1,250 (LoRA merged on save), EXP26 = exp26-align's step 2,500.
"""
import hashlib
import json
import os
import random
import sys

OUT = "runs/panel-bind"
VAL = "data/exp26/align_val.jsonl"
TOLERANCE = 0.003
ENDPOINTS = {"start": "runs/v04-k/cand", "d1": "runs/exp74-panel-d1/d1", "exp26": "runs/exp26-align/stage1"}
REFERENCE = {"start": ("runs/exp74-panel-d1/d1/align_report.json", 0), "d1": ("runs/exp74-panel-d1/d1/align_report.json", -1),
             "exp26": ("runs/exp26-align/stage1/align_report.json", -1)}
HEAD = "answerdotai/ModernBERT-large"


def sha(p: str) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def tree(p: str) -> str:
    h = hashlib.sha256()
    for d, _, fs in sorted(os.walk(p)):
        for f in sorted(fs):
            h.update(os.path.relpath(os.path.join(d, f), p).encode() + sha(os.path.join(d, f)).encode())
    return h.hexdigest()


def reference(name: str) -> dict:
    path, i = REFERENCE[name]
    v = json.load(open(path))["val"][i]
    return {"file": path, "step": v["step"], "real": v["val/mlm_acc_real"], "shuffled": v["val/mlm_acc_shuffled"]}


def start(out: str = OUT) -> dict:
    from huggingface_hub import hf_hub_download
    rows = [json.loads(x) for x in open(VAL) if x.strip()]
    pics = hashlib.sha256(json.dumps([sha(os.path.join("data", r["image"])) for r in rows]).encode()).hexdigest()
    wrong = {r["id"]: rows[b + ((i - b + 1) % len(rows[b:b + 16]))]["id"] for b in range(0, len(rows), 16) for i, r in enumerate(rows[b:b + 16], b)}
    here = os.path.dirname(os.path.abspath(__file__))
    st = {"endpoints": {n: {"path": p, "model.safetensors": sha(os.path.join(p, "model.safetensors")), "config.json": sha(os.path.join(p, "config.json")),
                            "tokenizer": tree(os.path.join(p, "tokenizer"))} for n, p in ENDPOINTS.items()},
          "mlm_head": {"repo": HEAD, "model.safetensors": sha(hf_hub_download(HEAD, "model.safetensors"))},
          "val": {"file": VAL, "sha256": sha(VAL), "rows": len(rows), "pictures_sha256": pics},
          "code": {f: sha(f) for f in (os.path.join(here, "probe.py"), os.path.join(here, "report.py"), "src/devision/train/align.py")},
          "wrong_picture": wrong, "references": {n: reference(n) for n in ENDPOINTS}, "tolerance": TOLERANCE}
    os.makedirs(out, exist_ok=False)
    json.dump(st, open(os.path.join(out, "START.json"), "w"), indent=1)
    return st


def boot(units, stat, n=2000, seed=0):
    rng = random.Random(seed)
    keys = sorted(units)
    vals = sorted(stat([units[keys[rng.randrange(len(keys))]] for _ in keys]) for _ in range(n))
    return {"value": stat([units[k] for k in keys]), "ci95": [vals[int(0.025 * n)], vals[int(0.975 * n) - 1]], "units": len(keys)}


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def number_stats(rows, mode):
    rs = [r for r in rows if r["mode"] == mode]
    diff = [r for r in rs if r["gold"]["A"] != r["gold"]["B"]]
    same = [r for r in rs if r["gold"]["A"] == r["gold"]["B"]]
    tok = [x for r in rs for e in ("A", "B") for x in r["token_right"][e]]
    wrong_other = [r["pred"][e] == r["gold"]["B" if e == "A" else "A"] for r in diff for e in ("A", "B") if not r["exact"][e]]
    return {"rows": len(rs), "number_token_accuracy": mean(tok), "A_exact": mean([r["exact"]["A"] for r in rs]),
            "B_exact": mean([r["exact"]["B"] for r in rs]),
            "different": {"rows": len(diff), "both_exact": mean([r["exact"]["A"] and r["exact"]["B"] for r in diff]),
                          "same_prediction": mean([r["pred"]["A"] == r["pred"]["B"] for r in diff]),
                          "swapped": mean([r["pred"]["A"] == r["gold"]["B"] and r["pred"]["B"] == r["gold"]["A"] for r in diff]),
                          "wrong_end_is_other_gold": mean(wrong_other), "wrong_ends": len(wrong_other)},
            "equal": {"rows": len(same), "both_exact": mean([r["exact"]["A"] and r["exact"]["B"] for r in same])}}


def report(out: str = OUT) -> dict:
    st = json.load(open(os.path.join(out, "START.json")))
    res: dict = {"endpoints": {}}
    both = {}
    for name in ENDPOINTS:
        d = os.path.join(out, name)
        e: dict = {}
        res["endpoints"][name] = e
        if not os.path.exists(os.path.join(d, "summary.json")):
            e["status"] = "missing"
            continue
        summ = json.load(open(os.path.join(d, "summary.json")))
        nums = [json.loads(x) for x in open(os.path.join(d, "numbers.jsonl"))]
        ref = st["references"][name]
        got = {m: summ["random"][m]["all"]["accuracy"] for m in ("real", "shuffled")}
        e["reproduction"] = {"reference": ref, "got": got, "within": all(abs(got[m] - ref[m]) <= st["tolerance"] for m in got)}
        e["random_split"] = summ["random"]
        ids = {r["id"] for r in nums if r["mode"] == "real"}
        if len(nums) != 2 * st["val"]["rows"] or len(ids) != st["val"]["rows"]:
            e["status"] = "invalid: numbers rows incomplete"
            continue
        if not e["reproduction"]["within"]:
            e["status"] = "invalid: the random-mask reproduction is outside the tolerance"
            continue
        e["status"] = "completed"
        e["numbers"] = {m: number_stats(nums, m) for m in ("real", "shuffled")}
        e["wrong_picture_same_quantity"] = mean([r["quantity_same_as_shown"] for r in nums if r["mode"] == "shuffled"])
        both[name] = {(r["id"], r["mode"]): r for r in nums}
        units = {}
        for r in nums:
            if r["mode"] == "real" and r["gold"]["A"] != r["gold"]["B"]:
                w = both[name][(r["id"], "shuffled")]
                units[r["family"]] = (float(r["exact"]["A"] and r["exact"]["B"]), float(w["exact"]["A"] and w["exact"]["B"]))
        e["true_minus_wrong_both_exact"] = boot(units, lambda us: mean([a - b for a, b in us]))
    for other in ("start", "exp26"):
        if "d1" in both and other in both:
            units = {}
            for (rid, mode), r in both["d1"].items():
                if mode == "real" and r["gold"]["A"] != r["gold"]["B"]:
                    o = both[other][(rid, mode)]
                    units[r["family"]] = (float(r["exact"]["A"] and r["exact"]["B"]), float(o["exact"]["A"] and o["exact"]["B"]))
            res["d1_minus_%s_both_exact" % other] = boot(units, lambda us: mean([a - b for a, b in us]))
    states = [e["status"] for e in res["endpoints"].values()]
    res["status"] = "completed" if all(s == "completed" for s in states) else \
        "invalid" if any(s.startswith("invalid") for s in states) else "incomplete"
    json.dump(res, open(os.path.join(out, "report.json"), "w"), indent=1)
    return res


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if argv[:1] == ["start"]:
        st = start(*argv[1:2])
        print(json.dumps({k: v for k, v in st.items() if k != "wrong_picture"}, indent=1))
        return 0
    res = report(*argv[1:2])
    print(json.dumps(res, indent=1))
    return 0 if res["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
