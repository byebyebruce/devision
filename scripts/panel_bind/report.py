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
    """Strict "exact" = every token of the number span right (the pre-written definition); "string_equal" (the decoded
    prediction equal to the gold text) is descriptive only (critic PANEL-BIND-NOTE-01)."""
    rs = [r for r in rows if r["mode"] == mode]
    diff = [r for r in rs if r["gold"]["A"] != r["gold"]["B"]]
    same = [r for r in rs if r["gold"]["A"] == r["gold"]["B"]]
    tok = [x for r in rs for e in ("A", "B") for x in r["token_right"][e]]
    wrong_other = [r["pred"][e] == r["gold"]["B" if e == "A" else "A"] for r in diff for e in ("A", "B") if not r["exact"][e]]
    return {"rows": len(rs), "number_token_accuracy": mean(tok), "A_exact_all_tokens": mean([r["exact"]["A"] for r in rs]),
            "B_exact_all_tokens": mean([r["exact"]["B"] for r in rs]),
            "A_string_equal_descriptive": mean([r["pred"]["A"] == r["gold"]["A"] for r in rs]),
            "B_string_equal_descriptive": mean([r["pred"]["B"] == r["gold"]["B"] for r in rs]),
            "different": {"rows": len(diff), "both_exact_all_tokens": mean([r["exact"]["A"] and r["exact"]["B"] for r in diff]),
                          "same_prediction": mean([r["pred"]["A"] == r["pred"]["B"] for r in diff]),
                          "swapped": mean([r["pred"]["A"] == r["gold"]["B"] and r["pred"]["B"] == r["gold"]["A"] for r in diff]),
                          "wrong_end_is_other_gold": mean(wrong_other), "wrong_ends": len(wrong_other)},
            "equal": {"rows": len(same), "both_exact_all_tokens": mean([r["exact"]["A"] and r["exact"]["B"] for r in same])}}


def read_json(path):
    """(object, None) or (None, "missing" | "invalid: ...") -- an empty / truncated file is invalid, never a crash."""
    if not os.path.exists(path):
        return None, "missing"
    try:
        return json.load(open(path)), None
    except ValueError as e:
        return None, "invalid: %s unreadable (%s)" % (os.path.basename(path), e)


def read_jsonl(path):
    if not os.path.exists(path):
        return None, "missing"
    try:
        return [json.loads(x) for x in open(path) if x.strip()], None
    except ValueError as e:
        return None, "invalid: %s unreadable or truncated (%s)" % (os.path.basename(path), e)


def expected_rows(st):
    """The frozen question list: id -> family, own picture, wrong picture, gold A / B (parsed as the probe parses)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("panel_bind_probe", os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe.py"))
    assert spec and spec.loader
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    rows = [json.loads(x) for x in open(st["val"]["file"]) if x.strip()]
    image = {r["id"]: r["image"] for r in rows}
    wrong = {r["id"]: rows[b + ((i - b + 1) % len(rows[b:b + 16]))]["id"] for b in range(0, len(rows), 16) for i, r in enumerate(rows[b:b + 16], b)}
    exp = {}
    for r in rows:
        sp = probe.spans(r["captions"][0])
        exp[r["id"]] = {"family": r["family"], "real": r["image"], "shuffled": image[wrong[r["id"]]],
                        "gold": {"A": sp["A"][2], "B": sp["B"][2]}}
    return exp, wrong


def check_random(recs, exp):
    """Recomputed counts per mode, or a problem: the (id, mode) set unique and complete, family / picture as the frozen
    map, token counts legal."""
    keys = [(r.get("id"), r.get("mode")) for r in recs]
    want = {(i, m) for i in exp for m in ("real", "shuffled")}
    if len(keys) != len(set(keys)) or set(keys) != want:
        return None, "random rows are not each (id, mode) exactly once (%d rows, %d unique, %d expected)" % (len(keys), len(set(keys)), len(want))
    cnt = {m: {k: [0, 0] for k in ("number", "other", "all")} for m in ("real", "shuffled")}
    for r in recs:
        e = exp[r["id"]]
        if r.get("family") != e["family"] or r.get("picture") != e[r["mode"]]:
            return None, "random %s/%s: family or picture is not the frozen one" % (r["id"], r["mode"])
        for k in ("number", "other"):
            v = (r.get("tokens") or {}).get(k)
            if not (isinstance(v, list) and len(v) == 2 and all(type(x) is int for x in v) and 0 <= v[0] <= v[1]):
                return None, "random %s/%s: token counts %r are not legal" % (r["id"], r["mode"], v)
            for kk in (k, "all"):
                cnt[r["mode"]][kk][0] += v[0]
                cnt[r["mode"]][kk][1] += v[1]
    return {m: {k: {"right": v[0], "n": v[1], "accuracy": v[0] / max(1, v[1])} for k, v in c.items()} for m, c in cnt.items()}, None


def check_numbers(recs, exp):
    keys = [(r.get("id"), r.get("mode")) for r in recs]
    want = {(i, m) for i in exp for m in ("real", "shuffled")}
    if len(keys) != len(set(keys)) or set(keys) != want:
        return "numbers rows are not each (id, mode) exactly once (%d rows, %d unique, %d expected)" % (len(keys), len(set(keys)), len(want))
    for r in recs:
        e = exp[r["id"]]
        if r.get("family") != e["family"] or r.get("picture") != e[r["mode"]] or r.get("gold") != e["gold"]:
            return "numbers %s/%s: family, picture or gold is not the frozen one" % (r["id"], r["mode"])
        tr, ex = r.get("token_right") or {}, r.get("exact") or {}
        for end in ("A", "B"):
            t = tr.get(end)
            if not (isinstance(t, list) and t and all(type(x) is bool for x in t)) or type(ex.get(end)) is not bool or ex[end] != all(t) \
                    or not isinstance((r.get("pred") or {}).get(end), str):
                return "numbers %s/%s: end %s's token_right / exact / pred are not consistent" % (r["id"], r["mode"], end)
    return None


def write(out, res):
    os.makedirs(out, exist_ok=True)
    tmp = os.path.join(out, "report.json.tmp")
    json.dump(res, open(tmp, "w"), indent=1)
    os.replace(tmp, os.path.join(out, "report.json"))
    return res


def report(out: str = OUT) -> dict:
    """Per end point, from its records (critic PANEL-BIND-R1-01 / 02): the frozen question list from START; the
    random records recomputed and equal to the summary before the reproduction is judged; the numbers records complete
    and consistent; only completed end points enter any comparison. A missing / truncated START, summary or record file
    gives that end point (or the whole report) its own state; a report is always written."""
    res: dict = {"endpoints": {}}
    st, why = read_json(os.path.join(out, "START.json"))
    if st is None:
        res["status"] = "incomplete" if why == "missing" else "invalid"
        res["reason"] = "START.json %s: no identity, no conclusion" % why
        return write(out, res)
    try:
        if sha(st["val"]["file"]) != st["val"]["sha256"]:
            raise ValueError("the validation file is not START's")
        exp, wrong = expected_rows(st)
        if st.get("wrong_picture") != wrong:
            raise ValueError("START's wrong-picture map is not the fixed one")
    except (KeyError, ValueError, TypeError) as err:
        res["status"], res["reason"] = "invalid", "START: %s" % err
        return write(out, res)
    both = {}
    for name in ENDPOINTS:
        d = os.path.join(out, name)
        e: dict = {}
        res["endpoints"][name] = e
        summ, w1 = read_json(os.path.join(d, "summary.json"))
        rnd, w2 = read_jsonl(os.path.join(d, "random.jsonl"))
        nums, w3 = read_jsonl(os.path.join(d, "numbers.jsonl"))
        problems = [w for w in (w1, w2, w3) if w]
        if problems:
            e["status"] = "missing" if all(w == "missing" for w in problems) else next(w for w in problems if w != "missing")
            if e["status"] == "missing" and len(problems) < 3:
                e["status"] = "incomplete: %s missing" % ", ".join(f for f, w in zip(("summary", "random", "numbers"), (w1, w2, w3)) if w)
            continue
        assert summ is not None and rnd is not None and nums is not None
        recount, bad = check_random(rnd, exp)
        bad = bad or check_numbers(nums, exp)
        if bad:
            e["status"] = "invalid: " + bad
            continue
        assert recount is not None
        try:
            same = all(summ["random"][m][k]["right"] == recount[m][k]["right"] and summ["random"][m][k]["n"] == recount[m][k]["n"]
                       for m in ("real", "shuffled") for k in ("number", "other", "all"))
        except (KeyError, TypeError):
            same = False
        if not same:
            e["status"] = "invalid: the summary is not its random records"
            continue
        ref = st["references"][name]
        got = {m: recount[m]["all"]["accuracy"] for m in ("real", "shuffled")}
        e["reproduction"] = {"reference": ref, "got": got, "difference": {m: got[m] - ref[m] for m in got},
                             "within": all(abs(got[m] - ref[m]) <= st["tolerance"] for m in got)}
        e["random_split"] = recount
        e["gap"] = got["real"] - got["shuffled"]
        if not e["reproduction"]["within"]:
            e["status"] = "invalid: the random-mask reproduction is outside the tolerance"
            continue
        e["status"] = "completed"
        e["numbers"] = {m: number_stats(nums, m) for m in ("real", "shuffled")}
        e["wrong_picture_same_quantity"] = mean([r["quantity_same_as_shown"] for r in nums if r["mode"] == "shuffled"])
        both[name] = {(r["id"], r["mode"]): r for r in nums}
        units = {}
        for (rid, mode), r in both[name].items():
            if mode == "real" and r["gold"]["A"] != r["gold"]["B"]:
                wr = both[name][(rid, "shuffled")]
                units[r["family"]] = (float(r["exact"]["A"] and r["exact"]["B"]), float(wr["exact"]["A"] and wr["exact"]["B"]))
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
    res["status"] = "completed" if all(x == "completed" for x in states) else \
        "invalid" if any(x.startswith("invalid") for x in states) else "incomplete"
    return write(out, res)


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
