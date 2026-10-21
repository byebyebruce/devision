"""exp47 residual-difference diagnosis of the frozen EXP45-G (critic EXP47-PREP-01..03; docs/experiments/2026-10-18-exp47-residual-plan.md).
No training, no weight written, eval / no_grad, temperature 1, EXP46's 48 units / 96 questions (each unit's two questions in
one forward of 2). For each question, on the same V: the module's residual with the attention restricted to the asked
field's box, r_T = out(attend_T), and to the other object's box, r_O (EXP46's oracle masks); c = (r_T + r_O) / 2,
d = (r_T - r_O) / 2, beta = max(1, min(1000, 0.1 x RMS(V + c) / max(RMS(d), 1e-12))) over the unit's two questions.
The visual tokens the module returns, by mode:

  baseline     V + r (the module as it is)         target_1   V + c + d (= V + r_T)    other_1   V + c - d (= V + r_O)
  center       V + c                                target_gain V + c + beta d          other_gain V + c - beta d

Fidelity (else the unit stops before any new mode, nothing loosened): the plain forward reproduces G's frozen raw records
and EXP46's baseline, the instrumented baseline the plain forward, target_1 / other_1 EXP46's oracle_target / oracle_other
(each <= 1e-5, same prediction, finite). Per question: probabilities, prediction, correct, P(gold) - P(other object's gold);
per unit: RMS(V), RMS(c), RMS(d), RMS(V + c), beta, capped, zero difference, the applied relative difference, and along the
path the half-difference / common RMS of target vs other at the module's output, at the encoder's option-marker states
(the same forward's return_hidden) and in the option logits centred per question. Written unit by unit; a model-free
--finalize gives "incomplete" with per-mode missing counts unless all six modes covered all questions.

    uv run python scripts/exp47/diagnose.py
"""
import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional

import torch
from laya.common import collate_items

HERE = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("exp46_diagnose", os.path.join(HERE, "..", "exp46", "diagnose.py"))
assert _spec is not None and _spec.loader is not None
e46 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e46)

MODES = ("baseline", "target_1", "other_1", "center", "target_gain", "other_gain")
BETA_SCALE, BETA_MAX = 0.1, 1000.0
G_RUN, G_SHA, SETS = e46.G_RUN, e46.G_SHA, e46.SETS
rms = e46.rms


def residual(cond, query, kv, support: Optional[torch.Tensor]) -> torch.Tensor:
    """The module's residual out(attend(query, kv, kv)), the attention restricted to `support` when given."""
    mask = None
    if support is not None:
        heads, allow = cond.attend.num_heads, support.to(query.device)
        mask = torch.zeros(allow.shape, dtype=query.dtype, device=query.device).masked_fill(~allow, float("-inf"))
        mask = mask[:, None, None, :].expand(-1, heads, query.size(1), -1).reshape(-1, query.size(1), allow.size(1))
    out = cond.attend(query, kv, kv, need_weights=False, attn_mask=mask)[0]
    return cond.out(out)


def visual_for(cond, visual, patches, word_embeds, qmask, mode: str, target, other):
    """(the visual tokens under `mode`, record) for a batch [A, B] of one picture."""
    m = qmask.unsqueeze(-1).to(word_embeds.dtype)
    q = (word_embeds * m).sum(1) / m.sum(1).clamp(min=1)
    query = visual + cond.question(q).unsqueeze(1)
    kv = cond.patches(patches.to(visual.dtype))
    if mode == "baseline":
        return visual + residual(cond, query, kv, None), {}
    r_t, r_o = residual(cond, query, kv, target), residual(cond, query, kv, other)
    c, d = (r_t + r_o) / 2, (r_t - r_o) / 2
    rd, rvc = float(rms(d)), float(rms(visual + c))
    beta = max(1.0, min(BETA_MAX, BETA_SCALE * rvc / max(rd, 1e-12)))
    k = {"target_1": 1.0, "other_1": -1.0, "center": 0.0, "target_gain": beta, "other_gain": -beta}[mode]
    rec = {"rms_v": float(rms(visual)), "rms_c": float(rms(c)), "rms_d": rd, "rms_vc": rvc, "beta": beta,
           "beta_capped": beta >= BETA_MAX, "zero_difference": rd == 0.0, "applied_relative_difference": abs(k) * rd / max(rvc, 1e-12)}
    return visual + c + k * d, rec


@contextlib.contextmanager
def patched(cond, mode: str, target, other, sink: list):
    original = cond.forward

    def forward(visual, patches, word_embeds, qmask):
        new, rec = visual_for(cond, visual, patches, word_embeds, qmask, mode, target, other)
        sink.append((new, rec))
        return new
    cond.forward = forward
    try:
        yield
    finally:
        cond.forward = original


def half(a: torch.Tensor, b: torch.Tensor):
    """(RMS of the half difference, RMS of the mean) of two tensors."""
    return float(rms((a - b) / 2)), float(rms((a + b) / 2))


def centred(logits: torch.Tensor, items) -> List[torch.Tensor]:
    return [logits[i, :len(it["markers"])].float() - logits[i, :len(it["markers"])].float().mean() for i, it in enumerate(items)]


def contrast(recs: List[Dict], plus: str, minus: str) -> Dict:
    """Per unit, the mean over its two questions of [P_plus(gold) - P_plus(other gold)] - [P_minus(gold) - P_minus(other gold)]."""
    by = defaultdict(dict)
    for r in recs:
        by[(r["unit"], r["object"])][r["mode"]] = r
    units = defaultdict(list)
    for (u, _), ms in by.items():
        if plus in ms and minus in ms and ms[plus]["margin"] is not None and ms[minus]["margin"] is not None:
            units[u].append(ms[plus]["margin"] - ms[minus]["margin"])
    return {u: sum(v) / len(v) for u, v in units.items() if len(v) == 2}


def mode_summary(recs: List[Dict]) -> Dict:
    """Per mode and slice (split x layout, split, layout, field): questions right, units with both right, units whose two
    questions got the same predicted option, and against baseline wrong -> right / right -> wrong / prediction changed.
    No attention quantity: exp47 does not measure any (critic EXP47-R1-01; version 1 printed EXP46's zero-filled ones)."""
    by = defaultdict(list)
    for r in recs:
        by[r["mode"]].append(r)
    base = {(r["unit"], r["object"]): r for r in by.get("baseline", [])}
    out: Dict = {}
    for mode, rs in by.items():
        groups = defaultdict(list)
        for r in rs:
            for g in ("all", r["split"], r["layout"], r["field"], "%s/%s" % (r["split"], r["layout"])):
                groups[g].append(r)
        out[mode] = {}
        for g, gr in sorted(groups.items()):
            units = defaultdict(list)
            for r in gr:
                units[r["unit"]].append(r)
            pairs = [v for v in units.values() if len(v) == 2]
            out[mode][g] = {"questions": len(gr), "right": sum(r["correct"] for r in gr), "units": len(pairs),
                            "both_right": sum(all(x["correct"] for x in v) for v in pairs),
                            "same_answer": sum(v[0]["prediction"] == v[1]["prediction"] for v in pairs),
                            "wrong_to_right": sum((not base[(r["unit"], r["object"])]["correct"]) and r["correct"] for r in gr) if base else None,
                            "right_to_wrong": sum(base[(r["unit"], r["object"])]["correct"] and not r["correct"] for r in gr) if base else None,
                            "prediction_changed": sum(base[(r["unit"], r["object"])]["prediction"] != r["prediction"] for r in gr) if base else None}
    return out


def analyse(recs: List[Dict], units_meta: Dict[str, Dict]) -> Dict:
    """The pre-written readout: per slice (split x layout, and field) each mode's right / both right / same answer /
    changes against baseline; C_beta, C_1, C_beta - C_1 per unit; the continuation condition."""
    c_b, c_1 = contrast(recs, "target_gain", "other_gain"), contrast(recs, "target_1", "other_1")
    out: Dict = {"report_version": 2, "modes": mode_summary(recs), "attention": "not measured in exp47", "slices": {}}
    for sl in ("fit/original", "fit/L", "tune/original", "tune/L"):
        us = [u for u, m in units_meta.items() if "%s/%s" % (m["split"], m["layout"]) == sl]
        diff = [c_b[u] - c_1[u] for u in us if u in c_b and u in c_1]
        by_field = {}
        for f in ("mass", "speed", "distance"):
            v = [c_b[u] - c_1[u] for u in us if units_meta[u]["field"] == f and u in c_b and u in c_1]
            by_field[f] = {"n": len(v), "mean_diff": sum(v) / len(v) if v else None}
        both = {}
        for mode in MODES:
            pairs = defaultdict(list)
            for r in recs:
                if r["mode"] == mode and r["unit"] in us:
                    pairs[r["unit"]].append(r["correct"])
            both[mode] = sum(all(v) for v in pairs.values() if len(v) == 2)
        out["slices"][sl] = {"units": len(us), "both_right": both,
                             "C_beta_mean": sum(c_b[u] for u in us if u in c_b) / max(1, sum(u in c_b for u in us)),
                             "C_1_mean": sum(c_1[u] for u in us if u in c_1) / max(1, sum(u in c_1 for u in us)),
                             "diff_mean": sum(diff) / len(diff) if diff else None, "diff_positive": sum(x > 0 for x in diff),
                             "diff_negative": sum(x < 0 for x in diff), "by_field": by_field}
    t, f = out["slices"]["tune/original"], out["slices"]["fit/original"]
    out["continue"] = {
        "target_gain_both_3": t["both_right"]["target_gain"] >= 3,
        "target_gain_minus_other_gain_2": t["both_right"]["target_gain"] - t["both_right"]["other_gain"] >= 2,
        "tune_diff_005": t["diff_mean"] is not None and t["diff_mean"] >= 0.05,
        "fit_diff_positive": f["diff_mean"] is not None and f["diff_mean"] > 0,
        "two_fields_positive": sum(1 for v in t["by_field"].values() if v["mean_diff"] is not None and v["mean_diff"] > 0) >= 2}
    out["continue"]["all"] = all(out["continue"].values())
    return out


def load_jsonl(path: str) -> List[Dict]:
    """Lines of a JSONL; a half-written last line (an interrupted run) is skipped."""
    out = []
    if os.path.exists(path):
        for line in open(path):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def finalize(out: str, rc: str, units_path: str = "data/exp46/units.jsonl") -> Dict:
    os.makedirs(out, exist_ok=True)                 # tolerate a run that stopped before creating it
    try:
        st: Dict = json.load(open(os.path.join(out, "status.json")))
    except (OSError, ValueError):
        st = {"state": "not started or status unreadable"}
    recs = load_jsonl(os.path.join(out, "details.jsonl"))
    n = int(st.get("questions", 0))
    done = {m: sum(r["mode"] == m for r in recs) for m in MODES}
    complete = st.get("state") == "done" and n > 0 and all(done[m] == n for m in MODES)
    meta = {u["key"]: u for u in load_jsonl(units_path)}
    rep: Dict = {"status": "complete" if complete else "incomplete", "state": st, "exit_code": rc,
                 "completed_questions_by_mode": done, "missing_by_mode": {m: n - done[m] for m in MODES},
                 "units": load_jsonl(os.path.join(out, "units.jsonl"))}
    if recs:
        rep["analysis"] = analyse(recs, {u: meta[u] for u in {r["unit"] for r in recs}})
        if not complete:
            rep["analysis"]["continue"] = "incomplete"
    json.dump(rep, open(os.path.join(out, "report.json"), "w"), indent=1, default=str)
    lines = ["status %s (%s; exit %s); questions by mode %s" % (rep["status"], st.get("state"), rc, done)]
    for sl, v in (rep.get("analysis") or {}).get("slices", {}).items():
        lines.append("%-14s both %s C_beta %.4f C_1 %.4f diff %s (+%d / -%d) fields %s" % (
            sl, v["both_right"], v["C_beta_mean"], v["C_1_mean"], None if v["diff_mean"] is None else round(v["diff_mean"], 4),
            v["diff_positive"], v["diff_negative"], {f: None if x["mean_diff"] is None else round(x["mean_diff"], 4) for f, x in v["by_field"].items()}))
    if rep.get("analysis"):
        lines.append("continue %s" % rep["analysis"]["continue"])
    open(os.path.join(out, "report.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--units", default="data/exp46/units.jsonl")
    ap.add_argument("--manifest", default="data/exp46/MANIFEST.json")
    ap.add_argument("--regions", default="data/exp45/regions_readout.jsonl")
    ap.add_argument("--exp46", default="runs/exp46", help="EXP46's frozen outputs (details.jsonl): the references")
    ap.add_argument("--out", default="runs/exp47")
    ap.add_argument("--run", default=G_RUN)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--expect-sha", default="1", help="0: skip the G sha / frozen checks (tiny tests only)")
    ap.add_argument("--sets", default="")
    ap.add_argument("--finalize", default=None, metavar="EXIT_CODE")
    a = ap.parse_args(argv)
    if a.finalize is not None:
        rep = finalize(a.out, a.finalize, a.units)
        return 0 if rep["status"] == "complete" else 1
    from devision.infer import Decider, best_device
    from devision.model import question_mask
    from devision.train.rlcd import _forward, _Images, _items, box_grid
    spec = importlib.util.spec_from_file_location("v04b_freeze", os.path.join(HERE, "..", "v04b", "freeze.py"))
    assert spec is not None and spec.loader is not None
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    sets = json.loads(a.sets) if a.sets else SETS
    if a.expect_sha == "1" and frozen.check(a.run, list(SETS))["weights"] != G_SHA:
        raise SystemExit("EXP45-G is not c34b43dd / 2a5dfa64")
    inputs = e46.check_inputs(a.manifest, a.units, a.regions, sets, a.data_root)
    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()  # noqa: E731
    ref_path = os.path.join(a.exp46, "details.jsonl")
    ref = {(r["mode"], r["unit"], r["object"]): r for r in load_jsonl(ref_path)}
    units = [json.loads(x) for x in open(a.units)]
    keys = [(q["id"], q["qid"], u["set"]) for u in units for q in u["questions"]]
    if (len(units) != 48 and a.expect_sha == "1") or len(set(keys)) != len(keys):
        raise SystemExit("units: %d units, %d distinct of %d questions" % (len(units), len(set(keys)), len(keys)))
    for u in units:
        for qd in u["questions"]:
            for m in ("baseline", "oracle_target", "oracle_other"):
                if (m, u["key"], qd["object"]) not in ref:
                    raise SystemExit("EXP46 has no %s record for %s %s" % (m, u["key"], qd["object"]))
    regs = {}
    for line in open(a.regions):
        r = json.loads(line)
        regs[(r["set"], r["id"], r["qid"])] = r
    rows = {n: {r["id"]: r for r in (json.loads(x) for x in open(p))} for n, p in sets.items() if any(u["set"] == n for u in units)}
    frozen_probs = {}
    for n in rows:
        for x in (json.loads(l) for l in open(os.path.join(a.run, "eval", n + ".details.jsonl"))):
            frozen_probs[(n, x["sample_id"], x["qid"])] = x
    os.makedirs(a.out, exist_ok=True)
    status: Dict = {"state": "loading", "questions": len(keys), "units_done": 0, "forwards": 0}

    def save_status():
        tmp = os.path.join(a.out, "status.json.tmp")
        json.dump(status, open(tmp, "w"), indent=1)
        os.replace(tmp, os.path.join(a.out, "status.json"))       # never a half-written status
    save_status()
    device = best_device() if a.device == "auto" else a.device
    start = {"model": {"run": a.run, **G_SHA}, "inputs": inputs, "exp46_details_sha256": sha(ref_path),
             "code_sha256": sha(os.path.abspath(__file__)), "device": device, "modes": list(MODES),
             "beta": {"scale": BETA_SCALE, "max": BETA_MAX}}
    json.dump(start, open(os.path.join(a.out, "START.json"), "w"), indent=1)
    d = Decider.load(os.path.join(a.run, "stage2"), device=device)
    model, cond = d.model, d.model.condition
    if cond is None:
        raise SystemExit("no question conditioning module")
    images = _Images(a.data_root, d)
    side = d.cfg.image_size // int(getattr(d.model.vision.config, "patch_size"))
    n_visual = side * side // (d.cfg.visual_shuffle ** 2)
    weights_before = {k: v.detach().clone() for k, v in cond.state_dict().items()}
    for f in ("details.jsonl", "units.jsonl"):
        open(os.path.join(a.out, f), "w").close()
    status["state"] = "running"
    save_status()

    def stop(why: str) -> int:
        status["state"], status["reason"] = "stopped", why
        save_status()
        finalize(a.out, "stopped", a.units)
        return 2

    def close(p, q, keys_):
        for i, kk in enumerate(keys_):
            if not all(x == x for x in p[i]) or max(abs(x - y) for x, y in zip(p[i], q[i])) > 1e-5:
                return False
            if kk[max(range(len(p[i])), key=p[i].__getitem__)] != kk[max(range(len(q[i])), key=q[i].__getitem__)]:
                return False
        return True
    with torch.no_grad():
        for u in units:
            samples = []
            for qd in u["questions"]:
                r = rows[u["set"]][qd["id"]]
                samples.append(dict(r, questions={qd["qid"]: r["questions"][qd["qid"]]}, gold={qd["qid"]: r["gold"][qd["qid"]]}))
            items = _items(d, samples)
            batch = collate_items([items], d.tok.pad_token_id)
            assert batch is not None
            pixels = torch.stack([images(it["image"]) for it in items])
            keys_ = [it["keys"] for it in items]
            sup = []
            for qd in u["questions"]:
                g = regs[(u["set"], qd["id"], qd["qid"])]
                sup.append([c > 0 for c in box_grid([g["box"]], g["width"], g["height"], d.cfg.image_size, grid=side, smooth=0.0)])
            target, other = torch.tensor(sup), torch.tensor(sup[::-1])
            fr = [frozen_probs[(u["set"], qd["id"], qd["qid"])] for qd in u["questions"]]
            plain = e46.probs_of(_forward(model, batch, pixels, device), items)
            status["forwards"] += 1
            refp = lambda m: [[ref[(m, u["key"], qd["object"])]["probabilities"][j] for j in range(len(it["keys"]))]  # noqa: E731
                              for qd, it in zip(u["questions"], items)]
            frozen_p = [[f["probabilities"][k] for k in kk] for f, kk in zip(fr, keys_)]
            if not close(plain, frozen_p, keys_) or not close(plain, refp("baseline"), keys_):
                return stop("%s: plain forward differs from G's frozen record or EXP46's baseline" % u["key"])
            res: Dict = {}
            for mode in MODES:
                sink: list = []
                with patched(cond, mode, target, other, sink):
                    logits, hidden = _forward(model, batch, pixels, device, return_hidden=True)
                status["forwards"] += 1
                pr = e46.probs_of(logits, items)
                ok = {"baseline": lambda: close(pr, plain, keys_), "target_1": lambda: close(pr, refp("oracle_target"), keys_),
                      "other_1": lambda: close(pr, refp("oracle_other"), keys_)}.get(mode, lambda: all(all(x == x for x in p) for p in pr))()
                if not ok:
                    return stop("%s: %s fails its fidelity check" % (u["key"], mode))
                mpos = batch["marker_pos"].to(hidden.device) + n_visual
                marks = [hidden[i, mpos[i][batch["marker_mask"][i].to(hidden.device)]] for i in range(2)]
                res[mode] = {"visual": sink[0][0], "rec": sink[0][1], "probs": pr, "marks": marks, "logits": centred(logits, items)}
            rec_unit = {"unit": u["key"], **{k: v for k, v in res["target_gain"]["rec"].items()},
                        "path": {}}
            for plus, minus in (("target_1", "other_1"), ("target_gain", "other_gain")):
                rec_unit["path"]["%s-%s" % (plus, minus)] = {
                    "module_output": half(res[plus]["visual"], res[minus]["visual"]),
                    "option_states": [half(res[plus]["marks"][i], res[minus]["marks"][i]) for i in range(2)],
                    "option_logits": [half(res[plus]["logits"][i], res[minus]["logits"][i]) for i in range(2)]}
            with open(os.path.join(a.out, "details.jsonl"), "a") as fh:
                for mode in MODES:
                    for i, (qd, it) in enumerate(zip(u["questions"], items)):
                        pr = res[mode]["probs"][i]
                        kk = it["keys"]
                        pred = kk[max(range(len(pr)), key=pr.__getitem__)]
                        other_gold = u["questions"][1 - i]["gold"]
                        margin = pr[kk.index(qd["gold"])] - pr[kk.index(other_gold)] if other_gold in kk else None
                        fh.write(json.dumps({"mode": mode, "unit": u["key"], "set": u["set"], "split": u["split"], "layout": u["layout"],
                                             "field": u["field"], "object": qd["object"], "side": qd["side"], "id": qd["id"], "qid": qd["qid"],
                                             "probabilities": pr, "prediction": pred, "gold": qd["gold"], "correct": pred == qd["gold"],
                                             "margin": margin}) + "\n")
            with open(os.path.join(a.out, "units.jsonl"), "a") as fh:
                fh.write(json.dumps(rec_unit) + "\n")
            status["units_done"] += 1
            save_status()
    if any(not torch.equal(v, cond.state_dict()[k]) for k, v in weights_before.items()):
        return stop("the module's weights changed")
    status["state"] = "done"
    save_status()
    rep = finalize(a.out, "0", a.units)
    return 0 if rep["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
