"""exp46 causal diagnosis of the frozen EXP45-G (critic EXP46-PREP-01..03; docs/experiments/2026-10-18-exp46-causal-plan.md).
No training, no weight written, eval / no_grad, temperature 1. On the 48 fixed units (data/exp46/units.jsonl; each unit
= one picture and field with both of its reading questions, A / Pair 1 then B / Pair 2, in one forward of 2), six modes
of the question conditioning step (model.QuestionConditioning), everything else -- picture, question tokens, options,
the text encoder and decision head -- untouched:

  baseline            the module as it is (recorded); first the unpatched forward must reproduce G's frozen raw
                      probabilities (max difference <= 1e-5, same predictions) or the diagnosis stops;
  swap_query          the two questions' projected question vectors p = question(q) exchanged;
  amplify_difference  c = (pA + pB) / 2, d = (pA - pB) / 2, queries V + c +- alpha d with alpha =
                      max(1, min(1000, 0.1 x RMS(V + c) / max(RMS(d), 1e-12))), one alpha per pair (fixed formula);
  oracle_target       the attention (all heads) restricted before the softmax to the patches the asked field's box
                      covers (sidecar box; this mode and the next only), renormalised inside;
  oracle_other        the same, restricted to the other object's same-field box;
  no_residual         the module's added residual set to zero (the original 64 visual tokens stay).

The module's forward is replaced only inside a `with` block and restored after each mode. Per question: probabilities,
prediction, correct; the attention per head averaged over the visual-token queries, its mass on the asked / other box;
the RMS of V, of the residual, the norm of p and of the pair difference d, alpha. Not measured: a separate query-key
"object difference score" (marked missing rather than recomputed outside the module). Writes runs/exp46/START.json,
details.jsonl, report.json / report.txt.

    uv run python scripts/exp46/diagnose.py
"""
import argparse
import contextlib
import hashlib
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional

import torch
from laya.common import collate_items

HERE = os.path.dirname(__file__)
MODES = ("baseline", "swap_query", "amplify_difference", "oracle_target", "oracle_other", "no_residual")
G_RUN = "runs/exp45-g"
G_SHA = {"model.safetensors": "c34b43ddde846af40e154aefb8ca1a381c27325e95b03aca55d42b75d4e7c15e",
         "config.json": "2a5dfa64d33846052bf8c13862bd0def53888fce7160a18c25d38279ce89ab08"}
SETS = {"fit_O": "data/exp42/fit_O.jsonl", "fit_L": "data/exp42/fit_L.jsonl",
        "rl_tune": "data/real_layout/questions_tune.jsonl", "rl_tune_L": "data/exp40/rl_tune_L.jsonl"}
ALPHA_SCALE, ALPHA_MAX = 0.1, 1000.0


def rms(x: torch.Tensor) -> torch.Tensor:
    return x.float().pow(2).mean().sqrt()


def conditioned(cond, visual, patches, word_embeds, qmask, mode: str, support: Optional[torch.Tensor] = None):
    """The module's step under `mode` for a batch [A, B] of one picture: (new visual tokens, record). Uses the module's
    own layers; baseline is the module's computation (with the attention weights returned)."""
    m = qmask.unsqueeze(-1).to(word_embeds.dtype)
    q = (word_embeds * m).sum(1) / m.sum(1).clamp(min=1)
    p = cond.question(q)                                            # [2, D]
    c0, d0 = (p[0] + p[1]) / 2, (p[0] - p[1]) / 2
    # pre-intervention quantities (the original q and p of the two questions)
    rec: Dict = {"q_norm": q.float().norm(dim=-1).tolist(), "q_diff_rms": float(rms((q[0] - q[1]) / 2)),
                 "p_norm": p.float().norm(dim=-1).tolist(), "d_rms": float(rms(d0)), "vc_rms": float(rms(visual + c0)),
                 "zero_difference": bool(float(rms(d0)) == 0.0), "alpha": None, "alpha_capped": None}
    if mode == "swap_query":
        p = p.flip(0)
    elif mode == "amplify_difference":
        c, d = (p[0] + p[1]) / 2, (p[0] - p[1]) / 2
        alpha = max(1.0, min(ALPHA_MAX, ALPHA_SCALE * float(rms(visual + c)) / max(float(rms(d)), 1e-12)))
        rec["alpha"], rec["alpha_capped"] = alpha, alpha >= ALPHA_MAX
        p = torch.stack([c + alpha * d, c - alpha * d])
    query = visual + p.unsqueeze(1)
    kv = cond.patches(patches.to(visual.dtype))
    heads = cond.attend.num_heads
    mask = None
    if mode in ("oracle_target", "oracle_other"):
        assert support is not None
        allow = support.to(query.device)                             # [2, patches] bool
        mask = torch.zeros(allow.shape, dtype=query.dtype, device=query.device).masked_fill(~allow, float("-inf"))
        mask = mask[:, None, None, :].expand(-1, heads, query.size(1), -1).reshape(-1, query.size(1), allow.size(1))
    out, w = cond.attend(query, kv, kv, need_weights=True, average_attn_weights=False, attn_mask=mask)
    res = cond.out(out)
    rec["v_rms"], rec["residual_rms_computed"] = float(rms(visual)), float(rms(res))
    rec["residual_rms_applied"] = 0.0 if mode == "no_residual" else float(rms(res))   # no_residual: computed, not added
    rec["attention"] = w.float().mean(2)                             # [2, heads, patches], averaged over queries
    new = visual if mode == "no_residual" else visual + res
    return new, rec


@contextlib.contextmanager
def patched(cond, mode: str, support=None, sink: Optional[list] = None):
    """Replace the module's forward by `conditioned` under `mode` inside the block; always restored."""
    original = cond.forward

    def forward(visual, patches, word_embeds, qmask):
        new, rec = conditioned(cond, visual, patches, word_embeds, qmask, mode, support)
        if sink is not None:
            sink.append(rec)
        return new
    cond.forward = forward
    try:
        yield
    finally:
        cond.forward = original


def probs_of(logits, items) -> List[List[float]]:
    return [torch.softmax(logits[i, :len(it["markers"])].float(), -1).tolist() for i, it in enumerate(items)]


def summarize(recs: List[Dict]) -> Dict:
    """Per mode (and split / layout / field): questions right, units with both right, mean attention mass on the asked
    and on the other box, units where both questions put more mass on their own box, A / B attention L1 and top-patch
    change; changes against baseline (wrong -> right, right -> wrong)."""
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
            out[mode][g] = {
                "questions": len(gr), "right": sum(r["correct"] for r in gr), "units": len(pairs),
                "both_right": sum(all(x["correct"] for x in v) for v in pairs),
                "mass_target": sum(r["mass_target"] for r in gr) / len(gr), "mass_other": sum(r["mass_other"] for r in gr) / len(gr),
                "both_prefer_own_box": sum(all(x["mass_target"] > x["mass_other"] for x in v) for v in pairs),
                "ab_attention_l1": sum(v[0]["ab_l1"] for v in pairs) / len(pairs) if pairs else None,
                "ab_top_patch_differs": sum(v[0]["top_patch"] != v[1]["top_patch"] for v in pairs),
                "wrong_to_right": sum((not base[(r["unit"], r["object"])]["correct"]) and r["correct"] for r in gr) if base else None,
                "right_to_wrong": sum(base[(r["unit"], r["object"])]["correct"] and not r["correct"] for r in gr) if base else None,
                "prediction_changed": sum(base[(r["unit"], r["object"])]["prediction"] != r["prediction"] for r in gr) if base else None}
    return out


def baseline_check(pr: List[List[float]], plain: List[List[float]], frozen: List[Dict], keys: List[List[str]]) -> str:
    """'' when the instrumented baseline is finite, within 1e-5 of the plain forward and of G's frozen raw
    probabilities, with the same predictions; else why not (critic EXP46-R1-01)."""
    for i, (a_, b_, f, k) in enumerate(zip(pr, plain, frozen, keys)):
        if not all(x == x and abs(x) != float("inf") for x in a_):
            return "question %d: instrumented probabilities not finite" % i
        if max(abs(x - y) for x, y in zip(a_, b_)) > 1e-5 or max(abs(x - f["probabilities"][kk]) for x, kk in zip(a_, k)) > 1e-5:
            return "question %d: instrumented baseline more than 1e-5 from the plain forward / frozen record" % i
        top = k[max(range(len(a_)), key=a_.__getitem__)]
        if top != k[max(range(len(b_)), key=b_.__getitem__)] or top != f["prediction"]:
            return "question %d: instrumented baseline prediction differs" % i
    return ""


def check_inputs(manifest_path: str, units_path: str, regions_path: str, sets: Dict[str, str], data_root: str) -> Dict:
    """The selection's own MANIFEST: units, the four sources, the sidecar and every picture still have the sha256 it
    recorded (critic EXP46-R1-04); before any model is loaded."""
    m = json.load(open(manifest_path))
    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()  # noqa: E731
    bad = []
    if sha(units_path) != m["sha256"]:
        bad.append("units")
    for n, p in sets.items():
        if n in m["sources"] and sha(p) != m["sources"][n]:
            bad.append(n)
    if sha(regions_path) != m["sources"]["regions_readout"]:
        bad.append("regions_readout")
    for pic, h in m["pictures_sha256"].items():
        if sha(os.path.join(data_root, pic)) != h:
            bad.append(pic)
    if bad:
        raise SystemExit("inputs differ from data/exp46/MANIFEST.json: %s" % bad)
    return {"manifest_sha256": sha(manifest_path), "checked": ["units", "regions_readout"] + sorted(sets) + ["%d pictures" % len(m["pictures_sha256"])]}


def finalize(out: str, rc: str = "") -> Dict:
    """The report from what was written so far (no model): complete only when every mode covered every question;
    otherwise "incomplete" with the status / exit code and each mode's completed and missing counts (EXP46-R1-03)."""
    st: Dict = json.load(open(os.path.join(out, "status.json"))) if os.path.exists(os.path.join(out, "status.json")) else {"state": "not started"}
    recs = [json.loads(x) for x in open(os.path.join(out, "details.jsonl"))] if os.path.exists(os.path.join(out, "details.jsonl")) else []
    n = int(st.get("questions", 0))
    done = {m: sum(r["mode"] == m for r in recs) for m in MODES}
    complete = st.get("state") == "done" and n > 0 and all(done[m] == n for m in MODES)
    rep = {"status": "complete" if complete else "incomplete", "state": st, "exit_code": rc,
           "completed_questions_by_mode": done, "missing_by_mode": {m: n - done[m] for m in MODES},
           "summary": summarize(recs) if recs else {}}
    if os.path.exists(os.path.join(out, "START.json")):
        rep["start"] = json.load(open(os.path.join(out, "START.json")))
    json.dump(rep, open(os.path.join(out, "report.json"), "w"), indent=1)
    lines = ["status %s (%s; exit %s); questions by mode %s" % (rep["status"], st.get("state"), rc, done),
             "baseline vs frozen %s; instrumented vs plain %s" % (st.get("max_plain_vs_frozen"), st.get("max_instrumented_vs_plain"))]
    for mode, s_ in rep["summary"].items():
        x = s_["all"]
        lines.append("%-19s right %d/%d both %d/%d target %.3f other %.3f own-box both %d top-differs %d w->r %s r->w %s" % (
            mode, x["right"], x["questions"], x["both_right"], x["units"], x["mass_target"], x["mass_other"],
            x["both_prefer_own_box"], x["ab_top_patch_differs"], x["wrong_to_right"], x["right_to_wrong"]))
    open(os.path.join(out, "report.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--units", default="data/exp46/units.jsonl")
    ap.add_argument("--manifest", default="data/exp46/MANIFEST.json")
    ap.add_argument("--regions", default="data/exp45/regions_readout.jsonl")
    ap.add_argument("--out", default="runs/exp46")
    ap.add_argument("--run", default=G_RUN)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--expect-sha", default="1", help="0: skip the G sha / frozen checks (tiny tests only)")
    ap.add_argument("--sets", default="", help="JSON {set: path}; default the four readout sets (tiny tests only)")
    ap.add_argument("--finalize", default=None, metavar="EXIT_CODE", help="write the report from what exists (no model)")
    a = ap.parse_args(argv)
    if a.finalize is not None:
        rep = finalize(a.out, a.finalize)
        return 0 if rep["status"] == "complete" else 1
    from devision.infer import Decider, best_device
    from devision.model import question_mask
    from devision.train.rlcd import _forward, _Images, _items, box_grid
    import importlib.util
    spec = importlib.util.spec_from_file_location("v04b_freeze", os.path.join(HERE, "..", "v04b", "freeze.py"))
    assert spec is not None and spec.loader is not None
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    sets = json.loads(a.sets) if a.sets else SETS
    if a.expect_sha == "1":
        fr = frozen.check(a.run, list(SETS))["weights"]
        if fr != G_SHA:
            raise SystemExit("EXP45-G is not c34b43dd / 2a5dfa64")
    inputs = check_inputs(a.manifest, a.units, a.regions, sets, a.data_root)
    units = [json.loads(x) for x in open(a.units)]
    keys = [(q["id"], q["qid"], u["set"]) for u in units for q in u["questions"]]
    if (len(units) != 48 and a.expect_sha == "1") or len(set(keys)) != len(keys):
        raise SystemExit("units: %d units, %d questions, %d distinct" % (len(units), len(keys), len(set(keys))))
    regs = {}
    for line in open(a.regions):
        r = json.loads(line)
        regs[(r["set"], r["id"], r["qid"])] = r
    rows = {n: {r["id"]: r for r in (json.loads(x) for x in open(p))} for n, p in sets.items() if any(u["set"] == n for u in units)}
    frozen_probs = {}
    for n in rows:
        for x in (json.loads(l) for l in open(os.path.join(a.run, "eval", n + ".details.jsonl"))):
            frozen_probs[(n, x["sample_id"], x["qid"])] = x
    for u in units:                                   # selection / source / frozen record agree, question by question
        for qd in u["questions"]:
            r, f = rows[u["set"]][qd["id"]], frozen_probs.get((u["set"], qd["id"], qd["qid"]))
            src_gold = next(o for o, p in r["gold"][qd["qid"]]["probabilities"].items() if p == 1.0)
            if f is None or r["image"] != u["picture"] or src_gold != qd["gold"] or f.get("gold_answer", src_gold) != src_gold \
                    or list(f.get("options", r["questions"][qd["qid"]]["criteria"])) != list(r["questions"][qd["qid"]]["criteria"]) \
                    or f.get("image_used", r["image"]) != r["image"]:
                raise SystemExit("%s %s: selection, source and frozen record disagree (picture, gold or options)" % (qd["id"], qd["qid"]))
    os.makedirs(a.out, exist_ok=True)
    start = {"model": {"run": a.run, **G_SHA}, "inputs": inputs, "units_sha256": hashlib.sha256(open(a.units, "rb").read()).hexdigest(),
             "regions_sha256": hashlib.sha256(open(a.regions, "rb").read()).hexdigest(), "modes": list(MODES),
             "alpha": {"scale": ALPHA_SCALE, "max": ALPHA_MAX}}
    status: Dict = {"state": "loading", "questions": len(keys), "units_done": 0, "max_plain_vs_frozen": 0.0, "max_instrumented_vs_plain": 0.0}

    def save_status():
        json.dump(status, open(os.path.join(a.out, "status.json"), "w"), indent=1)
    save_status()
    device = best_device() if a.device == "auto" else a.device
    start["device"] = device
    json.dump(start, open(os.path.join(a.out, "START.json"), "w"), indent=1)
    d = Decider.load(os.path.join(a.run, "stage2"), device=device)
    model, cond = d.model, d.model.condition
    if cond is None:
        raise SystemExit("no question conditioning module")
    images = _Images(a.data_root, d)
    side = d.cfg.image_size // int(getattr(d.model.vision.config, "patch_size"))
    weights_before = {k: v.detach().clone() for k, v in cond.state_dict().items()}
    open(os.path.join(a.out, "details.jsonl"), "w").close()
    status["state"] = "running"
    save_status()

    def stop(why: str) -> int:
        status["state"], status["reason"] = "stopped", why
        save_status()
        finalize(a.out, "stopped")
        return 2
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
            qm = question_mask(batch["input_ids"], batch["marker_pos"])
            diff = ((batch["input_ids"][0] != batch["input_ids"][1]) & qm[0]).nonzero().flatten().tolist()
            trace = {"input_ids_sha256": [hashlib.sha256(json.dumps(batch["input_ids"][i].tolist()).encode()).hexdigest() for i in range(2)],
                     "question_mask_sha256": [hashlib.sha256(json.dumps(qm[i].nonzero().flatten().tolist()).encode()).hexdigest() for i in range(2)],
                     "diff_positions": diff, "diff_token_ids": [[int(batch["input_ids"][i, j]) for j in diff] for i in range(2)],
                     "diff_tokens": [d.tok.decode([int(batch["input_ids"][i, j]) for j in diff]) for i in range(2)]}
            sup = []
            for qd in u["questions"]:
                g = regs[(u["set"], qd["id"], qd["qid"])]
                sup.append([c > 0 for c in box_grid([g["box"]], g["width"], g["height"], d.cfg.image_size, grid=side, smooth=0.0)])
            target, other = torch.tensor(sup), torch.tensor(sup[::-1])
            fr_recs = [frozen_probs[(u["set"], qd["id"], qd["qid"])] for qd in u["questions"]]
            plain = probs_of(_forward(model, batch, pixels, device), items)              # unpatched: decide's own path
            for i, (it, f) in enumerate(zip(items, fr_recs)):
                if not all(x == x for x in plain[i]):
                    return stop("%s: plain probabilities not finite" % u["key"])
                status["max_plain_vs_frozen"] = max(status["max_plain_vs_frozen"], max(abs(x - f["probabilities"][k]) for x, k in zip(plain[i], it["keys"])))
                if it["keys"][max(range(len(plain[i])), key=plain[i].__getitem__)] != f["prediction"]:
                    return stop("%s: plain prediction differs from G's frozen record" % u["key"])
            if status["max_plain_vs_frozen"] > 1e-5:
                return stop("plain forward more than 1e-5 from G's frozen probabilities (%.2e)" % status["max_plain_vs_frozen"])
            unit_recs = []
            for mode in MODES:
                sink: list = []
                support = target if mode == "oracle_target" else other if mode == "oracle_other" else None
                with patched(cond, mode, support, sink):
                    pr = probs_of(_forward(model, batch, pixels, device), items)
                rec = sink[0]
                if mode == "baseline":                      # the instrumented path must be the plain one (EXP46-R1-01)
                    status["max_instrumented_vs_plain"] = max(status["max_instrumented_vs_plain"],
                                                              max(abs(x - y) for a_, b_ in zip(pr, plain) for x, y in zip(a_, b_)))
                    why = baseline_check(pr, plain, fr_recs, [it["keys"] for it in items])
                    if why:
                        return stop("%s: %s" % (u["key"], why))
                att = rec["attention"]
                mean = att.mean(1)
                l1 = float((mean[0] - mean[1]).abs().sum())
                for i, (qd, it) in enumerate(zip(u["questions"], items)):
                    pred = it["keys"][max(range(len(pr[i])), key=pr[i].__getitem__)]
                    unit_recs.append({"mode": mode, "unit": u["key"], "set": u["set"], "split": u["split"], "layout": u["layout"],
                                      "field": u["field"], "object": qd["object"], "side": qd["side"], "id": qd["id"], "qid": qd["qid"],
                                      "probabilities": pr[i], "prediction": pred, "gold": qd["gold"], "correct": pred == qd["gold"],
                                      "mass_target": float(mean[i][target[i]].sum()), "mass_other": float(mean[i][other[i]].sum()),
                                      "head_mass_target": [float(att[i, h][target[i]].sum()) for h in range(att.size(1))],
                                      "head_mass_other": [float(att[i, h][other[i]].sum()) for h in range(att.size(1))],
                                      "top_patch": int(mean[i].argmax()), "ab_l1": l1,
                                      "q_norm": rec["q_norm"][i], "q_diff_rms": rec["q_diff_rms"], "p_norm": rec["p_norm"][i],
                                      "d_rms": rec["d_rms"], "vc_rms": rec["vc_rms"], "zero_difference": rec["zero_difference"],
                                      "alpha": rec["alpha"], "alpha_capped": rec["alpha_capped"], "v_rms": rec["v_rms"],
                                      "residual_rms_computed": rec["residual_rms_computed"], "residual_rms_applied": rec["residual_rms_applied"],
                                      "input_ids_sha256": trace["input_ids_sha256"][i], "question_mask_sha256": trace["question_mask_sha256"][i],
                                      "diff_positions": trace["diff_positions"], "diff_token_ids": trace["diff_token_ids"][i],
                                      "diff_tokens": trace["diff_tokens"][i], "qk_object_score": "missing"})
            with open(os.path.join(a.out, "details.jsonl"), "a") as fh:     # written unit by unit
                for r in unit_recs:
                    fh.write(json.dumps(r) + "\n")
            status["units_done"] += 1
            save_status()
    if any(not torch.equal(v, cond.state_dict()[k]) for k, v in weights_before.items()):
        return stop("the module's weights changed")
    status["state"] = "done"
    save_status()
    rep = finalize(a.out, "0")
    return 0 if rep["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
