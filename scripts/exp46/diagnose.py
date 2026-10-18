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
    rec: Dict = {"p_norm": p.float().norm(dim=-1).tolist(), "d_rms": float(rms((p[0] - p[1]) / 2)), "alpha": None}
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
    rec["v_rms"], rec["residual_rms"] = float(rms(visual)), float(rms(res))
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--units", default="data/exp46/units.jsonl")
    ap.add_argument("--regions", default="data/exp45/regions_readout.jsonl")
    ap.add_argument("--out", default="runs/exp46")
    ap.add_argument("--run", default=G_RUN)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--expect-sha", default="1", help="0: skip the G sha / frozen checks (tiny tests only)")
    ap.add_argument("--sets", default="", help="JSON {set: path}; default the four readout sets (tiny tests only)")
    a = ap.parse_args(argv)
    sys.path.insert(0, os.path.join(HERE, "..", "v04b"))
    from devision.infer import Decider, best_device
    from devision.model import question_mask
    from devision.train.rlcd import _forward, _Images, _items, box_grid
    import importlib.util
    spec = importlib.util.spec_from_file_location("v04b_freeze", os.path.join(HERE, "..", "v04b", "freeze.py"))
    assert spec is not None and spec.loader is not None
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    if a.expect_sha == "1":
        fr = frozen.check(a.run, list(SETS))["weights"]
        if fr != G_SHA:
            raise SystemExit("EXP45-G is not c34b43dd / 2a5dfa64")
    units = [json.loads(x) for x in open(a.units)]
    keys = [(q["id"], q["qid"], u["set"]) for u in units for q in u["questions"]]
    if (len(units) != 48 and a.expect_sha == "1") or len(set(keys)) != len(keys):
        raise SystemExit("units: %d units, %d questions, %d distinct" % (len(units), len(keys), len(set(keys))))
    regs = {}
    for line in open(a.regions):
        r = json.loads(line)
        regs[(r["set"], r["id"], r["qid"])] = r
    sets = json.loads(a.sets) if a.sets else SETS
    rows = {n: {r["id"]: r for r in (json.loads(x) for x in open(p))} for n, p in sets.items() if any(u["set"] == n for u in units)}
    frozen_probs = {}
    for n in rows:
        for x in (json.loads(l) for l in open(os.path.join(a.run, "eval", n + ".details.jsonl"))):
            frozen_probs[(n, x["sample_id"], x["qid"])] = x
    device = best_device() if a.device == "auto" else a.device
    d = Decider.load(os.path.join(a.run, "stage2"), device=device)
    model, cond = d.model, d.model.condition
    if cond is None:
        raise SystemExit("no question conditioning module")
    images = _Images(a.data_root, d)
    side = d.cfg.image_size // int(getattr(d.model.vision.config, "patch_size"))
    os.makedirs(a.out, exist_ok=True)
    start = {"model": {"run": a.run, **G_SHA}, "units_sha256": hashlib.sha256(open(a.units, "rb").read()).hexdigest(),
             "regions_sha256": hashlib.sha256(open(a.regions, "rb").read()).hexdigest(), "device": device, "modes": list(MODES),
             "alpha": {"scale": ALPHA_SCALE, "max": ALPHA_MAX}}
    json.dump(start, open(os.path.join(a.out, "START.json"), "w"), indent=1)
    weights_before = {k: v.detach().clone() for k, v in cond.state_dict().items()}
    recs, max_base_diff, max_instr_diff = [], 0.0, 0.0
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
            diff_tokens = [int(i) for i in ((batch["input_ids"][0] != batch["input_ids"][1]) & qm[0]).nonzero().flatten()]
            sup = []
            for qd in u["questions"]:
                g = regs[(u["set"], qd["id"], qd["qid"])]
                sup.append([c > 0 for c in box_grid([g["box"]], g["width"], g["height"], d.cfg.image_size, grid=side, smooth=0.0)])
            target, other = torch.tensor(sup), torch.tensor(sup[::-1])
            plain = probs_of(_forward(model, batch, pixels, device), items)            # unpatched: decide's own path
            for i, (qd, it) in enumerate(zip(u["questions"], items)):
                f = frozen_probs[(u["set"], qd["id"], qd["qid"])]
                diff = max(abs(x - f["probabilities"][k]) for x, k in zip(plain[i], it["keys"]))
                max_base_diff = max(max_base_diff, diff)
                if it["keys"][max(range(len(plain[i])), key=plain[i].__getitem__)] != f["prediction"]:
                    raise SystemExit("baseline prediction differs from G's frozen record at %s %s" % (qd["id"], qd["qid"]))
            if max_base_diff > 1e-5:
                raise SystemExit("baseline does not reproduce G's frozen probabilities (max difference %.2e)" % max_base_diff)
            for mode in MODES:
                sink: list = []
                support = target if mode == "oracle_target" else other if mode == "oracle_other" else None
                with patched(cond, mode, support, sink):
                    pr = probs_of(_forward(model, batch, pixels, device), items)
                rec = sink[0]
                if mode == "baseline":
                    max_instr_diff = max(max_instr_diff, max(abs(x - y) for a_, b_ in zip(pr, plain) for x, y in zip(a_, b_)))
                att = rec["attention"]                                               # [2, heads, patches]
                mean = att.mean(1)
                l1 = float((mean[0] - mean[1]).abs().sum())
                for i, (qd, it) in enumerate(zip(u["questions"], items)):
                    pred = it["keys"][max(range(len(pr[i])), key=pr[i].__getitem__)]
                    recs.append({"mode": mode, "unit": u["key"], "set": u["set"], "split": u["split"], "layout": u["layout"],
                                 "field": u["field"], "object": qd["object"], "side": qd["side"], "id": qd["id"], "qid": qd["qid"],
                                 "probabilities": pr[i], "prediction": pred, "gold": qd["gold"], "correct": pred == qd["gold"],
                                 "mass_target": float(mean[i][target[i]].sum()), "mass_other": float(mean[i][other[i]].sum()),
                                 "head_mass_target": [float(att[i, h][target[i]].sum()) for h in range(att.size(1))],
                                 "head_mass_other": [float(att[i, h][other[i]].sum()) for h in range(att.size(1))],
                                 "top_patch": int(mean[i].argmax()), "ab_l1": l1, "p_norm": rec["p_norm"][i], "d_rms": rec["d_rms"],
                                 "v_rms": rec["v_rms"], "residual_rms": rec["residual_rms"], "alpha": rec["alpha"],
                                 "diff_tokens": diff_tokens, "qk_object_score": "missing"})
    if any(not torch.equal(v, cond.state_dict()[k]) for k, v in weights_before.items()):
        raise SystemExit("the module's weights changed")
    with open(os.path.join(a.out, "details.jsonl"), "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    rep = {"start": start, "baseline_vs_frozen_max_diff": max_base_diff, "instrumented_vs_plain_max_diff": max_instr_diff,
           "questions": len(recs) // len(MODES), "summary": summarize(recs)}
    json.dump(rep, open(os.path.join(a.out, "report.json"), "w"), indent=1)
    lines = ["baseline vs frozen %.2e; instrumented vs plain %.2e" % (max_base_diff, max_instr_diff)]
    for mode, s in rep["summary"].items():
        x = s["all"]
        lines.append("%-19s right %d/%d both %d/%d target %.3f other %.3f own-box both %d top-differs %d w->r %s r->w %s" % (
            mode, x["right"], x["questions"], x["both_right"], x["units"], x["mass_target"], x["mass_other"],
            x["both_prefer_own_box"], x["ab_top_patch_differs"], x["wrong_to_right"], x["right_to_wrong"]))
    open(os.path.join(a.out, "report.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
