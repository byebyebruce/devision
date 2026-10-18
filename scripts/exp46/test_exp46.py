"""exp46 without a real model: the instrumented baseline is the module's own computation; swap / amplify change only the
projected question vectors; the oracle modes restrict and renormalise the attention to the given patches (the other
object's box for oracle_other); no_residual returns the visual tokens unchanged; the module's forward is restored after
every mode, nothing gets a gradient or a new weight; the fixed selection is complete and balanced; the full diagnosis
runs on a tiny checkpoint and refuses a baseline that does not reproduce the frozen records, or repeated questions."""
import importlib.util
import json
import os
import sys

import pytest
import torch
from PIL import Image

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tests"))
spec = importlib.util.spec_from_file_location("exp46_diagnose", os.path.join(HERE, "diagnose.py"))
assert spec is not None and spec.loader is not None
dg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dg)


def module():
    from conftest import tiny_decider  # type: ignore
    d = tiny_decider(seed=1)
    d.model.add_question_conditioning()
    cond = d.model.condition
    torch.manual_seed(5)
    torch.nn.init.normal_(cond.out.weight, std=0.1)                 # non-zero, so the residual matters
    torch.manual_seed(0)
    visual = torch.randn(1, 4, 64).expand(2, -1, -1).contiguous()   # one picture, two questions
    patches = torch.randn(1, 16, 32).expand(2, -1, -1).contiguous()
    words = torch.randn(2, 5, 64)
    qmask = torch.ones(2, 5, dtype=torch.bool)
    return d, cond, visual, patches, words, qmask


def test_baseline_is_the_modules_own_computation_and_no_residual_keeps_v():
    d, cond, v, pt, w, qm = module()
    with torch.no_grad():
        ref = cond(v, pt, w, qm)
        base, rec = dg.conditioned(cond, v, pt, w, qm, "baseline")
        none, _ = dg.conditioned(cond, v, pt, w, qm, "no_residual")
    assert torch.allclose(ref, base, atol=1e-6) and rec["residual_rms"] > 0
    assert torch.equal(none, v)


def test_swap_and_amplify_change_only_the_question_vectors():
    d, cond, v, pt, w, qm = module()
    with torch.no_grad():
        swapped, _ = dg.conditioned(cond, v, pt, w, qm, "swap_query")
        flipped, _ = dg.conditioned(cond, v, pt, w.flip(0), qm, "baseline")     # same as feeding the other question's words
        amp, rec = dg.conditioned(cond, v, pt, w, qm, "amplify_difference")
        p = cond.question((w * qm.unsqueeze(-1)).sum(1) / qm.sum(1, keepdim=True))
        c, dd = (p[0] + p[1]) / 2, (p[0] - p[1]) / 2
        alpha = max(1.0, min(1000.0, 0.1 * float(dg.rms(v + c)) / max(float(dg.rms(dd)), 1e-12)))
        base, _ = dg.conditioned(cond, v, pt, w, qm, "baseline")
    assert torch.allclose(swapped, flipped, atol=1e-6)
    assert rec["alpha"] == pytest.approx(alpha) and rec["alpha_capped"] == (alpha >= 1000)
    assert not torch.allclose(amp, base) or alpha == 1.0


def test_oracle_restricts_and_renormalises_the_attention():
    d, cond, v, pt, w, qm = module()
    support = torch.zeros(2, 16, dtype=torch.bool)
    support[0, :4], support[1, 8:12] = True, True
    with torch.no_grad():
        _, rec = dg.conditioned(cond, v, pt, w, qm, "oracle_target", support)
        _, rec2 = dg.conditioned(cond, v, pt, w, qm, "oracle_other", support.flip(0))
    att = rec["attention"]                                           # [2, heads, 16]
    assert torch.allclose(att.sum(-1), torch.ones_like(att.sum(-1)), atol=1e-5)
    assert (att[0][:, ~support[0]] == 0).all() and (att[1][:, ~support[1]] == 0).all()
    assert (rec2["attention"][0][:, ~support[1]] == 0).all()        # the other object's box


def test_forward_is_restored_and_nothing_is_trained():
    d, cond, v, pt, w, qm = module()
    original = cond.forward
    before = {k: x.clone() for k, x in cond.state_dict().items()}
    with pytest.raises(RuntimeError):
        with dg.patched(cond, "swap_query"):
            assert cond.forward is not original
            raise RuntimeError("inside")
    assert cond.forward == original
    with torch.no_grad(), dg.patched(cond, "amplify_difference", sink=[]):
        out = cond(v, pt, w, qm)
    assert not out.requires_grad and all(torch.equal(x, cond.state_dict()[k]) for k, x in before.items())


def test_selection_is_complete_and_balanced():
    m = json.load(open(os.path.join(ROOT, "data", "exp46", "MANIFEST.json")))
    units = [json.loads(x) for x in open(os.path.join(ROOT, "data", "exp46", "units.jsonl"))]
    assert m["units"] == len(units) == 48 and set(m["distribution"].values()) == {2} and len(m["distribution"]) == 24
    for u in units:
        a, b = u["questions"]
        assert a["gold"] != b["gold"] and a["object"] in ("A", "1") and a["side"] == u["first_side"]


def tiny_run(tmp_path, tamper=None, dup=False):
    from conftest import tiny_decider  # type: ignore
    from laya.common import collate_items
    from devision.train.rlcd import _forward, _Images, _items
    d = tiny_decider(seed=1)
    d.model.add_question_conditioning()
    d.cfg.question_conditioning = True
    torch.nn.init.normal_(d.model.condition.out.weight, std=0.1)
    run = tmp_path / "run"
    d.save(run / "stage2")
    (tmp_path / "data" / "images").mkdir(parents=True)
    rows, regs, units = [], [], []
    for i in range(2):
        Image.new("RGB", (60, 40), (90 * i, 90, 160)).save(tmp_path / "data" / "images" / ("%d.png" % i))
        qs = {"q1": {"type": "choice", "instructions": "what color is it ? left", "criteria": {"red": None, "blue": None}},
              "q2": {"type": "choice", "instructions": "what color is it ? right", "criteria": {"red": None, "blue": None}}}
        rows.append({"id": "r%d" % i, "image": "images/%d.png" % i, "image_id": "r%d" % i, "questions": qs,
                     "gold": {"q1": {"probabilities": {"red": 1.0, "blue": 0.0}}, "q2": {"probabilities": {"red": 0.0, "blue": 1.0}}}})
        for q, obj, x in (("q1", "A", 0), ("q2", "B", 30)):
            regs.append({"set": "s", "id": "r%d" % i, "qid": q, "box": [x, 10, 30, 20], "width": 60, "height": 40})
        units.append({"key": "s|%d" % i, "set": "s", "split": "fit", "layout": "original", "field": "mass", "first_side": "left",
                      "questions": [{"id": "r%d" % i, "qid": "q1", "object": "A", "side": "left", "gold": "red"},
                                    {"id": "r%d" % i, "qid": "q2", "object": "B", "side": "right", "gold": "blue"}]})
    if dup:
        units.append(units[0])
    (tmp_path / "set.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "regs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in regs))
    (tmp_path / "units.jsonl").write_text("".join(json.dumps(u) + "\n" for u in units))
    images = _Images(tmp_path / "data", d)
    (run / "eval").mkdir(parents=True)
    with open(run / "eval" / "s.details.jsonl", "w") as fh:
        with torch.no_grad():
            for r in rows:
                items = _items(d, [r])
                batch = collate_items([items], d.tok.pad_token_id)
                pr = dg.probs_of(_forward(d.model, batch, torch.stack([images(it["image"]) for it in items]), "cpu"), items)
                for it, p in zip(items, pr):
                    probs = dict(zip(it["keys"], p))
                    if tamper:
                        probs["red"] += 1e-3
                    fh.write(json.dumps({"sample_id": r["id"], "qid": it["qid"], "probabilities": probs,
                                         "prediction": max(probs, key=lambda k: probs[k])}) + "\n")
    return ["--units", str(tmp_path / "units.jsonl"), "--regions", str(tmp_path / "regs.jsonl"), "--out", str(tmp_path / "out"),
            "--run", str(run), "--data-root", str(tmp_path / "data"), "--device", "cpu", "--expect-sha", "0",
            "--sets", json.dumps({"s": str(tmp_path / "set.jsonl")})]


def test_full_diagnosis_on_a_tiny_checkpoint(tmp_path):
    assert dg.main(tiny_run(tmp_path)) == 0
    rep = json.load(open(tmp_path / "out" / "report.json"))
    assert rep["questions"] == 4 and rep["baseline_vs_frozen_max_diff"] <= 1e-5 and rep["instrumented_vs_plain_max_diff"] < 1e-5
    assert set(rep["summary"]) == set(dg.MODES) and rep["summary"]["baseline"]["all"]["prediction_changed"] == 0
    det = [json.loads(x) for x in open(tmp_path / "out" / "details.jsonl")]
    ot = [r for r in det if r["mode"] == "oracle_target"]
    assert all(abs(r["mass_target"] - 1) < 1e-4 and r["mass_other"] < 1e-4 for r in ot)


def test_refuses_a_wrong_baseline_and_repeated_questions(tmp_path):
    with pytest.raises(SystemExit):
        dg.main(tiny_run(tmp_path / "a", tamper=True))
    with pytest.raises(SystemExit):
        dg.main(tiny_run(tmp_path / "b", dup=True))
