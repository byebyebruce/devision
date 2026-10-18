"""exp47 without a real model: the six modes' visual tokens follow the formulas (beta = 1 gives back the two oracle end
points, target_1 = V + r_T, center = V + c, exchanging target / other flips d and keeps beta); return_hidden does not
change the logits; the whole diagnosis runs on a tiny checkpoint against EXP46's own tiny outputs, stops a unit whose
target_1 does not reproduce EXP46's oracle_target, never trains or leaves the module patched, and the model-free
finalize copes with a missing directory or a half-written file."""
import importlib.util
import json
import os
import sys

import pytest
import torch

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tests"))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


dg = load("exp47_diagnose", os.path.join(HERE, "diagnose.py"))
t46 = load("exp46_test", os.path.join(HERE, "..", "exp46", "test_exp46.py"))


def setup():
    d, cond, v, pt, w, qm = t46.module()
    s = torch.zeros(2, 16, dtype=torch.bool)
    s[0, :4], s[1, 8:12] = True, True
    return cond, v, pt, w, qm, s, s.flip(0)


def test_modes_follow_the_formulas():
    cond, v, pt, w, qm, tgt, oth = setup()
    with torch.no_grad():
        q = (w * qm.unsqueeze(-1)).sum(1) / qm.sum(1, keepdim=True)
        query = v + cond.question(q).unsqueeze(1)
        kv = cond.patches(pt)
        r_t, r_o = dg.residual(cond, query, kv, tgt), dg.residual(cond, query, kv, oth)
        out = {m: dg.visual_for(cond, v, pt, w, qm, m, tgt, oth) for m in dg.MODES}
        base_ref = cond(v, pt, w, qm)
    c, d = (r_t + r_o) / 2, (r_t - r_o) / 2
    assert torch.allclose(out["baseline"][0], base_ref, atol=1e-6)
    assert torch.allclose(out["target_1"][0], v + r_t, atol=1e-6) and torch.allclose(out["other_1"][0], v + r_o, atol=1e-6)
    assert torch.allclose(out["center"][0], v + c, atol=1e-6)
    beta = out["target_gain"][1]["beta"]
    assert beta == pytest.approx(max(1.0, min(1000.0, 0.1 * float(dg.rms(v + c)) / max(float(dg.rms(d)), 1e-12))))
    assert torch.allclose(out["target_gain"][0], v + c + beta * d, atol=1e-5) and torch.allclose(out["other_gain"][0], v + c - beta * d, atol=1e-5)
    with torch.no_grad():
        swapped = dg.visual_for(cond, v, pt, w, qm, "target_gain", oth, tgt)          # target and other exchanged
    assert swapped[1]["beta"] == pytest.approx(beta) and torch.allclose(swapped[0], out["other_gain"][0], atol=1e-5)


def test_beta_one_gives_back_the_end_points(monkeypatch):
    cond, v, pt, w, qm, tgt, oth = setup()
    monkeypatch.setattr(dg, "BETA_SCALE", 0.0)                                         # beta = max(1, 0) = 1
    with torch.no_grad():
        g, rec = dg.visual_for(cond, v, pt, w, qm, "target_gain", tgt, oth)
        one, _ = dg.visual_for(cond, v, pt, w, qm, "target_1", tgt, oth)
    assert rec["beta"] == 1.0 and torch.allclose(g, one, atol=1e-6)


def test_return_hidden_keeps_the_logits():
    from conftest import tiny_decider  # type: ignore
    from laya.common import collate_items
    from devision.train.rlcd import _forward, _items
    d = tiny_decider(seed=1)
    d.model.add_question_conditioning()
    s = {"id": "x", "image": "x.png", "image_id": "x", "questions": {"q": {"type": "choice", "instructions": "what color is it ?",
         "criteria": {"red": None, "blue": None}}}, "gold": {"q": {"probabilities": {"red": 1.0, "blue": 0.0}}}}
    items = _items(d, [s])
    batch = collate_items([items], d.tok.pad_token_id)
    assert batch is not None
    pixels = torch.zeros(1, 3, 256, 256)
    with torch.no_grad():
        a = _forward(d.model, batch, pixels, "cpu")
        b, h = _forward(d.model, batch, pixels, "cpu", return_hidden=True)
    assert torch.equal(a, b) and h.size(1) == batch["input_ids"].size(1) + 64


def tiny_both(tmp_path, tamper=False):
    args46 = t46.tiny_run(tmp_path)
    assert t46.dg.main(args46) == 0                                                   # EXP46's reference outputs
    if tamper:
        p = tmp_path / "out" / "details.jsonl"
        rows = [json.loads(x) for x in open(p)]
        for r in rows:
            if r["mode"] == "oracle_target":
                r["probabilities"] = [r["probabilities"][0] + 1e-3, r["probabilities"][1] - 1e-3]
        p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    a = dict(zip(args46[::2], args46[1::2]))
    return ["--units", a["--units"], "--manifest", a["--manifest"], "--regions", a["--regions"], "--exp46", str(tmp_path / "out"),
            "--out", str(tmp_path / "out47"), "--run", a["--run"], "--data-root", a["--data-root"], "--device", "cpu",
            "--expect-sha", "0", "--sets", a["--sets"]]


def test_full_diagnosis_on_a_tiny_checkpoint(tmp_path):
    assert dg.main(tiny_both(tmp_path)) == 0
    rep = json.load(open(tmp_path / "out47" / "report.json"))
    assert rep["status"] == "complete" and rep["completed_questions_by_mode"] == {m: 4 for m in dg.MODES}
    u = rep["units"][0]
    for k in ("rms_v", "rms_c", "rms_d", "rms_vc", "beta", "beta_capped", "zero_difference", "applied_relative_difference"):
        assert k in u
    p = u["path"]["target_gain-other_gain"]
    assert len(p["option_states"]) == 2 and len(p["option_logits"]) == 2 and p["module_output"][0] >= 0
    assert set(rep["analysis"]["continue"]) >= {"tune_diff_005", "all"}


def test_a_unit_whose_target_1_does_not_reproduce_exp46_stops(tmp_path):
    assert dg.main(tiny_both(tmp_path, tamper=True)) == 2
    rep = json.load(open(tmp_path / "out47" / "report.json"))
    assert rep["status"] == "incomplete" and "target_1" in rep["state"]["reason"]
    assert rep["completed_questions_by_mode"] == {m: 0 for m in dg.MODES}


def test_finalize_copes_with_missing_and_half_written_files(tmp_path):
    rep = dg.finalize(str(tmp_path / "never"), "125", os.path.join(ROOT, "data", "exp46", "units.jsonl"))
    assert rep["status"] == "incomplete"
    out = tmp_path / "half"
    out.mkdir()
    (out / "status.json").write_text('{"state": "runn')
    (out / "details.jsonl").write_text('{"mode": "baseline", "unit": "x"')
    rep = dg.finalize(str(out), "124", os.path.join(ROOT, "data", "exp46", "units.jsonl"))
    assert rep["status"] == "incomplete" and rep["exit_code"] == "124"
