# pyright: reportOptionalMemberAccess=false, reportAttributeAccessIssue=false, reportArgumentType=false
"""PROTECT-PREP-01's mixed input (docs/experiments/2026-10-19-protect-replay-plan.md): rows marked "input": "text" are
scored exactly as text_only / decide(image=None) -- no picture opened even when the row names one -- and rows marked
"picture" exactly as the ordinary picture path; a micro batch never mixes the two and units stay whole; an update
changes only LoRA and the decision head; rows without the mark, and mixed_input with text_only, are refused."""
import random

import pytest
import torch
from conftest import tiny_decider
from PIL import Image
from test_distill import CONFIG

from devision.train import rlcd
from devision.train.rlcd import TrainConfig, train

N = 4


def exp72_like(seed=0):
    d = tiny_decider(seed=seed)
    d.model.add_question_conditioning()
    d.cfg.question_conditioning = True
    d.model.enable_local_views(N)
    d.cfg.local_views, d.cfg.local_overlap, d.cfg.local_source = N, 0.5, "T"
    for p in d.model.condition.parameters():
        torch.nn.init.normal_(p, std=0.05)
    return d


def q(text, options):
    return {"type": "choice", "instructions": text, "criteria": {o: None for o in options}}


def rows(root, n_text=8, n_pic=8):
    """Text rows (half of them naming a picture file that does not exist) and picture rows on real files; groups of
    two questions-on-one-family to keep whole."""
    (root / "img").mkdir(exist_ok=True)
    out = []
    for i in range(n_text):
        gold = ["red", "blue"][i % 2]
        out.append({"id": "t%d" % i, "input": "text", "image": "missing/%d.png" % i if i % 2 else "text-only:k", "image_id": "t%d" % i,
                    "group": "tf%d" % (i // 2), "state_text": "the dog is %s ." % gold, "questions": {"q": q("what color is the dog ?", ["red", "blue"])},
                    "gold": {"q": {"probabilities": {"red": float(gold == "red"), "blue": float(gold == "blue")}}}})
    for i in range(n_pic):
        colour = [(200, 30, 30), (30, 30, 200)][i % 2]
        Image.new("RGB", (40, 30), colour).save(root / "img" / ("%d.png" % i))
        gold = ["red", "blue"][i % 2]
        out.append({"id": "p%d" % i, "input": "picture", "image": "img/%d.png" % i, "image_id": "p%d" % i, "group": "pf%d" % (i // 2),
                    "questions": {"q": q("what color is it ?", ["red", "blue"])},
                    "gold": {"q": {"probabilities": {"red": float(gold == "red"), "blue": float(gold == "blue")}}}})
    return out


def test_each_row_is_scored_by_its_own_input(tmp_path):
    d = exp72_like()
    rs = rows(tmp_path)
    items = rlcd._items(d, rs)
    images = rlcd._Images(tmp_path, d)
    mixed = rlcd._val_logits(d.model, items, images, d.tok.pad_token_id, "cpu", per=3, mixed=True)
    text = [i for i, it in enumerate(items) if it["input"] == "text"]
    pic = [i for i, it in enumerate(items) if it["input"] == "picture"]
    alone_t = rlcd._val_logits(d.model, [items[i] for i in text], images, d.tok.pad_token_id, "cpu", per=3, text_only=True)
    alone_p = rlcd._val_logits(d.model, [items[i] for i in pic], images, d.tok.pad_token_id, "cpu", per=3)
    assert [mixed[i] for i in text] == alone_t and [mixed[i] for i in pic] == alone_p
    for i in text:                                         # = decide(image=None) on the row's text
        r = rs[i]
        got = d.decide(r["state_text"], r["questions"], image=None)["answers"]["q"]["probabilities"]
        want = torch.softmax(torch.tensor(mixed[i]) / d.cfg.temperature_for(items[i]["qtype"], 2), -1).tolist()
        assert [got[o] for o in ("red", "blue")] == pytest.approx(want, abs=1e-5)
    for i in pic:                                          # = decide with the row's own picture
        r = rs[i]
        got = d.decide("", r["questions"], image=str(tmp_path / r["image"]))["answers"]["q"]["probabilities"]
        want = torch.softmax(torch.tensor(mixed[i]) / d.cfg.temperature_for(items[i]["qtype"], 2), -1).tolist()
        assert [got[o] for o in ("red", "blue")] == pytest.approx(want, abs=1e-5)


def test_plan_keeps_modes_apart_and_units_whole():
    items = [{"input": "text", "group": "a%d" % (i // 3), "qid": "q"} for i in range(13)] + \
            [{"input": "picture", "group": "b%d" % (i // 2), "qid": "q"} for i in range(21)]
    plan = rlcd.plan_mixed(items, 8, random.Random(1))
    assert sorted(i for b in plan for i in b) == list(range(len(items)))
    assert all(len({items[i]["input"] for i in b}) == 1 and len(b) <= 8 for b in plan)
    where = {i: n for n, b in enumerate(plan) for i in b}
    for g in {it["group"] for it in items}:
        assert len({where[i] for i, it in enumerate(items) if it["group"] == g}) == 1


def snapshot(d):
    return {n: p.detach().clone() for n, p in d.model.named_parameters()}


def test_training_updates_only_lora_and_head_and_encodes_only_picture_batches(tmp_path, monkeypatch):
    d = exp72_like(seed=1)
    start = snapshot(d)
    calls = {"n": 0}
    real = d.model.visual_inputs

    def counted(*a, **k):
        calls["n"] += 1
        return real(*a, **k)
    monkeypatch.setattr(d.model, "visual_inputs", counted)
    plans = []
    real_plan = rlcd.plan_mixed
    monkeypatch.setattr(rlcd, "plan_mixed", lambda *a, **k: (plans.append(real_plan(*a, **k)), plans[-1])[1])
    rs = rows(tmp_path)
    cfg = {**CONFIG, "epochs": 1, "objective": "ce", "micro_batch": 4, "save_every": 0, "mixed_input": True, "eval_every": 0}
    train(d, rs, data_root=tmp_path, out_dir=tmp_path / "out", val_samples=rs[:2] + rs[-2:], config=TrainConfig(**cfg))
    items = rlcd._items(d, rs)
    pic_batches = sum(items[b[0]]["input"] == "picture" for b in plans[0])
    assert pic_batches == 2 and len(plans[0]) == 4
    assert calls["n"] >= pic_batches                       # pictures encoded for picture batches (and validation) only
    end = snapshot(d)
    frozen = [n for n in start if n.startswith(("vision.", "projector.", "condition.")) or "embeddings" in n]
    assert frozen and all(torch.equal(start[n], end[n]) for n in frozen)
    assert any(not torch.equal(start[n], end[n]) for n in start if n.startswith("decision."))


def test_text_batches_never_encode(tmp_path, monkeypatch):
    d = exp72_like(seed=2)
    rs = [r for r in rows(tmp_path) if r["input"] == "text"]
    monkeypatch.setattr(d.model, "visual_inputs", lambda *a, **k: (_ for _ in ()).throw(AssertionError("vision called")))
    cfg = {**CONFIG, "epochs": 1, "objective": "ce", "micro_batch": 4, "save_every": 0, "mixed_input": True, "eval_every": 0}
    train(d, rs, data_root=tmp_path / "nowhere", out_dir=tmp_path / "out", val_samples=rs[:2], config=TrainConfig(**cfg))


@pytest.mark.parametrize("bad", ["unmarked", "with_text_only"])
def test_refusals(tmp_path, bad):
    d = exp72_like()
    rs = rows(tmp_path)
    over = {"mixed_input": True, "epochs": 1, "objective": "ce", "save_every": 0}
    if bad == "unmarked":
        rs[3].pop("input")
        match = "needs \"input\""
    else:
        over["text_only"] = True
        match = "exclusive"
    with pytest.raises(ValueError, match=match):
        train(d, rs, data_root=tmp_path, out_dir=tmp_path / "o", val_samples=rs[:2], config=TrainConfig(**{**CONFIG, **over}))
