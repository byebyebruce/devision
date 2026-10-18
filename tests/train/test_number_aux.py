"""exp44's number reconstruction loss (docs/experiments/2026-10-18-exp44-number-plan.md): off by default and then the
run of before; its input is the picture, state text and question with a fixed number of [MASK] slots -- never the
options or anything that depends on the answer; the two backward passes of a batch add up to the joint loss's gradient
on the same weights and make one update; a non-finite number loss stops before any update; the frozen MLM head gets
no gradient and is not saved, and the checkpoint still answers through decide."""
import os
import random

import pytest
import torch
from conftest import image_b64, tiny_decider
from PIL import Image
from test_distill import CONFIG

from devision.infer import Decider
from devision.train import rlcd
from devision.train.rlcd import TrainConfig, aux_example, aux_head, aux_logits, aux_loss, aux_table, train
from devision.model import text_encoder

REAL_TOK = "runs/exp40-m/stage2/tokenizer"
GOLDS = ["4 u", "46 u", "1,200 m/s", "300 m/s", "0.75 in", "12.25 cm", "70.2 mm"]


def number_samples(root, n=24):
    (root / "images").mkdir(parents=True)
    out = []
    for i in range(n):
        Image.new("RGB", (40 + i, 32), (10 * i % 255, 80, 160)).save(root / "images" / ("%d.png" % i))
        opts = random.Random(i).sample(GOLDS, 3)
        out.append({"id": "n:%d" % i, "image": "images/%d.png" % i, "image_id": "n:%d" % i, "source": "synthetic",
                    "group": "g%d" % (i // 2), "state_text": "two samples",
                    "questions": {"q": {"type": "choice", "instructions": "What is the mass of each particle in sample A?",
                                        "criteria": {o: None for o in opts}}},
                    "gold": {"q": {"probabilities": {o: float(k == i % 3) for k, o in enumerate(opts)}}}})
    return out


def run(tmp_path, name, samples, **over):
    cfg = {**CONFIG, "objective": "ce", "aux_head": "", **over}
    return train(tiny_decider(seed=1), samples[:16], data_root=tmp_path / "data", out_dir=tmp_path / name,
                 val_samples=samples[16:], config=TrainConfig(**cfg))


@pytest.fixture
def samples(tmp_path):
    return number_samples(tmp_path / "data")


@pytest.mark.skipif(not os.path.isdir(REAL_TOK), reason="needs exp40-M's tokenizer")
def test_input_never_depends_on_the_answer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(REAL_TOK)
    q = "What is the average particle speed in sample A?"
    examples = [aux_example(tok, "Two samples of gas.", q, g, 8) for g in GOLDS]
    assert all(ids == examples[0][0] for ids, _ in examples)                  # same input for every gold
    ids = examples[0][0]
    assert ids.count(tok.mask_token_id) == 8 and ids[-9:-1] == [tok.mask_token_id] * 8
    assert not any(ch.isdigit() for ch in tok.decode(ids))                     # no number, no option in the input
    lens = {len([t for t in tgt if t != -100]) for _, tgt in examples}
    assert len(lens) > 1                                                       # the targets' lengths differ
    for (_, tgt), g in zip(examples, GOLDS):
        real = [t for t in tgt if t != -100]
        assert real[-1] == tok.sep_token_id and tok.decode(real[:-1]).strip() == g.split()[0] and len(tgt) == 8
    with pytest.raises(ValueError):
        aux_example(tok, "", q, "red", 8)                                     # not a number
    with pytest.raises(ValueError):
        aux_example(tok, "", q, "1,200 m/s", 2)                               # more tokens than slots
    bad = [{"id": "x", "state_text": "", "questions": {"q": {"type": "choice", "instructions": q, "criteria": {"4 u": None, "5 u": None}}},
            "gold": {"q": {"probabilities": {"4 u": 0.5, "5 u": 0.5}}}}]
    with pytest.raises(ValueError):
        aux_table(tok, bad, 8)                                                # not a single gold


def test_off_is_the_run_before(tmp_path, samples):
    a = run(tmp_path, "a", samples, aux_head="answerdotai/ModernBERT-large")  # weight 0: the head is never loaded
    b = run(tmp_path, "b", samples)
    assert a["losses"] == b["losses"] and "aux" not in a


def test_on_keeps_order_and_first_step_and_saves_a_normal_checkpoint(tmp_path, samples, monkeypatch):
    orders = {}
    real_save = rlcd._save_state
    monkeypatch.setattr(rlcd, "_save_state", lambda path, state: (orders.setdefault(str(path), state["order"]), real_save(path, state)))
    base = run(tmp_path, "base", samples, save_every=1)
    on = run(tmp_path, "on", samples, save_every=1, aux_weight=1.0, vision_unfreeze_top=1)
    o_base, o_on = (v for k, v in sorted(orders.items()))
    assert o_base == o_on                                                       # the same batch order
    assert on["losses"][0] == base["losses"][0] and on["losses"][1:] != base["losses"][1:]
    assert on["aux"]["weight"] == 1.0 and on["aux"]["slots"] == 8
    d = Decider.load(str(tmp_path / "on"), device="cpu")
    assert not any("aux" in k or k.startswith("head.") for k in d.model.state_dict())
    ans = d.decide([{"type": "image", "base64": image_b64()}], {"q": {"type": "choice", "instructions": "what color is it?",
                                                                       "criteria": {"red": None, "blue": None}}})
    assert set(ans["answers"]["q"]["probabilities"]) == {"red", "blue"}


def test_nonfinite_number_loss_stops_before_any_update(tmp_path, samples, monkeypatch):
    monkeypatch.setattr(rlcd, "aux_loss", lambda logits, target: (torch.tensor(float("nan")), torch.tensor(0.0)))
    with pytest.raises(RuntimeError, match="number loss"):
        run(tmp_path, "nan", samples, aux_weight=1.0)
    assert not (tmp_path / "nan" / "model.safetensors").exists()


def test_two_backward_passes_equal_the_joint_gradient_and_the_head_stays_frozen(tmp_path, samples):
    d = tiny_decider(seed=1)
    model, tok = d.model, d.tok
    head, ident = aux_head(text_encoder(model.decision), "")
    table = aux_table(tok, samples[:8], 8)
    items = rlcd._items(d, samples[:8])
    images = rlcd._Images(tmp_path / "data", d)
    pixels = torch.stack([images(it["image"]) for it in items])
    from laya.common import collate_items
    batch = collate_items([items], tok.pad_token_id)
    assert batch is not None
    ids, tgt = zip(*(table[(it["sample_id"], it["qid"])] for it in items))
    params = [p for p in model.parameters() if p.requires_grad]

    def losses():
        logits = rlcd._forward(model, batch, pixels, "cpu").float()
        ce, _ = rlcd._rlcd_loss(logits, batch["marker_mask"], batch["target"], batch["qtype"], 0.3, 4, "ce")
        a, _ = aux_loss(aux_logits(model, head, ids, pixels, "cpu", 8, tok.pad_token_id), torch.tensor(tgt))
        return ce, a
    model.zero_grad()
    ce, a = losses()
    (ce + a).backward()
    joint = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in params]
    model.zero_grad()
    ce, _ = losses()
    ce.backward()
    _, a = losses()
    a.backward()
    split = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in params]
    assert all(torch.allclose(x, y, atol=1e-6) for x, y in zip(joint, split))
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.projector.parameters())   # reaches the projector
    assert all(not p.requires_grad for p in head.head.parameters()) and not head.bias.requires_grad and ident == {"repo": ""}
