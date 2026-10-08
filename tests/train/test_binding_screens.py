"""Round 13 screens S1 (grounding loss) and S2 (question-conditioned visual tokens), on the tiny model:
boxes map to the visual-token grid the way the picture is letterboxed; the grounding loss is finite and trains;
the S2 module starts as a no-op, trains, is saved and loaded with the checkpoint, and a request with several
questions answers each one as if asked alone."""
import random

import pytest
import torch
from conftest import tiny_decider
from PIL import Image

from devision.infer import Decider
from devision.train.rlcd import TrainConfig, box_grid, train

QUESTIONS = {"left": {"type": "choice", "instructions": "What color is the square?",
                      "criteria": {"red": None, "blue": None}},
             "dot": {"type": "noul", "instructions": "Is there a circle?"}}


def test_box_grid_follows_the_letterbox():
    # a square picture: a box on the left half lands in the left half of the grid
    g = box_grid([[0, 0, 128, 256]], 256, 256, 256, grid=8, smooth=0.0)
    left = sum(g[r * 8 + c] for r in range(8) for c in range(4))
    assert abs(left - 1.0) < 1e-6
    # a wide picture is padded top and bottom: its top edge maps to row 2, not row 0
    g = box_grid([[0, 0, 512, 64]], 512, 256, 256, grid=8, smooth=0.0)
    assert sum(g[:16]) == 0 and abs(sum(g[16:24]) - 1.0) < 1e-6
    assert box_grid([], 100, 100, 256) == []
    smooth = box_grid([[0, 0, 32, 32]], 256, 256, 256, grid=8)
    assert abs(sum(smooth) - 1) < 1e-6 and min(smooth) > 0


def scenes(root, n):
    """Red or blue square on the left or right of a grey picture, with its box."""
    (root / "img").mkdir(parents=True)
    out = []
    for i in range(n):
        colour, side = ["red", "blue"][i % 2], i // 2 % 2
        im = Image.new("RGB", (256, 256), (128, 128, 128))
        x = 20 if side == 0 else 150
        im.paste((220, 20, 20) if colour == "red" else (20, 20, 220), (x, 90, x + 80, 170))
        im.save(root / "img" / ("%d.png" % i))
        out.append({"id": "s%d" % i, "source": "syn", "image_id": "syn:%d" % i, "image": "img/%d.png" % i,
                    "group": "fam%d" % (i // 4), "width": 256, "height": 256,
                    "boxes": {"left": [[x, 90, 80, 80]]},
                    "questions": {"left": QUESTIONS["left"]},
                    "gold": {"left": {"probabilities": {"red": float(colour == "red"), "blue": float(colour == "blue")}}}})
    random.Random(0).shuffle(out)
    return out


def config(**over) -> TrainConfig:
    c = TrainConfig(epochs=3, micro_batch=4, lr_new=1e-3, lr_head=1e-3, lr_lora=1e-3, warmup=2, eval_every=0,
                    seed=0, device="cpu", save_every=0)
    for k, v in over.items():
        setattr(c, k, v)
    return c


@pytest.mark.slow
def test_grounding_loss_is_finite_trains_and_leaves_no_head_in_the_checkpoint(tmp_path):
    samples = scenes(tmp_path / "d", 16)
    report = train(tiny_decider(seed=2), samples, data_root=tmp_path / "d", out_dir=tmp_path / "ck",
                   val_samples=samples[:4], config=config(grounding_weight=0.5))
    assert all(map(lambda v: v == v, report["losses"]))
    loaded = Decider.load(str(tmp_path / "ck"), device="cpu")
    assert not hasattr(loaded.model, "grounding")
    assert not any(k.startswith("grounding") for k in loaded.model.state_dict())


def test_grounding_head_gets_gradients_from_boxed_questions(tmp_path):
    from laya.common import collate_items

    from devision.model import question_mask
    from devision.train.rlcd import GroundingHead, _forward, _Images, _items
    d = tiny_decider(seed=3)
    samples = scenes(tmp_path / "d", 4)
    items = _items(d, samples)
    assert all(len(it["grid"]) == 64 for it in items)   # 256 px, 16 px patches, 2x2 merge: 8 x 8
    batch = collate_items([items], d.tok.pad_token_id)
    assert batch is not None
    images = _Images(tmp_path / "d", d)
    pixels = torch.stack([images(it["image"]) for it in items])
    logits, hidden = _forward(d.model, batch, pixels, torch.device("cpu"), return_hidden=True)
    head = GroundingHead(hidden.size(-1))
    logp = head(hidden, hidden.size(1) - batch["input_ids"].size(1),
                question_mask(batch["input_ids"], batch["marker_pos"]))
    target = torch.tensor([it["grid"] for it in items])
    loss = (target * (target.log() - logp)).sum(-1).mean()
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    assert head.q.weight.grad is not None and head.q.weight.grad.abs().sum() > 0
    assert d.model.projector.mlp[1].weight.grad is not None   # reaches the visual tokens too


def test_question_mask_covers_the_question_words_only(tmp_path):
    from devision.infer import question_item
    from devision.model import question_mask
    d = tiny_decider()
    it = question_item(d.tok, d.cfg, "", QUESTIONS["left"])
    mask = question_mask(torch.tensor([it["ids"]]), torch.tensor([it["markers"]]))[0]
    words = d.tok.convert_ids_to_tokens([i for i, m in zip(it["ids"], mask) if m])
    assert "[SEP]" not in words and "[MASK]" not in words and "[CLS]" not in words
    assert len(words) >= 3


def test_question_conditioning_starts_as_a_no_op_and_is_saved(tmp_path):
    d = tiny_decider(seed=4)
    pic = Image.new("RGB", (64, 48), (200, 30, 30))
    before = d.decide(image=pic, state="", questions=QUESTIONS)
    d.model.add_question_conditioning()
    d.cfg.question_conditioning = True
    assert d.decide(image=pic, state="", questions=QUESTIONS) == before   # zero output layer: exactly as before
    assert d.model.condition is not None
    torch.nn.init.normal_(d.model.condition.out.weight, std=0.05)   # as if trained
    trained = d.decide(image=pic, state="", questions=QUESTIONS)
    assert trained != before
    d.save(str(tmp_path / "ck"))
    loaded = Decider.load(str(tmp_path / "ck"), device="cpu")
    assert loaded.cfg.question_conditioning and loaded.model.condition is not None
    assert loaded.decide(image=pic, state="", questions=QUESTIONS) == trained
    # several questions in one request = each asked alone (the condition is per question, the picture shared)
    for qid, q in QUESTIONS.items():
        alone = loaded.decide(image=pic, state="", questions={qid: q})["answers"][qid]
        both = trained["answers"][qid]
        for k in ("noul", "probabilities"):
            if k in alone:
                a, b = alone[k], both[k]
                if isinstance(a, dict):
                    assert all(abs(a[o] - b[o]) < 1e-5 for o in a)
                else:
                    assert abs(a - b) < 1e-5


@pytest.mark.slow
def test_question_conditioning_trains(tmp_path):
    samples = scenes(tmp_path / "d", 16)
    d = tiny_decider(seed=5)
    train(d, samples, data_root=tmp_path / "d", out_dir=tmp_path / "ck", val_samples=samples[:4],
          config=config(question_conditioning=True))
    loaded = Decider.load(str(tmp_path / "ck"), device="cpu")
    assert loaded.model.condition is not None
    assert loaded.model.condition.out.weight.abs().sum() > 0   # the zero layer moved
