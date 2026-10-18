# pyright: reportOptionalMemberAccess=false, reportAttributeAccessIssue=false, reportOptionalSubscript=false, reportArgumentType=false, reportCallIssue=false, reportOperatorIssue=false
"""exp52's number reader (docs/experiments/2026-10-18-exp52-number-reader-plan.md): off by default (old checkpoints
answer as before); its input is the picture and the question's own tokens only -- options (order, count), qid and gold do
not change it, a different question does; the decision loss and the character loss both reach the shared content path;
ordinary decide uses it (leaving the content out changes the answers), with the extra tokens and marker offsets right for
one or many questions per picture; save / load answers the same; both arms start from the same tensors and first step,
the weight-0 arm is the run without the loss; malformed sidecars and settings are refused."""
import json

import pytest
import torch
from conftest import tiny_decider
from PIL import Image, ImageDraw
from test_distill import CONFIG

from devision.infer import Decider
from devision.model import NUMBER_SLOTS
from devision.train import rlcd
from devision.train.rlcd import TrainConfig, number_labels, number_loss, train

N, OVERLAP = 4, 0.5


def reader_decider(seed=0):
    d = tiny_decider(seed=seed)
    d.model.add_question_conditioning()
    d.cfg.question_conditioning = True
    d.model.enable_local_views(N)
    d.cfg.local_views, d.cfg.local_overlap, d.cfg.local_source = N, OVERLAP, "T"
    torch.manual_seed(seed + 52)
    d.model.add_number_reader()
    d.cfg.number_reader = True
    for p in d.model.reader.parameters():                  # a trained-looking reader, so its tokens matter
        torch.nn.init.normal_(p, std=0.05)
    return d


def picture():
    img = Image.new("RGB", (120, 90), "white")
    ImageDraw.Draw(img).text((10, 60), "red 14 blue 9", fill="black")
    return img


def q(text, options, qtype="choice"):
    return {"type": qtype, "instructions": text, "criteria": {o: None for o in options}}


def floats(a):
    """The numbers of an answer, in a fixed order (probabilities, confidence, ...)."""
    if isinstance(a, dict):
        return [x for k in sorted(a) for x in floats(a[k])]
    return [float(a)] if isinstance(a, (int, float)) and not isinstance(a, bool) else []


def chars_of(d, questions, img):
    d.model.reader.keep_chars = True
    d.decide({}, questions, image=img)
    return d.model.reader.last_chars.clone()


def test_off_by_default_and_old_checkpoints_unchanged(tmp_path):
    d = tiny_decider(seed=2)
    assert d.cfg.number_reader is False and d.model.reader is None
    d.save(tmp_path / "old")
    cfg = json.loads((tmp_path / "old" / "config.json").read_text())
    cfg.pop("number_reader")
    (tmp_path / "old" / "config.json").write_text(json.dumps(cfg))
    back = Decider.load(str(tmp_path / "old"), device="cpu")
    qs = {"q": q("what color is it ?", ["red", "blue"])}
    assert back.model.reader is None and back.decide({}, qs, image=picture()) == d.decide({}, qs, image=picture())


def test_reader_input_is_the_picture_and_the_question_only():
    d = reader_decider()
    img = picture()
    base = chars_of(d, {"a": q("what color is the dog ?", ["red", "blue", "green"])}, img)
    for other in ({"a": q("what color is the dog ?", ["green", "red", "blue"])},              # options reordered
                  {"a": q("what color is the dog ?", ["red", "blue"])},                       # fewer options
                  {"zz": q("what color is the dog ?", ["red", "blue", "green", "black"])}):  # another qid, more options
        assert torch.allclose(chars_of(d, other, img), base, atol=1e-6)
    asked_b = chars_of(d, {"a": q("what color is the cat ?", ["red", "blue", "green"])}, img)
    assert not torch.allclose(asked_b, base, atol=1e-4)                                      # another question: other input


def test_decide_uses_the_content_tokens_one_or_many_questions(tmp_path):
    d = reader_decider()
    img = picture()
    qs = {"a": q("what color is the dog ?", ["red", "blue", "green"]), "b": q("is there a dog ?", [], "noul"),
          "c": q("what color is the cat ?", ["blue", "red"])}
    qs["b"].pop("criteria")
    together = d.decide({}, qs, image=img)["answers"]
    for k in qs:                                                  # same answers alone (offsets right with 64 + slots)
        alone = d.decide({}, {k: qs[k]}, image=img)["answers"][k]
        assert floats(alone) == pytest.approx(floats(together[k]), abs=1e-5)
    from laya.common import collate_items
    from devision.infer import image_tensor
    items = rlcd._items(d, [{"id": "s", "image": "x", "image_id": "x", "questions": {"a": qs["a"]},
                            "gold": {"a": {"probabilities": {"red": 1.0, "blue": 0.0, "green": 0.0}}}}])
    batch = collate_items([items], d.tok.pad_token_id)
    assert batch is not None
    with torch.no_grad():                         # the option logits decide turns into probabilities depend on the content
        on = rlcd._forward(d.model.eval(), batch, image_tensor(img, d.cfg)[None], "cpu")
        d.model.reader.off = True
        off = rlcd._forward(d.model, batch, image_tensor(img, d.cfg)[None], "cpu")
        d.model.reader.off = False
    assert not torch.allclose(on, off, atol=1e-6)
    calls, seen = [], []                           # decide itself runs the reader and feeds 64 + slots visual tokens
    h = d.model.reader.register_forward_hook(lambda m, i, o: calls.append(o[0].shape))
    real_encode = d.model.encode
    d.model.encode = lambda ids, mask, visual=None: (seen.append(visual.shape[1]), real_encode(ids, mask, visual))[1]
    try:
        d.decide({}, qs, image=img)
    finally:
        h.remove()
        d.model.encode = real_encode
    assert calls == [torch.Size([3, NUMBER_SLOTS, calls[0][2]])] and seen == [64 + NUMBER_SLOTS]
    plain = tiny_decider()
    n_plain = plain.decide({}, qs, image=img)["usage"]["input_tokens"] - 0
    assert d.decide({}, qs, image=img)["usage"]["input_tokens"] - n_plain == 3 * NUMBER_SLOTS     # 64 + slots per question
    d.save(tmp_path / "r")
    back = Decider.load(str(tmp_path / "r"), device="cpu")
    assert back.cfg.number_reader and back.decide({}, qs, image=img) == d.decide({}, qs, image=img)


def test_both_losses_reach_the_shared_content_path():
    from laya.common import collate_items
    from devision.infer import image_tensor
    d = reader_decider()
    items = rlcd._items(d, [{"id": "s", "image": "x", "image_id": "x", "questions": {"q": q("what color is the dog ?", ["red", "blue"])},
                            "gold": {"q": {"probabilities": {"red": 1.0, "blue": 0.0}}}}])
    batch = collate_items([items], d.tok.pad_token_id)
    assert batch is not None
    px = image_tensor(picture(), d.cfg)[None]
    shared = [d.model.reader.look.in_proj_weight, d.model.reader.ask.in_proj_weight, d.model.reader.kv.weight]
    for which in ("decision", "chars"):
        d.model.zero_grad()
        d.model.reader.keep_chars = True
        logits = rlcd._forward(d.model, batch, px, "cpu")
        if which == "decision":
            loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0]))
        else:
            loss, _, n = number_loss(d.model.reader.last_chars, [number_labels("14")])
            assert n == 1 and loss is not None
        loss.backward()
        assert all(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in shared), which
    head_only = d.model.reader.chars[-1].weight
    assert head_only.grad is not None and float(head_only.grad.abs().sum()) > 0        # chars from the same content


def test_number_targets_and_loss():
    assert number_labels("1360") == [1, 3, 6, 0, 11] and number_labels("0.75")[:5] == [0, 10, 7, 5, 11]
    for bad in ("", "12345", "1,360", "a"):
        with pytest.raises(ValueError):
            number_labels(bad)
    chars = torch.full((2, NUMBER_SLOTS, 12), -10.0)
    for s, c in enumerate([1, 4, 11]):
        chars[0, s, c] = 10.0
    loss, right, n = number_loss(chars, [number_labels("14"), None])
    assert n == 1 and right == 1.0 and loss is not None and float(loss) < 1e-3
    assert number_loss(chars, [None, None])[0] is None


def number_samples(root):
    (root / "images").mkdir(parents=True)
    out, side = [], []
    for i in range(20):
        img = Image.new("RGB", (120, 90), "white")
        ImageDraw.Draw(img).text((10, 60), "value %d" % (i % 3 + 7), fill="black")
        img.save(root / "images" / ("%d.png" % i))
        opts = ["red", "blue", "green"]
        gold = opts[i % 3]
        out.append({"id": "s%d" % i, "image": "images/%d.png" % i, "image_id": "s%d" % i, "source": "synthetic",
                    "questions": {"q": q("what color is it ?", opts)}, "gold": {"q": {"probabilities": {o: float(o == gold) for o in opts}}}})
        side.append({"id": "s%d" % i, "qid": "q", "target": str(i % 3 + 7)})
    (root / "numbers.jsonl").write_text("".join(json.dumps(r) + "\n" for r in side[:16]))
    return out


def run(tmp_path, name, samples, **over):
    cfg = {**CONFIG, "epochs": 1, "objective": "ce", "question_conditioning": True, "local_views": N, "local_overlap": OVERLAP,
           "local_source": "T", "number_reader": True, "save_every": 0, **over}
    d = tiny_decider(seed=1)
    d.model.add_question_conditioning()
    d.cfg.question_conditioning = True
    inits = []
    real = torch.optim.AdamW

    def opt(groups, **kw):
        inits.append([p.detach().clone() for g in groups for p in g["params"]])
        return real(groups, **kw)
    torch.optim.AdamW, saved = opt, torch.optim.AdamW
    try:
        report = train(d, samples[:16], data_root=tmp_path / "data", out_dir=tmp_path / name, val_samples=samples[16:],
                       config=TrainConfig(**cfg))
    finally:
        torch.optim.AdamW = saved
    return d, report, inits[0]


def test_two_arms_same_start_weight_zero_is_the_run_without_loss(tmp_path):
    samples = number_samples(tmp_path / "data")
    side = str(tmp_path / "data" / "numbers.jsonl")
    d1, n1, i1 = run(tmp_path, "n", samples, number_file=side, number_weight=1.0)
    d0, n0, i0 = run(tmp_path, "z", samples, number_file=side, number_weight=0.0)
    _, nf, _ = run(tmp_path, "nofile", samples)
    assert len(i1) == len(i0) and all(torch.equal(a, b) for a, b in zip(i1, i0))           # same start tensors
    assert n1["losses"][0] == n0["losses"][0] and n1["losses"][1:] != n0["losses"][1:]
    assert n0["losses"] == nf["losses"]                                                     # weight 0 = no loss at all
    assert n1["number"]["weight"] == 1.0 and n1["number"]["init_seed"] == CONFIG["seed"] + 52
    assert "reader" in n1["local"]["groups"] and n1["local"]["vision_trainable"] == 0
    assert d1.cfg.number_reader and d1.model.reader is not None


def test_bad_sidecars_and_settings(tmp_path):
    samples = number_samples(tmp_path / "data")
    bad = tmp_path / "bad.jsonl"
    for rows in ([{"id": "nope", "qid": "q", "target": "7"}], [{"id": "s0", "qid": "q", "target": "7"}] * 2,
                 [{"id": "s0", "qid": "q", "target": "1,360"}]):
        bad.write_text("".join(json.dumps(r) + "\n" for r in rows))
        with pytest.raises(ValueError):
            run(tmp_path, "bad", samples, number_file=str(bad), number_weight=1.0)
    with pytest.raises(ValueError):
        run(tmp_path, "nr", samples, number_reader=False, number_file=str(tmp_path / "data" / "numbers.jsonl"))
    with pytest.raises(ValueError):
        run(tmp_path, "aux", samples, aux_weight=1.0)
