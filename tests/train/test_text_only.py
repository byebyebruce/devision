# pyright: reportOptionalMemberAccess=false, reportAttributeAccessIssue=false, reportArgumentType=false
"""exp57's text-only training (docs/experiments/2026-10-18-decision-cmp-plan.md): on a model shaped like EXP50-T
(question conditioning + local views), training's forward is ordinary decide(image=None)'s scoring -- same logits, no
picture opened, the vision tower / visual_inputs never called; an update changes only LoRA and the decision head
(vision, projector, conditioning, word embeddings bit-identical); the gold follows its option when options are
reordered; the picture path is unchanged (default off, still opens the pictures); conflicting settings are refused."""
import pytest
import torch
from conftest import tiny_decider
from laya.common import collate_items
from test_distill import CONFIG

from devision.train import rlcd
from devision.train.rlcd import TrainConfig, train

N = 4


def exp50_like(seed=0):
    d = tiny_decider(seed=seed)
    d.model.add_question_conditioning()
    d.cfg.question_conditioning = True
    d.model.enable_local_views(N)
    d.cfg.local_views, d.cfg.local_overlap, d.cfg.local_source = N, 0.5, "T"
    for p in d.model.condition.parameters():         # a trained-looking conditioning, so skipping it would show
        torch.nn.init.normal_(p, std=0.05)
    return d


def q(text, options):
    return {"type": "choice", "instructions": text, "criteria": {o: None for o in options}}


def samples(n=8):
    out = []
    for i in range(n):
        opts = ["red", "blue", "green"] if i % 2 else ["green", "red", "blue"]
        gold = ["red", "blue", "green"][i % 3]
        out.append({"id": "t%d" % i, "image": "missing/%d.png" % i, "image_id": "t%d" % i, "source": "cmp",
                    "group": "f%d" % (i // 2), "state_text": "the dog is %s . the cat is white" % gold,
                    "questions": {"q": q("what color is the dog ?", opts)},
                    "gold": {"q": {"probabilities": {o: float(o == gold) for o in opts}}}})
    return out


class NoVision:
    """Fails the test if anything encodes a picture."""

    def __init__(self, d):
        self.d, self.calls = d, 0

    def __enter__(self):
        def boom(*a, **k):
            self.calls += 1
            raise AssertionError("vision called")
        self.saved = (self.d.model.visual_inputs, self.d.model.vision.forward)
        self.d.model.visual_inputs, self.d.model.vision.forward = boom, boom
        return self

    def __exit__(self, *exc):
        self.d.model.visual_inputs, self.d.model.vision.forward = self.saved


def test_training_forward_is_decide_without_image():
    d = exp50_like()
    ss = samples(4)
    items = rlcd._items(d, ss)
    batch = collate_items([items], d.tok.pad_token_id)
    assert batch is not None
    with NoVision(d) as nv:
        with torch.no_grad():
            logits = rlcd._forward(d.model.eval(), batch, None, "cpu")
        val = rlcd._val_logits(d.model, items, lambda rel: (_ for _ in ()).throw(AssertionError(rel)), d.tok.pad_token_id,
                               "cpu", per=3, text_only=True)
        for s, row, vrow in zip(ss, logits, val):
            got = d.decide(s["state_text"], s["questions"], image=None)["answers"]["q"]["probabilities"]
            t = d.cfg.temperature_for(items[0]["qtype"], 3)
            want = torch.softmax(row[:3] / t, -1).tolist()
            assert [got[o] for o in s["questions"]["q"]["criteria"]] == pytest.approx(want, abs=1e-6)
            assert vrow == pytest.approx(row[:3].tolist(), abs=1e-5)       # validation scores the same way
    assert nv.calls == 0


def test_gold_follows_its_option():
    d = exp50_like()
    a, b = samples(2)[1], dict(samples(2)[1])
    b["questions"] = {"q": q("what color is the dog ?", ["blue", "red", "green"])}
    b["gold"] = {"q": {"probabilities": {o: a["gold"]["q"]["probabilities"][o] for o in ["blue", "red", "green"]}}}
    ia, ib = rlcd._items(d, [a, b])
    gold = next(o for o, p in a["gold"]["q"]["probabilities"].items() if p == 1.0)
    assert ia["keys"][ia["target"].index(1.0)] == ib["keys"][ib["target"].index(1.0)] == gold
    assert ia["target"] != ib["target"]


def snapshot(d):
    return {n: p.detach().clone() for n, p in d.model.named_parameters()}


def head_or_lora(n):
    return "lora_" in n or (n.startswith("decision.") and not n.startswith("decision.encoder."))


def test_one_update_changes_only_lora_and_the_head(tmp_path, monkeypatch):
    """At the optimizer step the model is LoRA-wrapped: only lora_ and decision-head tensors move. After export (LoRA
    merged into the encoder) vision, projector, conditioning and word embeddings are still the start's."""
    d = exp50_like(seed=1)
    start = snapshot(d)
    seen = {}

    class Watched(torch.optim.AdamW):
        def __init__(self, groups, **kw):
            super().__init__(groups, **kw)
            seen["trainable"] = {n for n, p in d.model.named_parameters() if any(p is q for g in groups for q in g["params"])}
            seen["before"] = snapshot(d)

        def step(self, *a, **k):
            out = super().step(*a, **k)
            if "after" not in seen:
                seen["after"] = snapshot(d)
            return out
    monkeypatch.setattr(torch.optim, "AdamW", Watched)
    ss = samples(8)
    cfg = {**CONFIG, "epochs": 1, "objective": "ce", "micro_batch": 8, "save_every": 0, "text_only": True}
    with NoVision(d) as nv:
        report = train(d, ss, data_root=tmp_path / "nowhere", out_dir=tmp_path / "out", val_samples=ss[:2],
                       config=TrainConfig(**cfg))
    assert nv.calls == 0 and len(report["losses"]) == 1                       # one update, no picture
    assert seen["trainable"] and all(head_or_lora(n) for n in seen["trainable"]), sorted(seen["trainable"])
    changed = {n for n in seen["before"] if not torch.equal(seen["before"][n], seen["after"][n])}
    assert changed and all(head_or_lora(n) for n in changed), sorted(n for n in changed if not head_or_lora(n))
    assert any("lora_" in n for n in changed) and any("lora_" not in n for n in changed)
    end = snapshot(d)                                                         # exported (merged) model
    frozen = [n for n in start if n.startswith(("vision.", "projector.", "condition.")) or "embeddings" in n]
    assert any(n.startswith("decision.encoder.embeddings") for n in frozen)
    assert all(torch.equal(start[n], end[n]) for n in frozen)
    assert any(not torch.equal(start[n], end[n]) for n in start if n.startswith("decision.encoder."))   # LoRA merged in
    assert d.cfg.local_views == N and d.cfg.question_conditioning            # the start's picture path is kept as is


def test_picture_path_unchanged_by_default(tmp_path):
    assert TrainConfig().text_only is False
    d = exp50_like(seed=1)
    cfg = {**CONFIG, "epochs": 1, "objective": "ce", "micro_batch": 8, "save_every": 0}
    with pytest.raises(FileNotFoundError):                                     # the default still opens each picture
        train(d, samples(8), data_root=tmp_path / "nowhere", out_dir=tmp_path / "out", val_samples=samples(2),
              config=TrainConfig(**cfg))


@pytest.mark.parametrize("over", [{"local_views": N, "question_conditioning": True}, {"number_reader": True},
                                  {"grounding_weight": 0.1}, {"aux_weight": 0.1}, {"distill_weight": 0.1},
                                  {"vision_unfreeze_top": 1}, {"unfreeze_top": 1}, {"question_conditioning": True}])
def test_conflicting_settings_refused(tmp_path, over):
    with pytest.raises(ValueError, match="text_only"):
        train(exp50_like(), samples(4), data_root=tmp_path, out_dir=tmp_path / "o",
              config=TrainConfig(**{**CONFIG, "text_only": True, **over}))


def test_training_order_is_the_precomputed_plan(tmp_path, monkeypatch):
    """data/cmp/PLAN.json is computed as random.seed(seed) -> plan_batches(_items(...)); the run draws the same plan."""
    import random
    d = exp50_like(seed=1)
    ss = samples(8)
    random.seed(3)
    want = rlcd.plan_batches(rlcd._items(d, ss), 4)
    got, real = [], rlcd.plan_batches
    monkeypatch.setattr(rlcd, "plan_batches", lambda *a, **k: (got.append(real(*a, **k)), got[-1])[1])
    train(d, ss, data_root=tmp_path / "nowhere", out_dir=tmp_path / "out", val_samples=ss[:2],
          config=TrainConfig(**{**CONFIG, "epochs": 1, "objective": "ce", "micro_batch": 4, "seed": 3, "save_every": 0,
                                "text_only": True}))
    assert got[0] == want and all(len(set(b)) == len(b) for b in want)
