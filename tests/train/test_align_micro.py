"""exp39: alignment run in parts of micro_batch rows is the whole-batch step (tiny model, CPU). The same draws,
masks and number rewrites (parts with unequal masked-token counts), the same per-step loss, the same scheduler steps,
the same evaluation (its batch rotation kept), parameters within float tolerance; a non-finite part fails before the
step and nothing is saved; micro batches are refused with decision supervision or when they do not divide batch."""
import pytest
import torch
from conftest import tiny_decider
from test_pipeline import caption_samples, synthetic_samples

from devision.infer import Decider
from devision.train.align import AlignConfig, align


TEMP = ("Sample A: mass of each particle %d u, average particle speed %s m/s. "
        "Sample B: mass of each particle %d u, average particle speed %s m/s.")
FORCE = "Pair 1: two large magnets, %s in apart. Pair 2: one large magnet and one small magnet, %s in apart."


@pytest.fixture
def data(tmp_path):
    """24 training rows and 7 validation rows (a tail batch of 3 with batch 4) over the 16 synthetic pictures:
    real-layout number captions (two forms, numbers of different token lengths) mixed with plain captions, so the
    two parts of a batch have different numbers of masked tokens."""
    root = tmp_path / "data"
    samples = synthetic_samples(root, 16)
    pics = sorted({(s["image_id"], s["image"]) for s in samples})
    rows = []
    for i in range(31):
        image_id, image = pics[i % len(pics)]
        if i % 3 == 0:
            src, cap = "real-layout-temp", TEMP % (4 + i, "1,%03d" % (100 + 7 * i), 40 - i, str(300 + 10 * i))
        elif i % 3 == 1:
            src, cap = "real-layout-force", FORCE % ("%g" % (0.25 * (i % 7 + 1)), "%g" % (1.5 + i))
        else:
            src, cap = "coco-captions", "the image is red" if i % 2 else "a blue picture"
        rows.append({"id": "row:%d" % i, "source": src, "image_id": image_id, "image": image, "captions": [cap]})
    return root, rows[:24], rows[24:]


def run(tmp_path, root, caps, val, name, micro, monkeypatch=None, **kw):
    import devision.train.align as A
    logged = []
    if monkeypatch is not None:
        real_add = A.TrainLog.add
        monkeypatch.setattr(A.TrainLog, "add", lambda self, n, **m: (logged.append(dict(m)), real_add(self, n, **m))[1])
    report = align(tiny_decider(seed=1), caps, data_root=root, out_dir=tmp_path / name, val_samples=val,
                   config=AlignConfig(epochs=2, batch=4, lr=3e-3, warmup=2, mlm_head="", eval_every=3, lora_r=4,
                                      lr_lora=3e-3, mask_prob=0.5, seed=0, device="cpu", micro_batch=micro,
                                      number_mask_sources="real-layout-temp,real-layout-force",
                                      track_sources="real-layout-temp,real-layout-force", **kw))
    return report, Decider.load(tmp_path / name, device="cpu").model.state_dict(), logged


def test_parts_equal_the_whole_batch(tmp_path, data, monkeypatch):
    root, caps, val = data
    assert len(val) == 7 and len(val) % 4 == 3                          # a non-empty validation with a tail batch
    full, wf, lf = run(tmp_path, root, caps, val, "full", 0, monkeypatch)
    monkeypatch.undo()
    part, wp, lp = run(tmp_path, root, caps, val, "part", 2, monkeypatch)
    assert len(full["losses"]) == len(part["losses"]) == 12              # 24 rows / 4 x 2 epochs: steps and scheduler
    assert all(abs(a - b) < 1e-5 for a, b in zip(full["losses"], part["losses"]))
    assert len(full["val"]) == len(part["val"]) >= 3                     # step 0, 3, 6, 9, 12: validation really ran
    assert [e["step"] for e in full["val"]] == [e["step"] for e in part["val"]]
    for a, b in zip(full["val"], part["val"]):
        assert all(abs(a[k] - b[k]) < 1e-5 for k in a if k != "step")
        assert a["val/mlm_acc_real"] != a["val/mlm_acc_shuffled"] or a["val/mlm_nll_real"] != a["val/mlm_nll_shuffled"]
    assert all(torch.allclose(wf[k], wp[k], atol=1e-5) for k in wf)
    assert len(lf) == len(lp) == 12
    keys = [k for k in lf[0] if k.startswith(("tokens_", "loss_share_"))]
    assert {"tokens_real_layout_temp", "tokens_real_layout_force", "tokens_rest"} <= set(keys)
    for a, b in zip(lf, lp):
        assert all(abs(a[k] - b[k]) < 1e-5 for k in keys + ["accuracy"])
    # the number masking was really on and the parts of a batch carried different numbers of masked tokens
    assert any(r["tokens_real_layout_temp"] > 0 and r["tokens_real_layout_force"] > 0 for r in lf)


def test_parts_split_unequal_masked_tokens(data):
    """The fixture's first batch: the two halves hold different numbers of masked tokens after the number rewrite."""
    import random

    from devision.train.align import _masked_batch, _number_rows
    root, caps, _ = data
    tok = tiny_decider().tok
    c = AlignConfig(batch=4, mask_prob=0.5)
    chunk = caps[:4]
    texts = [r["captions"][0] for r in chunk]
    ids, att, tgt = _masked_batch(tok, texts, c, random.Random(0))
    _number_rows(tok, texts, [i for i, r in enumerate(chunk) if r["source"].startswith("real-layout")], ids, tgt, c)
    per_half = [int((tgt[h:h + 2, 1:] != -100).sum()) for h in (0, 2)]
    assert per_half[0] != per_half[1] and min(per_half) > 0


def test_a_non_finite_part_stops_before_the_step(tmp_path, data, monkeypatch):
    import devision.train.align as A
    root, caps, val = data
    real = A._mlm_logits
    calls = {"n": 0}

    def second_part_nan(*a, **k):
        calls["n"] += 1
        out = real(*a, **k)
        return out * float("nan") if calls["n"] == 2 else out
    monkeypatch.setattr(A, "_mlm_logits", second_part_nan)
    with pytest.raises(RuntimeError):
        align(tiny_decider(seed=1), caps, data_root=root, out_dir=tmp_path / "nan",
              config=AlignConfig(epochs=1, batch=4, lr=3e-3, warmup=1, mlm_head="", eval_every=0, seed=0, device="cpu",
                                 micro_batch=2))
    assert not (tmp_path / "nan" / "model.safetensors").exists()


def test_refused_combinations(tmp_path, data):
    import json
    root, caps, _ = data
    with open(tmp_path / "x.jsonl", "w") as fh:
        for s in synthetic_samples(tmp_path / "d2", 2):
            fh.write(json.dumps(dict(s, image="../d2/" + s["image"])) + "\n")
    for micro, dec in ((3, ""), (2, str(tmp_path / "x.jsonl"))):
        with pytest.raises(ValueError):
            align(tiny_decider(seed=1), caps, data_root=root, out_dir=tmp_path / "r",
                  config=AlignConfig(epochs=1, batch=4, mlm_head="", eval_every=0, seed=0, device="cpu",
                                     micro_batch=micro, decision_data=dec))
