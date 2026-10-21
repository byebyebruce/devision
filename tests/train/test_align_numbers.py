"""exp36: number-only supervision of real-layout captions in alignment. Only the listed sources' rows change, after
the random draw (the random stream and every other row stay as without it); every token of every number of both
objects is masked and is the only target, labels / units never; a source that is not in the data leaves the run
exactly as before (tiny model, CPU)."""
import os
import random

import pytest
import torch
from conftest import tiny_decider
from test_align_decisions import data  # noqa: F401  (fixture)

from devision.train.align import AlignConfig, _masked_batch, _number_rows, align, number_spans

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
TOKENIZER = os.path.join(ROOT, "runs", "exp32-align", "stage1", "tokenizer")
TEMP = ("Sample A: mass of each particle 46 u, average particle speed 1,890 m/s. "
        "Sample B: mass of each particle 12 u, average particle speed 980 m/s.")
FORCE = "Pair 1: two large magnets, 0.75 in apart. Pair 2: one large magnet and one small magnet, 42.8 mm apart."
COCO = "A batter swings at the ball at a game."


@pytest.mark.skipif(not os.path.isdir(TOKENIZER), reason="needs the checkpoint tokenizer")
def test_only_the_number_rows_change_and_only_numbers_are_targets():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    c = AlignConfig(max_caption_tokens=44, mask_prob=0.5)
    caps = [TEMP, COCO, FORCE]
    r1, r2 = random.Random(5), random.Random(5)
    a = _masked_batch(tok, caps, c, r1)
    b = _masked_batch(tok, caps, c, r2)
    _number_rows(tok, caps, [0, 2], b[0], b[2], c)
    assert r1.random() == r2.random()                                   # the random stream is untouched
    assert torch.equal(a[0][1], b[0][1]) and torch.equal(a[2][1], b[2][1])   # the COCO row as before
    for i in (0, 2):
        ids = tok(caps[i], add_special_tokens=False)["input_ids"]
        tgt = [int(t) for t in b[2][i, 1:1 + len(ids)]]
        masked = [t for t, x in zip(ids, tgt) if x != -100]
        assert tok.decode(masked).replace(" ", "") == "".join(s["text"] for s in number_spans(caps[i]))
        assert all(int(b[0][i, 1 + k]) == (tok.mask_token_id if x != -100 else ids[k]) for k, x in enumerate(tgt))


@pytest.mark.skipif(not os.path.isdir(TOKENIZER), reason="needs the checkpoint tokenizer")
def test_refuses_captions_without_their_numbers():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    c = AlignConfig(max_caption_tokens=44)
    bad = ["Sample A: mass of each particle 46 u. Sample B: mass of each particle 12 u.", COCO]
    batch = _masked_batch(tok, bad, c, random.Random(0))
    with pytest.raises(ValueError):
        _number_rows(tok, bad, [0], batch[0], batch[2], c)
    with pytest.raises(ValueError):
        _number_rows(tok, bad, [1], batch[0], batch[2], c)


def test_a_source_not_in_the_data_is_the_plain_run(tmp_path, data):  # noqa: F811
    root, captions, _ = data
    runs = []
    for name, sources in (("plain", ""), ("absent", "real-layout-temp")):
        runs.append(align(tiny_decider(seed=1), captions, data_root=root, out_dir=tmp_path / name,
                          config=AlignConfig(epochs=1, batch=4, lr=3e-3, warmup=2, mlm_head="", eval_every=0, lora_r=4,
                                             lr_lora=3e-3, seed=0, device="cpu", number_mask_sources=sources))["losses"])
    assert runs[0] == runs[1]
