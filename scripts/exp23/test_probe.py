"""exp23 probe without a real model: picture labels and ink heights, probes that can and cannot learn, each
judgement exit, and an end-to-end run on the tiny model."""
import os
import random
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "tests"))
sys.path.insert(0, os.path.join(HERE, "..", "data"))
from conftest import tiny_decider  # type: ignore  # noqa: E402
from exp21_reading import digit_ink  # type: ignore  # noqa: E402
from probe import CONTROL, correct, dataset, digest, judge, picture, run, train_probe  # noqa: E402


@pytest.mark.parametrize("h", (6, 8, 16, 48))
def test_pictures_hold_the_number_at_the_stated_height(h):
    rng = random.Random(h)
    for v in (10, 47, 88, 99):
        img, box = picture(v, h, rng)
        assert abs(digit_ink(img, box) - h) <= max(1, h // 20) and 0 <= box[0] and box[2] <= 256 and box[1] >= 0
    imgs, y = dataset(h, 50, seed=1)
    assert len(imgs) == 50 and y.shape == (50, 2) and y[:, 0].min() >= 1 and y.max() <= 9


def test_no_test_picture_repeats_a_training_picture():
    """At the real sizes, seeds 1 / 2 draw a few identical pictures (critic EXP23-N02): they are skipped."""
    a, _ = dataset(16, 3000, seed=1)
    train = {digest(x) for x in a}
    b, _ = dataset(16, 1000, seed=2, exclude=train)
    assert len(train) == 3000 and len({digest(x) for x in b}) == 1000 and not train & {digest(x) for x in b}


def test_probes_learn_a_present_signal_and_not_an_absent_one():
    """Like ink on a white page: one token per sample, at a random position, differs from the background (a
    marker dimension) and carries the two digits."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    y = np.stack([rng.integers(0, 10, 600), rng.integers(0, 10, 600)], 1)
    pos = torch.as_tensor(rng.integers(0, 8, 600))
    rows = torch.arange(600)
    signal = torch.randn(600, 8, 32)
    signal[rows, pos, 31] += 6.0
    signal[rows, pos, torch.as_tensor(y[:, 0])] += 8.0
    signal[rows, pos, 10 + torch.as_tensor(y[:, 1])] += 8.0
    noise = torch.randn(600, 8, 32)
    noise[rows, pos, 31] += 6.0                            # ink without the digits
    for kind in ("pooled", "attention"):
        assert correct(train_probe(kind, signal[:500], y[:500], epochs=100), signal[500:], y[500:]).mean() > 0.9
        assert correct(train_probe(kind, noise[:500], y[:500], epochs=100), noise[500:], y[500:]).mean() < 0.25


def acc(s, p, e, s48=0.95):
    return {"siglip": {"16": s, "48": s48}, "projector": {"16": p}, "encoder": {"16": e}}


@pytest.mark.parametrize("a,start", [
    (acc(0.9, 0.9, 0.9, s48=CONTROL - 0.01), "difficulty control not passed"),
    (acc(0.15, 0.1, 0.1), "low readability at all three levels"),
    (acc(0.9, 0.3, 0.3), "readability drops after the projector"),
    (acc(0.9, 0.9, 0.3), "readability drops in ModernBERT"),
    (acc(0.9, 0.85, 0.8), "all three levels readable"),
    (acc(0.5, 0.4, 0.4), "uncertain"),
    (acc(0.9, 0.75, 0.6), "all three levels readable"),
    (acc(0.7, 0.55, 0.5), "uncertain"),     # projector below 0.60 but within 0.20 of siglip
    # downstream recovery (critic EXP23-I01): never blamed on the upstream level
    (acc(0.15, 0.9, 0.9), "non-monotonic"),
    (acc(0.9, 0.3, 0.9), "non-monotonic"),
    (acc(0.4, 0.4, 0.9), "non-monotonic"),
])
def test_judgement(a, start):
    label = judge(a)["label"]
    assert label.startswith(start)
    for causal in ("lose", "does not carry", "does not use", "lost"):
        assert causal not in label


def test_end_to_end_on_the_tiny_model(tmp_path):
    tiny_decider().save(tmp_path / "m")
    r = run(str(tmp_path / "m"), str(tmp_path / "out"), base=str(tmp_path / "m"), device="cpu", heights=(16, 48),
            n_train=40, n_test=20, epochs=2)
    assert set(r["levels"]) == {"siglip", "projector", "encoder", "base_siglip"}
    assert r["levels"]["projector"]["16"]["best"]["n"] == 20
    assert "label" in r["judgement"] and os.path.exists(tmp_path / "out" / "report.json")
