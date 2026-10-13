"""exp17 interpolation on hand-made tensors, no model."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
from interp import interpolate  # noqa: E402


def test_endpoints_and_midpoint():
    a = {"w": torch.tensor([0.0, 2.0]), "n": torch.tensor([3])}
    b = {"w": torch.tensor([1.0, 4.0]), "n": torch.tensor([3])}
    assert torch.equal(interpolate(a, b, 0.0)["w"], a["w"])
    assert torch.equal(interpolate(a, b, 1.0)["w"], b["w"])
    assert torch.allclose(interpolate(a, b, 0.3)["w"], torch.tensor([0.3, 2.6]))
    assert torch.equal(interpolate(a, b, 0.5)["n"], a["n"])


def test_refuses_mismatches():
    a = {"w": torch.zeros(2)}
    with pytest.raises(ValueError):
        interpolate(a, {"v": torch.zeros(2)}, 0.5)
    with pytest.raises(ValueError):
        interpolate(a, {"w": torch.zeros(3)}, 0.5)
    with pytest.raises(ValueError):
        interpolate({"n": torch.tensor([1])}, {"n": torch.tensor([2])}, 0.5)
