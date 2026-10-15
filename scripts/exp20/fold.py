"""exp20 (docs/experiments/2026-10-15-exp20-tokens256-plan.md): turn a 64-visual-token checkpoint (2x2 pixel
shuffle) into a 256-token one (no shuffle) whose projector answers the same wherever a 2x2 neighbourhood of
SigLIP patches is uniform.

The old projector is LayerNorm(4d) -> Linear(4d, t) -> GELU -> Linear(t, t) over the 4 patches of a 2x2 block,
concatenated in order k = 0..3. Four copies of one patch x have the same mean and variance as x, so
LayerNorm(4d) of the copies is [g_k * xhat + b_k]_k with xhat = LayerNorm(d)(x) without affine; the first linear
layer then gives sum_k W_k (g_k * xhat + b_k) + c. The new projector is LayerNorm(d) with weight 1 and bias 0,
followed by Linear(d, t) with W' = sum_k W_k diag(g_k) and c' = c + sum_k W_k b_k; the rest is copied. Every other
weight (SigLIP, ModernBERT, the decision head) and the temperatures are unchanged.

    uv run python scripts/exp20/fold.py runs/r15-vision/stage2 runs/exp20-tokens256/init
"""
import argparse
import json
import os

import torch

from devision.infer import Decider
from devision.model import build_model, text_encoder


def fold_projector(state, d: int, s: int = 2):
    """New projector tensors (prefix "projector.mlp.") from an s x s shuffle projector's, for vision width d."""
    g, b = state["projector.mlp.0.weight"], state["projector.mlp.0.bias"]
    w, c = state["projector.mlp.1.weight"], state["projector.mlp.1.bias"]
    if g.numel() != s * s * d:
        raise ValueError("projector input %d is not %d x %d" % (g.numel(), s * s, d))
    w64, g64, b64 = w.double(), g.double(), b.double()
    new_w = sum(w64[:, k * d:(k + 1) * d] * g64[k * d:(k + 1) * d] for k in range(s * s))
    new_c = c.double() + sum(w64[:, k * d:(k + 1) * d] @ b64[k * d:(k + 1) * d] for k in range(s * s))
    return {"projector.mlp.0.weight": torch.ones(d, dtype=g.dtype), "projector.mlp.0.bias": torch.zeros(d, dtype=b.dtype),
            "projector.mlp.1.weight": new_w.to(w.dtype), "projector.mlp.1.bias": new_c.to(c.dtype)}


def fold(decider: Decider) -> Decider:
    """A new Decider with visual_shuffle 1 and the folded projector; `decider` is left as it was."""
    cfg = decider.cfg
    if cfg.visual_shuffle != 2:
        raise ValueError("expected a 2x2-shuffle checkpoint, got visual_shuffle=%d" % cfg.visual_shuffle)
    if cfg.question_conditioning:
        raise ValueError("question-conditioned checkpoints are not supported")
    model = decider.model
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    d = int(model.vision.config.hidden_size)
    state.update(fold_projector(state, d))
    new_cfg = type(cfg).from_dict(dict(cfg.to_dict(), visual_shuffle=1))
    new = build_model(text_encoder(model.decision).config, model.vision.config, new_cfg)
    new.load_state_dict(state, strict=True)
    return Decider(new, decider.tok, new_cfg)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src")
    p.add_argument("out")
    a = p.parse_args()
    if os.path.exists(a.out):
        raise SystemExit("%s exists; remove it first" % a.out)
    new = fold(Decider.load(a.src, device="cpu"))
    new.save(a.out)
    json.dump({"source": a.src, "visual_shuffle": "2 -> 1", "projector": "folded (scripts/exp20/fold.py)"},
              open(os.path.join(a.out, "fold.json"), "w"), indent=1)
    print("wrote %s (256 visual tokens)" % a.out)


if __name__ == "__main__":
    main()
