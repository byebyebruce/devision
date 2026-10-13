"""exp17 (docs/experiments/2026-10-13-exp17-interp.md): linear weight interpolation between two checkpoints of the
same structure, theta = a + alpha * (b - a). The output directory is a full checkpoint (config, tokenizer, encoder and
vision configs copied from b); its temperatures are b's until refitted with devision-calibrate.

    uv run python scripts/exp17/interp.py runs/v12-ext/stage2 runs/r14-count/stage2 0.5 runs/exp17-interp/a050
"""
import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file

WEIGHTS = "model.safetensors"


def interpolate(a, b, alpha):
    """State dicts with the same keys, shapes and dtypes -> a + alpha * (b - a), computed in float64."""
    if a.keys() != b.keys():
        raise ValueError("different tensors: only in a %s, only in b %s" % (sorted(a.keys() - b.keys())[:5],
                                                                          sorted(b.keys() - a.keys())[:5]))
    out = {}
    for k, x in a.items():
        y = b[k]
        if x.shape != y.shape or x.dtype != y.dtype:
            raise ValueError("%s: %s %s vs %s %s" % (k, x.dtype, tuple(x.shape), y.dtype, tuple(y.shape)))
        if not x.is_floating_point():
            if not torch.equal(x, y):
                raise ValueError("%s: non-float tensors differ" % k)
            out[k] = x.clone()
            continue
        out[k] = (x.double() + alpha * (y.double() - x.double())).to(x.dtype).contiguous()
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("a", help="checkpoint at alpha = 0")
    p.add_argument("b", help="checkpoint at alpha = 1 (its configs are copied)")
    p.add_argument("alpha", type=float)
    p.add_argument("out")
    args = p.parse_args()
    if os.path.exists(args.out):
        raise SystemExit("%s exists; remove it first" % args.out)
    w = interpolate(load_file(os.path.join(args.a, WEIGHTS)), load_file(os.path.join(args.b, WEIGHTS)), args.alpha)
    shutil.copytree(args.b, args.out, ignore=lambda d, names: [n for n in names if n == WEIGHTS or
                                                                (d == args.b and n not in ("config.json", "devision_config.json", "tokenizer", "encoder", "vision"))])
    save_file(w, os.path.join(args.out, WEIGHTS))
    json.dump({"a": args.a, "b": args.b, "alpha": args.alpha, "formula": "a + alpha * (b - a)"},
              open(os.path.join(args.out, "interp.json"), "w"), indent=1)
    print("wrote %s (alpha %.3f, %d tensors)" % (args.out, args.alpha, len(w)))


if __name__ == "__main__":
    main()
