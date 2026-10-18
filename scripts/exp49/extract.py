"""exp49 frozen region features (critic EXP49-PREP-02). EXP48-X's final checkpoint, eval / no_grad, its own 256 letterbox
preprocessing (no crop, no zoom, no redraw), one image forward per picture: the SigLIP tower's last-layer 256 patches
(model.vision_patches) and the projector's 64 visual tokens (model.projector), i.e. before QuestionConditioning and
ModernBERT. For every question of data/exp49/questions.jsonl only the tokens its box covers (box_grid on the actual
letterbox, 16 x 16 and 8 x 8, any coverage) are kept, with their position relative to the region's top-left cell, padded
to the largest region. Pictures go in batches and every batch is written out at once, so no large graph or cache builds
up. Writes runs/exp49/features.npz and features.json (model / config / questions / regions / per-picture sha256,
preprocessing, shapes).

    uv run python scripts/exp49/extract.py
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

RUN = "runs/exp48-x"
X_SHA = {"model.safetensors": "a8b92cb93a56aa4653f516cbf2a4a38e88fbd64d69ac0f820acc6307f519b232",
         "config.json": "c017a469b35fcccf7e2fb2953dcd334be345bfd4fac493d64f4fd302155140df"}


def sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def region(box, width, height, side: int):
    """(token indices row-major, (row, col) relative to the region's first cell) of the cells the box covers."""
    from devision.train.rlcd import box_grid
    cov = box_grid([box], width, height, 256, grid=side, smooth=0.0)
    idx = [i for i, c in enumerate(cov) if c > 0]
    if not idx:
        raise SystemExit("a box covers no cell")
    r0, c0 = min(i // side for i in idx), min(i % side for i in idx)
    return idx, [(i // side - r0, i % side - c0) for i in idx]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", default="data/exp49/questions.jsonl")
    ap.add_argument("--run", default=RUN)
    ap.add_argument("--out", default="runs/exp49")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--expect-sha", default="1")
    a = ap.parse_args(argv)
    from devision.infer import Decider, best_device
    from devision.train.rlcd import _Images
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "v04b"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("v04b_freeze", os.path.join(os.path.dirname(__file__), "..", "v04b", "freeze.py"))
    assert spec is not None and spec.loader is not None
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    if a.expect_sha == "1" and frozen.check(a.run, [])["weights"] != X_SHA:
        raise SystemExit("EXP48-X is not a8b92cb9 / c017a469")
    qs = [json.loads(x) for x in open(a.questions)]
    pics = sorted({q["picture"] for q in qs})
    regions = {16: [region(q["box"], q["width"], q["height"], 16) for q in qs], 8: [region(q["box"], q["width"], q["height"], 8) for q in qs]}
    n16, n8 = max(len(r[0]) for r in regions[16]), max(len(r[0]) for r in regions[8])
    device = best_device() if a.device == "auto" else a.device
    d = Decider.load(os.path.join(a.run, "stage2"), device=device)
    model = d.model.eval()
    images = _Images(a.data_root, d)
    feats = {"siglip": np.zeros((len(qs), n16, 0), np.float16), "proj": np.zeros((len(qs), n8, 0), np.float16)}
    masks = {"siglip": np.zeros((len(qs), n16), bool), "proj": np.zeros((len(qs), n8), bool)}
    rels = {"siglip": np.zeros((len(qs), n16, 2), np.int64), "proj": np.zeros((len(qs), n8, 2), np.int64)}
    by_pic: dict = {}
    for i, q in enumerate(qs):
        by_pic.setdefault(q["picture"], []).append(i)
    with torch.no_grad():
        for s in range(0, len(pics), a.batch):
            chunk = pics[s:s + a.batch]
            px = torch.stack([images(p) for p in chunk])
            patches = model.vision_patches(px.to(device))                       # [B, 256, Dv]
            tokens = model.projector(patches)                                    # [B, 64, Dt]
            pt, tk = patches.float().cpu().numpy(), tokens.float().cpu().numpy()
            if feats["siglip"].shape[2] == 0:
                feats["siglip"] = np.zeros((len(qs), n16, pt.shape[2]), np.float16)
                feats["proj"] = np.zeros((len(qs), n8, tk.shape[2]), np.float16)
            for b, p in enumerate(chunk):
                for i in by_pic[p]:
                    for layer, side, arr in (("siglip", 16, pt[b]), ("proj", 8, tk[b])):
                        idx, rel = regions[side][i]
                        feats[layer][i, :len(idx)] = arr[idx]
                        masks[layer][i, :len(idx)] = True
                        rels[layer][i, :len(idx)] = rel
    os.makedirs(a.out, exist_ok=True)
    np.savez(os.path.join(a.out, "features.npz"), siglip=feats["siglip"], siglip_mask=masks["siglip"], siglip_rel=rels["siglip"],
             proj=feats["proj"], proj_mask=masks["proj"], proj_rel=rels["proj"])
    info = {"model": {"run": a.run, **X_SHA}, "questions_sha256": sha(a.questions),
            "pictures_sha256": hashlib.sha256(json.dumps({p: sha(os.path.join(a.data_root, p)) for p in pics}, sort_keys=True).encode()).hexdigest(),
            "preprocessing": {"image_size": d.cfg.image_size, "letterbox": True, "grids": [16, 8]}, "device": device,
            "shapes": {k: list(v.shape) for k, v in feats.items()}, "features_sha256": sha(os.path.join(a.out, "features.npz"))}
    json.dump(info, open(os.path.join(a.out, "features.json"), "w"), indent=1)
    print(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
