"""exp53 region selector (critic EXP53-PREP-01): picture + natural question -> one normalised box. Inputs: the full
picture's 256 SigLIP patches (the first 256 keys / values of visual_inputs) with their whole-picture geometry
(image.patch_geometry of the full picture), and the question's own token embeddings (EXP50-T's frozen ModernBERT input
embeddings of the instruction text) with learned positions and one self-attention layer, so word order and context are
part of the question representation. No field / object label, question id, gold box, target, option, answer length or
donor identity is an input. One learned box query attends over the question tokens, then twice over the picture tokens;
an MLP gives (cx, cy, w, h) through a sigmoid, in picture fractions.

Box rule (fixed before any run): the box in pixels is clipped to the picture; a side shorter than 1 px after clipping
is widened to 1 px around its centre inside the picture (counted as "degenerate"). The pixel box then goes through
scripts/exp51/regions.region -- the same rule the oracle path uses -- which fixes the tokens, their relative rows /
columns and their continuous geometry relative to the box for the frozen EXP51-T head.

Training (fixed): fit_O + fit_L boxes only; loss L1(cx, cy, w, h) + (1 - GIoU) per question, batch mean; AdamW lr 1e-3,
weight decay 0.01, batch 64 from a seeded permutation (seed 0), 2,000 updates (128,000 question exposures, about 111 per
question), 100-update linear warm-up then cosine; the last step kept.
"""
import math
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

WIDTH, HEADS, MAX_Q = 256, 4, 48
LR, WD, BATCH, STEPS, WARMUP, SEED = 1e-3, 0.01, 64, 2000, 100, 0


class Selector(nn.Module):
    def __init__(self, vision_dim: int = 768, text_dim: int = 1024, width: int = WIDTH):
        super().__init__()
        self.patch = nn.Linear(vision_dim, width)
        self.geom = nn.Linear(4, width)
        self.word = nn.Linear(text_dim, width)
        self.pos = nn.Embedding(MAX_Q, width)
        self.context = nn.TransformerEncoderLayer(width, HEADS, 2 * width, dropout=0.0, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, width) * 0.02)
        self.ask = nn.MultiheadAttention(width, HEADS, batch_first=True)
        self.look1 = nn.MultiheadAttention(width, HEADS, batch_first=True)
        self.look2 = nn.MultiheadAttention(width, HEADS, batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.box = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 4))

    def forward(self, patches, geom, words, word_mask):
        """patches [B, 256, Dv], geom [B, 256, 4], words [B, L, Dt], word_mask [B, L] (True = token) ->
        boxes [B, 4] = (cx, cy, w, h) in picture fractions."""
        pos = self.pos(torch.arange(words.size(1), device=words.device))[None]
        q = self.context(self.word(words.float()) + pos, src_key_padding_mask=~word_mask)
        img = self.patch(patches.float()) + self.geom(geom.float())
        s = self.query[None].expand(words.size(0), -1, -1)
        s = s + self.ask(s, q, q, key_padding_mask=~word_mask, need_weights=False)[0]
        s = s + self.look1(s, img, img, need_weights=False)[0]
        s = s + self.look2(self.norm(s), img, img, need_weights=False)[0]
        return torch.sigmoid(self.box(s[:, 0]))


def cxcywh_to_xyxy(b: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)


def giou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Generalised IoU of xyxy boxes [B, 4]."""
    ix0, iy0 = torch.maximum(a[:, 0], b[:, 0]), torch.maximum(a[:, 1], b[:, 1])
    ix1, iy1 = torch.minimum(a[:, 2], b[:, 2]), torch.minimum(a[:, 3], b[:, 3])
    inter = (ix1 - ix0).clamp(min=0) * (iy1 - iy0).clamp(min=0)
    area = lambda x: (x[:, 2] - x[:, 0]).clamp(min=0) * (x[:, 3] - x[:, 1]).clamp(min=0)  # noqa: E731
    union = area(a) + area(b) - inter
    iou = inter / union.clamp(min=1e-9)
    cx0, cy0 = torch.minimum(a[:, 0], b[:, 0]), torch.minimum(a[:, 1], b[:, 1])
    cx1, cy1 = torch.maximum(a[:, 2], b[:, 2]), torch.maximum(a[:, 3], b[:, 3])
    hull = ((cx1 - cx0) * (cy1 - cy0)).clamp(min=1e-9)
    return iou - (hull - union) / hull


def box_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 on (cx, cy, w, h) plus 1 - GIoU, per question, then the batch mean."""
    return ((pred - target).abs().sum(-1) + 1 - giou(cxcywh_to_xyxy(pred), cxcywh_to_xyxy(target))).mean()


def target_of(box_xywh: Sequence[float], w: int, h: int) -> List[float]:
    x, y, bw, bh = box_xywh
    return [(x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h]


def pixel_box(frac: Sequence[float], w: int, h: int) -> Tuple[List[float], bool]:
    """([x, y, w, h] in pixels, degenerate) by the fixed rule of the module docstring."""
    cx, cy, bw, bh = (float(v) for v in frac)
    x0, x1 = max(0.0, (cx - bw / 2) * w), min(float(w), (cx + bw / 2) * w)
    y0, y1 = max(0.0, (cy - bh / 2) * h), min(float(h), (cy + bh / 2) * h)
    bad = False
    if x1 - x0 < 1:
        c = min(max((x0 + x1) / 2, 0.5), w - 0.5)
        x0, x1, bad = c - 0.5, c + 0.5, True
    if y1 - y0 < 1:
        c = min(max((y0 + y1) / 2, 0.5), h - 0.5)
        y0, y1, bad = c - 0.5, c + 0.5, True
    return [x0, y0, x1 - x0, y1 - y0], bad


def iou_xywh(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / (a[2] * a[3] + b[2] * b[3] - inter) if inter > 0 else 0.0


class Inputs:
    """Per question: its picture's global patches / geometry (by picture index) and its question words (padded)."""

    def __init__(self, patches: torch.Tensor, geom: torch.Tensor, pic_index: List[int], words: torch.Tensor, word_mask: torch.Tensor):
        self.patches, self.geom, self.pic, self.words, self.mask = patches, geom, pic_index, words, word_mask

    def batch(self, idx: List[int], device: str = "cpu"):
        p = [self.pic[i] for i in idx]
        return (self.patches[p].to(device), self.geom[p].to(device), self.words[idx].to(device), self.mask[idx].to(device))


def train(inp: Inputs, train_idx: List[int], targets: torch.Tensor, steps: int = STEPS, seed: int = SEED,
          device: str = "cpu", log=None, on_step=None) -> Tuple[Selector, Dict]:
    torch.manual_seed(seed)
    sel = Selector(inp.patches.shape[-1], inp.words.shape[-1]).to(device)
    opt = torch.optim.AdamW(sel.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / WARMUP) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    g = torch.Generator().manual_seed(seed)
    order: List[int] = []
    losses = []
    for step in range(steps):
        if len(order) < BATCH:
            order += torch.randperm(len(train_idx), generator=g).tolist()
        pick, order = order[:BATCH], order[BATCH:]
        idx = [train_idx[k] for k in pick]
        loss = box_loss(sel(*inp.batch(idx, device)), targets[idx].to(device))
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite selector loss at step %d" % step)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
        if on_step and (step + 1) % 50 == 0:
            on_step(step + 1)
        if log and (step + 1) % 250 == 0:
            log("selector step %d loss %.4f" % (step + 1, sum(losses[-250:]) / 250))
    return sel.eval(), {"steps": steps, "batch": BATCH, "exposures": steps * BATCH, "final_loss_250": sum(losses[-250:]) / min(250, len(losses))}


def predict(sel: Selector, inp: Inputs, idx: List[int], device: str = "cpu", batch: int = 256) -> List[List[float]]:
    out: List[List[float]] = []
    with torch.no_grad():
        for s in range(0, len(idx), batch):
            out += sel(*inp.batch(idx[s:s + batch], device)).cpu().tolist()
    return out
