"""MLM-REPRO-PREP-01: where does the saved N end point's wrong-picture accuracy (0.669519 through the probe) part
from its training-time validation (0.680539, align_report step 1,250)? One loaded instance of the saved N end point,
one tokenizer / MLM head / device / picture order / mask (batch 16, Random(1) per mode, mask 0.5, the next row's
picture), three passes, nothing retrained, no full-vocabulary logits kept:

  A  align._evaluate as training calls it (the reference function), aggregate real / shuffled;
  B  scripts/panel_bind/probe.run_random (the accepted probe path), aggregate real / shuffled;
  C  instrumented: the same loop; per batch the sha256 of input ids / attention / targets / the pictures shown, and on
     the same logits three scorings -- (1) align's: argmax of logits[mask] on the device, (2) the probe's: argmax of
     the whole logits on the device, then the masked positions on CPU, (3) a CPU float32 reference: argmax of the
     masked logits moved to CPU -- with their hits; for every token where they disagree: gold, the three predictions,
     the top-2 margin (CPU float32) and non-finite counts. One batch's logits at a time.

Interpretation, written before running (no root cause is claimed beyond it):
  - A != B on identical inputs  -> the scoring / call path differs; the variant that equals the reference says which;
  - A == B == C but != the training-time summary -> the saved end point now reads differently from training time;
    the training-time state (un-merged LoRA, optimizer, RNG) was not saved, so that difference is NOT located here;
  - (1) != (2) or != (3) on some tokens -> argmax over masked vs whole logits / device vs CPU differs on those tokens.
No retraining, no new tolerance, the original N / report / compare are not touched.

    uv run python scripts/mlm_repro/repro.py --checkpoint runs/exp75-panel-n/n --out runs/mlm-repro
"""
import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
from typing import Dict, List

import torch

VAL = "data/exp26/align_val.jsonl"
REFERENCE = "runs/exp75-panel-n/n/align_report.json"


def _probe():
    spec = importlib.util.spec_from_file_location("mr_probe", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "panel_bind", "probe.py"))
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def h(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def score(logits: torch.Tensor, y: torch.Tensor) -> Dict:
    """Three scorings of one batch's logits [B, T, V] against targets y [B, T] (-100 = not a target)."""
    m = (y != -100).to(logits.device)
    yd = y.to(logits.device)
    p1 = logits[m].argmax(-1)                                   # align: masked first, argmax on the device
    p2 = logits.argmax(-1).cpu()[m.cpu()]                       # probe: argmax of everything on the device, then mask
    lc = logits.detach().float().cpu()[m.cpu()]                 # CPU float32 reference on the same logits
    p3 = lc.argmax(-1)
    out = compare_predictions(p1.cpu(), p2, p3, y[m.cpu()], lc)
    out["hits_align"] = int((p1 == yd[m]).sum().item())         # counted on the device, as align does
    return out


def compare_predictions(p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor, g: torch.Tensor, lc: torch.Tensor) -> Dict:
    """Hits of the three prediction vectors (CPU) against the gold, non-finite logits, and every disagreeing token."""
    out = {"targets": int(g.numel()), "hits_align": int((p1 == g).sum().item()), "hits_probe": int((p2 == g).sum().item()),
           "hits_cpu": int((p3 == g).sum().item()), "nonfinite": int((~torch.isfinite(lc)).sum().item()), "disagree": []}
    diff = (p1 != p2) | (p1 != p3) | (p2 != p3)
    if diff.any():
        top2 = lc[diff].topk(2, -1).values
        for k, i in enumerate(diff.nonzero().flatten().tolist()):
            out["disagree"].append({"gold": int(g[i]), "align": int(p1[i]), "probe": int(p2[i]), "cpu": int(p3[i]),
                                    "margin": float(top2[k, 0] - top2[k, 1])})
    return out


def instrumented(d, head, images, dev, samples) -> Dict:
    from devision.train.align import AlignConfig, _logits_in_parts, _masked_batch
    c = AlignConfig(batch=16, max_caption_tokens=40, mask_prob=0.5, micro_batch=0)
    batches: List[Dict] = []
    tot: Dict = {}
    for mode in ("real", "shuffled"):
        rng = random.Random(1)
        agg: Dict = {"targets": 0, "hits_align": 0, "hits_probe": 0, "hits_cpu": 0, "nonfinite": 0, "disagree": 0}
        with torch.no_grad():
            for b in range(0, len(samples), c.batch):
                chunk = samples[b:b + c.batch]
                caps = [rng.choice(s["captions"]) for s in chunk]
                pix_src = chunk if mode == "real" else chunk[1:] + chunk[:1]
                input_ids, attention, target = _masked_batch(d.tok, caps, c, rng)
                pixels = torch.stack([images(s["image"]) for s in pix_src])
                logits = _logits_in_parts(d.model, head, input_ids, attention, pixels, dev, 0)
                sc = score(logits, target[:, 1:])
                del logits
                batches.append({"mode": mode, "batch": b // c.batch, "input_ids": h(input_ids), "attention": h(attention), "targets": h(target),
                                "pictures": h(pixels), "rows": [s["id"] for s in chunk], "shown": [s["image"] for s in pix_src], **sc})
                for k in agg:
                    agg[k] += len(sc["disagree"]) if k == "disagree" else sc[k]
        agg.update({"acc_align": agg["hits_align"] / agg["targets"], "acc_probe": agg["hits_probe"] / agg["targets"], "acc_cpu": agg["hits_cpu"] / agg["targets"]})
        tot[mode] = agg
    return {"summary": tot, "batches": batches}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="runs/exp75-panel-n/n")
    ap.add_argument("--out", default="runs/mlm-repro")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        raise SystemExit("%s exists (fresh only)" % a.out)
    probe = _probe()
    samples = [json.loads(x) for x in open(VAL) if x.strip()]
    d, head, images, dev = probe.load(a.checkpoint, a.device)
    os.makedirs(a.out)
    ref = json.load(open(REFERENCE))["val"][-1]
    from devision.train.align import AlignConfig, _evaluate
    A = _evaluate(d.model, head, d.tok, samples, images, AlignConfig(batch=16, max_caption_tokens=40, mask_prob=0.5, micro_batch=0), dev)
    B = probe.run_random(d, head, images, dev, samples)["summary"]
    C = instrumented(d, head, images, dev, samples)
    with open(os.path.join(a.out, "batches.jsonl"), "w") as fh:
        fh.write("".join(json.dumps(x) + "\n" for x in C["batches"]))
    res = {"checkpoint": a.checkpoint, "device": str(dev),
           "reference": {"step": ref["step"], "real": ref["val/mlm_acc_real"], "shuffled": ref["val/mlm_acc_shuffled"]},
           "A_align_evaluate": {m: A["val/mlm_acc_" + m] for m in ("real", "shuffled")},
           "B_probe": {m: B[m]["all"]["accuracy"] for m in ("real", "shuffled")},
           "C_instrumented": C["summary"]}
    tmp = os.path.join(a.out, "summary.json.tmp")
    json.dump(res, open(tmp, "w"), indent=1)
    os.replace(tmp, os.path.join(a.out, "summary.json"))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
