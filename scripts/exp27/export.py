"""exp27 (docs/experiments/2026-10-17-exp27-transfer-plan.md): turn a stage-2 resume state (<stage>/resume.pt)
into a normal checkpoint, without training. exp26's chain stops a run once its step-5000 records are written, so
the run never reached its final save; resume.pt holds the trained weights of step 5000.

The model is set up exactly as src/devision/train/rlcd.py's train() does before the first step (vision tower frozen
except its top `vision_unfreeze_top` layers, ModernBERT wrapped in the same LoRA, its top `unfreeze_top` layers
trainable), the trainable weights are copied from the state -- their names must match one for one -- and the LoRA
is merged, as train() does before saving. Temperatures stay the init checkpoint's: accuracy does not depend on them.

    uv run python scripts/exp27/export.py runs/exp26-align/stage1 runs/exp26-read-compare/stage2/resume.pt \
        runs/exp27-transfer/exp26-step5000
"""
import argparse
import json
import os
import sys
from dataclasses import fields

import torch
from peft import LoraConfig, get_peft_model

from devision.infer import Decider
from devision.model import text_encoder
from devision.train.rlcd import LORA_TARGETS, TrainConfig, _vision_params


def export(decider: Decider, state: dict) -> Decider:
    names = {f.name for f in fields(TrainConfig)}
    c = TrainConfig(**{k: v for k, v in state["config"].items() if k in names})
    if c.question_conditioning or c.grounding_weight > 0:
        raise ValueError("question conditioning / grounding runs are not supported")
    model = decider.model
    model.vision.requires_grad_(False)
    for p in (_vision_params(model.vision, c.vision_unfreeze_top) if c.vision_unfreeze_top else []):
        p.requires_grad_(True)
    peft_encoder = get_peft_model(text_encoder(model.decision), LoraConfig(
        r=c.lora_r, lora_alpha=c.lora_alpha, lora_dropout=0.0, target_modules=LORA_TARGETS))
    model.decision.encoder = peft_encoder
    n_layers = text_encoder(model.decision).config.num_hidden_layers
    top = tuple("layers.%d." % i for i in range(n_layers - c.unfreeze_top, n_layers))
    for n, p in peft_encoder.named_parameters():
        if "lora_" not in n and any(t in n for t in top):
            p.requires_grad_(True)
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if set(trainable) != set(state["weights"]):
        missing, extra = set(trainable) - set(state["weights"]), set(state["weights"]) - set(trainable)
        raise ValueError("the state does not match this setup: %d missing, %d extra, e.g. %s"
                         % (len(missing), len(extra), sorted(missing | extra)[:3]))
    with torch.no_grad():
        for n, p in trainable.items():
            p.copy_(state["weights"][n].to(p.dtype))
    model.decision.encoder = peft_encoder.merge_and_unload()
    decider.model = model.cpu().float().eval().requires_grad_(False)
    return decider


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("init")
    p.add_argument("resume")
    p.add_argument("out")
    a = p.parse_args()
    if os.path.exists(a.out):
        raise SystemExit("%s exists; remove it first" % a.out)
    state = torch.load(a.resume, map_location="cpu", weights_only=False)
    d = export(Decider.load(a.init, device="cpu"), state)
    d.save(a.out)
    json.dump({"init": a.init, "resume": a.resume, "step": len(state["losses"]),
               "trained_tensors": len(state["weights"])}, open(os.path.join(a.out, "export.json"), "w"), indent=1)
    print("wrote %s (step %d)" % (a.out, len(state["losses"])))


if __name__ == "__main__":
    sys.exit(main())
