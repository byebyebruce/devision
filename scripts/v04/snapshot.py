"""A loadable checkpoint from a running stage-2 training's resume.pt (look at a run mid-way without stopping it).

    uv run python scripts/v04/snapshot.py runs/r19-n2/train/resume.pt --init runs/v03/model --out runs/r19-n2/snap

Rebuilds the starting checkpoint with the same LoRA wrapping as rlcd.train (r / alpha / targets from the state's
config), copies the trainable weights stored in resume.pt, merges the LoRA and saves. Temperatures stay the starting
checkpoint's (not refitted), so only accuracies -- not NLL -- are meaningful. Runs on the CPU, so it does not touch
the GPU the training uses. Only for runs without the training-only extras (grounding, conditioning, reader, ...).
"""
import argparse
import json
import os
from typing import Any

import torch


def main(argv=None) -> None:
    from peft import LoraConfig, get_peft_model
    from devision.infer import Decider
    from devision.model import text_encoder
    from devision.train.rlcd import LORA_TARGETS
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("resume")
    p.add_argument("--init", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    state = torch.load(a.resume, map_location="cpu", weights_only=False)
    c = state["config"]
    for k in ("grounding_weight", "question_conditioning", "number_file", "region_file", "unfreeze_top"):
        if c.get(k):
            raise SystemExit("resume.pt uses %s; this script only rebuilds plain runs" % k)
    decider = Decider.load(a.init, device="cpu")
    model: Any = decider.model
    peft_encoder = get_peft_model(text_encoder(model.decision), LoraConfig(
        r=c["lora_r"], lora_alpha=c["lora_alpha"], lora_dropout=0.0, target_modules=LORA_TARGETS))
    model.decision.encoder = peft_encoder
    params = dict(model.named_parameters())
    missing = [n for n in state["weights"] if n not in params]
    if missing:
        raise SystemExit("weights not in the rebuilt model: %s" % missing[:5])
    with torch.no_grad():
        for n, w in state["weights"].items():
            params[n].copy_(w.to(params[n].dtype))
    model.decision.encoder = peft_encoder.merge_and_unload()
    decider.model = model.cpu().float().eval()
    decider.save(a.out)
    with open(os.path.join(a.out, "SNAPSHOT.json"), "w") as f:
        json.dump({"resume": a.resume, "step": len(state["losses"]), "init": a.init,
                   "note": "temperatures are the init's, not refitted"}, f, indent=1)
    print("step %d -> %s (%d tensors copied)" % (len(state["losses"]), a.out, len(state["weights"])))


if __name__ == "__main__":
    main()
