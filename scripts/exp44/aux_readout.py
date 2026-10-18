"""exp44 number readout through the frozen MLM head (critic EXP44-PREP-03; a diagnosis of the shared encoder, not a
product output): for every reading question of a set, the training example's input (rlcd.aux_example: picture, state
text, question, 8 [MASK] slots, no option) -> the most probable token per slot, cut at the first [SEP]; no end token
or no number: invalid; never corrected towards an option. Writes <out>/<name>.json (run identity, as devision-eval
writes it: checkpoint, data sha256, head identity) and <name>.details.jsonl (per question: the aux input's sha256,
gold number, prediction, valid, correct).

    uv run python scripts/exp44/aux_readout.py runs/exp44-j/stage2 data/exp42/fit_O.jsonl runs/exp44-j/eval aux_fit_O
"""
import argparse
import hashlib
import json
import os
import re
import sys

import torch

from devision.infer import Decider, best_device
from devision.model import text_encoder
from devision.train.rlcd import _Images, aux_example, aux_head, aux_logits, aux_number

SLOTS = 8
NUMBER = re.compile(r"^\d[\d,]*(?:\.\d+)?$")


def reading(rows):
    for r in rows:
        for q, v in r["questions"].items():
            if any(w in v["instructions"] for w in ("mass", "speed", "distance")):
                yield r, q, v


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("data")
    p.add_argument("out")
    p.add_argument("name")
    p.add_argument("--data-root", default="data")
    p.add_argument("--head", default="answerdotai/ModernBERT-large", help='"" = a random head (tiny tests only)')
    p.add_argument("--device", default="auto")
    a = p.parse_args(argv)
    ckpt, data, out, name = a.checkpoint, a.data, a.out, a.name
    device = best_device() if a.device == "auto" else a.device
    d = Decider.load(ckpt, device=device)
    model, tok = d.model, d.tok
    head, ident = aux_head(text_encoder(model.decision), a.head)
    head.to(device).eval()
    images = _Images(a.data_root, d)
    rows = [json.loads(x) for x in open(data)]
    todo = []
    for r, q, v in reading(rows):
        gold = next(o for o, p in r["gold"][q]["probabilities"].items() if p == 1.0)
        ids, _ = aux_example(tok, r.get("state_text", ""), v["instructions"], gold, SLOTS)
        todo.append((r, q, gold, ids))
    recs = []
    with torch.no_grad():
        for i in range(0, len(todo), 8):
            chunk = todo[i:i + 8]
            pixels = torch.stack([images(r["image"]) for r, _, _, _ in chunk])
            pred = aux_logits(model, head, [c[3] for c in chunk], pixels, device, SLOTS, tok.pad_token_id).argmax(-1).cpu()
            for (r, q, gold, ids), p in zip(chunk, pred.tolist()):
                end = p.index(tok.sep_token_id) if tok.sep_token_id in p else None
                text = tok.decode(p[:end]).strip() if end is not None else None
                valid = text is not None and bool(NUMBER.match(text))
                g = aux_number(gold)
                recs.append({"sample_id": r["id"], "qid": q, "image_id": r["image_id"], "image_used": r["image"],
                             "aux_input_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                             "gold_number": g, "prediction": text, "valid": valid, "correct": valid and text == g})
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, name + ".details.jsonl"), "w") as fh:
        for x in recs:
            fh.write(json.dumps(x) + "\n")
    info = {"checkpoint": ckpt, "data": data, "data_sha256": hashlib.sha256(open(data, "rb").read()).hexdigest(),
            "device": device, "slots": SLOTS, "head": ident, "questions": len(recs)}
    json.dump({"run": info, "summary": {"correct": sum(x["correct"] for x in recs), "valid": sum(x["valid"] for x in recs),
                                        "questions": len(recs)}}, open(os.path.join(out, name + ".json"), "w"), indent=1)
    print("%s: %d / %d correct, %d valid" % (name, sum(x["correct"] for x in recs), len(recs), sum(x["valid"] for x in recs)))


if __name__ == "__main__":
    main()
