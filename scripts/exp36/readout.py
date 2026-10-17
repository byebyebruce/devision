"""exp36 readout of one checkpoint on one caption list (critic EXP36-READOUT-01): exp35's two conditions with the
same rules -- the original random-mask evaluation (reproduction / token classes only, not a capability gate) and the
full-number mask (every number token of both objects masked, nothing else) on the real picture and on the same
fixed wrong picture (rotation by one inside each batch of 16) -- plus, from the same forward pass, one record per
number: sample, family, field, object, gold and predicted tokens, whether the whole number is right, the picture
shown and the condition. Read only (no update, no temperature, MLM head from the local cache, HF_HUB_OFFLINE=1).

    HF_HUB_OFFLINE=1 uv run python scripts/exp36/readout.py --checkpoint runs/exp36-b/stage1 \\
        --data data/real_layout/captions_tune.jsonl --out runs/exp36-b/numbers_tune
"""
import argparse
import importlib.util
import json
import os
import random
import sys
import time
from collections import defaultdict

_spec = importlib.util.spec_from_file_location("exp35_numbers", os.path.join(os.path.dirname(__file__), "..", "exp35", "mlm_numbers.py"))
assert _spec is not None and _spec.loader is not None
exp35 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp35)


def run(checkpoint: str, data: str, out_dir: str, device: str = "mps") -> dict:
    import torch

    from devision.infer import Decider
    from devision.model import text_encoder
    from devision.train.align import AlignConfig, _masked_batch, _mlm_logits, _MLMHead
    from devision.train.rlcd import _Images
    from devision.train.samples import read_jsonl

    started = time.time()
    c = AlignConfig(batch=16, max_caption_tokens=44, mask_prob=0.5)
    decider = Decider.load(checkpoint, device=device)
    model, tok = decider.model, decider.tok
    head = _MLMHead(text_encoder(model.decision))
    head.load_pretrained(exp35.HEAD_REPO)
    dev = next(model.parameters()).device
    model.eval()
    head.to(dev).eval()
    images = _Images("data", decider)
    samples = [s for s in read_jsonl(data) if s["captions"]]
    pairs = exp35.pairing(samples, c.batch)
    kind_of = [p["wrong_picture"] for p in pairs]
    shown = [p["wrong_picture_id"] for p in pairs]
    os.makedirs(out_dir, exist_ok=True)
    start = {"checkpoint": checkpoint, "data": data, "rows": len(samples),
             "sha256": {"model.safetensors": exp35._sha(os.path.join(checkpoint, "model.safetensors")), "data": exp35._sha(data)},
             "head": exp35.head_identity(), "order_and_wrong_pictures": pairs}
    json.dump(start, open(os.path.join(out_dir, "START.json"), "w"), indent=1)   # before any forward

    # original random-mask evaluation (align._evaluate), token classes
    orig = {}
    for mode in ("real", "shuffled"):
        rng = random.Random(1)
        acc = defaultdict(exp35.Acc)
        with torch.no_grad():
            for b in range(0, len(samples), c.batch):
                chunk = samples[b:b + c.batch]
                caps = [rng.choice(s["captions"]) for s in chunk]
                pix = chunk if mode == "real" else chunk[1:] + chunk[:1]
                input_ids, attention, target = _masked_batch(tok, caps, c, rng)
                logits = _mlm_logits(model, head, input_ids, attention, torch.stack([images(s["image"]) for s in pix]), dev)
                y = target[:, 1:].to(dev)
                logp = torch.log_softmax(logits, -1)
                for i, cap in enumerate(caps):
                    spans = exp35.number_spans(cap)
                    _, offs = exp35.encode(tok, cap, c.max_caption_tokens)
                    which = exp35.token_spans(offs, spans)
                    for t in range(len(offs)):
                        if int(y[i, t]) == -100:
                            continue
                        hit, nll = int(logits[i, t].argmax()) == int(y[i, t]), -float(logp[i, t, y[i, t]])
                        acc["all"].add(hit, nll)
                        acc["number" if which[t] is not None else "other"].add(hit, nll)
        orig[mode] = {k: v.out() for k, v in sorted(acc.items())}

    # full-number mask, one record per number
    records = []
    with torch.no_grad():
        for mode in ("real", "wrong"):
            for b in range(0, len(samples), c.batch):
                idx = list(range(b, min(b + c.batch, len(samples))))
                chunk = [samples[i] for i in idx]
                caps = [s["captions"][0] for s in chunk]
                pix = chunk if mode == "real" else chunk[1:] + chunk[:1]
                seqs, labels, info = [], [], []
                for cap in caps:
                    ids, offs = exp35.encode(tok, cap, c.max_caption_tokens)
                    spans = exp35.number_spans(cap)
                    which = exp35.token_spans(offs, spans)
                    seqs.append([tok.cls_token_id] + [tok.mask_token_id if w is not None else t for t, w in zip(ids, which)] + [tok.sep_token_id])
                    labels.append([-100] + [t if w is not None else -100 for t, w in zip(ids, which)] + [-100])
                    info.append((ids, spans, which))
                L = max(map(len, seqs))
                input_ids = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long)
                attention = torch.zeros((len(seqs), L), dtype=torch.long)
                for i, sq in enumerate(seqs):
                    input_ids[i, :len(sq)], attention[i, :len(sq)] = torch.tensor(sq), 1
                logits = _mlm_logits(model, head, input_ids, attention, torch.stack([images(s["image"]) for s in pix]), dev)
                pred = logits.argmax(-1).cpu()
                for i, (ids, spans, which) in enumerate(info):
                    s = chunk[i]
                    for k, sp in enumerate(spans):
                        toks = [t for t, w in enumerate(which) if w == k]
                        gold = [ids[t] for t in toks]
                        got = [int(pred[i, t]) for t in toks]
                        records.append({"sample_id": s["id"], "family": s["family"], "source": s.get("source"),
                                        "field": sp["field"], "object": sp["object"], "text": sp["text"],
                                        "gold_tokens": gold, "pred_tokens": got, "right": gold == got,
                                        "picture": s["id"] if mode == "real" else shown[idx[i]], "mode": mode,
                                        "wrong_picture_kind": kind_of[idx[i]], "condition": "full-number"})
    with open(os.path.join(out_dir, "numbers.jsonl"), "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    report = dict(start, original=orig, seconds=round(time.time() - started, 1), records=len(records),
                  records_sha256=exp35._sha(os.path.join(out_dir, "numbers.jsonl")))
    report.pop("order_and_wrong_pictures")
    json.dump(report, open(os.path.join(out_dir, "report.json"), "w"), indent=1)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args(argv)
    r = run(a.checkpoint, a.data, a.out, a.device)
    print("%s on %s: original real %.4f shuffled %.4f (numbers %.3f / %.3f); %d number records; %.0f s" % (
        a.checkpoint, a.data, r["original"]["real"]["all"]["accuracy"], r["original"]["shuffled"]["all"]["accuracy"],
        r["original"]["real"]["number"]["accuracy"], r["original"]["shuffled"]["number"]["accuracy"], r["records"], r["seconds"]))


if __name__ == "__main__":
    sys.exit(main())
