"""PANEL-BIND-PREP-01: a direct number-binding probe of a frozen end point on EXP26's 1,000 held-out panel pictures
(data/exp26/align_val.jsonl), through align's own caption path (align._mlm_logits: the whole picture, the cached
ModernBERT MLM head, batch 16, captions <= 40 tokens). No training, no new data, no oracle.

  random   align._evaluate reproduced step for step (Random(1) per mode, rng.choice of the caption, _masked_batch with
           mask 0.5, the wrong picture = the next row's in the same batch of 16), with each target token marked as a
           number token (inside the A / B number span) or not: real / shuffled accuracy overall, on number tokens and
           on the rest, with their denominators (they recompose the total).
  numbers  every token of both number spans masked (A and B together), every other token visible; the same inputs and
           masks with the true picture and with the fixed wrong picture; per row the decoded A / B predictions.

Spans: "A: <quantity> <number> <unit>. B: <quantity> <number> <unit>." -- a caption that does not parse, a number whose
tokens do not cover exactly its characters, or a caption longer than 40 tokens is refused before any forward pass.
Writes <out>/<mode>.jsonl (one line per row and picture mode) and <out>/summary.json.

    uv run python scripts/panel_bind/probe.py --checkpoint DIR --out DIR [--device auto]
"""
import argparse
import json
import os
import random
import re
import sys
from typing import Dict, List, Tuple

import torch

VAL = "data/exp26/align_val.jsonl"
PATTERN = re.compile(r"^A: (?P<qa>.+?) (?P<a>[\d,]+(?:\.\d+)?) (?P<ua>\S+)\. B: (?P<qb>.+?) (?P<b>[\d,]+(?:\.\d+)?) (?P<ub>\S+)\.$")
BATCH, MAX_TOKENS, MASK_PROB = 16, 40, 0.5


def quantity(caption: str) -> str:
    m = PATTERN.match(caption)
    if not m:
        raise ValueError("caption does not parse: %r" % caption)
    return m.group("qa")


def spans(caption: str) -> Dict[str, Tuple[int, int, str]]:
    m = PATTERN.match(caption)
    if not m:
        raise ValueError("caption does not parse: %r" % caption)
    return {"A": (m.start("a"), m.end("a"), m.group("a")), "B": (m.start("b"), m.end("b"), m.group("b"))}


def number_tokens(tok, caption: str) -> Tuple[List[int], Dict[str, List[int]]]:
    """(token ids, {"A": [token positions], "B": [...]}); refused unless each number's tokens cover exactly its text."""
    enc = tok(caption, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    if len(ids) > MAX_TOKENS:
        raise ValueError("caption longer than %d tokens: %r" % (MAX_TOKENS, caption))
    out = {}
    for end, (s, e, text) in spans(caption).items():
        pos = [i for i, (a, b) in enumerate(offs) if a < e and b > s and b > a]
        covered = "".join(caption[a:b] for i, (a, b) in enumerate(offs) if i in pos).strip()
        if not pos or covered != text:
            raise ValueError("number %s of %r: tokens cover %r" % (end, caption, covered))
        out[end] = pos
    return ids, out


def load(checkpoint: str, device: str):
    from devision.infer import Decider
    from devision.model import text_encoder
    from devision.train.align import _MLMHead
    from devision.train.rlcd import _device, _Images
    d = Decider.load(checkpoint)
    dev = _device(device)
    head = _MLMHead(text_encoder(d.model.decision))
    head.load_pretrained("answerdotai/ModernBERT-large")
    d.model.to(dev).eval()
    head.to(dev).eval()
    return d, head, _Images("data", d), dev


def run_random(d, head, images, dev, samples) -> Dict:
    """align._evaluate, with number / other token accounting."""
    from devision.train.align import AlignConfig, _logits_in_parts, _masked_batch
    c = AlignConfig(batch=BATCH, max_caption_tokens=MAX_TOKENS, mask_prob=MASK_PROB, micro_batch=0)
    tok = d.tok
    numpos = {s["id"]: set(sum(number_tokens(tok, s["captions"][0])[1].values(), [])) for s in samples}
    out: Dict = {}
    rows = []
    for mode in ("real", "shuffled"):
        rng = random.Random(1)
        cnt = {"all": [0, 0], "number": [0, 0], "other": [0, 0]}
        with torch.no_grad():
            for b in range(0, len(samples), c.batch):
                chunk = samples[b:b + c.batch]
                caps = [rng.choice(s["captions"]) for s in chunk]
                pix_src = chunk if mode == "real" else chunk[1:] + chunk[:1]
                input_ids, attention, target = _masked_batch(tok, caps, c, rng)
                logits = _logits_in_parts(d.model, head, input_ids, attention, torch.stack([images(s["image"]) for s in pix_src]), dev, 0)
                y = target[:, 1:]
                pred = logits.argmax(-1).cpu()
                for i, s in enumerate(chunk):
                    hit = {"number": [0, 0], "other": [0, 0]}
                    for t in range(y.size(1)):
                        if y[i, t] == -100:
                            continue
                        k = "number" if t in numpos[s["id"]] else "other"     # y[:, t] is caption token t
                        hit[k][0] += int(pred[i, t] == y[i, t])
                        hit[k][1] += 1
                    for k in ("number", "other"):
                        cnt[k][0] += hit[k][0]
                        cnt[k][1] += hit[k][1]
                        cnt["all"][0] += hit[k][0]
                        cnt["all"][1] += hit[k][1]
                    rows.append({"id": s["id"], "family": s["family"], "mode": mode, "picture": pix_src[i]["image"], "tokens": hit})
        out[mode] = {k: {"right": v[0], "n": v[1], "accuracy": v[0] / max(1, v[1])} for k, v in cnt.items()}
    out["gap"] = out["real"]["all"]["accuracy"] - out["shuffled"]["all"]["accuracy"]
    return {"summary": out, "rows": rows}


def run_numbers(d, head, images, dev, samples) -> Dict:
    """Both number spans masked together; the rest visible; true and fixed wrong picture with identical inputs."""
    from devision.train.align import _logits_in_parts
    tok = d.tok
    rows = []
    for b in range(0, len(samples), BATCH):
        chunk = samples[b:b + BATCH]
        enc = [number_tokens(tok, s["captions"][0]) for s in chunk]
        seqs, labels = [], []
        for ids, pos in enc:
            masked = set(pos["A"]) | set(pos["B"])
            seqs.append([tok.cls_token_id] + [tok.mask_token_id if i in masked else t for i, t in enumerate(ids)] + [tok.sep_token_id])
            labels.append([-100] + [t if i in masked else -100 for i, t in enumerate(ids)] + [-100])
        n, L = len(seqs), max(map(len, seqs))
        input_ids = torch.full((n, L), tok.pad_token_id, dtype=torch.long)
        attention = torch.zeros((n, L), dtype=torch.long)
        for i, sq in enumerate(seqs):
            input_ids[i, :len(sq)], attention[i, :len(sq)] = torch.tensor(sq), 1
        for mode in ("real", "shuffled"):
            pix_src = chunk if mode == "real" else chunk[1:] + chunk[:1]
            with torch.no_grad():
                pred = _logits_in_parts(d.model, head, input_ids, attention, torch.stack([images(s["image"]) for s in pix_src]), dev, 0).argmax(-1).cpu()
            for i, (s, (ids, pos)) in enumerate(zip(chunk, enc)):
                got = {e: tok.decode([int(pred[i, t]) for t in pos[e]]).strip() for e in ("A", "B")}
                gold = {e: spans(s["captions"][0])[e][2] for e in ("A", "B")}
                tok_right = {e: [int(pred[i, t]) == ids[t] for t in pos[e]] for e in ("A", "B")}
                rows.append({"id": s["id"], "family": s["family"], "mode": mode, "picture": pix_src[i]["image"], "gold": gold, "pred": got,
                             "token_right": tok_right, "exact": {e: all(tok_right[e]) for e in ("A", "B")},
                             "quantity_same_as_shown": quantity(s["captions"][0]) == quantity(pix_src[i]["captions"][0])})
    return {"rows": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    samples = [json.loads(x) for x in open(VAL) if x.strip()]
    d, head, images, dev = load(a.checkpoint, a.device)
    for s in samples:                                   # every caption parses and its numbers tokenise exactly: before any forward
        number_tokens(d.tok, s["captions"][0])
    if os.path.exists(a.out):
        raise SystemExit("%s exists" % a.out)
    os.makedirs(a.out)
    r = run_random(d, head, images, dev, samples)
    open(os.path.join(a.out, "random.jsonl"), "w").write("".join(json.dumps(x) + "\n" for x in r["rows"]))
    n = run_numbers(d, head, images, dev, samples)
    open(os.path.join(a.out, "numbers.jsonl"), "w").write("".join(json.dumps(x) + "\n" for x in n["rows"]))
    tmp = os.path.join(a.out, "summary.json.tmp")             # written last and atomically: a summary means the end point finished
    json.dump({"checkpoint": a.checkpoint, "random": r["summary"], "rows": len(samples)}, open(tmp, "w"), indent=1)
    os.replace(tmp, os.path.join(a.out, "summary.json"))
    print(json.dumps(r["summary"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
