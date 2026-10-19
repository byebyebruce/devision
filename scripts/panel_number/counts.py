"""PANEL-NUMBER-PREP-01: align's training stream replayed on CPU with the real tokenizer (no model): the same
Random(seed) shuffle, caption choice and _masked_batch draw align makes, then -- for arm N -- _number_rows on every
exp26-panels row. Reports, per arm, the updates, the batch order digest (must be equal for N and R), and the supervised
number / other target tokens (number = inside the A / B spans). Refuses (via _number_rows) any caption whose numbers
are missing, do not tokenise exactly or overflow max_caption_tokens -- on every training caption, before any run.

    uv run python scripts/panel_number/counts.py [--tokenizer runs/exp74-panel-d1/d1/tokenizer]
"""
import argparse
import hashlib
import json
import random

import yaml

from devision.train.align import AlignConfig, _masked_batch, _number_rows, number_spans


def replay(config: str, tok):
    st = yaml.safe_load(open(config))["stages"][0]
    p = st["params"]
    c = AlignConfig(batch=p["batch"], max_caption_tokens=p["max_caption_tokens"], mask_prob=p["mask_prob"], seed=p["seed"],
                    number_mask_sources=p.get("number_mask_sources", ""))
    samples = [s for s in map(json.loads, open(st["data"])) if s["captions"]]
    sources = {x for x in c.number_mask_sources.split(",") if x}
    rng = random.Random(c.seed)
    order_digest = hashlib.sha256()
    cnt = {"number": 0, "other": 0}
    updates = 0
    for _ in range(p["epochs"]):
        order = list(samples)
        rng.shuffle(order)
        for b in range(0, len(order) - c.batch + 1, c.batch):
            chunk = order[b:b + c.batch]
            caps = [rng.choice(s["captions"]) for s in chunk]
            order_digest.update(json.dumps([s["id"] for s in chunk]).encode())
            input_ids, attention, target = _masked_batch(tok, caps, c, rng)
            if sources:
                _number_rows(tok, caps, [i for i, s in enumerate(chunk) if s.get("source") in sources], input_ids, target, c)
            for i, cap in enumerate(caps):
                enc = tok(cap, add_special_tokens=False, return_offsets_mapping=True)
                spans = number_spans(cap)
                for t, (a, bb) in enumerate(enc["offset_mapping"][:c.max_caption_tokens]):
                    if target[i, 1 + t] != -100:
                        num = any(a < s["end"] and bb > s["start"] and bb > a for s in spans)
                        cnt["number" if num else "other"] += 1
            updates += 1
    return {"updates": updates, "order_sha256": order_digest.hexdigest(), "targets": cnt}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="runs/exp74-panel-d1/d1/tokenizer")
    a = ap.parse_args(argv)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    out = {arm: replay("configs/exp75-panel-%s.yaml" % arm, tok) for arm in ("n", "r")}
    out["same_order"] = out["n"]["order_sha256"] == out["r"]["order_sha256"]
    print(json.dumps(out, indent=1))
    return 0 if out["same_order"] and out["n"]["targets"]["other"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
