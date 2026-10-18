"""exp44 input / target check on every question, no model (critic EXP44-PREP-01): with exp40-M's own tokenizer, the
number-reconstruction example (rlcd.aux_example: picture, state text, question, 8 [MASK] slots, no option) of every
training question (data/exp42/train.jsonl) and every reading question of the four readout sets (fit_O, fit_L, rl_tune,
rl_tune_L): none refused (a number gold, one gold option, fits the slots), the largest target span (number tokens +
end), the formats seen; the frozen MLM head's file and tensor sha256 (from the local cache, nothing downloaded); and
12 examples (aux input / slots / target and the ordinary decision input) to runs/exp44-preview/examples.md. Writes
data/exp44/CHECK.json.

    HF_HUB_OFFLINE=1 uv run python scripts/exp44/check.py
"""
import hashlib
import json
import os
import re
from collections import Counter

from transformers import AutoTokenizer

from devision.infer import question_item
from devision.model import ModelConfig
from devision.train.rlcd import aux_table

TOK = "runs/exp40-m/stage2/tokenizer"
SETS = {"train": "data/exp42/train.jsonl", "fit_O": "data/exp42/fit_O.jsonl", "fit_L": "data/exp42/fit_L.jsonl",
        "rl_tune": "data/real_layout/questions_tune.jsonl", "rl_tune_L": "data/exp40/rl_tune_L.jsonl"}
SLOTS = 8


def reading_only(rows):
    out = []
    for r in rows:
        qs = {q: v for q, v in r["questions"].items() if any(w in v["instructions"] for w in ("mass", "speed", "distance"))}
        if qs:
            out.append(dict(r, questions=qs, gold={q: r["gold"][q] for q in qs}))
    return out


def head_identity():
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    path = hf_hub_download("answerdotai/ModernBERT-large", "model.safetensors")
    out = {"file": os.path.realpath(path), "file_sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}
    with safe_open(path, "pt") as f:
        for k in ("head.dense.weight", "head.norm.weight", "decoder.bias"):
            out[k] = hashlib.sha256(f.get_tensor(k).float().numpy().tobytes()).hexdigest()
    return out


def main():
    tok = AutoTokenizer.from_pretrained(TOK)
    cfg_d = json.load(open("runs/exp40-m/stage2/config.json"))
    cfg = ModelConfig(**{k: v for k, v in cfg_d.items() if k in ModelConfig.__dataclass_fields__})
    res = {"slots": SLOTS, "tokenizer": TOK, "sets": {}, "head": head_identity()}
    tables, rows_by = {}, {}
    for name, path in SETS.items():
        rows = reading_only([json.loads(x) for x in open(path)])
        table = aux_table(tok, rows, SLOTS)                    # raises on any refused question
        spans = Counter(sum(t != -100 for t in tgt) for _, tgt in table.values())
        inputs = {len(ids) for ids, _ in table.values()}
        res["sets"][name] = {"questions": len(table), "target_span_tokens": dict(sorted(spans.items())),
                             "max_span": max(spans), "input_tokens": [min(inputs), max(inputs)],
                             "formats": dict(Counter(re.sub(r"\d", "9", next(o for o, p in r["gold"][q]["probabilities"].items() if p == 1.0).split()[0])
                                                     for r in rows for q in r["questions"]))}
        tables[name], rows_by[name] = table, rows
    os.makedirs("data/exp44", exist_ok=True)
    json.dump(res, open("data/exp44/CHECK.json", "w"), indent=1)
    # 12 examples from rl_tune by a fixed rule: per field, the first two pictures (by id) whose two objects differ,
    # both objects -- integers, decimals, long speeds, different gold lengths
    lines = ["# exp44 examples (rl_tune; per field the first two different-value pictures by id, both objects)", ""]
    picked = []
    for field in ("mass", "speed", "distance"):
        n = 0
        for r in sorted(rows_by["rl_tune"], key=lambda r: r["id"]):
            qs = [q for q, v in r["questions"].items() if field in v["instructions"]]
            golds = {next(o for o, p in r["gold"][q]["probabilities"].items() if p == 1.0) for q in qs}
            if len(qs) == 2 and len(golds) == 2:
                picked += [(r, q) for q in qs]
                n += 1
            if n == 2:
                break
    for r, q in picked:
        ids, tgt = tables["rl_tune"][(r["id"], q)]
        v = r["questions"][q]
        ce = question_item(tok, cfg, r["state_text"], v)
        lines += ["## %s %s" % (r["id"], q), "",
                  "- aux input (%d tokens): `%s`" % (len(ids), tok.decode(ids).replace("\n", " ")),
                  "- slots: 8 x [MASK] at positions %d..%d" % (len(ids) - 9, len(ids) - 2),
                  "- target: %s -> `%s`" % ([t for t in tgt], tok.decode([t for t in tgt if t != -100])),
                  "- decision (CE) input: `%s`" % tok.decode(ce["ids"]).replace("\n", " "), ""]
    os.makedirs("runs/exp44-preview", exist_ok=True)
    open("runs/exp44-preview/examples.md", "w").write("\n".join(lines))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
