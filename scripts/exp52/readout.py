"""exp52 number readout and content-use check (critic EXP52-PREP-01 items 3-4), after an arm's weights are frozen. On the
reading questions of fit_O and rl_tune (data/exp49/questions.jsonl), per picture: visual_inputs once, then ONE ordinary
forward (rlcd._forward's path, temperature 1, no oracle: the question's own text and the picture only) that returns
the option logits and -- from the same content tokens -- the character head's string; then the same forward with the
reader's content tokens left out of ModernBERT's input (reader.off), a fixed removal used only as dependence evidence.
Writes runs/<arm>/eval/readout_details.jsonl (EXP49's details format: predicted characters; donor = the value-changed
twin's own prediction, i.e. the same question on the twin picture) and readout_use.jsonl (per question: gold option,
option logits with and without the content), and readout.json (identity, timing).

    uv run python scripts/exp52/readout.py --run runs/exp52-n
"""
import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict

import torch

HERE = os.path.dirname(__file__)
SETS = {"fit_O": "data/exp42/fit_O.jsonl", "rl_tune": "data/real_layout/questions_tune.jsonl"}


def sha(p: str) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--questions", default="data/exp49/questions.jsonl")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--families", type=int, default=0, help="CPU smoke tests only: the first N families (0: all)")
    a = ap.parse_args(argv)
    from laya.common import collate_items

    from devision.infer import Decider, best_device
    from devision.train import rlcd
    sys.path.insert(0, os.path.join(HERE, "..", "exp49"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("exp49_probe", os.path.join(HERE, "..", "exp49", "probe.py"))
    assert spec is not None and spec.loader is not None
    p49 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p49)
    t0 = time.time()
    qs = [json.loads(x) for x in open(a.questions) if x.strip()]
    qs = [q for q in qs if q["set"] in SETS]
    if a.families:
        keep = sorted({q["family"] for q in qs})[:a.families]
        qs = [q for q in qs if q["family"] in keep]
    rows = {n: {r["id"]: r for r in map(json.loads, open(p))} for n, p in SETS.items()}
    ck = os.path.join(a.run, "stage2")
    device = best_device() if a.device == "auto" else a.device
    d = Decider.load(ck, device=device)
    model = d.model.eval()
    reader = model.reader
    if reader is None:
        raise SystemExit("%s has no number reader" % ck)
    reader.keep_chars = True
    images = rlcd._Images(a.data_root, d)
    by_pic = defaultdict(list)
    for i, q in enumerate(qs):
        by_pic[(q["set"], q["picture"])].append(i)
    pred: list = [None] * len(qs)
    use: list = [None] * len(qs)
    with torch.no_grad():
        for (setname, pic), idx in sorted(by_pic.items()):
            samples = []
            for i in idx:
                r = rows[setname][qs[i]["id"]]
                samples.append({**r, "questions": {qs[i]["qid"]: r["questions"][qs[i]["qid"]]},
                                "gold": {qs[i]["qid"]: r["gold"][qs[i]["qid"]]}})
            items = rlcd._items(d, samples)
            batch = collate_items([items], d.tok.pad_token_id)
            assert batch is not None
            px = images(pic)[None]
            index = torch.zeros(len(items), dtype=torch.long)
            reader.off = False
            on = rlcd._forward(model, batch, px, device, index=index).float().cpu()
            assert reader.last_chars is not None
            chars = reader.last_chars.argmax(-1).cpu().tolist()
            reader.off = True
            off = rlcd._forward(model, batch, px, device, index=index).float().cpu()
            reader.off = False
            for k, i in enumerate(idx):
                n = len(items[k]["keys"])
                pred[i] = chars[k]
                use[i] = {"key": qs[i]["key"], "gold": items[k]["target"].index(max(items[k]["target"])),
                          "on": on[k, :n].tolist(), "off": off[k, :n].tolist()}
    key_i = {q["key"]: i for i, q in enumerate(qs)}
    donor_idx = [key_i[q["twin"]] if q["twin"] and q["twin"] in key_i else i for i, q in enumerate(qs)]
    donor = [pred[j] for j in donor_idx]
    ev = os.path.join(a.run, "eval")
    os.makedirs(ev, exist_ok=True)
    ident = {"checkpoint": ck, "model.safetensors": sha(os.path.join(ck, "model.safetensors")), "config.json": sha(os.path.join(ck, "config.json")),
             "questions_sha256": sha(a.questions), "sets": {n: sha(p) for n, p in SETS.items()}, "path": "rlcd._forward, temperature 1"}
    dsha = p49.write_details(os.path.join(ev, "readout_details.jsonl"), qs, pred, donor, donor_idx, ident)
    with open(os.path.join(ev, "readout_use.jsonl"), "w") as fh:
        fh.write(json.dumps({"identity": ident}) + "\n")
        fh.write("".join(json.dumps(u) + "\n" for u in use))
    info = {"identity": ident, "questions": len(qs), "pictures": len(by_pic), "seconds": time.time() - t0, "device": device,
            "details_sha256": dsha, "use_sha256": sha(os.path.join(ev, "readout_use.jsonl"))}
    json.dump(info, open(os.path.join(ev, "readout.json"), "w"), indent=1)
    print(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
