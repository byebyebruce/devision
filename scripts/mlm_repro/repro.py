"""MLM-REPRO-PREP-01 (round 1: MLM-REPRO-R1-01 / R1-02): where does the saved N end point's wrong-picture accuracy
(0.669519 through the probe) part from its training-time validation (0.680539, align_report step 1,250)?

  run       One loaded instance of the saved N end point; identity first (no model): the accepted EXP75 START
            (runs/panel-number/START.json 1827f65d...) for N's model / config / tokenizer, the cached MLM head, the
            validation file, the 1,000 pictures' bytes and the fixed wrong-picture map -- any difference, nothing is
            loaded. Then three passes, each written as soon as it finishes (nothing retrained, no full-vocabulary logits
            kept):
              A  align._evaluate as training calls it, with a runtime wrapper (not an edit of align.py) on its
                 _masked_batch / _logits_in_parts that records, per batch, the sha256 of input ids / attention / targets
                 / pictures and align's hits -> a/batches.jsonl, a/summary.json;
              B  scripts/panel_bind/probe.run_random, the same wrapper -> b/batches.jsonl, b/summary.json;
              C  the same loop, instrumented; per batch the same hashes and, on the same logits, (1) align's scoring
                 (masked first, argmax on the device, counted on the device), the same predictions counted on CPU,
                 (2) the probe's (argmax of the whole logits, then the masked positions), (3) a CPU float32 reference;
                 every token where they disagree with its row / token position, gold, predictions, top-2 margin;
                 appended batch by batch -> c/batches.jsonl, then c/summary.json.
  finalize  No model, always run (also after a failure or the limit): checks each segment's completeness (126 batches,
            63 per mode, a summary), aligns A / B / C batch by batch (input_equal, hits) and writes report.json --
            "complete" only when every segment is; otherwise "partial" with what exists and no conclusion.

Interpretation, written before running (no root cause beyond it):
  - inputs equal in A / B / C but A != B -> the scoring / call path differs; the variant equal to the training-time
    summary says which;
  - A == B == C but != the training-time summary -> the saved end point now reads differently from training time; the
    training-time state (un-merged LoRA, optimizer, RNG) was not saved, so that difference is NOT located here;
  - inputs differ somewhere -> named per batch (mode, batch, which hash);
  - (1) / (2) / (3) differ on some tokens -> those tokens are listed; not by itself a claim about the device.

    uv run python scripts/mlm_repro/repro.py run --checkpoint runs/exp75-panel-n/n --out runs/mlm-repro
    uv run python scripts/mlm_repro/repro.py finalize --out runs/mlm-repro
"""
import argparse
import contextlib
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
START = ("runs/panel-number/START.json", "1827f65d422247b2ec02e7e937bf0a3e9e0a45ecd7997b394f8cbac4358e3041")
ENDPOINT = "n"
BATCHES = 126                    # 2 modes x 63 batches of 16 (1,000 rows)


def _probe():
    spec = importlib.util.spec_from_file_location("mr_probe", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "panel_bind", "probe.py"))
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _bind_report():
    spec = importlib.util.spec_from_file_location("mr_bind_report", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "panel_bind", "report.py"))
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def sha(p: str) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def h(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def write_atomic(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, path)


def identity_problems(checkpoint: str, start=START, val: str = VAL) -> List[str]:
    """Before any load: the accepted EXP75 START names this end point, head, question list, pictures and wrong map."""
    rep = _bind_report()
    path, want = start
    if not os.path.exists(path) or sha(path) != want:
        return ["%s is not the accepted EXP75 START" % path]
    st = json.load(open(path))
    out = []
    e = st["endpoints"].get(ENDPOINT, {})
    if os.path.normpath(e.get("path", "")) != os.path.normpath(checkpoint):
        out.append("START's %s end point is %r, not %r" % (ENDPOINT, e.get("path"), checkpoint))
    for k in ("model.safetensors", "config.json"):
        p = os.path.join(checkpoint, k)
        if not os.path.exists(p) or sha(p) != e.get(k):
            out.append("%s %s is not START's" % (checkpoint, k))
    if rep.tree(os.path.join(checkpoint, "tokenizer")) != e.get("tokenizer"):
        out.append("the tokenizer is not START's")
    try:
        from huggingface_hub import hf_hub_download
        if sha(hf_hub_download(st["mlm_head"]["repo"], "model.safetensors")) != st["mlm_head"]["model.safetensors"]:
            out.append("the cached MLM head is not START's")
    except Exception as ex:                                 # offline cache missing etc.: refused, never downloaded silently
        out.append("the MLM head cannot be checked (%s)" % ex)
    if not os.path.exists(val) or sha(val) != st["val"]["sha256"]:
        out.append("the validation file is not START's")
        return out
    rows = [json.loads(x) for x in open(val) if x.strip()]
    pics = hashlib.sha256(json.dumps([sha(os.path.join("data", r["image"])) for r in rows]).encode()).hexdigest()
    if pics != st["val"]["pictures_sha256"]:
        out.append("the 1,000 pictures' bytes are not START's")
    wrong = {r["id"]: rows[b + ((i - b + 1) % len(rows[b:b + 16]))]["id"] for b in range(0, len(rows), 16) for i, r in enumerate(rows[b:b + 16], b)}
    if wrong != st["wrong_picture"]:
        out.append("the wrong-picture map is not START's")
    return out


@contextlib.contextmanager
def recording(records: List[Dict]):
    """Wraps align's _masked_batch / _logits_in_parts for the duration (both A and B call them through the align
    module), recording per call the hashes of what was actually run and align's hits on it."""
    import devision.train.align as al
    real_mb, real_lp = al._masked_batch, al._logits_in_parts
    pending: Dict = {}

    def mb(tok, captions, c, rng):
        out = real_mb(tok, captions, c, rng)
        pending["target"] = out[2]
        return out

    def lp(model, head, input_ids, attention, pixels, device, micro):
        logits = real_lp(model, head, input_ids, attention, pixels, device, micro)
        y = pending.pop("target")[:, 1:].to(logits.device)
        m = y != -100
        records.append({"input_ids": h(input_ids), "attention": h(attention), "target_hash": h(y), "pictures": h(pixels),
                        "targets_n": int(m.sum().item()), "hits": int((logits[m].argmax(-1) == y[m]).sum().item())})
        return logits
    al._masked_batch, al._logits_in_parts = mb, lp
    try:
        yield records
    finally:
        al._masked_batch, al._logits_in_parts = real_mb, real_lp


def score(logits: torch.Tensor, y: torch.Tensor) -> Dict:
    """Three scorings of one batch's logits [B, T, V] against targets y [B, T] (-100 = not a target)."""
    m = (y != -100).to(logits.device)
    yd = y.to(logits.device)
    p1 = logits[m].argmax(-1)                                   # align: masked first, argmax on the device
    p2 = logits.argmax(-1).cpu()[m.cpu()]                       # probe: argmax of everything on the device, then mask
    lc = logits.detach().float().cpu()[m.cpu()]                 # CPU float32 reference on the same logits
    p3 = lc.argmax(-1)
    where = m.cpu().nonzero().tolist()                          # (row, token) of every target, in mask order
    out = compare_predictions(p1.cpu(), p2, p3, y[m.cpu()], lc, where)
    out["hits_align_device"] = int((p1 == yd[m]).sum().item())  # counted on the device, as align does
    return out


def compare_predictions(p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor, g: torch.Tensor, lc: torch.Tensor, where=None) -> Dict:
    """Hits of the three prediction vectors (CPU) against the gold, non-finite logits, and every disagreeing token
    with its (row, token) position."""
    out = {"targets": int(g.numel()), "hits_align": int((p1 == g).sum().item()), "hits_probe": int((p2 == g).sum().item()),
           "hits_cpu": int((p3 == g).sum().item()), "nonfinite": int((~torch.isfinite(lc)).sum().item()), "disagree": []}
    diff = (p1 != p2) | (p1 != p3) | (p2 != p3)
    if diff.any():
        top2 = lc[diff].topk(2, -1).values
        for k, i in enumerate(diff.nonzero().flatten().tolist()):
            rec = {"gold": int(g[i]), "align": int(p1[i]), "probe": int(p2[i]), "cpu": int(p3[i]), "margin": float(top2[k, 0] - top2[k, 1])}
            if where is not None:
                rec["row"], rec["token"] = where[i]
            out["disagree"].append(rec)
    return out


def instrumented(d, head, images, dev, samples, sink) -> Dict:
    """C: one record per batch passed to sink as soon as it is scored."""
    from devision.train.align import AlignConfig, _logits_in_parts, _masked_batch
    c = AlignConfig(batch=16, max_caption_tokens=40, mask_prob=0.5, micro_batch=0)
    tot: Dict = {}
    for mode in ("real", "shuffled"):
        rng = random.Random(1)
        agg: Dict = {"targets": 0, "hits_align": 0, "hits_align_device": 0, "hits_probe": 0, "hits_cpu": 0, "nonfinite": 0, "disagree": 0}
        with torch.no_grad():
            for b in range(0, len(samples), c.batch):
                chunk = samples[b:b + c.batch]
                caps = [rng.choice(s["captions"]) for s in chunk]
                pix_src = chunk if mode == "real" else chunk[1:] + chunk[:1]
                input_ids, attention, target = _masked_batch(d.tok, caps, c, rng)
                pixels = torch.stack([images(s["image"]) for s in pix_src])
                logits = _logits_in_parts(d.model, head, input_ids, attention, pixels, dev, 0)
                y = target[:, 1:]
                sc = score(logits, y)
                del logits
                sink({"mode": mode, "batch": b // c.batch, "input_ids": h(input_ids), "attention": h(attention), "target_hash": h(y),
                      "pictures": h(pixels), "rows": [s["id"] for s in chunk], "shown": [s["image"] for s in pix_src], **sc})
                for k in agg:
                    agg[k] += len(sc["disagree"]) if k == "disagree" else sc[k]
        agg.update({"acc_align": agg["hits_align"] / agg["targets"], "acc_probe": agg["hits_probe"] / agg["targets"],
                    "acc_cpu": agg["hits_cpu"] / agg["targets"]})
        tot[mode] = agg
    return tot


def run(checkpoint: str, out: str, device: str) -> int:
    if os.path.exists(out):
        raise SystemExit("%s exists (fresh only)" % out)
    bad = identity_problems(checkpoint)
    os.makedirs(out)
    if bad:
        write_atomic(os.path.join(out, "identity.json"), {"ok": False, "problems": bad})
        print("refused before loading: %s" % bad)
        return 12
    probe = _probe()
    samples = [json.loads(x) for x in open(VAL) if x.strip()]
    write_atomic(os.path.join(out, "identity.json"), {"ok": True, "problems": [], "start": START[1],
                                                       "code": {f: sha(f) for f in (os.path.abspath(__file__), "scripts/panel_bind/probe.py", "src/devision/train/align.py")}})
    d, head, images, dev = probe.load(checkpoint, device)
    from devision.train.align import AlignConfig, _evaluate
    for seg, call in (("a", lambda: {m: v for m, v in _evaluate(d.model, head, d.tok, samples, images,
                                                                   AlignConfig(batch=16, max_caption_tokens=40, mask_prob=0.5, micro_batch=0), dev).items()}),
                      ("b", lambda: probe.run_random(d, head, images, dev, samples)["summary"])):
        recs: List[Dict] = []
        with recording(recs):
            summ = call()
        os.makedirs(os.path.join(out, seg))
        with open(os.path.join(out, seg, "batches.jsonl"), "w") as fh:
            fh.write("".join(json.dumps(r) + "\n" for r in recs))
        write_atomic(os.path.join(out, seg, "summary.json"), {"device": str(dev), "summary": summ})
    os.makedirs(os.path.join(out, "c"))
    with open(os.path.join(out, "c", "batches.jsonl"), "w") as fh:
        def sink(rec):
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
        tot = instrumented(d, head, images, dev, samples, sink)
    write_atomic(os.path.join(out, "c", "summary.json"), {"device": str(dev), "summary": tot})
    return 0


def read_lines(path: str):
    if not os.path.exists(path):
        return None, "missing"
    try:
        return [json.loads(x) for x in open(path) if x.strip()], None
    except ValueError:
        return None, "truncated"


def finalize(out: str) -> Dict:
    """No model: completeness of each segment, A / B / C aligned batch by batch, report.json (complete or partial)."""
    res: Dict = {"segments": {}}
    ident = os.path.join(out, "identity.json")
    res["identity"] = json.load(open(ident)) if os.path.exists(ident) else {"ok": None, "problems": ["no identity record"]}
    ref = json.load(open(REFERENCE))["val"][-1] if os.path.exists(REFERENCE) else {}
    res["reference"] = {"step": ref.get("step"), "real": ref.get("val/mlm_acc_real"), "shuffled": ref.get("val/mlm_acc_shuffled")}
    seg_rows: Dict = {}
    for seg in ("a", "b", "c"):
        rows, why = read_lines(os.path.join(out, seg, "batches.jsonl"))
        summ_path = os.path.join(out, seg, "summary.json")
        state: Dict = {"batches": len(rows) if rows is not None else 0}
        if rows is None:
            state["state"] = why
        elif len(rows) != BATCHES or not os.path.exists(summ_path):
            state["state"] = "partial"
        else:
            state["state"] = "complete"
            seg_rows[seg] = rows
            state["summary"] = json.load(open(summ_path))["summary"]
            per = {m: [r for i, r in enumerate(rows) if (i < BATCHES // 2) == (m == "real")] for m in ("real", "shuffled")}
            key = "hits_align" if seg == "c" else "hits"
            state["recounted"] = {m: sum(r[key] for r in v) / max(1, sum(r["targets_n" if seg != "c" else "targets"] for r in v)) for m, v in per.items()}
        res["segments"][seg] = state
    if len(seg_rows) == 3:
        diffs = []
        for i in range(BATCHES):
            a, b, c = seg_rows["a"][i], seg_rows["b"][i], seg_rows["c"][i]
            for k in ("input_ids", "attention", "target_hash", "pictures"):
                if not a[k] == b[k] == c[k]:
                    diffs.append({"batch": i, "mode": c["mode"], "hash": k})
            if not a["hits"] == b["hits"] == c["hits_align"] == c["hits_align_device"]:
                diffs.append({"batch": i, "mode": c["mode"], "hits": [a["hits"], b["hits"], c["hits_align"], c["hits_align_device"], c["hits_probe"], c["hits_cpu"]]})
        res["input_equal"] = not any("hash" in x for x in diffs)
        res["differences"] = diffs
        res["disagreeing_tokens"] = sum(len(r["disagree"]) for r in seg_rows["c"])
        res["status"] = "complete"
    else:
        res["status"] = "partial: %s" % {s: v["state"] for s, v in res["segments"].items()}
        res["note"] = "no conclusion from a partial run"
    os.makedirs(out, exist_ok=True)
    write_atomic(os.path.join(out, "report.json"), res)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("run", "finalize"))
    ap.add_argument("--checkpoint", default="runs/exp75-panel-n/n")
    ap.add_argument("--out", default="runs/mlm-repro")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    if a.command == "run":
        return run(a.checkpoint, a.out, a.device)
    res = finalize(a.out)
    print(json.dumps({k: v for k, v in res.items() if k != "differences"}, indent=1))
    return 0 if res["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
