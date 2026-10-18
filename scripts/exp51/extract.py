"""exp51 local region features (critic EXP51-PREP-01). One frozen SigLIP tower -- the one EXP48-X, EXP50-T and EXP50-C
share tensor for tensor (scripts/exp51/vision_identity.py, checked before loading) -- loaded from EXP50-T's frozen end
(1a8e5439 / 99c6246e), eval / no_grad. Per picture, the real product path: image.local_pixels(picture, 256, 12, 0.25,
source) and model.visual_inputs -> the full picture's 256 SigLIP patches then the 12 views' (3,328 tokens, the
conditioning's keys / values before its projection). Condition T: source "T" (views cut from the original picture);
condition C: source "C" (views cut from the picture reduced to full-letterbox detail); nothing else differs. For every
exp49 question only its region tokens (scripts/exp51/regions.py) are kept, ragged: a flat float16 array [tokens, 768]
with offsets, plus rel / geom / view per token.

Pictures go in batches of 4 (52 images per tower call); only the region tokens are kept, one condition at a time, then
written (about 1.1 GB per condition) -- no padding to the largest region, no model graph. Writes
runs/exp51/features_<T|C>.npz and features.json (model / vision identity, questions, regions, per-picture sha256,
preprocessing, shapes, file sha256, timing, MPS / available memory after the first batch).

    uv run python scripts/exp51/extract.py
"""
import argparse
import gzip
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(__file__)
RUN = "runs/exp50-t"
T_SHA = {"model.safetensors": "1a8e5439d5c4f64fd1dbce551f9c2df55152b2aea7cc3a93fc01eb5d364ca007",
         "config.json": "99c6246e201c5c38c7f62c6831050792858b5aecf2b4be352c5cb7cbc3231fc6"}


def sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def gather(kv: np.ndarray, reg: dict) -> np.ndarray:
    """The region's tokens from one picture's visual_inputs keys / values [3328, 768], in the region's order."""
    return kv[np.asarray(reg["index"], dtype=np.int64)]


def extract(model, qs, regs, pictures, root: str, source: str, device: str, batch: int = 4, log=print):
    """(flat float16 [tokens, 768], offsets [questions + 1]) for one condition."""
    from PIL import Image

    from devision.image import local_pixels
    by_pic: dict = {}
    for i, q in enumerate(qs):
        by_pic.setdefault(q["picture"], []).append(i)
    offsets = np.zeros(len(qs) + 1, np.int64)
    for i, r in enumerate(regs):
        offsets[i + 1] = offsets[i] + len(r["index"])
    flat = None
    t0, mem = time.time(), None
    with torch.no_grad():
        for s in range(0, len(pictures), batch):
            chunk = pictures[s:s + batch]
            px = []
            for p in chunk:
                with Image.open(os.path.join(root, p)) as img:
                    px.append(local_pixels(img, 256, 12, 0.25, source))
            kv = model.visual_inputs(torch.stack(px).to(device))[0].float().cpu().numpy()    # [B, 3328, 768]
            if flat is None:
                flat = np.zeros((int(offsets[-1]), kv.shape[2]), np.float16)
            for b, p in enumerate(chunk):
                for i in by_pic[p]:
                    flat[offsets[i]:offsets[i + 1]] = gather(kv[b], regs[i])
            if mem is None:
                mem = memory()
                log("first batch %.1f s, memory %s" % (time.time() - t0, mem))
    assert flat is not None
    return flat, offsets, {"seconds": time.time() - t0, "memory_first_batch": mem}


def memory() -> dict:
    out = {}
    if torch.backends.mps.is_available():
        out["mps_driver_gb"] = torch.mps.driver_allocated_memory() / 2 ** 30
    try:
        import psutil
        out["available_gb"] = psutil.virtual_memory().available / 2 ** 30
    except ImportError:
        pass
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", default="data/exp49/questions.jsonl")
    ap.add_argument("--regions", default="data/exp51")
    ap.add_argument("--run", default=RUN)
    ap.add_argument("--out", default="runs/exp51")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch", type=int, default=4)
    a = ap.parse_args(argv)
    from devision.infer import Decider, best_device
    sys.path.insert(0, HERE)
    import vision_identity  # type: ignore
    ident = vision_identity.check()                       # T / C / X vision tensors identical, before any model load
    man = json.load(open(os.path.join(a.regions, "MANIFEST.json")))
    rpath = os.path.join(a.regions, "regions.json.gz")
    if sha(a.questions) != man["questions_sha256"] or sha(rpath) != man["regions_sha256"]:
        raise SystemExit("questions / regions are not the MANIFEST's")
    ck = os.path.join(a.run, "stage2")
    if {f: sha(os.path.join(ck, f)) for f in T_SHA} != T_SHA:
        raise SystemExit("EXP50-T is not 1a8e5439 / 99c6246e")
    spath = os.path.join(a.out, "START.json")                # written by start.py before any model forward (R1-02)
    if not os.path.exists(spath):
        raise SystemExit("no %s: run scripts/exp51/start.py first" % spath)
    st = json.load(open(spath))
    if st["questions_sha256"] != man["questions_sha256"] or st["regions_sha256"] != man["regions_sha256"] \
            or {f: st["model"][f] for f in T_SHA} != T_SHA or st["vision_identity"] != ident:
        raise SystemExit("START.json does not bind these questions / regions / model / vision tower")
    qs = [json.loads(x) for x in open(a.questions) if x.strip()]
    regs = json.load(gzip.open(rpath, "rt"))
    if [r["key"] for r in regs] != [q["key"] for q in qs]:
        raise SystemExit("regions are not in question order")
    pictures = sorted({q["picture"] for q in qs})
    device = best_device() if a.device == "auto" else a.device
    model = Decider.load(ck, device=device).model.eval()
    info = {"start_sha256": sha(spath), "model": {"run": a.run, **T_SHA}, "vision_identity": ident, "questions_sha256": man["questions_sha256"],
            "regions_sha256": man["regions_sha256"], "device": device, "grid": man["grid"],
            "pictures_sha256": hashlib.sha256(json.dumps({p: sha(os.path.join(a.data_root, p)) for p in pictures}, sort_keys=True).encode()).hexdigest(),
            "preprocessing": "devision.image.local_pixels(img, 256, 12, 0.25, source) -> model.visual_inputs", "conditions": {}}
    for source in ("T", "C"):
        flat, offsets, timing = extract(model, qs, regs, pictures, a.data_root, source, device, a.batch)
        path = os.path.join(a.out, "features_%s.npz" % source)
        np.savez(path, x=flat, offsets=offsets)
        info["conditions"][source] = {"shape": list(flat.shape), "sha256": sha(path), **timing}
        del flat
        tmp = os.path.join(a.out, "features.json.tmp")                              # after each condition, atomic
        json.dump(info, open(tmp, "w"), indent=1)
        os.replace(tmp, os.path.join(a.out, "features.json"))
        print(source, json.dumps(info["conditions"][source]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
