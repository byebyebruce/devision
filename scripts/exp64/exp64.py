"""exp64: the character head's cross-layout adaptation, everything else frozen (critic EXP64-PREP-01;
docs/experiments/2026-10-19-exp64-head-plan.md). Fixed: EXP62's selector c92b7b7e (a research control only, never a
product end), EXP57's vision / decision (962503fb / 09a5b447), grid n12 / 0.25, the region rule. The only trained part:
the character head (devision.reading.CharHead), initialised from EXP51-T's 0b548142, which is also the baseline.

  start     model-free: identities (EXP62 run / selector / registered boxes and reads, EXP51 head, EXP58 bundle + EXP57
            end, EXP60 oracle boxes / details, EXP56 reads, data/exp64 MANIFEST / train / plan / audit, sources, code).
  features  MPS: EXP57's frozen visual inputs of the 1,360 training pictures; per training record the tokens of its
            region (EXP62's box), float16, to a disk array (no dev cache).
  train     the head from 0b548142: AdamW lr 1e-4 / wd 0.01, 50-update linear warm-up then constant, seed 0, the fixed
            plan (1,000 x 64: 24 parent / crop pairs + 16 replay), EXP49's per-sample slot CE, last step; fit validity.
  reads     MPS: per dev picture ONE visual encoding, both heads on the same tokens: rl_tune (EXP62 selector boxes),
            tune_V (EXP62 selector boxes and the oracle boxes), dev24 (EXP62 selector boxes and EXP60's oracle boxes);
            the old head must reproduce EXP62's / EXP60's registered strings on every slice (else stop); dev24's 24
            comparison requests with the new head through ordinary decide.
  report    model-free: EXP61's statistics; old / new head x auto / oracle; the pre-written gates and branch.

    uv run python scripts/exp64/exp64.py start --deadline EPOCH | features | train | reads | report [--finalize CODES]
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(__file__)
OUT, E62, E60 = "runs/exp64", "runs/exp62", "runs/exp60"
E62_SHA = {"START.json": "57f91d898811ec7ee8d3315036952f9d96c19c04764ca7f0a6d77341bb1750da",
           "registry.json": "87a73bb18c185823e8c38cb20e2d5c5671ef9571fc17aa06eece1c2657df925f",
           "report.json": "db231acf06d92b49b5832d6c4fad5c9a59bc23cc423fde6ecc3934afa9648fa1",
           "selector.pt": "c92b7b7e322ab6c07ea9c7ea44149fec6f3cf94beb5c1916d721d481cb2ebefd"}
HEAD_SHA = "0b548142ed3697742ca9efaec7498e3aaa37d2872415d6c8c77f6f2ba9d0e3fa"
D = "data/exp64"
LR, WD, WARMUP, STEPS, SEED = 1e-4, 0.01, 50, 1000, 0
FILES = ["scripts/exp64/exp64.py", "scripts/exp64/build.py", "scripts/exp64/run.sh", "scripts/exp62/build.py", "scripts/exp61/exp61.py",
         "scripts/exp60/exp60.py", "scripts/exp58/live.py", "scripts/exp49/probe.py", "src/devision/reading.py", "src/devision/infer.py",
         "src/devision/model.py", "src/devision/image.py", "scripts/exp53/watch.py"]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, rel))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


e61 = _load("exp61", "../exp61/exp61.py")
p49 = _load("exp49_probe", "../exp49/probe.py")
b64 = _load("exp64_build", "build.py")
e60, e59, e58, e54 = e61.e60, e61.e59, e61.e58, e61.e54
sha, write_json, register, jsonl = e61.sha, e61.write_json, e61.register, e61.jsonl


# ------------------------------------------------------------------ identities

def bound_e62() -> Dict[str, str]:
    bad = [f for f, h in E62_SHA.items() if sha(os.path.join(E62, f)) != h]
    if bad:
        raise SystemExit("EXP62's %s changed" % bad)
    for files in json.load(open(os.path.join(E62, "registry.json"))).values():
        for f, h in files.items():
            if sha(os.path.join(E62, f)) != h:
                raise SystemExit("EXP62's registered %s changed" % f)
    return dict(E62_SHA)


def bound_data() -> Dict[str, str]:
    m = json.load(open(os.path.join(D, "MANIFEST.json")))
    for k, f in (("train_sha256", "train.jsonl"), ("plan_sha256", "plan.jsonl"), ("audit_sha256", "audit.json")):
        if sha(os.path.join(D, f)) != m[k]:
            raise SystemExit("data/exp64/%s differs from its MANIFEST" % f)
    if m["exp62"]["registry"] != E62_SHA["registry.json"]:
        raise SystemExit("data/exp64 was built on another EXP62")
    return {"manifest": sha(os.path.join(D, "MANIFEST.json")), "train": m["train_sha256"], "plan": m["plan_sha256"], "audit": m["audit_sha256"]}


def pictures_digest(recs: List[Dict], root: str) -> str:
    import hashlib
    pics = sorted({r["picture"] for r in recs})
    return hashlib.sha256(json.dumps({p: sha(os.path.join(root, p)) for p in pics}, sort_keys=True).encode()).hexdigest()


def stage_start(deadline: int, out: str = OUT, data_root: str = "data") -> Dict:
    path = os.path.join(out, "START.json")
    if os.path.exists(path):
        raise SystemExit("%s exists (fresh only)" % path)
    st58 = e58.check_run(e59.E58, data_root=data_root, code=False)
    if sha(e58.HEAD) != HEAD_SHA or st58["bundle"]["head.pt"] != HEAD_SHA:
        raise SystemExit("the head is not EXP51-T's 0b548142")
    recs = jsonl(os.path.join(D, "train.jsonl"))
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    s = {"git_head": git("rev-parse", "HEAD"), "dirty_files": [f for f in FILES if git("status", "--porcelain", "--", f)],
         "files": {f: sha(f) for f in FILES}, "exp62": bound_e62(), "head_sha256": HEAD_SHA, "model": st58["model"], "bundle": st58["bundle"],
         "exp60": e61.bound_e60(), "exp56_reads": e54.check_registered(e61.E56, "reads")["reads.jsonl"], "data": bound_data(),
         "sources": {"exp49": sha(e61.QUESTIONS), "exp61": sha(e61.V_QUESTIONS), "exp62_parents": sha("data/exp62/parents.jsonl")},
         "train_pictures": pictures_digest(recs, data_root),
         "recipe": {"init": HEAD_SHA, "lr": LR, "weight_decay": WD, "warmup": WARMUP, "schedule": "linear warm-up then constant",
                    "steps": STEPS, "batch": "24 pairs + 16 replay (data/exp64/plan.jsonl)", "seed": SEED, "loss": "EXP49 per-sample slot CE"},
         "requests": {"train_pictures": len({r["picture"] for r in recs}), "dev_pictures": 600 + 600 + 24, "dev24_decide": 24},
         "candidate_eligible": False, "started": time.strftime("%F %T"), "deadline": deadline,
         "deadline_text": time.strftime("%F %T", time.localtime(deadline))}
    os.makedirs(out, exist_ok=True)
    write_json(path, s)
    return s


def check_start(out: str, data_root: str = "data", code: bool = True) -> Dict:
    st = json.load(open(os.path.join(out, "START.json")))
    if code:
        bad = [f for f, h in st["files"].items() if sha(f) != h]
        if bad:
            raise SystemExit("code differs from START: %s" % bad)
    if bound_e62() != st["exp62"] or bound_data() != st["data"] or e61.bound_e60() != st["exp60"]:
        raise SystemExit("EXP62 / data/exp64 / EXP60 differ from START")
    if {"exp49": sha(e61.QUESTIONS), "exp61": sha(e61.V_QUESTIONS), "exp62_parents": sha("data/exp62/parents.jsonl")} != st["sources"]:
        raise SystemExit("sources differ from START")
    if sha(e58.HEAD) != HEAD_SHA:
        raise SystemExit("EXP51's head changed")
    if data_root and pictures_digest(jsonl(os.path.join(D, "train.jsonl")), data_root) != st["train_pictures"]:
        raise SystemExit("training pictures differ from START")
    return st


# ------------------------------------------------------------------ features / train

def encode(model, cfg, img):
    """EXP57's frozen visual inputs of one picture (Reading.read's first step): [3328, 768] on the model's device."""
    import torch

    from devision.infer import image_tensor
    dev = next(model.parameters()).device
    with torch.no_grad():
        return model.visual_inputs(image_tensor(img, cfg)[None].to(dev))[0][0]


def stage_features(out: str = OUT, data_root: str = "data") -> Dict:
    import numpy as np
    import torch
    from PIL import Image

    from devision.infer import Decider, best_device
    from devision.reading import region
    st = check_start(out, data_root)
    if e58.fz.weights(e58.NEW_RUN) != st["model"]:
        raise SystemExit("EXP57's weights are not START's: refused before loading")
    recs = jsonl(os.path.join(D, "train.jsonl"))
    regs = [region(r["box"], r["width"], r["height"]) for r in recs]
    off = np.zeros(len(recs) + 1, np.int64)
    off[1:] = np.cumsum([len(g["index"]) for g in regs])
    flat = np.lib.format.open_memmap(os.path.join(out, "train_tokens.npy"), mode="w+", dtype=np.float16, shape=(int(off[-1]), 768))
    by_pic: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(recs):
        by_pic[r["picture"]].append(i)
    d = Decider.load(os.path.join(e58.NEW_RUN, "stage2"), device=best_device())
    m = d.model.eval()
    for n, (p, idx) in enumerate(sorted(by_pic.items())):
        with Image.open(os.path.join(data_root, p)) as img:
            kv = encode(m, d.cfg, img.convert("RGB"))
        for i in idx:
            flat[off[i]:off[i + 1]] = kv[torch.tensor(regs[i]["index"], device=kv.device)].half().cpu().numpy()
        e54.progress(out, "features", n + 1, len(by_pic))
    flat.flush()
    np.savez(os.path.join(out, "train_regions.npz"), off=off, rel=np.concatenate([np.asarray(g["rel"], np.int64) for g in regs]),
             geom=np.concatenate([np.asarray(g["geom"], np.float32) for g in regs]))
    register(out, "features", {f: sha(os.path.join(out, f)) for f in ("train_tokens.npy", "train_regions.npz")})
    return {"pictures": len(by_pic), "tokens": int(off[-1])}


class Tokens:
    """Per training record: its region tokens (float16 on disk), relative rows / columns and geometry."""

    def __init__(self, out: str):
        import numpy as np
        e54.check_registered(out, "features")
        self.flat = np.load(os.path.join(out, "train_tokens.npy"), mmap_mode="r")
        z = np.load(os.path.join(out, "train_regions.npz"))
        self.off, self.rel, self.geom = z["off"], z["rel"], z["geom"]

    def batch(self, idx: List[int]):
        import torch
        n = max(int(self.off[i + 1] - self.off[i]) for i in idx)
        x = torch.zeros(len(idx), n, self.flat.shape[1])
        mask = torch.zeros(len(idx), n, dtype=torch.bool)
        rel = torch.zeros(len(idx), n, 2, dtype=torch.long)
        geom = torch.zeros(len(idx), n, 4)
        for k, i in enumerate(idx):
            a, b = int(self.off[i]), int(self.off[i + 1])
            x[k, :b - a] = torch.from_numpy(self.flat[a:b].astype("float32"))
            mask[k, :b - a] = True
            rel[k, :b - a] = torch.from_numpy(self.rel[a:b])
            geom[k, :b - a] = torch.from_numpy(self.geom[a:b])
        return x, mask, rel, geom


def load_head(path: str):
    import torch

    from devision.reading import CharHead
    h = CharHead(768)
    h.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
    return h


def train_head(head, tok: Tokens, y, batches: List[List[int]], device: str = "cpu", on_step=None):
    """The head from its given state; AdamW lr 1e-4 / wd 0.01, linear warm-up over 50 updates then constant, the plan's
    batches, EXP49's per-sample slot CE; the last step."""
    import torch
    torch.manual_seed(SEED)
    head = head.to(device).train()
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / WARMUP))
    losses = []
    for step, idx in enumerate(batches):
        x, m, r, g = tok.batch(idx)
        loss = p49.sample_loss(head(x.to(device), m.to(device), r.to(device), g.to(device)), y[idx].to(device))
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite head loss at step %d" % step)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
        if on_step and (step + 1) % 50 == 0:
            on_step(step + 1)
    return head.eval(), {"steps": len(batches), "exposures": sum(map(len, batches)),
                         "final_loss_100": sum(losses[-100:]) / min(100, len(losses)) if losses else None}


def predict(head, tok: Tokens, idx: List[int], device: str = "cpu", batch: int = 128) -> List[str]:
    import torch
    out = []
    with torch.no_grad():
        for s in range(0, len(idx), batch):
            x, m, r, g = tok.batch(idx[s:s + batch])
            for ids in head.to(device)(x.to(device), m.to(device), r.to(device), g.to(device)).argmax(-1).cpu().tolist():
                d = p49.decode(ids)
                out.append(d["text"] if d["valid"] else None)
    return out


def stage_train(out: str = OUT) -> Dict:
    import torch

    from devision.infer import best_device
    check_start(out, data_root="")
    recs = jsonl(os.path.join(D, "train.jsonl"))
    at = {r["key"]: i for i, r in enumerate(recs)}
    batches = [[at[k] for k in b] for b in jsonl(os.path.join(D, "plan.jsonl"))]
    y = torch.tensor([p49.labels(r["target"]) for r in recs])
    tok = Tokens(out)
    dev = str(best_device())
    t0 = time.time()
    head, stats = train_head(load_head(e58.HEAD), tok, y, batches, dev, on_step=lambda k: e54.progress(out, "train", k, len(batches)))
    torch.save(head.cpu().state_dict(), os.path.join(out, "head.pt"))
    old = load_head(e58.HEAD).eval()
    fit = {}
    for who, h in (("old", old), ("new", head.cpu())):
        pred = predict(h, tok, list(range(len(recs))))
        for s_ in ("fit_O", "fit_L", "train_V", "train_P"):
            for f in ("all",) + e61.FIELDS:
                ii = [i for i, r in enumerate(recs) if r["set"] == s_ and (f == "all" or r["field"] == f)]
                fit["%s/%s/%s" % (who, s_, f)] = sum(pred[i] == recs[i]["target"] for i in ii) / len(ii)
    info = {**stats, "seconds": time.time() - t0, "device": dev, "head_sha256": sha(os.path.join(out, "head.pt")), "fit": fit}
    write_json(os.path.join(out, "train.json"), info)
    register(out, "head", {f: sha(os.path.join(out, f)) for f in ("head.pt", "train.json")})
    return info


# ------------------------------------------------------------------ reads

def read_heads(reading, kv, w: int, h: int, templates: List[str], boxes: Optional[Dict[str, List[float]]], heads: Dict[str, Any]) -> Dict[str, List[Dict]]:
    """Reading.read's steps after the encoding (exp60.read_with's mirror), every head on the same region tokens."""
    import torch

    from devision.image import patch_geometry
    from devision.reading import GRID, UNREADABLE, decode, fmt, pixel_box, region
    dev = kv.device
    with torch.no_grad():
        if boxes is None:
            patches = kv[:256].half()[None].expand(len(templates), -1, -1)
            geom = patch_geometry((0, 0, w, h), w, h, GRID["size"], GRID["side"]).to(dev)[None].expand(len(templates), -1, -1)
            words, mask = reading._words(templates)
            given = [pixel_box(f, w, h) for f in reading.selector(patches, geom, words, mask).float().cpu().tolist()]
        else:
            given = [(list(boxes[t]), False) for t in templates]
        regs = [region(b, w, h) for b, _ in given]
        n = max(len(r["index"]) for r in regs)
        x = torch.zeros(len(regs), n, kv.shape[1], device=dev)
        msk = torch.zeros(len(regs), n, dtype=torch.bool, device=dev)
        rel = torch.zeros(len(regs), n, 2, dtype=torch.long, device=dev)
        gm = torch.zeros(len(regs), n, 4, device=dev)
        for i, r in enumerate(regs):
            k = len(r["index"])
            x[i, :k] = kv[torch.tensor(r["index"], device=dev)].half().float()
            msk[i, :k] = True
            rel[i, :k] = torch.tensor(r["rel"], device=dev)
            gm[i, :k] = torch.tensor(r["geom"], dtype=torch.float32, device=dev)
        out = {}
        for name, hd in heads.items():
            ids = hd(x, msk, rel, gm).argmax(-1).cpu().tolist()
            reads = []
            for t, (b, bad), c in zip(templates, given, ids):
                d = decode(c)
                reads.append({"question": t, "box": b, "degenerate": bad, "decoded": d["text"], "valid": d["valid"],
                              "value": fmt(d["text"]) if d["valid"] else UNREADABLE})
            out[name] = reads
    return out


def registered_reads(path: str, key) -> Dict[str, Dict[str, Dict]]:
    return {key(x): {r["question"]: r for r in x["reads"]} for x in jsonl(path)[1:]}


def same_strings(a: List[Dict], b: Dict[str, Dict]) -> bool:
    return all(r["decoded"] == b[r["question"]]["decoded"] and r["valid"] == b[r["question"]]["valid"] for r in a) and len(a) == len(b)


def stage_reads(out: str = OUT, data_root: str = "data") -> Dict:
    import copy

    import torch

    from devision.reading import reads_for, serialise
    from devision.train.evaluate import _probabilities
    st = check_start(out, data_root)
    reg = e54.check_registered(out, "head")
    if sha(os.path.join(out, "head.pt")) != reg["head.pt"]:
        raise SystemExit("head.pt is not the trained one")
    reading, st58, pics58 = e58.load_reading(e59.E58, data_root=data_root)       # EXP57 + EXP58 bundle (old head), verified
    reading.selector.load_state_dict(torch.load(os.path.join(E62, "selector.pt"), map_location="cpu"), strict=True)
    dev = next(reading.decider.model.parameters()).device
    heads = {"old": reading.head, "new": copy.deepcopy(reading.head)}
    heads["new"].load_state_dict(torch.load(os.path.join(out, "head.pt"), map_location="cpu"), strict=True)
    heads["new"].to(dev).eval()
    m, cfg = reading.decider.model, reading.decider.cfg
    tune = {r["id"]: r for r in jsonl(e54.TUNE)}
    vq = [q for q in jsonl(e61.V_QUESTIONS) if q["set"] == "tune_V"]
    vby: Dict[str, List[Dict]] = defaultdict(list)
    for q in vq:
        vby[q["picture"]].append(q)
    vpics = {p: sha(os.path.join(data_root, p)) for p in vby}
    ob = {k: v for k, v in json.load(open(e60.BOXES)).items() if not k.startswith("_")}
    total = len(tune) + len(vby) + 24
    done = 0
    results: Dict[str, List[Dict]] = defaultdict(list)

    def finish(slice_, name_old, ref, key):
        bad = [x for x in results[name_old] if not same_strings(x["reads"], ref[key(x)])]
        if bad:
            raise SystemExit("%s: the old head does not reproduce the registered reads on %d pictures -- stopped" % (slice_, len(bad)))
    for rid in sorted(tune):                                                      # rl_tune, EXP62 selector boxes
        img = e60.picture(data_root, tune[rid]["image"], pics58)
        kv = encode(m, cfg, img)
        r = read_heads(reading, kv, img.size[0], img.size[1], reads_for(tune[rid]["questions"]), None, heads)
        for who in heads:
            results["rl_tune_" + who].append({"id": rid, "reads": r[who]})
        done += 1
        e54.progress(out, "reads", done, total)
    finish("rl_tune", "rl_tune_old", registered_reads(os.path.join(E62, "rl_tune_new.jsonl"), lambda x: x["id"]), lambda x: x["id"])
    for p in sorted(vby):                                                         # tune_V, auto and oracle boxes
        img = e60.picture(data_root, p, vpics)
        kv = encode(m, cfg, img)
        ts = e61.templates_of(vby[p])
        a = read_heads(reading, kv, img.size[0], img.size[1], ts, None, heads)
        o = read_heads(reading, kv, img.size[0], img.size[1], ts, {q["text"]: q["box"] for q in vby[p]}, heads)
        for who in heads:
            results["tune_V_auto_" + who].append({"picture": p, "reads": a[who]})
            results["tune_V_oracle_" + who].append({"picture": p, "reads": o[who]})
        done += 1
        e54.progress(out, "reads", done, total)
    finish("tune_V auto", "tune_V_auto_old", registered_reads(os.path.join(E62, "tune_V_new.jsonl"), lambda x: x["picture"]), lambda x: x["picture"])
    finish("tune_V oracle", "tune_V_oracle_old", registered_reads(os.path.join(E62, "tune_V_oracle.jsonl"), lambda x: x["picture"]), lambda x: x["picture"])
    dev24 = []
    for it in e59.items():                                                        # dev24: auto and EXP60's oracle boxes
        row = it["row"]
        img = e60.picture(data_root, row["image"], pics58)
        kv = encode(m, cfg, img)
        ts = reads_for(row["questions"])
        a = read_heads(reading, kv, img.size[0], img.size[1], ts, None, heads)
        o = read_heads(reading, kv, img.size[0], img.size[1], ts, ob[row["id"]]["boxes"], heads)
        for who in heads:
            results["dev24_auto_" + who].append({"id": row["id"], "reads": a[who]})
            results["dev24_oracle_" + who].append({"id": row["id"], "reads": o[who]})
        from devision.reading import TEMPLATES
        table = [[*TEMPLATES[x["question"]], x["value"]] for x in a["new"]]
        state = serialise(row.get("state_text") or "", table)
        res = reading.decider.decide(state, {it["key"]: row["questions"][it["qid"]]}, image=None)
        dev24.append(e59.record(it, _probabilities(res["answers"][it["key"]]), {"diagnostics": {"reads": a["new"], "table": table, "state": state}}))
        done += 1
        e54.progress(out, "reads", done, total)
    ref_auto = {r["sample_id"]: {x["question"]: x for x in r["diagnostics"]["reads"]} for r in jsonl(os.path.join(E62, "details_dev24_new.jsonl"))[1:]}
    ref_or = {r["sample_id"]: {x["question"]: x for x in r["diagnostics"]["reads"]} for r in jsonl(os.path.join(E60, "details_oracle.jsonl"))[1:]}
    finish("dev24 auto", "dev24_auto_old", ref_auto, lambda x: x["id"])
    finish("dev24 oracle", "dev24_oracle_old", ref_or, lambda x: x["id"])
    s_sha = sha(os.path.join(out, "START.json"))
    for name, recs in results.items():
        f = "%s.jsonl" % name
        ident = {"start_sha256": s_sha, "set": name, "selector_sha256": E62_SHA["selector.pt"],
                 "head_sha256": HEAD_SHA if name.endswith("_old") else reg["head.pt"]}
        with open(os.path.join(out, f), "w") as fh:
            fh.write(json.dumps({"identity": ident}) + "\n" + "".join(json.dumps(x) + "\n" for x in recs))
        register(out, "reads_" + name, {f: sha(os.path.join(out, f))})
    e59.check_diagnostics(dev24, e59.items())
    f = "details_dev24_new_head.jsonl"
    with open(os.path.join(out, f), "w") as fh:
        fh.write(json.dumps({"identity": {"start_sha256": s_sha, "selector_sha256": E62_SHA["selector.pt"], "head_sha256": reg["head.pt"],
                                          "model": st["model"], "temperature": "none (raw)"}}) + "\n" + "".join(json.dumps(r) + "\n" for r in dev24))
    register(out, "decide_dev24", {f: sha(os.path.join(out, f))})
    return {"pictures": done}


# ------------------------------------------------------------------ report

def slice_recs(out: str, name: str, qs: List[Dict], key, head_sha: str) -> List[Dict]:
    ident = {"start_sha256": sha(os.path.join(out, "START.json")), "set": name, "selector_sha256": E62_SHA["selector.pt"], "head_sha256": head_sha}
    return e61.slice_records(out, name, qs, key, ident)


def dev24_hits(out: str, name: str, head_sha: str) -> Dict[str, Dict[str, bool]]:
    its = e59.items()
    gold = e59.gold_numbers(its)
    f = os.path.join(out, name + ".jsonl")
    if e54.check_registered(out, "reads_" + name).get(name + ".jsonl") != sha(f):
        raise SystemExit("%s not registered" % name)
    lines = jsonl(f)
    if lines[0]["identity"].get("head_sha256") != head_sha or lines[0]["identity"].get("start_sha256") != sha(os.path.join(out, "START.json")):
        raise SystemExit("%s: identity does not bind" % name)
    got = {x["id"]: x["reads"] for x in lines[1:]}
    if set(got) != set(gold):
        raise SystemExit("%s: pictures missing" % name)
    return {rid: {x["question"]: bool(x["valid"] and x["value"] == e59.value_of(gold[rid][x["question"]])) for x in rs} for rid, rs in got.items()}


def gates(res: Dict) -> Dict:
    g: Dict[str, Any] = {}
    fit = res.get("fit_validity")
    g["head_learned"] = "incomplete" if fit is None else fit
    tv = res.get("tune_V") or {}
    g["new_layout"] = "incomplete" if not tv.get("auto_new") else e61.invest(tv["auto_new"])
    d = tv.get("new_minus_old_head")
    g["head_attribution"] = "incomplete" if d is None or d["ci95"][0] != d["ci95"][0] else bool(d["ci95"][0] > 0)
    rn, ro = (res.get("rl_tune") or {}).get("new"), (res.get("rl_tune") or {}).get("baseline")
    if rn and ro:
        pairs = [(rn["exact"], ro["exact"])] + [(rn["exact:" + f]["value"], ro["exact:" + f]["value"]) for f in e61.FIELDS]
        g["old_layout"] = "incomplete" if any(a is None or b is None for a, b in pairs) else bool(all(a >= b - 0.03 for a, b in pairs))
    else:
        g["old_layout"] = "incomplete"
    dv = res.get("dev24")
    g["dev24_speed"] = "incomplete" if not dv else bool(dv["speed_new"] >= 18 and dv["speed_new_minus_original"]["ci95"][0] > 0)
    return g


def branch(g: Dict) -> str:
    if g["head_learned"] is False:
        return "the head did not fit its own training data (EXP49 validity line): the bottleneck is not located"
    missing = [k for k, v in g.items() if v == "incomplete"]
    if missing:
        return "incomplete: %s -- missing is not a low score" % missing
    if g["old_layout"] is False:
        return "old-layout regression beyond 0.03 against EXP56: this head is not usable"
    if g["new_layout"] is False:
        return "the adapted head does not reach the new-layout investment line"
    if g["head_attribution"] is False:
        return "new-layout line met but the gain over the frozen old head is not established (lower bound <= 0): not attributed to the head"
    if g["dev24_speed"] is False:
        return "generated layouts pass, EXP60's 24 real pictures do not: a region / coverage gap remains"
    return ("the head adaptation passes with EXP62's selector: next (separate permission) the live reading + decision "
            "combination; EXP62's selector stays a research control, not a product end; not a candidate")


def report(out: str = OUT, finalize: str = "") -> Dict:
    res: Dict[str, Any] = {"states": {}, "candidate_eligible": False}
    if finalize:
        res["finalized"] = {"exit_codes": finalize, "note": "missing items are not low scores"}
    if not os.path.exists(os.path.join(out, "START.json")):
        res["branch"] = "incomplete: no START (nothing ran)"
        return write_report(out, res)
    try:
        st = check_start(out, data_root="", code=False)
    except (SystemExit, OSError, ValueError, KeyError) as e:
        res["branch"] = "invalid: the run's identity does not verify (%s) -- no verdict" % e
        return write_report(out, res)
    new_sha = None
    try:
        reg = e54.check_registered(out, "head")
        tj = json.load(open(os.path.join(out, "train.json")))
        if sha(os.path.join(out, "head.pt")) != reg["head.pt"] or tj["head_sha256"] != reg["head.pt"]:
            raise SystemExit("head.pt / train.json / registry disagree")
        new_sha = reg["head.pt"]
        res["train"] = tj
        f = tj["fit"]
        res["fit_validity"] = bool(all(f["new/%s/all" % s] >= 0.90 and all(f["new/%s/%s" % (s, x)] >= 0.75 for x in e61.FIELDS) for s in ("fit_O", "fit_L")))
        res["states"]["train"] = "completed"
    except (SystemExit, OSError, ValueError, KeyError) as e:
        res["states"]["train"] = "missing / invalid: %s" % e
    sets = e61.question_sets()
    rk, vk = (lambda x: x["id"]), (lambda x: x["picture"])
    recs: Dict[str, List[Dict]] = {}
    for name, slice_, key in (("rl_tune_old", "rl_tune", rk), ("rl_tune_new", "rl_tune", rk), ("tune_V_auto_old", "tune_V", vk),
                              ("tune_V_auto_new", "tune_V", vk), ("tune_V_oracle_old", "tune_V", vk), ("tune_V_oracle_new", "tune_V", vk)):
        try:
            if new_sha is None:
                raise SystemExit("no verified trained head")
            recs[name] = slice_recs(out, name, sets[slice_], key, HEAD_SHA if name.endswith("_old") else new_sha)
            res["states"][name] = "completed"
        except (SystemExit, OSError, ValueError, KeyError) as e:
            res["states"][name] = "missing / invalid: %s" % e
    res["rl_tune"] = {"baseline": e61.stats(e61.old_rl_tune(sets["rl_tune"]))}
    for who in ("old", "new"):
        if "rl_tune_" + who in recs:
            res["rl_tune"]["exp62_selector_" + who + "_head" if who == "old" else "new"] = e61.stats(recs["rl_tune_" + who])
    if "rl_tune_new" in recs:
        base = e61.old_rl_tune(sets["rl_tune"])
        res["rl_tune"]["new_minus_baseline"] = {f: e61.paired(recs["rl_tune_new"], base, None if f == "all" else f) for f in ("all",) + e61.FIELDS}
    res["tune_V"] = {}
    for name in ("auto_old", "auto_new", "oracle_old", "oracle_new"):
        if "tune_V_" + name in recs:
            res["tune_V"][name] = e61.stats(recs["tune_V_" + name])
    if "tune_V_auto_new" in recs and "tune_V_auto_old" in recs:
        res["tune_V"]["new_minus_old_head"] = e61.paired(recs["tune_V_auto_new"], recs["tune_V_auto_old"])
        res["tune_V"]["new_minus_old_head_fields"] = {f: e61.paired(recs["tune_V_auto_new"], recs["tune_V_auto_old"], f) for f in e61.FIELDS}
    try:
        hn = dev24_hits(out, "dev24_auto_new", new_sha or "")
        ho = dev24_hits(out, "dev24_auto_old", HEAD_SHA)
        orig = {r["sample_id"]: r for r in jsonl(os.path.join(E60, "details_auto.jsonl"))[1:]}
        gold = e59.gold_numbers(e59.items())
        horig = {rid: {x["question"]: bool(x["valid"] and x["value"] == e59.value_of(gold[rid][x["question"]])) for x in r["diagnostics"]["reads"]}
                 for rid, r in orig.items()}
        fld = e60.field_of
        cnt = lambda hs: {f: sum(v for k in hs for t, v in hs[k].items() if fld(t) == f) for f in e61.FIELDS}  # noqa: E731
        res["dev24"] = {"fields_new_head": cnt(hn), "fields_old_head": cnt(ho), "fields_original": cnt(horig), "speed_new": cnt(hn)["speed"],
                        "speed_new_minus_original": e60.speed_difference(horig, hn), "speed_new_minus_old_head": e60.speed_difference(ho, hn),
                        "oracle": {who: cnt(dev24_hits(out, "dev24_oracle_" + who, HEAD_SHA if who == "old" else new_sha or "")) for who in ("old", "new")}}
        res["states"]["dev24"] = "completed"
    except (SystemExit, OSError, ValueError, KeyError) as e:
        res["states"]["dev24"] = "missing / invalid: %s" % e
    res["gates"] = gates(res)
    res["branch"] = branch(res["gates"])
    return write_report(out, res)


def write_report(out: str, res: Dict) -> Dict:
    os.makedirs(out, exist_ok=True)
    write_json(os.path.join(out, "report.json"), res)
    text = json.dumps({k: res[k] for k in ("finalized", "states", "fit_validity", "gates", "branch", "candidate_eligible") if k in res}, indent=1, default=str)
    for s in ("rl_tune", "tune_V"):
        for who, v in (res.get(s) or {}).items():
            if isinstance(v, dict) and "exact" in v:
                text += "\n%s %s: exact %s %s" % (s, who, v["exact"] and round(v["exact"], 4), {f: v["exact:" + f]["value"] and round(v["exact:" + f]["value"], 3) for f in e61.FIELDS})
    open(os.path.join(out, "report.txt"), "w").write(text + "\n")
    print(text)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["start", "features", "train", "reads", "report"])
    ap.add_argument("--deadline", type=int, default=0)
    ap.add_argument("--finalize", default="")
    a = ap.parse_args(argv)
    if a.stage == "start":
        print(json.dumps({k: v for k, v in stage_start(a.deadline).items() if k != "files"}, indent=1))
    elif a.stage == "features":
        print(json.dumps(stage_features(), indent=1))
    elif a.stage == "train":
        print(json.dumps(stage_train(), indent=1))
    elif a.stage == "reads":
        print(json.dumps(stage_reads(), indent=1))
    else:
        try:
            report(finalize=a.finalize)
        except Exception as e:
            os.makedirs(OUT, exist_ok=True)
            open(os.path.join(OUT, "report.txt"), "w").write("REPORT FAILED (invalid, no verdict): %s\n" % e)
            print("REPORT FAILED: %s" % e, file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
