"""exp69: a local speed-region corrector on top of the frozen c92 selector (critic EXP69-PREP-01 / -R1-ACCEPT-01;
docs/experiments/2026-10-19-exp69-speed-region-plan.md). Frozen: EXP62's selector c92b7b7e, EXP64's head 0e275bdf, EXP57's
vision / decision (962503fb / 09a5b447), grid n12 / 0.25, the region rule. The only trained part: a corrector with c92's
architecture plus a padding mask on the picture tokens, initialised from c92, that looks only at the local-view tokens
of a window around c92's box and returns a box in that window. It runs only for the two speed templates; mass and
distance keep c92's box (their reads must equal c92's string for string). candidate_eligible is always false.

  start     model-free: identities (EXP62 run / c92, EXP64 run / head / registered reads, EXP58 bundle + EXP57 end, EXP60,
            data/exp61, the bound audit a94f8f35 / 62c95dc7, data/exp69 train / dev targets / MANIFEST, code).
  features  MPS: each training picture encoded once; per record the union of its 8 windows' local tokens (float16, disk);
            per sample its positions in that union and region()'s geom against the window; the two templates' words.
  train     CPU: the corrector from c92; L1 + (1 - GIoU) on the window-fraction labels; AdamW 1e-3 / 0.01; batch 64 from
            one seeded permutation; 1,000 updates, 100 warm-up then cosine; seed 0; last step.
  reads     MPS: per dev picture ONE encoding, EXP64's head on: c92's boxes (must reproduce EXP64's registered reads), the
            corrected boxes (speed only), and on rl_tune / tune_V the target-region control (speed = the tight drawn-text
            box; descriptive, not an upper bound). rl_tune, tune_V, dev24.
  report    model-free: validity first; EXP61's statistics, corrected - c92 per field (case clusters); dev24 per field and
            speed corrected - c92 (12 pictures resampled); box localisation (descriptive); the pre-written gates.

    uv run python scripts/exp69/exp69.py start --deadline EPOCH | features | train | reads | report [--finalize CODES]
"""
import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(__file__)
OUT, E62, E64 = "runs/exp69", "runs/exp62", "runs/exp64"
D69 = "data/exp69"
M69_KEYS = ("train.jsonl", "dev_targets.jsonl")
AUDIT = {"data/exp66/audit.details.jsonl": "a94f8f355040f8147236a4a72abea6ef3ffd90dbf1fa8a87af9fde56b3c5a85d",
         "data/exp69/audit.json": "62c95dc7f945130c503a906918bba6525c254efcb238034cd0a75ca0c2014a99"}
LR, WD, BATCH, STEPS, WARMUP, SEED = 1e-3, 0.01, 64, 1000, 100, 0
SPEED = ("What is the average particle speed in sample A?", "What is the average particle speed in sample B?")
FILES = ["scripts/exp69/exp69.py", "scripts/exp69/build.py", "scripts/exp69/audit.py", "scripts/exp69/run.sh", "scripts/exp64/exp64.py",
         "scripts/exp61/exp61.py", "scripts/exp60/exp60.py", "scripts/exp59/exp59.py", "scripts/exp58/live.py", "scripts/exp53/selector.py",
         "src/devision/reading.py", "src/devision/infer.py", "src/devision/model.py", "src/devision/image.py", "scripts/exp53/watch.py"]
GROUPS = {"rl_tune": ["rl_tune_c92", "rl_tune_corr", "rl_tune_target"], "tune_V": ["tune_V_c92", "tune_V_corr", "tune_V_target"],
          "dev24": ["dev24_c92", "dev24_corr"]}
REF = {"rl_tune": ("rl_tune_new", lambda x: x["id"]), "tune_V": ("tune_V_auto_new", lambda x: x["picture"]), "dev24": ("dev24_auto_new", lambda x: x["id"])}
KEYS = {"rl_tune": lambda x: x["id"], "tune_V": lambda x: x["picture"], "dev24": lambda x: x["id"]}


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, rel))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


e64 = _load("exp64", "../exp64/exp64.py")
b69 = _load("exp69_build", "build.py")
e61, e60, e59, e58, e54 = e64.e61, e64.e60, e64.e59, e64.e58, e64.e54
sel = e61.sel
sha, write_json, register, jsonl = e61.sha, e61.write_json, e61.register, e61.jsonl
E64_SHA = {"START.json": "5ac4da33924376e19e0e450d157cccdf204cc60680f9969df7fae300e8b6d8ad",
           "registry.json": "d743f3e290db25d45091cd27fde3335974d3123e7cdf0411ad7ae31674f5dff3",
           "report.json": "ae61798133c40deef6c728790e4dc9f451dcf1fc3224aa86f807e6163bda3bdf",
           "head.pt": "0e275bdf3b47bff9b8786160b090b79ae903734d784c828d5de65e61c6a2d7c3"}
HEAD_SHA, C92 = E64_SHA["head.pt"], e64.E62_SHA["selector.pt"]


# ------------------------------------------------------------------ the corrector

def corrector_forward(m, patches, geom, pmask, words, wmask):
    """devision.reading.Selector.forward with a key padding mask on the picture tokens (pmask: True = token)."""
    import torch
    pos = m.pos(torch.arange(words.size(1), device=words.device))[None]
    q = m.context(m.word(words.float()) + pos, src_key_padding_mask=~wmask)
    img = m.patch(patches.float()) + m.geom(geom.float())
    s = m.query[None].expand(words.size(0), -1, -1)
    s = s + m.ask(s, q, q, key_padding_mask=~wmask, need_weights=False)[0]
    s = s + m.look1(s, img, img, key_padding_mask=~pmask, need_weights=False)[0]
    s = s + m.look2(m.norm(s), img, img, key_padding_mask=~pmask, need_weights=False)[0]
    return torch.sigmoid(m.box(s[:, 0]))


def window_tokens(win: List[float], w: int, h: int) -> Dict[str, List]:
    """region() of the window, local views only: token indices and their geom against the window."""
    from devision.reading import region
    r = region([win[0], win[1], win[2] - win[0], win[3] - win[1]], w, h)
    keep = [i for i, v in enumerate(r["view"]) if v > 0]
    return {"index": [r["index"][i] for i in keep], "geom": [r["geom"][i] for i in keep]}


def to_pixels(frac: List[float], win: List[float], w: int, h: int) -> List[float]:
    """A window-fraction (cx, cy, w, h) to a pixel box [x, y, w, h]: clipped to the window, at least 1 px each side
    (centred, inside the window and the canvas) -- selector.pixel_box's rule in the window's frame."""
    cx, cy, bw, bh = frac
    ww, wh = win[2] - win[0], win[3] - win[1]
    x0, x1 = max(win[0], win[0] + (cx - bw / 2) * ww), min(win[2], win[0] + (cx + bw / 2) * ww)
    y0, y1 = max(win[1], win[1] + (cy - bh / 2) * wh), min(win[3], win[1] + (cy + bh / 2) * wh)
    if x1 - x0 < 1:
        c = min(max((x0 + x1) / 2, win[0] + 0.5), win[2] - 0.5)
        x0, x1 = c - 0.5, c + 0.5
    if y1 - y0 < 1:
        c = min(max((y0 + y1) / 2, win[1] + 0.5), win[3] - 0.5)
        y0, y1 = c - 0.5, c + 0.5
    x0, y0, x1, y1 = max(0.0, x0), max(0.0, y0), min(float(w), x1), min(float(h), y1)
    return [x0, y0, x1 - x0, y1 - y0]


def window_of(box_xywh: List[float], w: int, h: int) -> List[float]:
    x, y, bw, bh = box_xywh
    return b69.window([x, y, x + bw, y + bh], w, h)


# ------------------------------------------------------------------ identities

def bound_e64() -> Dict[str, str]:
    bad = [f for f, hh in E64_SHA.items() if sha(os.path.join(E64, f)) != hh]
    if bad:
        raise SystemExit("EXP64's %s changed" % bad)
    for files in json.load(open(os.path.join(E64, "registry.json"))).values():
        for f, hh in files.items():
            if sha(os.path.join(E64, f)) != hh:
                raise SystemExit("EXP64's registered %s changed" % f)
    return dict(E64_SHA)


def bound_d69() -> Dict[str, str]:
    m = json.load(open(os.path.join(D69, "MANIFEST.json")))
    for f in M69_KEYS:
        if sha(os.path.join(D69, f)) != m["sha256"][f]:
            raise SystemExit("data/exp69/%s differs from its MANIFEST" % f)
    bad = [f for f, hh in AUDIT.items() if sha(f) != hh]
    if bad or m["sources"]["audit_details"] != AUDIT["data/exp66/audit.details.jsonl"] or m["sources"]["exp64_train"] != sha("data/exp64/train.jsonl"):
        raise SystemExit("the audit / training records differ from data/exp69's sources: %s" % bad)
    return {"manifest": sha(os.path.join(D69, "MANIFEST.json")), **m["sha256"], **AUDIT}


def train_pictures() -> List[str]:
    return sorted({r["picture"] for r in jsonl(os.path.join(D69, "train.jsonl"))})


def stage_start(deadline: int, out: str = OUT, data_root: str = "data") -> Dict:
    path = os.path.join(out, "START.json")
    if os.path.exists(path):
        raise SystemExit("%s exists (fresh only)" % path)
    st58 = e58.check_run(e59.E58, data_root=data_root, code=False)
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    tr = jsonl(os.path.join(D69, "train.jsonl"))
    s = {"git_head": git("rev-parse", "HEAD"), "dirty_files": [f for f in FILES if git("status", "--porcelain", "--", f)],
         "files": {f: sha(f) for f in FILES}, "exp62": e64.bound_e62(), "exp64": bound_e64(), "exp60": e61.bound_e60(), "model": st58["model"],
         "bundle": st58["bundle"], "variants": e64.bound_variants(data_root), "data": bound_d69(),
         "train_pictures": e64.pictures_digest([{"picture": p} for p in train_pictures()], data_root),
         "recipe": {"init": "c92 " + C92, "lr": LR, "weight_decay": WD, "batch": BATCH, "steps": STEPS, "warmup": WARMUP, "seed": SEED,
                    "loss": "L1(cx, cy, w, h) + (1 - GIoU) in window fractions", "device": "cpu", "samples": len(tr)},
         "requests": {"train_pictures": len(train_pictures()), "dev_pictures": {"rl_tune": 600, "tune_V": 600, "dev24": 24}},
         "candidate_eligible": False, "started": time.strftime("%F %T"), "deadline": deadline, "deadline_text": time.strftime("%F %T", time.localtime(deadline))}
    os.makedirs(out, exist_ok=True)
    write_json(path, s)
    return s


def check_start(out: str, data_root: str = "data", code: bool = True) -> Dict:
    st = json.load(open(os.path.join(out, "START.json")))
    if code:
        bad = [f for f, hh in st["files"].items() if sha(f) != hh]
        if bad:
            raise SystemExit("code differs from START: %s" % bad)
    if e64.bound_e62() != st["exp62"] or bound_e64() != st["exp64"] or e61.bound_e60() != st["exp60"] or bound_d69() != st["data"]:
        raise SystemExit("EXP62 / EXP64 / EXP60 / data/exp69 differ from START")
    if e64.bound_variants(data_root) != st["variants"]:
        raise SystemExit("data/exp61 differs from START")
    if data_root and e64.pictures_digest([{"picture": p} for p in train_pictures()], data_root) != st["train_pictures"]:
        raise SystemExit("training pictures differ from START")
    return st


# ------------------------------------------------------------------ features / train

def load_reading(data_root: str):
    """EXP57 + EXP58's verified bundle, then c92 as the selector and EXP64's head (both checked)."""
    import torch
    if sha(os.path.join(E62, "selector.pt")) != C92 or sha(os.path.join(E64, "head.pt")) != HEAD_SHA:
        raise SystemExit("c92 / EXP64's head changed")
    reading, st58, pics58 = e58.load_reading(e59.E58, data_root=data_root)
    dev = next(reading.decider.model.parameters()).device
    reading.selector.load_state_dict(torch.load(os.path.join(E62, "selector.pt"), map_location="cpu"), strict=True)
    reading.head.load_state_dict(torch.load(os.path.join(E64, "head.pt"), map_location="cpu"), strict=True)
    reading.selector.to(dev).eval()
    reading.head.to(dev).eval()
    return reading, pics58


def stage_features(out: str = OUT, data_root: str = "data") -> Dict:
    import numpy as np
    import torch
    st = check_start(out, data_root)
    if e58.fz.weights(e58.NEW_RUN) != st["model"]:
        raise SystemExit("EXP57's weights are not START's: refused before loading")
    reading, _ = load_reading(data_root)
    m, cfg = reading.decider.model, reading.decider.cfg
    tr = jsonl(os.path.join(D69, "train.jsonl"))
    by_rec: Dict[str, List[int]] = defaultdict(list)
    for i, s in enumerate(tr):
        by_rec[s["record"]].append(i)
    recs = sorted(by_rec)
    wins = [window_tokens(s["window"], s["width"], s["height"]) for s in tr]
    unions = {r: sorted({j for i in by_rec[r] for j in wins[i]["index"]}) for r in recs}
    off = np.zeros(len(recs) + 1, np.int64)
    off[1:] = np.cumsum([len(unions[r]) for r in recs])
    flat = np.lib.format.open_memmap(os.path.join(out, "union_tokens.npy"), mode="w+", dtype=np.float16, shape=(int(off[-1]), 768))
    rec_of_pic: Dict[str, List[int]] = defaultdict(list)
    for k, r in enumerate(recs):
        rec_of_pic[tr[by_rec[r][0]]["picture"]].append(k)
    from PIL import Image
    for n, (p, ks) in enumerate(sorted(rec_of_pic.items())):
        with Image.open(os.path.join(data_root, p)) as img:
            kv = e64.encode(m, cfg, img.convert("RGB"))
        for k in ks:
            flat[off[k]:off[k + 1]] = kv[torch.tensor(unions[recs[k]], device=kv.device)].half().cpu().numpy()
        e54.progress(out, "features", n + 1, len(rec_of_pic))
    flat.flush()
    pos, geom, soff, srec = [], [], [0], []
    rec_index = {r: k for k, r in enumerate(recs)}
    for i, s in enumerate(tr):
        k = rec_index[s["record"]]
        where = {j: q for q, j in enumerate(unions[s["record"]])}
        pos += [where[j] for j in wins[i]["index"]]
        geom += wins[i]["geom"]
        soff.append(len(pos))
        srec.append(k)
    words, wmask = reading._words(list(SPEED))
    np.savez(os.path.join(out, "samples.npz"), rec_off=off, pos=np.asarray(pos, np.int64), geom=np.asarray(geom, np.float32),
             sample_off=np.asarray(soff, np.int64), sample_rec=np.asarray(srec, np.int64), words=words.float().cpu().numpy(),
             wmask=wmask.cpu().numpy(), label=np.asarray([s["label"] for s in tr], np.float32),
             template=np.asarray([SPEED.index(s["template"]) for s in tr], np.int64))
    register(out, "features", {f: sha(os.path.join(out, f)) for f in ("union_tokens.npy", "samples.npz")})
    return {"pictures": len(rec_of_pic), "records": len(recs), "samples": len(tr), "union_tokens": int(off[-1])}


class Samples:
    def __init__(self, out: str):
        import numpy as np
        e54.check_registered(out, "features")
        self.flat = np.load(os.path.join(out, "union_tokens.npy"), mmap_mode="r")
        self.z = {k: v for k, v in np.load(os.path.join(out, "samples.npz")).items()}

    def __len__(self):
        return len(self.z["sample_rec"])

    def batch(self, idx: List[int]):
        import torch
        z = self.z
        lens = [int(z["sample_off"][i + 1] - z["sample_off"][i]) for i in idx]
        n = max(lens)
        x = torch.zeros(len(idx), n, self.flat.shape[1])
        g = torch.zeros(len(idx), n, 4)
        mask = torch.zeros(len(idx), n, dtype=torch.bool)
        for k, i in enumerate(idx):
            a, b = int(z["sample_off"][i]), int(z["sample_off"][i + 1])
            base = int(z["rec_off"][z["sample_rec"][i]])
            x[k, :b - a] = torch.from_numpy(self.flat[base + z["pos"][a:b]].astype("float32"))
            g[k, :b - a] = torch.from_numpy(z["geom"][a:b])
            mask[k, :b - a] = True
        t = torch.from_numpy(z["template"][idx])
        return x, g, mask, torch.from_numpy(z["words"])[t], torch.from_numpy(z["wmask"])[t], torch.from_numpy(z["label"][idx])


def train_corrector(init_state: Dict, data: "Samples", steps: int = STEPS, on_step=None):
    import torch
    torch.manual_seed(SEED)
    m = sel.Selector(768, 1024)
    m.load_state_dict(init_state, strict=True)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / WARMUP) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, steps)))))
    g = torch.Generator().manual_seed(SEED)
    order: List[int] = []
    losses = []
    for step in range(steps):
        if len(order) < BATCH:
            order += torch.randperm(len(data), generator=g).tolist()
        pick, order = order[:BATCH], order[BATCH:]
        x, gm, pm, w, wm, y = data.batch(pick)
        loss = sel.box_loss(corrector_forward(m, x, gm, pm, w, wm), y)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite corrector loss at step %d" % step)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
        if on_step and (step + 1) % 50 == 0:
            on_step(step + 1)
    return m.eval(), {"steps": steps, "exposures": steps * BATCH, "final_loss_100": sum(losses[-100:]) / min(100, len(losses)) if losses else None}


def stage_train(out: str = OUT) -> Dict:
    import torch
    check_start(out, data_root="")
    data = Samples(out)
    t0 = time.time()
    init = torch.load(os.path.join(E62, "selector.pt"), map_location="cpu")
    m, stats = train_corrector(init, data, on_step=lambda k: e54.progress(out, "train", k, STEPS))
    torch.save(m.state_dict(), os.path.join(out, "corrector.pt"))
    hold = []
    with torch.no_grad():
        for s in range(0, len(data), 256):
            idx = list(range(s, min(len(data), s + 256)))
            x, gm, pm, w, wm, y = data.batch(idx)
            p = corrector_forward(m, x, gm, pm, w, wm)
            hold += sel.giou(sel.cxcywh_to_xyxy(p), sel.cxcywh_to_xyxy(y)).tolist()
    info = {**stats, "seconds": time.time() - t0, "corrector_sha256": sha(os.path.join(out, "corrector.pt")),
            "train_giou_mean": sum(hold) / len(hold)}
    write_json(os.path.join(out, "train.json"), info)
    register(out, "corrector", {f: sha(os.path.join(out, f)) for f in ("corrector.pt", "train.json")})
    return info


def corrector_sha(out: str) -> str:
    reg = e54.check_registered(out, "corrector")
    tj = json.load(open(os.path.join(out, "train.json")))
    actual = sha(os.path.join(out, "corrector.pt"))
    if not (actual == reg["corrector.pt"] == tj["corrector_sha256"]):
        raise SystemExit("corrector.pt / train.json / registry do not name one corrector")
    return actual


# ------------------------------------------------------------------ reads

def corrected_boxes(reading, corr, kv, w: int, h: int, c92_reads: List[Dict]) -> Dict[str, List[float]]:
    """Speed templates: c92's box -> window -> local tokens -> corrector -> box; any other template: c92's box."""
    import torch
    out = {}
    for r in c92_reads:
        if r["question"] not in SPEED:
            out[r["question"]] = list(r["box"])
            continue
        win = window_of(r["box"], w, h)
        wt = window_tokens(win, w, h)
        dev = kv.device
        x = kv[torch.tensor(wt["index"], device=dev)].half().float()[None]
        g = torch.tensor(wt["geom"], dtype=torch.float32, device=dev)[None]
        pm = torch.ones(1, len(wt["index"]), dtype=torch.bool, device=dev)
        words, wmask = reading._words([r["question"]])
        with torch.no_grad():
            f = corrector_forward(corr, x, g, pm, words, wmask)[0].float().cpu().tolist()
        out[r["question"]] = to_pixels(f, win, w, h)
    return out


def identity(out: str, name: str, corr_sha: str) -> Dict:
    src = name.rsplit("_", 1)[1]
    ident = {"start_sha256": sha(os.path.join(out, "START.json")), "set": name, "selector_sha256": C92, "head_sha256": HEAD_SHA}
    if src == "corr":
        ident["corrector_sha256"] = corr_sha
    if src == "target":
        ident["targets"] = "data/exp69/dev_targets.jsonl " + json.load(open(os.path.join(D69, "MANIFEST.json")))["sha256"]["dev_targets.jsonl"]
    return ident


def write_reads(out: str, name: str, recs: List[Dict], corr_sha: str) -> None:
    f = "%s.jsonl" % name
    with open(os.path.join(out, f), "w") as fh:
        fh.write(json.dumps({"identity": identity(out, name, corr_sha)}) + "\n" + "".join(json.dumps(x) + "\n" for x in recs))
    register(out, "reads_" + name, {f: sha(os.path.join(out, f))})


def reference(group: str) -> Dict[str, Dict[str, Dict]]:
    name, key = REF[group]
    f = os.path.join(E64, name + ".jsonl")
    if e54.check_registered(E64, "reads_" + name).get(name + ".jsonl") != sha(f):
        raise SystemExit("EXP64's %s is not registered" % name)
    lines = jsonl(f)
    if lines[0]["identity"].get("head_sha256") != HEAD_SHA or lines[0]["identity"].get("selector_sha256") != C92:
        raise SystemExit("EXP64's %s is not c92 + 0e275bdf" % name)
    return {key(x): {r["question"]: r for r in x["reads"]} for x in lines[1:]}


def others_identical(c92: List[Dict], corr: List[Dict]) -> List[Dict]:
    """Mass / distance reads of the corrected path must equal c92's (decoded, valid, value, box)."""
    bad = []
    for x, y in zip(c92, corr):
        for a, b in zip(x["reads"], y["reads"]):
            if a["question"] != b["question"] or (a["question"] not in SPEED and (a["decoded"], a["valid"], a["value"], a["box"]) != (b["decoded"], b["valid"], b["value"], b["box"])):
                bad.append({"key": KEYS_ANY(x), "question": a["question"]})
    return bad


def KEYS_ANY(x: Dict) -> str:
    return x.get("id") or x.get("picture") or ""


def close_group(out: str, group: str, results: Dict[str, List[Dict]], corr_sha: str) -> Dict:
    """Files written and registered; then c92's reproduction of EXP64 and the corrected path's identical mass / distance
    recorded (check_<group>.json, registered); a failure raises only after both are on disk."""
    for n in GROUPS[group]:
        write_reads(out, n, results[n], corr_sha)
    rep = e64.reproduce(results[group + "_c92"], reference(group), REF[group][1])
    other = others_identical(results[group + "_c92"], results[group + "_corr"])
    chk = {"c92_reproduces_exp64": {k: v for k, v in rep.items() if k != "mismatches"} | {"mismatches": rep["mismatches"]}, "others_differ": other}
    f = "check_%s.json" % group
    write_json(os.path.join(out, f), chk)
    register(out, "check_" + group, {f: sha(os.path.join(out, f))})
    if not rep["reproduced"] or other:
        raise SystemExit("%s: c92 does not reproduce EXP64 (%d) or mass / distance changed (%d) -- stopped" % (group, len(rep["mismatches"]), len(other)))
    return chk


def stage_reads(out: str = OUT, data_root: str = "data") -> Dict:
    import torch

    from devision.reading import reads_for
    check_start(out, data_root)
    c_sha = corrector_sha(out)                                                    # before any load
    reading, pics58 = load_reading(data_root)
    dev = next(reading.decider.model.parameters()).device
    corr = sel.Selector(768, 1024)
    corr.load_state_dict(torch.load(os.path.join(out, "corrector.pt"), map_location="cpu"), strict=True)
    corr.to(dev).eval()
    heads = {"head": reading.head}
    m, cfg = reading.decider.model, reading.decider.cfg
    tgt = {(t["set"], t["key"], t["template"]): t["target"] for t in jsonl(os.path.join(D69, "dev_targets.jsonl"))}
    tune = {r["id"]: r for r in jsonl(e54.TUNE)}
    vby: Dict[str, List[Dict]] = defaultdict(list)
    for q in jsonl(e61.V_QUESTIONS):
        if q["set"] == "tune_V":
            vby[q["picture"]].append(q)
    vpics = {p: sha(os.path.join(data_root, p)) for p in vby}
    total, done = len(tune) + len(vby) + 24, 0
    results: Dict[str, List[Dict]] = defaultdict(list)

    def three(img, kv, ts, slice_, key, rec_key):
        w, h = img.size
        a = e64.read_heads(reading, kv, w, h, ts, None, heads)["head"]
        cb = corrected_boxes(reading, corr, kv, w, h, a)
        c = e64.read_heads(reading, kv, w, h, ts, cb, heads)["head"]
        results[slice_ + "_c92"].append({rec_key: key, "reads": a})
        results[slice_ + "_corr"].append({rec_key: key, "reads": c})
        if slice_ != "dev24":
            tb = {r["question"]: (list(map(float, [tgt[(slice_, key, r["question"])][0], tgt[(slice_, key, r["question"])][1],
                                                   tgt[(slice_, key, r["question"])][2] - tgt[(slice_, key, r["question"])][0],
                                                   tgt[(slice_, key, r["question"])][3] - tgt[(slice_, key, r["question"])][1]]))
                                  if r["question"] in SPEED else list(r["box"])) for r in a}
            results[slice_ + "_target"].append({rec_key: key, "reads": e64.read_heads(reading, kv, w, h, ts, tb, heads)["head"]})

    def tick():
        nonlocal done
        done += 1
        e54.progress(out, "reads", done, total)
    for rid in sorted(tune):
        img = e60.picture(data_root, tune[rid]["image"], pics58)
        three(img, e64.encode(m, cfg, img), reads_for(tune[rid]["questions"]), "rl_tune", rid, "id")
        tick()
    close_group(out, "rl_tune", results, c_sha)
    for p in sorted(vby):
        img = e60.picture(data_root, p, vpics)
        three(img, e64.encode(m, cfg, img), e61.templates_of(vby[p]), "tune_V", p, "picture")
        tick()
    close_group(out, "tune_V", results, c_sha)
    for it in e59.items():
        row = it["row"]
        img = e60.picture(data_root, row["image"], pics58)
        three(img, e64.encode(m, cfg, img), reads_for(row["questions"]), "dev24", row["id"], "id")
        tick()
    close_group(out, "dev24", results, c_sha)
    return {"pictures": done}


# ------------------------------------------------------------------ report

def read_items(out: str, name: str, corr_sha: str) -> List[Dict]:
    f = os.path.join(out, name + ".jsonl")
    if e54.check_registered(out, "reads_" + name).get(name + ".jsonl") != sha(f):
        raise SystemExit("%s not registered" % name)
    lines = jsonl(f)
    want = identity(out, name, corr_sha)
    if lines[0]["identity"] != want:
        raise SystemExit("%s: identity does not bind" % name)
    return lines[1:]


def templates_by_key() -> Dict[str, Dict[str, List[str]]]:
    t = e64.templates_by_key()
    return {"rl_tune": t["rl_tune"], "tune_V": t["tune_V_auto"], "dev24": t["dev24_auto"]}


def validity(out: str, corr_sha: Optional[str]) -> Dict:
    """Every read file present, registered, bound, complete; c92 reproducing EXP64 and mass / distance identical,
    both recomputed here. missing -> incomplete; anything else -> invalid."""
    res: Dict[str, Any] = {"groups": {}, "missing": [], "invalid": []}
    want = templates_by_key()
    for group, names in GROUPS.items():
        items: Dict[str, List[Dict]] = {}
        for n in names:
            if not os.path.exists(os.path.join(out, n + ".jsonl")):
                res["missing"].append(n)
                continue
            try:
                if corr_sha is None:
                    raise SystemExit("no verified corrector")
                items[n] = read_items(out, n, corr_sha)
                seen = set()
                for x in items[n]:
                    k = KEYS[group](x)
                    if k in seen or k not in want[group]:
                        raise SystemExit("%s: %s twice / unknown" % (n, k))
                    seen.add(k)
                    e64.check_reads_list(x["reads"], want[group][k])
                if seen != set(want[group]):
                    raise SystemExit("%s: pictures missing" % n)
            except SystemExit as e:
                res["invalid"].append("%s: %s" % (n, e))
        if group + "_c92" in items:
            rep = e64.reproduce(items[group + "_c92"], reference(group), KEYS[group])
            res["groups"][group] = {"reproduced": rep["reproduced"], "mismatches": len(rep["mismatches"])}
            if not rep["reproduced"]:
                res["invalid"].append("%s: c92 does not reproduce EXP64" % group)
            if group + "_corr" in items:
                c = {KEYS[group](x): x for x in items[group + "_c92"]}
                other = others_identical([c[KEYS[group](y)] for y in items[group + "_corr"]], items[group + "_corr"])
                res["groups"][group]["others_differ"] = len(other)
                if other:
                    res["invalid"].append("%s: mass / distance reads differ from c92" % group)
    res["state"] = "invalid" if res["invalid"] else "incomplete" if res["missing"] else "valid"
    return res


def no_worse(new: Dict, ref: Dict) -> Any:
    pairs = [(new["exact"], ref["exact"])] + [(new["exact:" + f]["value"], ref["exact:" + f]["value"]) for f in e61.FIELDS]
    if any(a is None or b is None for a, b in pairs):
        return "incomplete"
    return bool(all(a >= b - 0.03 for a, b in pairs))


def gates(res: Dict) -> Dict:
    g: Dict[str, Any] = {}
    for gate, s in (("old_layout", "rl_tune"), ("exp61_layout", "tune_V")):
        x = res.get(s) or {}
        g[gate] = no_worse(x["corr"], x["c92"]) if x.get("corr") and x.get("c92") else "incomplete"
    dv = res.get("dev24")
    g["dev24_speed"] = "incomplete" if not dv else bool(dv["corr"]["speed"] >= 18 and dv["speed_corr_minus_c92"]["ci95"][0] > 0)
    return g


def branch(g: Dict, valid: Optional[Dict] = None) -> str:
    if valid is not None and valid["state"] != "valid":
        return "%s: the run's required evidence is %s (%s) -- no verdict" % (valid["state"], valid["state"], "; ".join(valid["invalid"] or ["missing %s" % valid["missing"]]))
    missing = [k for k, v in g.items() if v == "incomplete"]
    if missing:
        return "incomplete: %s -- missing is not a low score" % missing
    if g["old_layout"] is False:
        return "old-layout regression beyond 0.03 against c92 + the same head: this corrector is not usable"
    if g["exp61_layout"] is False:
        return "EXP61-layout (tune_V) regression beyond 0.03 against c92 + the same head: this corrector is not usable"
    if g["dev24_speed"] is False:
        return "generated layouts hold, real speed does not reach 18 / 24 with a paired gain over c92: this corrector does not close the gap"
    return ("the speed corrector passes with EXP64's head: next (separate permission) the live reading + decision combination; "
            "dev24 / the dev slices are repeatedly used development diagnostics, not a confirmation; not a candidate")


def localisation(out: str, corr_sha: str) -> Dict:
    """Descriptive: on rl_tune / tune_V, the share of corrected / c92 speed boxes holding the tight target, and mean IoU."""
    tgt = {(t["set"], t["key"], t["template"]): t["target"] for t in jsonl(os.path.join(D69, "dev_targets.jsonl"))}
    res: Dict[str, Any] = {}
    for s in ("rl_tune", "tune_V"):
        for who in ("c92", "corr"):
            hold, ious = [], []
            for x in read_items(out, "%s_%s" % (s, who), corr_sha):
                for r in x["reads"]:
                    if r["question"] in SPEED:
                        t = tgt[(s, KEYS[s](x), r["question"])]
                        bx, by, bw, bh = r["box"]
                        b = [bx, by, bx + bw, by + bh]
                        hold.append(b69.inside(t, [b[0] - 1, b[1] - 1, b[2] + 1, b[3] + 1]))
                        ix = max(0.0, min(b[2], t[2]) - max(b[0], t[0]))
                        iy = max(0.0, min(b[3], t[3]) - max(b[1], t[1]))
                        i = ix * iy
                        ious.append(i / ((b[2] - b[0]) * (b[3] - b[1]) + (t[2] - t[0]) * (t[3] - t[1]) - i))
            res["%s/%s" % (s, who)] = {"holds_target": round(sum(hold) / len(hold), 4), "mean_iou": round(sum(ious) / len(ious), 4), "n": len(hold)}
    return res


def report(out: str = OUT, finalize: str = "") -> Dict:
    res: Dict[str, Any] = {"states": {}, "candidate_eligible": False}
    if finalize:
        res["finalized"] = {"exit_codes": finalize, "note": "missing items are not low scores"}
    if not os.path.exists(os.path.join(out, "START.json")):
        res["branch"] = "incomplete: no START (nothing ran)"
        return write_report(out, res)
    try:
        check_start(out, data_root="", code=False)
    except (SystemExit, OSError, ValueError, KeyError) as e:
        res["branch"] = "invalid: the run's identity does not verify (%s) -- no verdict" % e
        return write_report(out, res)
    c_sha = None
    try:
        c_sha = corrector_sha(out)
        res["train"] = json.load(open(os.path.join(out, "train.json")))
        res["states"]["train"] = "completed"
    except (SystemExit, OSError, ValueError, KeyError) as e:
        res["states"]["train"] = "missing / invalid: %s" % e
    sets = e61.question_sets()
    recs: Dict[str, List[Dict]] = {}
    for group in ("rl_tune", "tune_V"):
        for n in GROUPS[group]:
            try:
                if c_sha is None:
                    raise SystemExit("no verified corrector")
                recs[n] = e61.slice_records(out, n, sets[group], KEYS[group], identity(out, n, c_sha))
                res["states"][n] = "completed"
            except (SystemExit, OSError, ValueError, KeyError) as e:
                res["states"][n] = "missing / invalid: %s" % e
        res[group] = {n.rsplit("_", 1)[1]: e61.stats(recs[n]) for n in GROUPS[group] if n in recs}
        if group + "_corr" in recs and group + "_c92" in recs:
            res[group]["corr_minus_c92"] = {f: e61.paired(recs[group + "_corr"], recs[group + "_c92"], None if f == "all" else f) for f in ("all",) + e61.FIELDS}
    try:
        hc, hn = e64.dev24_hits(out, "dev24_c92", HEAD_SHA), e64.dev24_hits(out, "dev24_corr", HEAD_SHA)
        cnt = lambda hs: {f: sum(v for k in hs for t, v in hs[k].items() if e60.field_of(t) == f) for f in e61.FIELDS}  # noqa: E731
        res["dev24"] = {"c92": cnt(hc), "corr": cnt(hn), "speed_corr_minus_c92": e60.speed_difference(hc, hn)}
        res["states"]["dev24"] = "completed"
    except (SystemExit, OSError, ValueError, KeyError) as e:
        res["states"]["dev24"] = "missing / invalid: %s" % e
    if c_sha is not None and all(n in recs for n in ("rl_tune_corr", "rl_tune_c92", "tune_V_corr", "tune_V_c92")):
        try:
            res["localisation"] = localisation(out, c_sha)
        except (SystemExit, OSError, ValueError, KeyError) as e:
            res["localisation"] = "invalid: %s" % e
    res["validity"] = validity(out, c_sha)
    res["gates"] = gates(res)
    res["branch"] = branch(res["gates"], res["validity"])
    return write_report(out, res)


def write_report(out: str, res: Dict) -> Dict:
    os.makedirs(out, exist_ok=True)
    write_json(os.path.join(out, "report.json"), res)
    text = json.dumps({k: res[k] for k in ("finalized", "states", "validity", "gates", "branch", "candidate_eligible") if k in res}, indent=1, default=str)
    for s in ("rl_tune", "tune_V"):
        for who, v in (res.get(s) or {}).items():
            if isinstance(v, dict) and "exact" in v:
                text += "\n%s %s: exact %s %s" % (s, who, v["exact"] and round(v["exact"], 4), {f: v["exact:" + f]["value"] and round(v["exact:" + f]["value"], 3) for f in e61.FIELDS})
    if "dev24" in res:
        text += "\ndev24 c92 %s corr %s" % (res["dev24"]["c92"], res["dev24"]["corr"])
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
