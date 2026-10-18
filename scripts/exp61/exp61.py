"""exp61: repair the automatic region selector with layout coverage from training sources (critic EXP61-PREP-01;
docs/experiments/2026-10-18-exp61-selector-plan.md). Only the selector changes: it starts from EXP53's registered end
(runs/exp53/selector.pt 8f10f1fd) and trains EXP53's fixed recipe (L1 + GIoU, AdamW 1e-3 / wd 0.01, batch 64, seed 0,
100 warm-up, cosine, 2,000 updates, last step kept) on replay (fit_O + fit_L, EXP53's 1,152 boxes and its registered
patches) + the layout variants (data/exp61, train_V: 1,200 boxes). Vision tower, character head, decision model frozen:
the reading runs through devision.reading.Reading with EXP57's end and EXP58's bundle; only its selector module is
swapped. No dev picture, box or value of EXP59 / EXP60 is trained on.

  start   model-free: identities (EXP53 START / registry / selector / global, EXP51 head, EXP57 end, EXP58 bundle, EXP56
          reads, EXP60 START / registry / boxes, data/exp61 MANIFEST and pictures, data/exp49 questions, this code).
  global  MPS: EXP57's frozen vision patches + question-word embeddings for the 1,000 variant pictures / 3,000 questions.
  train   CPU: the new selector (the recipe above), its boxes on every selector set, a new reading bundle.
  reads   MPS, the production read (Reading.read; the old or new selector module): rl_tune original (new; old = EXP56's
          registered live reads), tune_V (old, new, and oracle boxes through exp60's mirrored read), EXP60's 24 pictures
          (new, through Reading.decide: 24 comparison requests).
  report  model-free: per slice and selector EXP53's statistics (all, fields, both objects, changed-value twins, true -
          donor with a family-bootstrap interval), new - old paired by family; box validity on the training sets; the
          pre-written gates and branch.

    uv run python scripts/exp61/exp61.py start --deadline EPOCH | global | train | reads | report [--finalize CODES]
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(__file__)
OUT, E53, E56, E60 = "runs/exp61", "runs/exp53", "runs/exp56", "runs/exp60"
E53_START_SHA = "c640dd9a03cb1e3ec988690965e8839b0a069fcb21dec995adc070829a09c753"
SELECTOR_SHA = "8f10f1fd3de347dd7d5d2c375b7c4660ba97b75e65920e295132845d5cd2518a"
E60_START_SHA = "371d1e5cb012393c69d09d6aca4abd85e350e2a0f9ac31e6a65ccb6c8a76a658"
E60_REGISTRY_SHA = "5f19c98652baa6545942d536b7d4e24c119dddca5a35e141917c613bc27fdd36"
QUESTIONS, V_QUESTIONS, V_MANIFEST = "data/exp49/questions.jsonl", "data/exp61/questions.jsonl", "data/exp61/MANIFEST.json"
REPLAY = ("fit_O", "fit_L")
STEPS = 2000
FILES = ["scripts/exp61/exp61.py", "scripts/exp61/build.py", "scripts/exp61/run.sh", "scripts/exp53/selector.py", "scripts/exp53/stages.py",
         "scripts/exp60/exp60.py", "scripts/exp58/live.py", "src/devision/reading.py", "src/devision/infer.py", "src/devision/model.py",
         "src/devision/image.py", "scripts/exp53/watch.py"]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, rel))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


e60 = _load("exp60", "../exp60/exp60.py")
e59, e58, e57, e54 = e60.e59, e60.e58, e60.e57, e60.e54
sel = _load("exp53_selector", "../exp53/selector.py")
sha, write_json, register, jsonl = e60.sha, e60.write_json, e60.register, e60.jsonl
r41 = e57.r41


def progress(out, stage, done, total):
    e54.progress(out, stage, done, total)


# ------------------------------------------------------------------ identities

def pictures_digest(qs: List[Dict], root: str = "data") -> str:
    pics = sorted({q["picture"] for q in qs})
    return hashlib.sha256(json.dumps({p: sha(os.path.join(root, p)) for p in pics}, sort_keys=True).encode()).hexdigest()


def check_variants(root: str = "data") -> Dict:
    """The variant list against its MANIFEST, and (with root) every picture's bytes against the MANIFEST digest."""
    m = json.load(open(V_MANIFEST))
    if sha(V_QUESTIONS) != m["questions_sha256"]:
        raise SystemExit("data/exp61/questions.jsonl differs from its MANIFEST")
    if root and pictures_digest(jsonl(V_QUESTIONS), root) != m["pictures_sha256"]:
        raise SystemExit("data/exp61 pictures differ from their MANIFEST")
    return {"manifest": sha(V_MANIFEST), "questions": m["questions_sha256"], "pictures": m["pictures_sha256"]}


def bound_e53() -> Dict[str, str]:
    if sha(os.path.join(E53, "START.json")) != E53_START_SHA:
        raise SystemExit("EXP53's START changed")
    reg = json.load(open(os.path.join(E53, "registry.json")))
    for files in reg.values():
        for f, h in files.items():
            if sha(os.path.join(E53, f)) != h:
                raise SystemExit("EXP53's registered %s changed" % f)
    if reg["selector"]["selector.pt"] != SELECTOR_SHA:
        raise SystemExit("EXP53's selector is not 8f10f1fd")
    return {"start": E53_START_SHA, "selector": SELECTOR_SHA, "global.npz": reg["global"]["global.npz"]}


def bound_e60() -> Dict[str, str]:
    if sha(os.path.join(E60, "START.json")) != E60_START_SHA or sha(os.path.join(E60, "registry.json")) != E60_REGISTRY_SHA:
        raise SystemExit("EXP60's START / registry changed")
    for files in json.load(open(os.path.join(E60, "registry.json"))).values():
        for f, h in files.items():
            if sha(os.path.join(E60, f)) != h:
                raise SystemExit("EXP60's registered %s changed" % f)
    return {"start": E60_START_SHA, "registry": E60_REGISTRY_SHA, "oracle_boxes": sha(e60.BOXES)}


def stage_start(deadline: int, out: str = OUT, data_root: str = "data") -> Dict:
    path = os.path.join(out, "START.json")
    if os.path.exists(path):
        raise SystemExit("%s exists (fresh only)" % path)
    st58 = e58.check_run(e59.E58, data_root=data_root, code=False)       # EXP57 end, EXP58 bundle (its selector = EXP53's)
    if st58["bundle"]["selector.pt"] != SELECTOR_SHA:
        raise SystemExit("EXP58's bundle does not hold EXP53's selector")
    ident = {"exp53": bound_e53(), "exp60": bound_e60(), "variants": check_variants(data_root), "exp49_questions": sha(QUESTIONS),
             "exp56_reads": e54.check_registered(E56, "reads")["reads.jsonl"], "model": st58["model"], "bundle": st58["bundle"]}
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    m = json.load(open(V_MANIFEST))
    s = {"git_head": git("rev-parse", "HEAD"), "dirty_files": [f for f in FILES if git("status", "--porcelain", "--", f)],
         "files": {f: sha(f) for f in FILES}, **ident,
         "recipe": {"init": "runs/exp53/selector.pt", "steps": STEPS, "batch": sel.BATCH, "lr": sel.LR, "weight_decay": sel.WD,
                    "warmup": sel.WARMUP, "seed": sel.SEED, "loss": "L1(cx, cy, w, h) + (1 - GIoU)", "device": "cpu",
                    "train": {"replay": list(REPLAY), "variants": "train_V", "questions": {"replay": 1152, "train_V": m["questions"]["train_V"]}}},
         "requests": {"global_pictures": sum(m["pictures"].values()), "reads": {"rl_tune_new": 600, "tune_V_old": 600, "tune_V_new": 600,
                                                                               "tune_V_oracle": 600, "dev24_new_decide": 24}},
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
    if bound_e53() != st["exp53"] or bound_e60() != st["exp60"] or sha(QUESTIONS) != st["exp49_questions"]:
        raise SystemExit("EXP53 / EXP60 / exp49 identities differ from START")
    if check_variants(data_root) != st["variants"]:              # the list / MANIFEST always; the pictures with data_root
        raise SystemExit("data/exp61 differs from START")
    st58 = e58.check_run(e59.E58, data_root=data_root, code=False)
    if st58["model"] != st["model"] or st58["bundle"] != st["bundle"]:
        raise SystemExit("model / bundle differ from START")
    return st


# ------------------------------------------------------------------ global / train

def stage_global(out: str = OUT, data_root: str = "data") -> Dict:
    import numpy as np
    import torch
    from PIL import Image

    from devision.image import patch_geometry, to_pixel_values
    from devision.infer import Decider, best_device
    st = check_start(out, data_root)
    if e58.fz.weights(e58.NEW_RUN) != st["model"]:
        raise SystemExit("EXP57's weights are not START's: refused before loading")
    qs = jsonl(V_QUESTIONS)
    pics = sorted({q["picture"] for q in qs})
    dev = best_device()
    d = Decider.load(os.path.join(e58.NEW_RUN, "stage2"), device=dev)
    m = d.model.eval()
    patches = np.zeros((len(pics), 256, 768), np.float16)
    geom = np.zeros((len(pics), 256, 4), np.float32)
    with torch.no_grad():
        for s in range(0, len(pics), 16):
            chunk = pics[s:s + 16]
            px, sizes = [], []
            for p in chunk:
                with Image.open(os.path.join(data_root, p)) as img:
                    px.append(to_pixel_values(img, 256))
                    sizes.append(img.size)
            pt = m.vision_patches(torch.stack(px).to(dev)).float().cpu().numpy()
            for k, (w, h) in enumerate(sizes):
                patches[s + k] = pt[k]
                geom[s + k] = patch_geometry((0, 0, w, h), w, h, 256, 16).numpy()
            progress(out, "global", s + len(chunk), len(pics))
    from devision.model import text_encoder
    emb: Any = text_encoder(d.model.decision).get_input_embeddings()
    ids = [d.tok(q["text"], add_special_tokens=False)["input_ids"][:sel.MAX_Q] for q in qs]
    n = max(map(len, ids))
    words = np.zeros((len(qs), n, emb.weight.shape[1]), np.float16)
    mask = np.zeros((len(qs), n), bool)
    with torch.no_grad():
        for i, x in enumerate(ids):
            words[i, :len(x)] = emb(torch.tensor(x, device=emb.weight.device)).half().cpu().numpy()
            mask[i, :len(x)] = True
    path = os.path.join(out, "variant_global.npz")
    np.savez(path, patches=patches, geom=geom, words=words, mask=mask, pic_index=np.array([pics.index(q["picture"]) for q in qs]))
    register(out, "global", {"variant_global.npz": sha(path)})
    return {"pictures": len(pics), "questions": len(qs)}


def combined_inputs(out: str):
    """(Inputs, records): EXP53's registered global inputs for the exp49 questions, then the variants', words padded to
    one length; records in the same order (exp49 questions, then data/exp61 questions)."""
    import numpy as np
    import torch
    a = np.load(os.path.join(E53, "global.npz"))
    e54.check_registered(out, "global")
    b = np.load(os.path.join(out, "variant_global.npz"))
    L = max(a["words"].shape[1], b["words"].shape[1])
    pad = lambda w, m: (np.pad(w, ((0, 0), (0, L - w.shape[1]), (0, 0))), np.pad(m, ((0, 0), (0, L - m.shape[1]))))  # noqa: E731
    wa, ma = pad(a["words"], a["mask"])
    wb, mb = pad(b["words"], b["mask"])
    n_pics = a["patches"].shape[0]
    inp = sel.Inputs(torch.from_numpy(np.concatenate([a["patches"], b["patches"]])), torch.from_numpy(np.concatenate([a["geom"], b["geom"]])),
                     a["pic_index"].tolist() + (b["pic_index"] + n_pics).tolist(), torch.from_numpy(np.concatenate([wa, wb])),
                     torch.from_numpy(np.concatenate([ma, mb])))
    return inp, jsonl(QUESTIONS) + jsonl(V_QUESTIONS)


def train_from(init_state: Dict, inp, train_idx: List[int], targets, steps: int = STEPS, on_step=None):
    """EXP53's selector.train loop unchanged, except the model starts from `init_state` (EXP53's end)."""
    import torch
    torch.manual_seed(sel.SEED)
    model = sel.Selector(inp.patches.shape[-1], inp.words.shape[-1])
    model.load_state_dict(init_state, strict=True)
    opt = torch.optim.AdamW(model.parameters(), lr=sel.LR, weight_decay=sel.WD)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / sel.WARMUP) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, steps)))))
    g = torch.Generator().manual_seed(sel.SEED)
    order: List[int] = []
    losses = []
    for step in range(steps):
        if len(order) < sel.BATCH:
            order += torch.randperm(len(train_idx), generator=g).tolist()
        pick, order = order[:sel.BATCH], order[sel.BATCH:]
        idx = [train_idx[k] for k in pick]
        loss = sel.box_loss(model(*inp.batch(idx)), targets[idx])
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite selector loss at step %d" % step)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
        if on_step and (step + 1) % 50 == 0:
            on_step(step + 1)
    return model.eval(), {"steps": steps, "exposures": steps * sel.BATCH, "final_loss_250": sum(losses[-250:]) / min(250, len(losses)) if losses else None}


def stage_train(out: str = OUT, steps: int = STEPS) -> Dict:
    import torch

    from devision.reading import write_bundle
    st = check_start(out, data_root="")
    inp, qs = combined_inputs(out)
    targets = torch.tensor([sel.target_of(q["box"], q["width"], q["height"]) for q in qs])
    train_idx = [i for i, q in enumerate(qs) if q["set"] in REPLAY + ("train_V",)]
    init = torch.load(os.path.join(E53, "selector.pt"), map_location="cpu")
    t0 = time.time()
    model, stats = train_from(init, inp, train_idx, targets, steps, on_step=lambda k: progress(out, "train", k, steps))
    torch.save(model.state_dict(), os.path.join(out, "selector.pt"))
    old = sel.Selector(inp.patches.shape[-1], inp.words.shape[-1])
    old.load_state_dict(init, strict=True)
    with open(os.path.join(out, "boxes.jsonl"), "w") as fh:
        for who, mdl in (("old", old.eval()), ("new", model)):
            for q, f in zip(qs, sel.predict(mdl, inp, list(range(len(qs))))):
                px, bad = sel.pixel_box(f, q["width"], q["height"])
                fh.write(json.dumps({"selector": who, "key": q["key"], "set": q["set"], "box": px, "degenerate": bad}) + "\n")
    bundle = write_bundle(os.path.join(out, "bundle"), os.path.join(e58.NEW_RUN, "stage2"), os.path.join(out, "selector.pt"), e58.HEAD,
                          {"selector": {"path": os.path.join(out, "selector.pt"), "init": SELECTOR_SHA, "recipe": st["recipe"]},
                           "head": {"path": e58.HEAD, "sha256": e58.HEAD_SHA}})
    info = {**stats, "train_questions": len(train_idx), "seconds": time.time() - t0, "selector_sha256": sha(os.path.join(out, "selector.pt")),
            "bundle_reading_json": sha(os.path.join(out, "bundle", "reading.json")), "bundle": bundle["sha256"]}
    write_json(os.path.join(out, "train.json"), info)
    register(out, "selector", {f: sha(os.path.join(out, f)) for f in ("selector.pt", "boxes.jsonl", "train.json")})
    return info


# ------------------------------------------------------------------ reads

def new_selector_sha(out: str) -> str:
    """The trained selector's sha256, after checking that the file, train.json, the new bundle's selector and the
    registry all name the same file (critic EXP61-R1-02)."""
    reg = e54.check_registered(out, "selector")
    actual = sha(os.path.join(out, "selector.pt"))
    tj = json.load(open(os.path.join(out, "train.json")))
    bundle = json.load(open(os.path.join(out, "bundle", "reading.json")))["sha256"]
    if not (actual == reg["selector.pt"] == tj["selector_sha256"] == bundle["selector.pt"] == sha(os.path.join(out, "bundle", "selector.pt"))):
        raise SystemExit("selector.pt / train.json / bundle / registry do not name one selector")
    if tj["bundle_reading_json"] != sha(os.path.join(out, "bundle", "reading.json")):
        raise SystemExit("the bundle's reading.json is not the one train.json recorded")
    return actual


def read_identity(out: str, name: str, new_sha: str, st: Dict) -> Dict:
    """What a read file is about: START, the set, the selector endpoint (old = EXP53's, new = this run's end) or, for
    oracle, the fixed boxes of the variant list START bound."""
    who = name.rsplit("_", 1)[1]
    ident = {"start_sha256": sha(os.path.join(out, "START.json")), "set": name}
    if who == "oracle":
        ident["boxes"] = "data/exp61/questions.jsonl " + st["variants"]["questions"]
    else:
        ident["selector_sha256"] = SELECTOR_SHA if who == "old" else new_sha
    return ident


def templates_of(qs: List[Dict]) -> List[str]:
    from devision.reading import TEMPLATES
    ts = {q["text"] for q in qs}
    return [t for t in TEMPLATES if t in ts]


def stage_reads(out: str = OUT, data_root: str = "data") -> Dict:
    import copy

    import torch

    from devision.reading import Selector, reads_for
    st = check_start(out, data_root)
    new_sha = new_selector_sha(out)                                               # before any load
    reading, st58, pics58 = e58.load_reading(e59.E58, data_root=data_root)      # EXP57 end + EXP58 bundle, verified
    old_sel = reading.selector
    new_sel = copy.deepcopy(old_sel)
    new_sel.load_state_dict(torch.load(os.path.join(out, "selector.pt"), map_location="cpu"), strict=True)
    assert isinstance(new_sel, Selector)
    tune = {r["id"]: r for r in jsonl(e54.TUNE)}
    vq = [q for q in jsonl(V_QUESTIONS) if q["set"] == "tune_V"]
    by_pic: Dict[str, List[Dict]] = defaultdict(list)
    for q in vq:
        by_pic[q["picture"]].append(q)
    vpics = {p: sha(os.path.join(data_root, p)) for p in by_pic}       # check_start just verified them against the MANIFEST
    total, done = 600 + 3 * len(by_pic) + 24, 0

    def tick():
        nonlocal done
        done += 1
        progress(out, "reads", done, total)
    results: Dict[str, List[Dict]] = {"rl_tune_new": [], "tune_V_old": [], "tune_V_new": [], "tune_V_oracle": []}
    reading.selector = new_sel
    for rid in sorted(tune):                                                  # rl_tune original, new selector
        r = tune[rid]
        img = e60.picture(data_root, r["image"], pics58)
        res = reading.read(img, reads_for(r["questions"]))
        results["rl_tune_new"].append({"id": rid, "reads": res["reads"]})
        tick()
    for p in sorted(by_pic):                                                  # tune_V: old, new, oracle
        ts = templates_of(by_pic[p])
        img = e60.picture(data_root, p, vpics)
        for name, s in (("tune_V_old", old_sel), ("tune_V_new", new_sel)):
            reading.selector = s
            results[name].append({"picture": p, "reads": reading.read(img, ts)["reads"]})
            tick()
        boxes = {q["text"]: q["box"] for q in by_pic[p]}
        results["tune_V_oracle"].append({"picture": p, "reads": e60.read_with(reading, img, ts, boxes)["reads"]})
        tick()
    reading.selector = new_sel                                                # EXP60's 24 pictures through the product path
    its = e59.items()
    dev = e59.p_condition(reading, its, data_root, pics58, on_request=lambda n: tick())
    e59.check_diagnostics(dev, its)
    reading.selector = old_sel
    for name, recs in results.items():
        f = "%s.jsonl" % name
        with open(os.path.join(out, f), "w") as fh:
            fh.write(json.dumps({"identity": read_identity(out, name, new_sha, st)}) + "\n")
            fh.write("".join(json.dumps(x) + "\n" for x in recs))
        register(out, "reads_" + name, {f: sha(os.path.join(out, f))})
    f = "details_dev24_new.jsonl"
    ident = {"condition": "dev24_new", "start_sha256": sha(os.path.join(out, "START.json")), "model": st["model"],
             "selector_sha256": new_sha, "temperature": "none (raw)", "path": "Reading.decide(image=actual picture)"}
    with open(os.path.join(out, f), "w") as fh:
        fh.write(json.dumps({"identity": ident}) + "\n" + "".join(json.dumps(r) + "\n" for r in dev))
    register(out, "decide_dev24_new", {f: sha(os.path.join(out, f))})
    return {"pictures": done}


# ------------------------------------------------------------------ report

def exact(read: Dict, target: str) -> bool:
    return bool(read["valid"] and read["decoded"] == target)


FIELDS = ("mass", "speed", "distance")


def _cells(qrecs: List[Dict]):
    """exp49 probe.readout's per-family cells on decoded strings (critic EXP61-R1-01): exact per field; for changed-value
    questions real - donor (own read right minus the twin picture's read equal to THIS question's target) and donor
    follows (the twin picture's read equal to the twin's target); both objects only where the two golds differ; same;
    changed-value twins once per pair (view v0)."""
    by = {q["key"]: q for q in qrecs}
    fam: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))

    def add(f, k, v):
        fam[f][k][0] += float(v)
        fam[f][k][1] += 1
    hit = lambda q, t: bool(q["read"] is not None and q["read"] == t)  # noqa: E731
    for q in qrecs:
        add(q["family"], "exact:" + q["field"], q["right"])
        if q["twin_changed"]:
            d = by.get(q["twin"])
            if d is None:
                raise SystemExit("%s: its twin %s is not in the set" % (q["key"], q["twin"]))
            add(q["family"], "real_minus_donor:" + q["field"], float(q["right"]) - float(hit(d, q["target"])))
            add(q["family"], "donor_follows:" + q["field"], hit(d, d["target"]))
    pairs = defaultdict(list)
    for q in qrecs:
        pairs[(q["picture"], q["field"])].append(q)
    for (_, f), v in pairs.items():
        if len(v) == 2 and v[0]["target"] != v[1]["target"]:
            add(v[0]["family"], "both:" + f, v[0]["right"] and v[1]["right"])
            add(v[0]["family"], "same:" + f, v[0]["read"] == v[1]["read"])
    for q in qrecs:
        d = by.get(q["twin"])
        if q["twin_changed"] and d is not None and q["view"] == "v0":
            add(q["family"], "twin:" + q["field"], q["right"] and d["right"])
    return fam


def _macro(fam, fs, stat) -> Optional[float]:
    vals = []
    for f in FIELDS:
        k = "%s:%s" % (stat, f)
        d = sum(fam[x][k][1] for x in fs if k in fam[x])
        if d:
            vals.append(sum(fam[x][k][0] for x in fs if k in fam[x]) / d)
    return sum(vals) / len(vals) if vals else None


def _clusters(qrecs: List[Dict]) -> Dict[str, set]:
    """Families linked through a shared case (exp49's union-find), keyed by the component root."""
    parent: Dict[str, str] = {}

    def find(a):
        while parent.setdefault(a, a) != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for q in qrecs:
        a, b = find("f:" + q["family"]), find("c:" + q["case"])
        if a != b:
            parent[max(a, b)] = min(a, b)
    cl: Dict[str, set] = defaultdict(set)
    for q in qrecs:
        cl[find("f:" + q["family"])].add(q["family"])
    return cl


def stats(qrecs: List[Dict]) -> Dict:
    """exp49 probe.readout's statistics (macro over the three fields; per-field values with their denominators; the
    real - donor interval by case-linked clusters). A statistic with no legal sample is None (missing, never 0)."""
    fam = _cells(qrecs)
    fams = sorted(fam)
    res: Dict[str, Any] = {s: _macro(fam, fams, s) for s in ("exact", "both", "twin", "same", "real_minus_donor", "donor_follows")}
    for st in ("exact", "both", "twin", "real_minus_donor", "donor_follows"):
        for f in FIELDS:
            k = "%s:%s" % (st, f)
            d = sum(fam[x][k][1] for x in fams if k in fam[x])
            res[k] = {"value": sum(fam[x][k][0] for x in fams if k in fam[x]) / d if d else None, "n": d}
    cl = _clusters(qrecs)
    keys = sorted(cl)
    expand = lambda ks: [f for k in ks for f in sorted(cl[k])]  # noqa: E731
    res["real_minus_donor_ci"] = r41.interval(lambda fs: _macro(fam, fs, "real_minus_donor") or float("nan"), keys, expand)["ci95"] \
        if res["real_minus_donor"] is not None else None
    res["clusters"] = len(keys)
    return res


def paired(new: List[Dict], old: List[Dict], field: Optional[str] = None) -> Dict:
    """new - old exact (macro over the fields, or one field) on the same questions, case-linked clusters resampled."""
    if {q["key"] for q in new} != {q["key"] for q in old} or len(new) != len(old):
        raise SystemExit("new and old are not on the same questions")
    fn_, fo = _cells(new), _cells(old)
    cl = _clusters(new)
    keys = sorted(cl)
    expand = lambda ks: [f for k in ks for f in sorted(cl[k])]  # noqa: E731

    def val(fam, fs):
        if field is None:
            return _macro(fam, fs, "exact")
        k = "exact:" + field
        d = sum(fam[x][k][1] for x in fs if k in fam[x])
        return sum(fam[x][k][0] for x in fs if k in fam[x]) / d if d else None
    fn = lambda fs: (lambda a, b: float("nan") if a is None or b is None else a - b)(val(fn_, fs), val(fo, fs))  # noqa: E731
    res = r41.interval(fn, keys, expand)
    res["clusters"] = len(keys)
    return res


def slice_records(out: str, name: str, qs: List[Dict], by_pic_key, ident: Optional[Dict] = None) -> List[Dict]:
    """A read file's records after its checks: registered and unchanged; identity = `ident` (START, set, the selector
    endpoint or the oracle boxes; at least START); every question's read present once."""
    f = os.path.join(out, "%s.jsonl" % name)
    if e54.check_registered(out, "reads_" + name).get("%s.jsonl" % name) != sha(f):
        raise SystemExit("%s not registered" % name)
    lines = jsonl(f)
    want = ident or {"start_sha256": sha(os.path.join(out, "START.json"))}
    if any(lines[0]["identity"].get(k) != v for k, v in want.items()):
        raise SystemExit("%s: identity does not bind" % name)
    got = {}
    for x in lines[1:]:
        k = by_pic_key(x)
        if k in got:
            raise SystemExit("%s: %s twice" % (name, k))
        got[k] = {r["question"]: r for r in x["reads"]}
    recs = []
    for q in qs:
        k, t = q["_pic"], q["_template"]
        if k not in got or t not in got[k]:
            raise SystemExit("%s: %s %s missing" % (name, k, t))
        rd = got[k][t]
        recs.append(dict(q, read=rd["decoded"] if rd["valid"] else None, right=exact(rd, q["target"])))
    if len(got) != len({q["_pic"] for q in qs}):
        raise SystemExit("%s: extra pictures" % name)
    return recs


def old_rl_tune(qs: List[Dict]) -> List[Dict]:
    """EXP56's registered live reads (old selector) on the rl_tune questions."""
    if e54.check_registered(E56, "reads")["reads.jsonl"] != sha(os.path.join(E56, "reads.jsonl")):
        raise SystemExit("EXP56's reads changed")
    got = {x["id"]: {r["question"]: r for r in x["reads"]} for x in jsonl(os.path.join(E56, "reads.jsonl"))}
    return [dict(q, read=got[q["_pic"]][q["_template"]]["decoded"], right=exact(got[q["_pic"]][q["_template"]], q["target"])) for q in qs]


def question_sets() -> Dict[str, List[Dict]]:
    tune = {r["id"]: r for r in jsonl(e54.TUNE)}
    rl = [dict(q, _pic=q["id"], _template=tune[q["id"]]["questions"][q["qid"]]["instructions"])
          for q in jsonl(QUESTIONS) if q["set"] == "rl_tune"]
    tv = [dict(q, _pic=q["picture"], _template=q["text"]) for q in jsonl(V_QUESTIONS) if q["set"] == "tune_V"]
    return {"rl_tune": rl, "tune_V": tv}


def box_validity(out: str) -> Dict:
    """Per training / dev selector set and selector: mean IoU with the target box and the share whose box holds the
    drawn value (variants: text_box; exp49 sets: IoU only)."""
    qs = {q["key"]: q for q in jsonl(QUESTIONS) + jsonl(V_QUESTIONS)}
    acc: Dict[tuple, List[float]] = defaultdict(list)
    inside: Dict[tuple, List[float]] = defaultdict(list)
    for b in jsonl(os.path.join(out, "boxes.jsonl")):
        q = qs[b["key"]]
        acc[(b["set"], b["selector"])].append(sel.iou_xywh(b["box"], q["box"]))
        if "text_box" in q:
            x, y, w, h = b["box"]
            t = q["text_box"]
            inside[(b["set"], b["selector"])].append(float(x <= t[0] + 1 and y <= t[1] + 1 and x + w >= t[2] - 1 and y + h >= t[3] - 1))
    return {"%s/%s" % k: {"mean_iou": round(sum(v) / len(v), 4), "n": len(v),
                          **({"text_inside": round(sum(inside[k]) / len(inside[k]), 4)} if k in inside else {})} for k, v in sorted(acc.items())}


def invest(x: Dict) -> Any:
    """EXP53's investment line (exp49 probe.conditions): exact >= 0.80, every field >= 0.60, both >= 0.50, twin >= 0.50,
    real - donor >= 0.20 with its lower bound > 0. A missing statistic -> "incomplete" (never a pass)."""
    need = [x.get("exact"), x.get("both"), x.get("twin"), x.get("real_minus_donor"), x.get("real_minus_donor_ci")] + \
        [(x.get("exact:" + f) or {}).get("value") for f in FIELDS]
    if any(v is None for v in need):
        return "incomplete"
    return bool(x["exact"] >= 0.80 and all(x["exact:" + f]["value"] >= 0.60 for f in FIELDS) and x["both"] >= 0.50
                and x["twin"] >= 0.50 and x["real_minus_donor"] >= 0.20 and x["real_minus_donor_ci"][0] > 0)


def gates(res: Dict) -> Dict:
    g: Dict[str, Any] = {}
    tv = res.get("tune_V") or {}
    if tv.get("new") and tv.get("new_minus_old"):
        inv = invest(tv["new"])
        lo = tv["new_minus_old"]["all"]["ci95"][0]
        g["new_layout"] = inv if inv == "incomplete" or lo != lo else bool(inv and lo > 0)
    else:
        g["new_layout"] = "incomplete"
    rn, ro = (res.get("rl_tune") or {}).get("new"), (res.get("rl_tune") or {}).get("old")
    if rn and ro:
        pairs = [(rn["exact"], ro["exact"])] + [(rn["exact:" + f]["value"], ro["exact:" + f]["value"]) for f in FIELDS]
        g["old_layout"] = "incomplete" if any(a is None or b is None for a, b in pairs) else bool(all(a >= b - 0.03 for a, b in pairs))
    else:
        g["old_layout"] = "incomplete"
    orc = tv.get("oracle")
    if orc:
        vals = [orc.get("exact")] + [(orc.get("exact:" + f) or {}).get("value") for f in FIELDS]
        g["oracle_new_layout"] = "incomplete" if any(v is None for v in vals) else bool(vals[0] >= 0.80 and all(v >= 0.60 for v in vals[1:]))
    else:
        g["oracle_new_layout"] = "incomplete"
    dv = res.get("dev24")
    g["dev24_speed"] = "incomplete" if not dv else bool(dv["speed_new"] >= 18 and dv["speed_new_minus_old"]["ci95"][0] > 0)
    return g


def branch(g: Dict) -> str:
    if any(v == "incomplete" for v in g.values()):
        return "incomplete: %s -- missing is not a low score" % [k for k, v in g.items() if v == "incomplete"]
    if g["oracle_new_layout"] is False:
        return "the frozen head does not read the new layout even from the right boxes: region reading unresolved, the selector's part not yet located"
    if g["old_layout"] is False:
        return "old-layout regression beyond 0.03: this selector is not usable"
    if g["new_layout"] is False:
        return "this recipe did not repair the selector on the frozen new-layout slice"
    if g["dev24_speed"] is False:
        return "repaired on the generated new layout, not on EXP60's 24 real pictures: a coverage gap remains"
    return ("selector repaired by this recipe (replay + layout variants; no single augmentation claimed); next, with a "
            "separate permission: EXP58-style live B1 with the new bundle; not a candidate")


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
        new_sha = new_selector_sha(out)
        res["train"] = json.load(open(os.path.join(out, "train.json")))
        res["box_validity"] = box_validity(out)
        res["states"]["train"] = "completed"
    except (SystemExit, OSError, ValueError, KeyError) as e:
        res["states"]["train"] = "missing / invalid: %s" % e
    sets = question_sets()
    for slice_, names in (("rl_tune", {"new": "rl_tune_new"}), ("tune_V", {"old": "tune_V_old", "new": "tune_V_new", "oracle": "tune_V_oracle"})):
        res[slice_] = {}
        recs = {}
        key = (lambda x: x["id"]) if slice_ == "rl_tune" else (lambda x: x["picture"])
        for who, name in names.items():
            try:
                if new_sha is None:
                    raise SystemExit("no verified trained selector")
                recs[who] = slice_records(out, name, sets[slice_], key, read_identity(out, name, new_sha, st))
                res[slice_][who] = stats(recs[who])
                res["states"][name] = "completed"
            except (SystemExit, OSError, ValueError, KeyError) as e:
                res["states"][name] = "missing / invalid: %s" % e
        if slice_ == "rl_tune":
            recs["old"] = old_rl_tune(sets["rl_tune"])
            res[slice_]["old"] = stats(recs["old"])
        if "new" in recs and "old" in recs:
            res[slice_]["new_minus_old"] = {k: paired(recs["new"], recs["old"], None if k == "all" else k) for k in ("all",) + FIELDS}
    try:
        its = e59.items()
        f = os.path.join(out, "details_dev24_new.jsonl")
        if e54.check_registered(out, "decide_dev24_new").get("details_dev24_new.jsonl") != sha(f):
            raise SystemExit("dev24 not registered")
        dl = jsonl(f)
        if dl[0]["identity"].get("start_sha256") != sha(os.path.join(out, "START.json")) or \
                new_sha is None or dl[0]["identity"].get("selector_sha256") != new_sha:
            raise SystemExit("dev24: identity does not bind")
        e57.r41.check_details("exp61/dev24", [it["row"] for it in its], dl[1:])
        lines = {r["sample_id"]: r for r in dl[1:]}
        e59.check_diagnostics(list(lines.values()), its)
        old = {r["sample_id"]: r for r in jsonl(os.path.join(E60, "details_auto.jsonl"))[1:]}
        gold = e59.gold_numbers(its)
        hit = lambda r: {x["question"]: bool(x["valid"] and x["value"] == e59.value_of(gold[r["sample_id"]][x["question"]]))  # noqa: E731
                         for x in r["diagnostics"]["reads"]}
        hn, ho = {k: hit(v) for k, v in lines.items()}, {k: hit(v) for k, v in old.items()}
        fld = lambda t: e60.field_of(t)  # noqa: E731
        res["dev24"] = {"fields_new": {f: sum(v for k in hn for t, v in hn[k].items() if fld(t) == f) for f in ("mass", "speed", "distance")},
                        "fields_old": {f: sum(v for k in ho for t, v in ho[k].items() if fld(t) == f) for f in ("mass", "speed", "distance")},
                        "speed_new": sum(v for k in hn for t, v in hn[k].items() if fld(t) == "speed"),
                        "speed_new_minus_old": e60.speed_difference(ho, hn),
                        "comparison": {k: e59.transitions({i: old[i] for i in old}, {i: lines[i] for i in lines},
                                                          [it["row"]["id"] for it in its if it["kind"] == k]) for k in ("temp", "force")}}
        res["states"]["dev24_new"] = "completed"
    except (SystemExit, OSError, ValueError, KeyError) as e:
        res["states"]["dev24_new"] = "missing / invalid: %s" % e
    res["gates"] = gates(res)
    res["branch"] = branch(res["gates"])
    return write_report(out, res)


def write_report(out: str, res: Dict) -> Dict:
    os.makedirs(out, exist_ok=True)
    write_json(os.path.join(out, "report.json"), res)
    keys = ("finalized", "states", "train", "box_validity", "gates", "branch", "candidate_eligible")
    text = json.dumps({k: res[k] for k in keys if k in res}, indent=1, default=str)
    for s in ("rl_tune", "tune_V"):
        for who, v in (res.get(s) or {}).items():
            if who != "new_minus_old":
                r3 = lambda x: None if x is None else round(x, 3)  # noqa: E731
                text += "\n%s %s: exact %s mass %s speed %s distance %s both %s twin %s real-donor %s %s clusters %s" % (
                    s, who, r3(v["exact"]), *[r3(v["exact:" + f]["value"]) for f in FIELDS], r3(v["both"]), r3(v["twin"]),
                    r3(v["real_minus_donor"]), v["real_minus_donor_ci"], v["clusters"])
        if (res.get(s) or {}).get("new_minus_old"):
            text += "\n%s new-old %s" % (s, {k: "%+.3f [%+.3f, %+.3f]" % (x["value"], *x["ci95"]) for k, x in res[s]["new_minus_old"].items()})
    if res.get("dev24"):
        text += "\ndev24 %s" % json.dumps(res["dev24"])
    open(os.path.join(out, "report.txt"), "w").write(text + "\n")
    print(text)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["start", "global", "train", "reads", "report"])
    ap.add_argument("--deadline", type=int, default=0)
    ap.add_argument("--finalize", default="")
    a = ap.parse_args(argv)
    if a.stage == "start":
        print(json.dumps({k: v for k, v in stage_start(a.deadline).items() if k not in ("files",)}, indent=1))
    elif a.stage == "global":
        print(json.dumps(stage_global(), indent=1))
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
