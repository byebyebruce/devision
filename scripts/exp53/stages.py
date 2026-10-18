"""exp53 stages (critic EXP53-PREP-01; docs/experiments/2026-10-18-exp53-selector-plan.md). One frozen SigLIP tower and
EXP50-T's pixel path (local_pixels n12 / 0.25 / T -> visual_inputs), EXP51-T's frozen character head
(runs/exp51/head_T.pt, 0b548142...), a trained region selector (scripts/exp53/selector.py). Stages, each its own
process, all writing under runs/exp53 (fresh only):

  start   model-free: identities (EXP50-T, vision X / T / C, EXP51 head_T and its registry entry, questions, oracle
          regions, code files) -> START.json, before any model forward.
  global  MPS: per picture the full picture's 256 SigLIP patches (= the first 256 keys / values of visual_inputs) and
          their whole-picture geometry; per question the instruction text's token embeddings (EXP50-T's frozen
          ModernBERT input embeddings) -> global.npz (about 0.7 GB fp16). Whole pictures, not oracle regions.
  train   CPU: the selector on fit_O + fit_L boxes only (1,152 questions), fixed budget; then one box per question for
          all 4,752 (real and twin pictures alike, the same function) -> selector.pt, boxes.jsonl.
  local   MPS: per picture visual_inputs once; for each question the oracle region (data/exp51/regions.json.gz) and the
          predicted region (exp51 regions.region of the predicted pixel box) gathered -> oracle.npz, auto.npz,
          auto_regions.json.gz.
  readout CPU: the frozen head on both paths -> details_oracle.jsonl (EXP51's donor: the twin's tokens with the
          question's own geometry) and details_auto.jsonl (donor: the twin question's own automatic readout), each
          registered with its identity in registry.json.
  report  model-free: EXP51's readout statistics per path, conditions, oracle reproduction against EXP51 details_T,
          box diagnostics (IoU, other object / field, degenerate boxes), the pre-written branch; with --finalize after
          a failed / stopped stage, whatever is complete and bound, the rest stated.

    uv run python scripts/exp53/stages.py start --deadline EPOCH | global | train | local | readout | report [--finalize CODES]
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List

import numpy as np
import torch

HERE = os.path.dirname(__file__)
OUT = "runs/exp53"
QUESTIONS = "data/exp49/questions.jsonl"
ORACLE_REGIONS, REGIONS_MANIFEST = "data/exp51/regions.json.gz", "data/exp51/MANIFEST.json"
HEAD, EXP51_REGISTRY, EXP51_DETAILS = "runs/exp51/head_T.pt", "runs/exp51/registry.json", "runs/exp51/details_T.jsonl"
HEAD_SHA = "0b548142ed3697742ca9efaec7498e3aaa37d2872415d6c8c77f6f2ba9d0e3fa"
RUN, T_SHA = "runs/exp50-t", {"model.safetensors": "1a8e5439d5c4f64fd1dbce551f9c2df55152b2aea7cc3a93fc01eb5d364ca007",
                              "config.json": "99c6246e201c5c38c7f62c6831050792858b5aecf2b4be352c5cb7cbc3231fc6"}
FILES = ["scripts/exp53/selector.py", "scripts/exp53/stages.py", "scripts/exp53/run.sh", "scripts/exp51/regions.py",
         "scripts/exp51/probe.py", "scripts/exp51/vision_identity.py", "scripts/exp49/probe.py", "scripts/exp41/report.py",
         "src/devision/image.py", "src/devision/model.py", "src/devision/infer.py"]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, rel))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sel = _load("exp53_selector", "selector.py")
reg51 = _load("exp51_regions", "../exp51/regions.py")
p51 = _load("exp51_probe", "../exp51/probe.py")
p49 = p51.p49


def sha(p: str) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, default=str)
    os.replace(tmp, path)


def questions() -> List[Dict]:
    return [json.loads(x) for x in open(QUESTIONS) if x.strip()]


def register(out: str, name: str, files: Dict[str, str]) -> None:
    """registry.json: per finished product its files' sha256 (atomic)."""
    path = os.path.join(out, "registry.json")
    r = json.load(open(path)) if os.path.exists(path) else {}
    r[name] = files
    write_json(path, r)


def check_registered(out: str, name: str) -> Dict[str, str]:
    r = json.load(open(os.path.join(out, "registry.json")))
    if name not in r:
        raise SystemExit("%s not registered" % name)
    for f, h in r[name].items():
        if sha(os.path.join(out, f)) != h:
            raise SystemExit("%s changed after it was registered" % f)
    return r[name]


# ------------------------------------------------------------------ start

def stage_start(deadline: int, out: str = OUT, vision=None) -> Dict:
    path = os.path.join(out, "START.json")
    if os.path.exists(path):
        raise SystemExit("%s exists (fresh only)" % path)
    vid = _load("exp51_vision", "../exp51/vision_identity.py")
    ident = vision if vision is not None else vid.check()
    man = json.load(open(REGIONS_MANIFEST))
    if sha(QUESTIONS) != man["questions_sha256"] or sha(ORACLE_REGIONS) != man["regions_sha256"]:
        raise SystemExit("questions / oracle regions are not EXP51's")
    if sha(HEAD) != HEAD_SHA or json.load(open(EXP51_REGISTRY))["T"]["head_sha256"] != HEAD_SHA:
        raise SystemExit("EXP51-T's head is not 0b548142 / not its registered head")
    weights = {f: sha(os.path.join(RUN, "stage2", f)) for f in T_SHA}
    if weights != T_SHA:
        raise SystemExit("EXP50-T is not 1a8e5439 / 99c6246e")
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    s = {"git_head": git("rev-parse", "HEAD"), "dirty_files": [f for f in FILES if git("status", "--porcelain", "--", f)],
         "files": {f: sha(f) for f in FILES}, "model": {"run": RUN, **weights}, "vision_identity": ident, "head_sha256": HEAD_SHA,
         "questions_sha256": man["questions_sha256"], "oracle_regions_sha256": man["regions_sha256"], "grid": man["grid"],
         "selector": {"width": sel.WIDTH, "heads": sel.HEADS, "lr": sel.LR, "weight_decay": sel.WD, "batch": sel.BATCH,
                      "steps": sel.STEPS, "warmup": sel.WARMUP, "seed": sel.SEED, "device": "cpu",
                      "loss": "L1(cx, cy, w, h) + (1 - GIoU)", "train_sets": ["fit_O", "fit_L"]},
         "started": time.strftime("%F %T"), "deadline": deadline, "deadline_text": time.strftime("%F %T", time.localtime(deadline))}
    os.makedirs(out, exist_ok=True)
    write_json(path, s)
    return s


def started(out: str = OUT) -> str:
    path = os.path.join(out, "START.json")
    if not os.path.exists(path):
        raise SystemExit("no START.json")
    return sha(path)


# ------------------------------------------------------------------ global

def question_words(d, qs: List[Dict]):
    """(embeddings [Q, L, D] float16, mask [Q, L]) of each question's instruction text: the tokenizer's ids without
    special tokens through the model's frozen input embeddings. The question text only."""
    from devision.model import text_encoder
    emb: torch.nn.Embedding = text_encoder(d.model.decision).get_input_embeddings()  # type: ignore[assignment]
    ids = [d.tok(q_text(q), add_special_tokens=False)["input_ids"][:sel.MAX_Q] for q in qs]
    n = max(map(len, ids))
    words = torch.zeros(len(qs), n, emb.weight.shape[1], dtype=torch.float16)
    mask = torch.zeros(len(qs), n, dtype=torch.bool)
    with torch.no_grad():
        for i, x in enumerate(ids):
            words[i, :len(x)] = emb(torch.tensor(x, device=emb.weight.device)).half().cpu()
            mask[i, :len(x)] = True
    return words, mask


_TEXT: Dict[str, str] = {}


def q_text(q: Dict) -> str:
    """The question's natural instruction text, from its source row (the only per-question input of the selector)."""
    return _TEXT[q["key"]]


def load_texts(qs: List[Dict]) -> None:
    src = {"fit_O": "data/exp42/fit_O.jsonl", "fit_L": "data/exp42/fit_L.jsonl", "rl_tune": "data/real_layout/questions_tune.jsonl",
           "rl_tune_L": "data/exp40/rl_tune_L.jsonl"}
    rows = {n: {r["id"]: r for r in map(json.loads, open(p))} for n, p in src.items()}
    for q in qs:
        _TEXT[q["key"]] = rows[q["set"]][q["id"]]["questions"][q["qid"]]["instructions"]


def stage_global(out: str = OUT, data_root: str = "data", device: str = "auto", batch: int = 16) -> Dict:
    from PIL import Image

    from devision.image import patch_geometry, to_pixel_values
    from devision.infer import Decider, best_device
    s_sha = started(out)
    qs = questions()
    load_texts(qs)
    pics = sorted({q["picture"] for q in qs})
    dev = best_device() if device == "auto" else device
    d = Decider.load(os.path.join(RUN, "stage2"), device=dev)
    m = d.model.eval()
    t0 = time.time()
    patches = np.zeros((len(pics), 256, 768), np.float16)
    geom = np.zeros((len(pics), 256, 4), np.float32)
    with torch.no_grad():
        for s in range(0, len(pics), batch):
            chunk = pics[s:s + batch]
            px, sizes = [], []
            for p in chunk:
                with Image.open(os.path.join(data_root, p)) as img:
                    px.append(to_pixel_values(img, 256))
                    sizes.append(img.size)
            pt = m.vision_patches(torch.stack(px).to(dev)).float().cpu().numpy()
            for k, (w, h) in enumerate(sizes):
                patches[s + k] = pt[k]
                geom[s + k] = patch_geometry((0, 0, w, h), w, h, 256, 16).numpy()
    words, mask = question_words(d, qs)
    path = os.path.join(out, "global.npz")
    np.savez(path, patches=patches, geom=geom, words=words.numpy(), mask=mask.numpy(),
             pic_index=np.array([pics.index(q["picture"]) for q in qs]))
    info = {"start_sha256": s_sha, "pictures": len(pics), "questions": len(qs), "seconds": time.time() - t0, "device": dev,
            "words": list(words.shape), "texts_sha256": hashlib.sha256(json.dumps([q_text(q) for q in qs]).encode()).hexdigest()}
    write_json(os.path.join(out, "global.json"), info)
    register(out, "global", {"global.npz": sha(path), "global.json": sha(os.path.join(out, "global.json"))})
    return info


def load_inputs(out: str = OUT):
    check_registered(out, "global")
    z = np.load(os.path.join(out, "global.npz"))
    return sel.Inputs(torch.from_numpy(z["patches"]), torch.from_numpy(z["geom"]), z["pic_index"].tolist(),
                      torch.from_numpy(z["words"]), torch.from_numpy(z["mask"]))


# ------------------------------------------------------------------ train

def stage_train(out: str = OUT, steps: int = sel.STEPS) -> Dict:
    s_sha = started(out)
    qs = questions()
    inp = load_inputs(out)
    targets = torch.tensor([sel.target_of(q["box"], q["width"], q["height"]) for q in qs])
    train_idx = [i for i, q in enumerate(qs) if q["set"] in ("fit_O", "fit_L")]
    t0 = time.time()
    model, stats = sel.train(inp, train_idx, targets, steps=steps, log=print)
    torch.save(model.state_dict(), os.path.join(out, "selector.pt"))
    frac = sel.predict(model, inp, list(range(len(qs))))
    with open(os.path.join(out, "boxes.jsonl"), "w") as fh:
        for q, f in zip(qs, frac):
            px, bad = sel.pixel_box(f, q["width"], q["height"])
            fh.write(json.dumps({"key": q["key"], "frac": f, "box": px, "degenerate": bad}) + "\n")
    info = {"start_sha256": s_sha, **stats, "train_questions": len(train_idx), "families": len({qs[i]["family"] for i in train_idx}),
            "seconds": time.time() - t0, "selector_params": sum(p.numel() for p in model.parameters())}
    write_json(os.path.join(out, "train.json"), info)
    register(out, "selector", {f: sha(os.path.join(out, f)) for f in ("selector.pt", "boxes.jsonl", "train.json")})
    return info


def boxes(out: str = OUT) -> Dict[str, Dict]:
    check_registered(out, "selector")
    return {r["key"]: r for r in map(json.loads, open(os.path.join(out, "boxes.jsonl")))}


# ------------------------------------------------------------------ local

def auto_regions(qs: List[Dict], bx: Dict[str, Dict]) -> List[Dict]:
    return [reg51.region(bx[q["key"]]["box"], q["width"], q["height"]) for q in qs]


def stage_local(out: str = OUT, data_root: str = "data", device: str = "auto", batch: int = 4) -> Dict:
    from PIL import Image

    from devision.image import local_pixels
    from devision.infer import Decider, best_device
    s_sha = started(out)
    qs = questions()
    bx = boxes(out)
    oracle = json.load(gzip.open(ORACLE_REGIONS, "rt"))
    auto = auto_regions(qs, bx)
    with gzip.open(os.path.join(out, "auto_regions.json.gz"), "wt") as fh:
        json.dump([{"key": q["key"], **r} for q, r in zip(qs, auto)], fh)
    dev = best_device() if device == "auto" else device
    m = Decider.load(os.path.join(RUN, "stage2"), device=dev).model.eval()
    by_pic: Dict[str, List[int]] = defaultdict(list)
    for i, q in enumerate(qs):
        by_pic[q["picture"]].append(i)
    pics = sorted(by_pic)
    offs = {}
    flats = {}
    for name, regs in (("oracle", oracle), ("auto", auto)):
        o = np.zeros(len(qs) + 1, np.int64)
        for i, r in enumerate(regs):
            o[i + 1] = o[i] + len(r["index"])
        offs[name], flats[name] = o, np.zeros((int(o[-1]), 768), np.float16)
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(pics), batch):
            chunk = pics[s:s + batch]
            px = []
            for p in chunk:
                with Image.open(os.path.join(data_root, p)) as img:
                    px.append(local_pixels(img, 256, 12, 0.25, "T"))
            kv = m.visual_inputs(torch.stack(px).to(dev))[0].float().cpu().numpy()
            for b, p in enumerate(chunk):
                for i in by_pic[p]:
                    for name, regs in (("oracle", oracle), ("auto", auto)):
                        flats[name][offs[name][i]:offs[name][i + 1]] = kv[b][np.asarray(regs[i]["index"], np.int64)]
    files = {}
    for name in ("oracle", "auto"):
        np.savez(os.path.join(out, name + ".npz"), x=flats[name], offsets=offs[name])
        files[name + ".npz"] = sha(os.path.join(out, name + ".npz"))
    info = {"start_sha256": s_sha, "seconds": time.time() - t0, "device": dev, "tokens": {k: int(v[-1]) for k, v in offs.items()},
            "pixels": "local_pixels(img, 256, 12, 0.25, 'T') -> visual_inputs"}
    write_json(os.path.join(out, "local.json"), info)
    files.update({"local.json": sha(os.path.join(out, "local.json")), "auto_regions.json.gz": sha(os.path.join(out, "auto_regions.json.gz"))})
    register(out, "local", files)
    return info


# ------------------------------------------------------------------ readout

def frozen_head(path: str = HEAD):
    if sha(path) != HEAD_SHA:
        raise SystemExit("head is not 0b548142")
    head = p49.Head(768)
    head.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
    for p in head.parameters():
        p.requires_grad_(False)
    return head.eval()


def stage_readout(out: str = OUT) -> Dict:
    s_sha = started(out)
    check_registered(out, "local")
    qs = questions()
    oracle = json.load(gzip.open(ORACLE_REGIONS, "rt"))
    auto = [r for r in json.load(gzip.open(os.path.join(out, "auto_regions.json.gz"), "rt"))]
    if [r["key"] for r in auto] != [q["key"] for q in qs]:
        raise SystemExit("auto regions are not in question order")
    head = frozen_head()
    key_i = {q["key"]: i for i, q in enumerate(qs)}
    twin = [key_i[q["twin"]] if q["twin"] else i for i, q in enumerate(qs)]
    allq = list(range(len(qs)))
    reg = {}
    for name, regs in (("oracle", oracle), ("auto", auto)):
        z = np.load(os.path.join(out, name + ".npz"))
        tok = p51.Tokens(z["x"], z["offsets"], regs)
        pred = p51.predict(head, tok, allq)
        if name == "oracle":                 # EXP51's donor: the twin's tokens with this question's own geometry
            donor = p51.predict(head, tok, allq, twin)
        else:                                # automatic: the same question on the twin picture, its own prediction
            donor = [pred[j] for j in twin]
        ident = {"path": name, "start_sha256": s_sha, "head_sha256": HEAD_SHA, "questions_sha256": sha(QUESTIONS),
                 "features_sha256": sha(os.path.join(out, name + ".npz")), "model": T_SHA,
                 "regions_sha256": sha(ORACLE_REGIONS) if name == "oracle" else sha(os.path.join(out, "auto_regions.json.gz")),
                 "boxes_sha256": sha(os.path.join(out, "boxes.jsonl")) if name == "auto" else None}
        dsha = p49.write_details(os.path.join(out, "details_%s.jsonl" % name), qs, pred, donor, twin, ident)
        reg["details_%s.jsonl" % name] = dsha
        del tok, z
    register(out, "readout", reg)
    return reg


# ------------------------------------------------------------------ report

def collect(out: str = OUT):
    """(qs, {path: (pred, donor)}, states) from registered, identity-checked details."""
    qs = questions()
    preds, states = {}, {}
    for name in ("oracle", "auto"):
        path = os.path.join(out, "details_%s.jsonl" % name)
        try:
            files = check_registered(out, "readout")
            if files.get(os.path.basename(path)) != sha(path):
                raise SystemExit("details not registered")
            pred, donor, ident = p49.read_details(path, qs)
            if ident.get("path") != name or ident.get("start_sha256") != started(out) or ident.get("head_sha256") != HEAD_SHA \
                    or ident.get("features_sha256") != sha(os.path.join(out, name + ".npz")):
                raise SystemExit("details identity does not bind")
            if name == "auto" and ident.get("boxes_sha256") != check_registered(out, "selector")["boxes.jsonl"]:
                raise SystemExit("auto details are not about the registered boxes")
            preds[name], states[name] = (pred, donor), {"state": "completed"}
        except (SystemExit, OSError, ValueError, KeyError) as e:
            states[name] = {"state": "missing" if not os.path.exists(path) else "invalid", "reason": str(e)}
    return qs, preds, states


def box_diagnostics(qs: List[Dict], bx: Dict[str, Dict]) -> Dict:
    by_pic = defaultdict(list)
    for q in qs:
        by_pic[(q["set"], q["picture"])].append(q)
    out: Dict = {}
    for s in ("fit_O", "fit_L", "rl_tune", "rl_tune_L"):
        sel_q = [q for q in qs if q["set"] == s]
        ious, other, deg = [], Counter(), 0
        for q in sel_q:
            b = bx[q["key"]]["box"]
            ious.append(sel.iou_xywh(b, q["box"]))
            deg += bx[q["key"]]["degenerate"]
            best = max(by_pic[(s, q["picture"])], key=lambda o: sel.iou_xywh(b, o["box"]))
            if best["key"] != q["key"] and sel.iou_xywh(b, best["box"]) > 0:
                other["other_object" if best["field"] == q["field"] else "other_field"] += 1
        ious.sort()
        out[s] = {"n": len(sel_q), "iou_mean": sum(ious) / len(ious), "iou_median": ious[len(ious) // 2],
                  "iou_ge_0.5": sum(i >= 0.5 for i in ious) / len(ious), "best_match_other": dict(other), "degenerate": deg}
    return out


def conditions_of(ro: Dict) -> Dict:
    c = p49.conditions(ro)
    c["fit_O_valid"] = p51.validity(ro, "fit_O")
    return c


def branch(st: Dict, auto_c, oracle_ok) -> str:
    if auto_c is None:
        return "incomplete: the automatic path is missing / invalid (oracle or training scores never stand in for it)"
    if auto_c["invest_original"]:
        return "automatic path readable on the original pictures: prepare connecting these selected, readable contents to ordinary decide (use, comparison, transfer, protection); no release of the probe"
    if not oracle_ok or not (auto_c["validity"] and auto_c["fit_O_valid"]):
        return "not located: the automatic selection did not learn on fit or the oracle path did not reproduce EXP51 -- no long candidate training, no more steps or other thresholds"
    return "oracle reproduced and automatic fit valid, tune below the condition: locate the selection / geometry generalisation first; local pixels are not re-judged unreadable"


def report(out: str = OUT, finalize: str = "") -> Dict:
    qs, preds, states = collect(out)
    y = [p49.labels(q["target"]) for q in qs]
    res: Dict = {"states": states, "paths": {}}
    for name, (pred, donor) in preds.items():
        ro = p49.readout(qs, pred, donor, y)
        res["paths"][name] = {"readout": ro, "conditions": conditions_of(ro), "cut_slice": p51.cut_slice(qs, pred)}
    oracle_ok = False
    if "oracle" in preds:                    # reproduction of EXP51-T's registered per-question readout
        p51_pred, _, _ = p49.read_details(EXP51_DETAILS, qs)
        same = sum(p49.decode(a)["text"] == p49.decode(b)["text"] and p49.decode(a)["valid"] == p49.decode(b)["valid"]
                   for a, b in zip(preds["oracle"][0], p51_pred)) / len(qs)
        e51 = 0.827
        e = res["paths"]["oracle"]["readout"]["tune_O"]["exact"]
        oracle_ok = bool(same >= 0.99 and abs(e - e51) <= 0.01)
        res["oracle_reproduction"] = {"same_strings": same, "tune_O_exact": e, "exp51_tune_O_exact": e51, "reproduced": oracle_ok,
                                      "rule": ">= 99% questions decode to EXP51-T's string and tune_O exact within 0.01 of 0.827"}
    try:
        res["boxes"] = box_diagnostics(qs, boxes(out))
    except (SystemExit, OSError, ValueError, KeyError) as e:
        res["boxes"] = {"state": "invalid", "reason": str(e)}
    auto_c = res["paths"].get("auto", {}).get("conditions")
    res["branch"] = branch(states, auto_c, oracle_ok)
    res["oracle_note"] = "the oracle path is a fidelity / capability control, never the result"
    if finalize:
        res["finalized"] = {"exit_codes": finalize, "note": "the run did not complete; missing parts are not low scores"}
    write_json(os.path.join(out, "report.json"), res)
    lines = ["INCOMPLETE run: %s" % finalize] if finalize else []
    for name, st in states.items():
        if st["state"] != "completed":
            lines.append("%s %s: %s" % (name, st["state"], st["reason"]))
    for name, v in res["paths"].items():
        for s_, x in v["readout"].items():
            lines.append("%-6s %-6s exact %.3f (%s) valid %.3f both %.3f twin %.3f same %.3f real-donor %+.3f %s follows %.3f" % (
                name, s_, x["exact"], {f: None if x["exact:" + f]["value"] is None else round(x["exact:" + f]["value"], 3) for f in p49.FIELDS},
                x["valid"], x["both"], x["twin"], x["same"], x["real_minus_donor"], [round(c, 3) for c in x["real_minus_donor_ci"]], x["donor_follows"]))
        lines.append("%s conditions %s cut slice %s" % (name, v["conditions"], v["cut_slice"]))
    lines.append("oracle reproduction %s" % res.get("oracle_reproduction"))
    lines.append("boxes %s" % res["boxes"])
    lines.append("branch: %s" % res["branch"])
    open(os.path.join(out, "report.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["start", "global", "train", "local", "readout", "report"])
    ap.add_argument("--deadline", type=int, default=0)
    ap.add_argument("--finalize", default="")
    a = ap.parse_args(argv)
    if a.stage == "start":
        print(json.dumps(stage_start(a.deadline), indent=1))
    elif a.stage == "global":
        print(json.dumps(stage_global(), indent=1))
    elif a.stage == "train":
        print(json.dumps(stage_train(), indent=1))
    elif a.stage == "local":
        print(json.dumps(stage_local(), indent=1))
    elif a.stage == "readout":
        print(json.dumps(stage_readout(), indent=1))
    else:
        report(finalize=a.finalize)
    return 0


if __name__ == "__main__":
    sys.exit(main())
